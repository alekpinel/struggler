"""M3: card events fire.

Unit tests pin each implemented event's effect; the property test proves a
full game with events enabled still terminates and keeps every mandated
invariant; the golden replay is the diffable regression that events resolve
deterministically through the seeded dice-as-CHANCE decisions.

Events are unit-tested by driving the public decision loop where practical and,
for a fixed board setup, by calling ``engine._fire_event`` on a bare engine —
the same entry point the engine uses internally — so the assertion is about the
effect, not the plumbing that routes to it.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from struggler.cards import action_rounds, cards_entering
from struggler.engine import CHINA_CARD_ID, Engine
from struggler.events import EVENTS
from struggler.replay import run_with_checkpoints
from struggler.types import Action, DecisionKind, Period, Region, Side

MAX_INT32 = 2**31 - 1
REPLAY_DIR = Path(__file__).parent / "replays"


def _bare(seed: int = 0) -> Engine:
    """A minimal engine with the event layer on but no turn loop running."""
    engine = Engine(seed=seed)
    engine.events_enabled = True
    return engine


# -- tier 1: immediate state change -----------------------------------------


def test_duck_and_cover_degrades_defcon_and_scores_us():
    engine = _bare()
    engine.defcon = 5
    engine._fire_event(Side.US, "Duck_and_Cover")
    assert engine.defcon == 4
    assert engine.vp == 1  # 5 - new DEFCON (4)


def test_fidel_hands_cuba_to_the_ussr():
    engine = _bare()
    engine.board.influence["Cuba"] = {"US": 2, "USSR": 0}
    engine._fire_event(Side.USSR, "Fidel")
    assert engine.board.influence["Cuba"]["US"] == 0
    assert engine.board.control("Cuba") is Side.USSR


def test_nasser_adds_two_and_halves_us_rounding_up():
    engine = _bare()
    engine.board.influence["Egypt"] = {"US": 3, "USSR": 0}
    engine._fire_event(Side.USSR, "Nasser")
    assert engine.board.influence["Egypt"]["US"] == 1  # 3 - ceil(3/2)=2
    assert engine.board.influence["Egypt"]["USSR"] == 2


def test_de_gaulle_shifts_france():
    engine = _bare()
    engine.board.influence["France"] = {"US": 3, "USSR": 0}
    engine._fire_event(Side.USSR, "De_Gaulle_Leads_France")
    assert engine.board.influence["France"] == {"US": 1, "USSR": 1}


def test_captured_nazi_scientist_advances_space_race_with_vp():
    engine = _bare()
    engine._fire_event(Side.US, "Captured_Nazi_Scientist")
    assert engine.space_race["US"] == 1
    assert engine.vp == 2  # box 1, first to reach it


def test_nuclear_test_ban_scores_then_improves_defcon():
    engine = _bare()
    engine.defcon = 3
    engine._fire_event(Side.US, "Nuclear_Test_Ban")
    assert engine.vp == 1  # DEFCON 3 - 2
    assert engine.defcon == 5  # +2, clamped at the ceiling


# -- tier 1: the "war" family (seeded CHANCE roll) --------------------------


def _resolve_pending_war(engine: Engine) -> int:
    """Step the pending WAR_ROLL and return the die value it carried."""
    decision = engine.pending_decision
    assert decision is not None and decision.kind is DecisionKind.WAR_ROLL
    action = decision.options[0]
    value = action.payload["value"]
    engine.step(action)
    return value


def test_korean_war_seizes_south_korea_on_a_win():
    engine = _bare(seed=7)
    engine.board.influence["South_Korea"] = {"US": 3, "USSR": 0}  # US-controlled
    engine._fire_event(Side.USSR, "Korean_War")
    assert engine.military_ops["USSR"] == 2  # war always counts as military ops
    roll = _resolve_pending_war(engine)
    # No US-controlled country is adjacent, so there is no roll penalty.
    if roll >= 4:  # USSR wins: it takes over all US influence in the target
        assert engine.board.influence["South_Korea"] == {"US": 0, "USSR": 3}
        assert engine.vp == -2  # +2 for the USSR is negative on the US-positive track
    else:
        assert engine.board.influence["South_Korea"] == {"US": 3, "USSR": 0}
        assert engine.vp == 0


def test_arab_israeli_war_counts_target_control_as_a_penalty():
    # Israel US-controlled and every US-controlled neighbor adds a penalty; with
    # enough penalty the USSR cannot win regardless of the die.
    engine = _bare(seed=1)
    for cid in ("Israel", "Lebanon", "Syria", "Jordan", "Egypt"):
        engine.board.influence[cid] = {"US": 9, "USSR": 0}
    engine._fire_event(Side.USSR, "Arab_Israeli_War")
    _resolve_pending_war(engine)
    # Penalty is at least 5 (target + four neighbors), so even a 6 fails.
    assert engine.board.influence["Israel"]["US"] == 9
    assert engine.vp == 0


# -- tier 3: persistent per-turn modifiers ----------------------------------


def test_containment_boosts_us_ops_only():
    engine = _bare()
    engine._fire_event(Side.US, "Containment")
    duck = engine.cards["Duck_and_Cover"]  # 3 ops
    assert engine._effective_ops(Side.US, duck) == 4
    assert engine._effective_ops(Side.USSR, duck) == 3  # opponent unaffected


def test_red_scare_reduces_opponent_ops_to_a_floor_of_one():
    engine = _bare()
    engine._fire_event(Side.US, "Red_Scare_Purge")  # US plays it -> hurts USSR
    one_op = engine.cards["Nasser"]  # 1 op
    assert engine._effective_ops(Side.USSR, one_op) == 1  # max(1, 1-1)
    assert engine._effective_ops(Side.US, one_op) == 1  # US unaffected


def test_turn_effects_lapse_at_end_of_turn():
    engine = Engine.new_game(seed=3, events=True)
    engine.turn_effects["containment"] = True
    engine._end_of_turn()
    assert engine.turn_effects == {}


# -- tier 2: player-choice events -------------------------------------------


def _drain_event_influence(engine: Engine, taker=lambda opts: opts[0]) -> int:
    """Step through a run of EVENT_INFLUENCE decisions, returning how many."""
    steps = 0
    while (
        engine.pending_decision is not None
        and engine.pending_decision.kind is DecisionKind.EVENT_INFLUENCE
    ):
        engine.step(taker(engine.pending_decision.options))
        steps += 1
    return steps


def _eastern_europe(engine: Engine) -> list[str]:
    return [c for c, i in engine.board.countries.items()
            if i.subregion is not None and i.subregion.value == "EASTERN_EUROPE"]


def test_comecon_places_four_in_non_us_eastern_europe():
    engine = _bare()
    engine.board.influence["East_Germany"] = {"US": 5, "USSR": 0}  # US-controlled
    engine._fire_event(Side.USSR, "COMECON")
    # Every offered country is USSR's choice and never the US-controlled one.
    assert engine.pending_decision.actor is Side.USSR
    assert "East_Germany" not in [
        a.payload["country"] for a in engine.pending_decision.options
    ]
    assert _drain_event_influence(engine) == 4  # one point into each of 4 countries
    placed = sum(
        1 for c in _eastern_europe(engine) if engine.board.influence[c]["USSR"] > 0
    )
    assert placed == 4


def test_marshall_plan_places_seven_and_skips_ussr_controlled():
    engine = _bare()
    engine.board.influence["Italy"] = {"US": 0, "USSR": 5}  # USSR-controlled
    engine._fire_event(Side.US, "Marshall_Plan")
    assert "Italy" not in [a.payload["country"] for a in engine.pending_decision.options]
    assert _drain_event_influence(engine) == 7


def test_suez_crisis_caps_removal_at_two_per_country():
    engine = _bare()
    engine.board.influence["France"] = {"US": 5, "USSR": 0}
    engine.board.influence["UK"] = {"US": 0, "USSR": 0}
    engine.board.influence["Israel"] = {"US": 0, "USSR": 0}
    engine._fire_event(Side.USSR, "Suez_Crisis")
    # Only France has US influence; the 2-per-country cap stops removal at 2 even
    # though the card allows 4 total.
    removed = _drain_event_influence(engine)
    assert removed == 2
    assert engine.board.influence["France"]["US"] == 3


def test_truman_doctrine_only_offers_uncontrolled_europe():
    engine = _bare()
    engine.board.influence["Italy"] = {"US": 1, "USSR": 2}   # uncontrolled
    engine.board.influence["Poland"] = {"US": 0, "USSR": 5}  # USSR-controlled
    engine._fire_event(Side.US, "Truman_Doctrine")
    offered = [a.payload["country"] for a in engine.pending_decision.options]
    assert "Italy" in offered and "Poland" not in offered
    engine.step(Action(DecisionKind.EVENT_INFLUENCE, {"country": "Italy"}))
    assert engine.board.influence["Italy"]["USSR"] == 0  # all USSR removed
    assert engine.pending_decision is None  # single-country event is done


def test_warsaw_pact_remove_branch_clears_us_from_eastern_europe():
    engine = _bare()
    engine.board.influence["East_Germany"] = {"US": 3, "USSR": 0}
    engine.board.influence["Poland"] = {"US": 2, "USSR": 0}
    engine._fire_event(Side.USSR, "Warsaw_Pact_Formed")
    assert engine.pending_decision.kind is DecisionKind.EVENT_CHOICE
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "remove"}))
    # Only two EE countries have US influence, so removal stops after both.
    assert _drain_event_influence(engine) == 2
    assert engine.board.influence["East_Germany"]["US"] == 0
    assert engine.board.influence["Poland"]["US"] == 0


def test_warsaw_pact_add_branch_places_five_capped_at_two():
    engine = _bare()
    engine._fire_event(Side.USSR, "Warsaw_Pact_Formed")
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "add"}))
    # Always take East Germany when offered to prove the per-country cap of 2.
    def prefer_east_germany(opts):
        for a in opts:
            if a.payload["country"] == "East_Germany":
                return a
        return opts[0]
    assert _drain_event_influence(engine, prefer_east_germany) == 5
    assert engine.board.influence["East_Germany"]["USSR"] == 2  # capped


# -- tier 3: persistent game-long legality (NATO family) --------------------


def test_nato_requires_marshall_or_warsaw_first():
    engine = _bare()
    engine._fire_event(Side.US, "NATO")
    assert not engine.game_effects.get("nato")  # precondition unmet -> no effect
    engine.game_effects["marshall_or_warsaw"] = True
    engine._fire_event(Side.US, "NATO")
    assert engine.game_effects.get("nato") is True


def test_nato_blocks_ussr_coup_and_realign_on_us_europe_only():
    engine = _bare()
    engine.game_effects["marshall_or_warsaw"] = True
    engine._fire_event(Side.US, "NATO")
    engine.board.influence["West_Germany"] = {"US": 5, "USSR": 0}  # US-controlled
    ussr_coup = {a.payload["country"] for a in engine._coup_target_options(Side.USSR)}
    ussr_realign = {
        a.payload["country"] for a in engine._realignment_target_options(Side.USSR)
    }
    us_coup = {a.payload["country"] for a in engine._coup_target_options(Side.US)}
    assert "West_Germany" not in ussr_coup
    assert "West_Germany" not in ussr_realign
    assert "West_Germany" in us_coup  # the US is never locked out


def test_de_gaulle_lifts_nato_for_france():
    engine = _bare()
    engine.game_effects["marshall_or_warsaw"] = True
    engine._fire_event(Side.US, "NATO")
    engine.board.influence["France"] = {"US": 5, "USSR": 0}
    assert "France" not in {
        a.payload["country"] for a in engine._coup_target_options(Side.USSR)
    }
    engine._fire_event(Side.USSR, "De_Gaulle_Leads_France")  # removes 2 US, +1 USSR
    engine.board.influence["France"] = {"US": 5, "USSR": 0}  # re-establish US control
    assert "France" in {
        a.payload["country"] for a in engine._coup_target_options(Side.USSR)
    }


def test_us_japan_pact_controls_and_shields_japan():
    engine = _bare()
    engine.board.influence["Japan"] = {"US": 0, "USSR": 4}  # USSR-controlled first
    engine._fire_event(Side.US, "US_Japan_Mutual_Defense_Pact")
    assert engine.board.control("Japan") is Side.US
    assert "Japan" not in {
        a.payload["country"] for a in engine._coup_target_options(Side.USSR)
    }


def test_willy_brandt_scores_and_lifts_nato_for_west_germany():
    engine = _bare()
    engine.game_effects["marshall_or_warsaw"] = True
    engine._fire_event(Side.US, "NATO")
    engine._fire_event(Side.USSR, "Willy_Brandt")
    assert engine.vp == -1  # +1 VP for the USSR
    assert engine.board.influence["West_Germany"]["USSR"] == 1
    engine.board.influence["West_Germany"] = {"US": 5, "USSR": 0}  # US-controlled
    assert "West_Germany" in {
        a.payload["country"] for a in engine._coup_target_options(Side.USSR)
    }


def test_game_effects_persist_across_turns():
    engine = Engine.new_game(seed=5, events=True)
    engine.game_effects["nato"] = True
    engine._end_of_turn()
    assert engine.game_effects.get("nato") is True  # not cleared with turn_effects


# -- tier 4: UN Intervention (rule modifier) --------------------------------


def test_un_intervention_cancels_an_opponent_event_played_for_ops():
    engine = _bare()
    engine.defcon = 5
    engine.hands["USSR"] = ["Duck_and_Cover", "UN_Intervention"]
    modes = engine._play_modes(Side.USSR, "Duck_and_Cover")
    assert "un_intervention" in modes
    _play_card_for(engine, Side.USSR, "Duck_and_Cover", "un_intervention")
    assert engine.defcon == 5  # the US event did NOT fire
    assert "UN_Intervention" in engine.discard_pile  # spent
    assert "Duck_and_Cover" in engine.discard_pile
    assert engine.pending_decision.kind is DecisionKind.OPS_TYPE  # used for Ops


def test_un_intervention_not_offered_without_the_card_or_for_own_event():
    engine = _bare()
    engine.hands["USSR"] = ["Duck_and_Cover"]  # no UN Intervention held
    assert "un_intervention" not in engine._play_modes(Side.USSR, "Duck_and_Cover")
    # Own-side event card never offers it (there is no opponent event to cancel).
    engine.hands["USSR"] = ["Fidel", "UN_Intervention"]
    assert "un_intervention" not in engine._play_modes(Side.USSR, "Fidel")


# -- the China Card's "+1 Op if used entirely in Asia" bonus ----------------


def _play_china_ops(engine: Engine, side: Side) -> None:
    engine.hands[side.value] = [CHINA_CARD_ID]
    engine.china_card_owner = side.value
    engine.china_card_available = True
    _play_card_for(engine, side, CHINA_CARD_ID, "ops")


def test_china_card_grants_five_ops_used_entirely_in_asia():
    engine = _bare()
    engine.board.influence["North_Korea"]["USSR"] = 1  # a reachable Asian foothold
    _play_china_ops(engine, Side.USSR)
    engine.step(Action(DecisionKind.OPS_TYPE, {"type": "influence"}))

    def asian(opts):
        return next(
            a for a in opts
            if engine.board.countries[a.payload["country"]].region is not None
            and engine.board.countries[a.payload["country"]].region.value == "ASIA"
        )
    steps = 0
    while (engine.pending_decision is not None
           and engine.pending_decision.kind is DecisionKind.PLACE_INFLUENCE):
        engine.step(asian(engine.pending_decision.options))
        steps += 1
    assert steps == 5  # 4 base + 1 Asia bonus


def test_china_card_bonus_forfeited_by_leaving_asia():
    engine = _bare()
    engine.board.influence["North_Korea"]["USSR"] = 1
    engine.board.influence["Mexico"]["USSR"] = 1  # a non-Asian foothold too
    _play_china_ops(engine, Side.USSR)
    engine.step(Action(DecisionKind.OPS_TYPE, {"type": "influence"}))
    steps = 0
    while (engine.pending_decision is not None
           and engine.pending_decision.kind is DecisionKind.PLACE_INFLUENCE):
        opts = engine.pending_decision.options
        non_asia = [
            a for a in opts
            if engine.board.countries[a.payload["country"]].region.value != "ASIA"
        ]
        engine.step(non_asia[0] if non_asia else opts[0])
        steps += 1
    assert steps == 4  # leaving Asia forfeits the +1


# -- more registered cards (representative sample) ---------------------------


def test_immediate_fixed_influence_cards():
    engine = _bare()
    engine._fire_event(Side.USSR, "Allende")
    assert engine.board.influence["Chile"]["USSR"] == 2
    engine._fire_event(Side.US, "Panama_Canal_Returned")
    for cid in ("Panama", "Costa_Rica", "Venezuela"):
        assert engine.board.influence[cid]["US"] == 1


def test_camp_david_scores_places_and_blocks_arab_israeli_war():
    engine = _bare()
    engine._fire_event(Side.US, "Camp_David_Accords")
    assert engine.vp == 1
    assert engine.board.influence["Israel"]["US"] == 1
    # Arab-Israeli War is now ineligible, so firing it does nothing.
    engine.board.influence["Israel"] = {"US": 0, "USSR": 0}
    engine._fire_event(Side.USSR, "Arab_Israeli_War")
    assert engine.pending_decision is None  # no war roll enqueued


def test_solidarity_requires_john_paul_ii():
    engine = _bare()
    engine._fire_event(Side.US, "Solidarity")  # precondition unmet
    assert engine.board.influence["Poland"]["US"] == 0
    engine._fire_event(Side.US, "John_Paul_II_Elected_Pope")  # itself adds 1 US
    engine._fire_event(Side.US, "Solidarity")
    assert engine.board.influence["Poland"]["US"] == 4  # 1 (John Paul) + 3


def test_opec_scores_per_ussr_controlled_field():
    engine = _bare()
    engine.board.influence["Iran"] = {"US": 0, "USSR": 3}   # controlled
    engine.board.influence["Libya"] = {"US": 0, "USSR": 3}  # controlled
    engine._fire_event(Side.USSR, "OPEC")
    assert engine.vp == -2  # 2 fields, USSR-favouring


def test_cia_created_conducts_one_op_of_us_operations():
    engine = _bare()
    engine.board.influence["France"]["US"] = 1  # a reachable US foothold
    engine._fire_event(Side.US, "CIA_Created")
    d = engine.pending_decision
    assert d is not None and d.kind is DecisionKind.OPS_TYPE
    assert d.actor is Side.US and d.context["ops"] == 1


def test_the_reformer_places_more_when_ussr_is_ahead():
    engine = _bare()
    engine.vp = -3  # USSR ahead
    engine._fire_event(Side.USSR, "The_Reformer")
    assert engine.pending_decision.context["remaining"] == 6
    assert engine.game_effects.get("reformer") is True


def test_reformer_bars_ussr_coups_in_europe_but_not_realignment():
    engine = _bare()
    engine.game_effects["reformer"] = True
    coup = {a.payload["country"] for a in engine._coup_target_options(Side.USSR)}
    realign = {a.payload["country"] for a in engine._realignment_target_options(Side.USSR)}
    assert "France" not in coup       # Europe coups barred
    assert "France" in realign        # realignment still allowed
    assert "Vietnam" in coup          # non-Europe coups unaffected


def test_brush_war_only_targets_low_stability_countries():
    engine = _bare(seed=8)
    engine._fire_event(Side.US, "Brush_War")
    d = engine.pending_decision
    assert d.kind is DecisionKind.WAR_TARGET and d.actor is Side.US
    for a in d.options:
        assert engine.board.countries[a.payload["country"]].stability <= 2


def test_indo_pakistani_war_target_choice_resolves_to_a_roll():
    engine = _bare(seed=9)
    engine._fire_event(Side.USSR, "Indo_Pakistani_War")
    d = engine.pending_decision
    assert {a.payload["country"] for a in d.options} == {"India", "Pakistan"}
    engine.step(Action(DecisionKind.WAR_TARGET, {"country": "Pakistan"}))
    assert engine.pending_decision.kind is DecisionKind.WAR_ROLL
    assert engine.military_ops["USSR"] == 2


def test_independent_reds_matches_us_to_ussr_influence():
    engine = _bare()
    engine.board.influence["Romania"] = {"US": 0, "USSR": 3}
    engine._fire_event(Side.US, "Independent_Reds")
    assert engine.pending_decision.kind is DecisionKind.EVENT_CHOICE
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "Romania"}))
    assert engine.board.influence["Romania"]["US"] == 3  # matched


def test_puppet_governments_only_targets_empty_countries():
    engine = _bare()
    engine.board.influence["Angola"] = {"US": 1, "USSR": 0}   # not empty
    engine.board.influence["Chile"] = {"US": 0, "USSR": 2}    # not empty
    engine._fire_event(Side.US, "Puppet_Governments")
    offered = {a.payload["country"] for a in engine.pending_decision.options}
    assert "Angola" not in offered and "Chile" not in offered


# -- forced random discard subsystem (CHANCE) -------------------------------


def test_five_year_plan_fires_a_discarded_ussr_event():
    engine = _bare(seed=2)
    engine.hands["USSR"] = ["Fidel"]  # single card -> deterministic draw
    engine.board.influence["Cuba"] = {"US": 2, "USSR": 0}
    engine._fire_event(Side.US, "Five_Year_Plan")
    d = engine.pending_decision
    assert d.kind is DecisionKind.RANDOM_DISCARD and d.actor is Side.CHANCE
    assert len(d.options) == 1  # only the drawn card, never the rest of the hand
    engine.step(d.options[0])
    assert engine.board.control("Cuba") is Side.USSR  # Fidel fired
    assert "Fidel" in engine.removed_cards


def test_five_year_plan_just_discards_a_non_ussr_card():
    engine = _bare(seed=2)
    engine.hands["USSR"] = ["Duck_and_Cover"]  # a US event: discarded, not fired
    engine.defcon = 5
    engine._fire_event(Side.US, "Five_Year_Plan")
    engine.step(engine.pending_decision.options[0])
    assert engine.defcon == 5  # Duck and Cover did NOT fire
    assert "Duck_and_Cover" in engine.discard_pile


def test_random_discard_leaks_only_the_drawn_card():
    engine = _bare(seed=3)
    engine.hands["USSR"] = ["Fidel", "Nasser", "Allende", "COMECON"]
    engine._fire_event(Side.US, "Five_Year_Plan")
    visible = {a.payload["card"] for a in engine.observe(Side.US).pending_decision.options}
    assert len(visible) == 1  # the other three hidden cards never appear


def test_terrorism_discards_twice_after_iranian_hostage_crisis():
    engine = _bare(seed=5)
    engine.game_effects["iranian_hostage"] = True
    engine.hands["US"] = ["Duck_and_Cover", "NATO", "Containment"]
    engine._fire_event(Side.USSR, "Terrorism")  # USSR vs US -> two discards
    discards = 0
    while (engine.pending_decision is not None
           and engine.pending_decision.kind is DecisionKind.RANDOM_DISCARD):
        engine.step(engine.pending_decision.options[0])
        discards += 1
    assert discards == 2
    assert len(engine.hands["US"]) == 1


# -- per-turn coup modifiers -------------------------------------------------


def test_nuclear_subs_spares_defcon_on_us_battleground_coup():
    engine = _bare(seed=1)
    engine.defcon = 5
    engine.turn_effects["nuclear_subs"] = True
    engine.board.influence["Italy"] = {"US": 0, "USSR": 1}  # Italy is a battleground
    engine._push(Side.US, DecisionKind.COUP_TARGET,
                 (Action(DecisionKind.COUP_TARGET, {"country": "Italy"}),),
                 {"ops": 4, "china": False})
    engine.step(Action(DecisionKind.COUP_TARGET, {"country": "Italy"}))
    engine.step(engine.pending_decision.options[0])  # coup roll
    assert engine.defcon == 5  # DEFCON untouched
    # A non-battleground US coup still degrades DEFCON.
    engine.board.influence["Lebanon"] = {"US": 0, "USSR": 1}  # not a battleground
    engine._push(Side.US, DecisionKind.COUP_TARGET,
                 (Action(DecisionKind.COUP_TARGET, {"country": "Lebanon"}),),
                 {"ops": 4, "china": False})
    engine.step(Action(DecisionKind.COUP_TARGET, {"country": "Lebanon"}))
    engine.step(engine.pending_decision.options[0])
    assert engine.defcon == 4


def _resolve_coup_roll(engine: Engine, side: Side, country: str, ops: int, value: int):
    """Drive a coup on `country` with a fixed die `value` (bypassing the RNG)."""
    engine._push(Side.CHANCE, DecisionKind.COUP_ROLL,
                 (Action(DecisionKind.COUP_ROLL, {"value": value}),),
                 {"side": side.value, "country": country, "ops": ops})
    engine.step(Action(DecisionKind.COUP_ROLL, {"value": value}))


def test_latin_american_death_squads_shifts_coup_margins():
    # Cuba (stability 3): a die of 3 with ops 3 gives margin 0 (a miss) normally,
    # but +1 from Death Squads for its player makes it a hit.
    plain = _bare(seed=1)
    plain.board.influence["Cuba"] = {"US": 1, "USSR": 0}
    _resolve_coup_roll(plain, Side.USSR, "Cuba", ops=3, value=3)
    assert plain.board.influence["Cuba"]["US"] == 1  # margin 0: no removal

    boosted = _bare(seed=1)
    boosted.turn_effects["la_death_squads"] = Side.USSR.value
    boosted.board.influence["Cuba"] = {"US": 1, "USSR": 0}
    _resolve_coup_roll(boosted, Side.USSR, "Cuba", ops=3, value=3)
    assert boosted.board.influence["Cuba"]["US"] == 0  # +1 margin: removed


# -- set-DEFCON branch -------------------------------------------------------


def test_salt_negotiations_defcon_coup_penalty_and_reclaim():
    engine = _bare(seed=1)
    engine.defcon = 3
    engine.discard_pile = ["Duck_and_Cover", "Asia_Scoring", "Fidel"]
    engine.hands["US"] = []
    engine._fire_event(Side.US, "Salt_Negotiations")
    assert engine.defcon == 5  # +2
    assert engine.turn_effects.get("salt") is True
    choices = {a.payload["choice"] for a in engine.pending_decision.options}
    assert "Asia_Scoring" not in choices  # scoring cards are not reclaimable
    assert {"Duck_and_Cover", "Fidel", "none"} <= choices
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "Fidel"}))
    assert "Fidel" in engine.hands["US"] and "Fidel" not in engine.discard_pile


def test_salt_coup_penalty_applies_to_both_sides():
    engine = _bare(seed=1)
    engine.turn_effects["salt"] = True
    from struggler.board import CountryInfo  # info object carries region/battleground
    info = engine.board.countries["Cuba"]
    assert engine._coup_roll_modifier(Side.US, info) == -1
    assert engine._coup_roll_modifier(Side.USSR, info) == -1


def test_how_i_learned_sets_defcon_and_adds_military_ops():
    engine = _bare(seed=1)
    engine.defcon = 5
    engine._fire_event(Side.US, "How_I_Learned_to_Stop_Worrying")
    assert {a.payload["choice"] for a in engine.pending_decision.options} == {
        "1", "2", "3", "4", "5"
    }
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "3"}))
    assert engine.defcon == 3
    assert engine.military_ops["US"] == 5


# -- per-turn regional Ops bonus (Vietnam Revolts) ---------------------------


def test_vietnam_revolts_places_and_grants_se_asia_ops_bonus():
    engine = _bare()
    engine._fire_event(Side.USSR, "Vietnam_Revolts")
    assert engine.board.influence["Vietnam"]["USSR"] == 2
    # A USSR Ops play now earns a "+1 if all in Southeast Asia" bonus.
    engine.board.influence["Vietnam"]["USSR"] = 2  # a reachable SE Asia foothold
    engine.hands["USSR"] = ["Socialist_Governments"]  # 3-Ops card
    _play_card_for(engine, Side.USSR, "Socialist_Governments", "ops")
    assert engine.pending_decision.context["bonus"] == "se_asia"
    engine.step(Action(DecisionKind.OPS_TYPE, {"type": "influence"}))

    def se_asia(opts):
        return next(
            a for a in opts
            if engine.board.countries[a.payload["country"]].subregion is not None
            and engine.board.countries[a.payload["country"]].subregion.value == "SOUTHEAST_ASIA"
        )
    steps = 0
    while (engine.pending_decision is not None
           and engine.pending_decision.kind is DecisionKind.PLACE_INFLUENCE):
        engine.step(se_asia(engine.pending_decision.options))
        steps += 1
    assert steps == 4  # base 3 + 1 all-in-SE-Asia bonus


def test_region_bonus_does_not_apply_to_us_or_outside_se_asia():
    engine = _bare()
    engine.turn_effects["vietnam_revolts"] = True
    # US plays are unaffected; only the USSR gets the SE Asia bonus.
    assert engine._ops_bonus_region(Side.US, china=False) is None
    assert engine._ops_bonus_region(Side.USSR, china=False) == "se_asia"


# -- influence + optional free operation (Junta) -----------------------------


def test_junta_places_two_then_offers_a_free_regional_operation():
    engine = _bare(seed=1)
    engine.defcon = 5
    engine._fire_event(Side.USSR, "Junta")
    placed = 0
    while (engine.pending_decision is not None
           and engine.pending_decision.kind is DecisionKind.EVENT_INFLUENCE):
        cid = engine.pending_decision.options[0].payload["country"]
        assert engine.board.countries[cid].region.value in ("CENTRAL_AMERICA", "SOUTH_AMERICA")
        engine.step(engine.pending_decision.options[0])
        placed += 1
    assert placed == 2
    choice = engine.pending_decision
    assert choice.kind is DecisionKind.EVENT_CHOICE
    assert {a.payload["choice"] for a in choice.options} == {"none", "coup", "realign"}
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "realign"}))
    target = engine.pending_decision
    assert target.kind is DecisionKind.REALIGNMENT_TARGET
    assert all(
        engine.board.countries[a.payload["country"]].region.value
        in ("CENTRAL_AMERICA", "SOUTH_AMERICA")
        for a in target.options
    )


def test_junta_free_op_can_be_declined():
    engine = _bare(seed=1)
    engine.defcon = 5
    engine._fire_event(Side.US, "Junta")
    while engine.pending_decision.kind is DecisionKind.EVENT_INFLUENCE:
        engine.step(engine.pending_decision.options[0])
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "none"}))
    assert engine.pending_decision is None  # nothing further enqueued


# -- more per-turn / game-long coup & realignment modifiers ------------------


def test_yuri_and_samantha_scores_ussr_on_us_coups():
    engine = _bare(seed=1)
    engine.defcon = 5
    engine.game_effects["yuri_samantha"] = True
    engine.board.influence["Cuba"] = {"US": 0, "USSR": 1}
    _resolve_coup_roll(engine, Side.US, "Cuba", ops=3, value=1)
    assert engine.vp == -1  # 1 VP to the USSR for the US coup attempt
    # A USSR coup does not trigger it.
    engine.vp = 0
    _resolve_coup_roll(engine, Side.USSR, "Cuba", ops=3, value=1)
    assert engine.vp == 0


def test_iran_contra_penalises_only_us_realignment():
    engine = _bare()
    engine.turn_effects["iran_contra"] = True
    assert engine._realignment_modifier(Side.US) == -1
    assert engine._realignment_modifier(Side.USSR) == 0


def test_flower_power_scores_ussr_when_us_plays_a_war_card():
    engine = _bare()
    engine.game_effects["flower_power"] = True
    engine.hands["US"] = ["Brush_War"]
    _play_card_for(engine, Side.US, "Brush_War", "event")
    assert engine.vp == -2  # 2 VP to the USSR
    # The USSR playing a war card does not trigger it.
    engine2 = _bare()
    engine2.game_effects["flower_power"] = True
    engine2.hands["USSR"] = ["Korean_War"]
    _play_card_for(engine2, Side.USSR, "Korean_War", "event")
    assert engine2.vp == 0


def test_an_evil_empire_cancels_flower_power():
    engine = _bare()
    engine.game_effects["flower_power"] = True
    engine._fire_event(Side.US, "An_Evil_Empire")
    assert "flower_power" not in engine.game_effects
    engine.hands["US"] = ["Brush_War"]
    engine.vp = 0
    _play_card_for(engine, Side.US, "Brush_War", "event")
    assert engine.vp == 0  # no longer scored (An Evil Empire itself gave +1 above)


def test_chernobyl_blocks_ussr_ops_influence_in_the_named_region():
    engine = _bare()
    engine._fire_event(Side.US, "Chernobyl")
    assert engine.pending_decision.kind is DecisionKind.EVENT_CHOICE
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "EUROPE"}))
    engine.board.influence["Poland"]["USSR"] = 3  # reachable European foothold
    ussr = {a.payload["country"] for a in engine._place_influence_options(Side.USSR, 5)}
    assert "Poland" not in ussr  # Europe blocked for the USSR
    # The US is unaffected, and the block is Europe-only for the USSR.
    engine.board.influence["Vietnam"]["USSR"] = 1
    assert "Vietnam" in {a.payload["country"] for a in engine._place_influence_options(Side.USSR, 5)}
    assert engine._chernobyl_blocks(Side.US, "Poland") is False


# -- dice-contest / branch events -------------------------------------------


def test_olympic_games_participate_runs_a_contest_awarding_two_vp():
    engine = _bare(seed=1)
    engine._fire_event(Side.US, "Olympic_Games")  # US sponsors; USSR decides
    decision = engine.pending_decision
    assert decision.actor is Side.USSR
    assert {a.payload["choice"] for a in decision.options} == {"participate", "boycott"}
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "participate"}))
    assert engine.pending_decision.kind is DecisionKind.CONTEST_ROLL
    engine.step(engine.pending_decision.options[0])
    assert abs(engine.vp) == 2  # exactly one side won 2 VP
    assert engine.pending_decision is None


def test_olympic_games_boycott_degrades_defcon_and_gives_sponsor_ops():
    engine = _bare(seed=1)
    engine.defcon = 5
    engine._fire_event(Side.US, "Olympic_Games")
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "boycott"}))
    assert engine.defcon == 4
    assert engine.pending_decision.kind is DecisionKind.OPS_TYPE
    assert engine.pending_decision.actor is Side.US
    assert engine.pending_decision.context["ops"] == 4


def test_summit_contest_then_winner_adjusts_defcon():
    engine = _bare(seed=3)
    engine.defcon = 3
    engine._fire_event(Side.US, "Summit")
    assert engine.pending_decision.kind is DecisionKind.CONTEST_ROLL
    engine.step(engine.pending_decision.options[0])
    follow = engine.pending_decision
    assert follow.kind is DecisionKind.EVENT_CHOICE
    assert {a.payload["choice"] for a in follow.options} == {"raise", "lower", "none"}
    winner = follow.actor
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "raise"}))
    assert engine.defcon == 4
    assert abs(engine.vp) == 2  # the winner also took 2 VP
    assert winner in (Side.US, Side.USSR)


def test_summit_reroll_on_a_tie_is_deterministic_and_resolves():
    # Equal modifiers make ties possible; the contest must still resolve.
    engine = _bare(seed=11)
    engine._fire_event(Side.USSR, "Summit")
    guard = 0
    while engine.pending_decision.kind is DecisionKind.CONTEST_ROLL:
        engine.step(engine.pending_decision.options[0])
        guard += 1
        assert guard < 100
    assert engine.pending_decision.kind is DecisionKind.EVENT_CHOICE  # a winner emerged


def test_wargames_only_playable_at_defcon_two_and_can_end_the_game():
    engine = _bare(seed=1)
    engine.defcon = 3
    engine._fire_event(Side.US, "Wargames")  # ineligible above DEFCON 2
    assert engine.pending_decision is None
    engine.defcon = 2
    engine._fire_event(Side.US, "Wargames")
    assert {a.payload["choice"] for a in engine.pending_decision.options} == {
        "end_game", "decline"
    }
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "end_game"}))
    assert engine.is_terminal  # the US gave the USSR 6 VP and the game was scored


# -- revealing / taking cards from the opponent's hand -----------------------


def test_aldrich_ames_lets_ussr_discard_a_chosen_us_card():
    engine = _bare()
    engine.hands["US"] = ["Duck_and_Cover", "NATO", "Containment"]
    engine._fire_event(Side.USSR, "Aldrich_Ames_Remix")
    decision = engine.pending_decision
    assert decision.actor is Side.USSR
    assert {a.payload["choice"] for a in decision.options} == set(engine.hands["US"])
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "NATO"}))
    assert "NATO" not in engine.hands["US"]
    assert "NATO" in engine.discard_pile


def test_grain_sales_reveals_exactly_one_ussr_card():
    engine = _bare(seed=4)
    engine.hands["USSR"] = ["Fidel", "Nasser", "Allende", "COMECON"]
    engine._fire_event(Side.US, "Grain_Sales_to_Soviets")
    reveal = engine.pending_decision
    assert reveal.kind is DecisionKind.RANDOM_DISCARD and reveal.actor is Side.CHANCE
    assert len(reveal.options) == 1  # only the drawn card, not the whole hand
    revealed = reveal.options[0].payload["card"]
    engine.step(reveal.options[0])
    choice = engine.pending_decision
    assert choice.actor is Side.US
    assert {a.payload["choice"] for a in choice.options} == {"take", "return"}
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "take"}))
    assert revealed not in engine.hands["USSR"] and revealed in engine.discard_pile
    assert engine.pending_decision.kind is DecisionKind.OPS_TYPE  # US uses its Ops


def test_grain_sales_return_leaves_the_card_and_gives_two_ops():
    engine = _bare(seed=4)
    engine.hands["USSR"] = ["Fidel"]
    engine._fire_event(Side.US, "Grain_Sales_to_Soviets")
    engine.step(engine.pending_decision.options[0])  # reveal
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "return"}))
    assert "Fidel" in engine.hands["USSR"]  # returned
    assert engine.pending_decision.context["ops"] == 2  # Grain Sales' own Ops


def test_ask_not_discards_chosen_cards_and_redraws_the_same_number():
    engine = _bare(seed=5)
    engine.draw_pile = ["Blockade", "Defectors", "Quagmire"]
    engine.hands["US"] = ["Containment", "NATO"]
    engine._fire_event(Side.US, "Ask_Not_What_Your_Country_Can_Do_For_You")
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "Containment"}))
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "stop"}))
    assert len(engine.hands["US"]) == 2  # one discarded, one drawn
    assert "Containment" in engine.discard_pile
    assert "Containment" not in engine.hands["US"]


def test_cambridge_five_places_in_a_revealed_scoring_region():
    engine = _bare()
    engine.hands["US"] = ["Asia_Scoring", "NATO"]  # US holds the Asia scoring card
    engine._fire_event(Side.USSR, "The_Cambridge_Five")
    decision = engine.pending_decision
    assert decision.kind is DecisionKind.EVENT_INFLUENCE and decision.actor is Side.USSR
    assert all(
        engine.board.countries[a.payload["country"]].region.value == "ASIA"
        for a in decision.options
    )


def test_cambridge_five_no_op_without_us_scoring_cards():
    engine = _bare()
    engine.hands["US"] = ["NATO", "Containment"]
    engine._fire_event(Side.USSR, "The_Cambridge_Five")
    assert engine.pending_decision is None


# -- the "opponent event fires when played for Ops" rule --------------------


def _play_card_for(engine: Engine, side: Side, cid: str, mode: str) -> None:
    """Drive a single PLAY_MODE decision for `side` playing `cid` as `mode`."""
    engine._push(
        side,
        DecisionKind.PLAY_MODE,
        (Action(DecisionKind.PLAY_MODE, {"mode": mode}),),
        {"card": cid},
    )
    engine.step(Action(DecisionKind.PLAY_MODE, {"mode": mode}))


def test_owner_event_play_fires_the_event():
    engine = _bare()
    engine.defcon = 5
    engine.hands["US"] = ["Duck_and_Cover"]
    _play_card_for(engine, Side.US, "Duck_and_Cover", "event")
    assert engine.defcon == 4  # the event fired
    assert "Duck_and_Cover" in engine.discard_pile


def test_opponent_card_for_ops_triggers_an_order_choice_then_both_halves():
    # USSR plays the US card Duck and Cover for Ops: the US event fires too, and
    # the USSR chooses whether it happens before or after its own operations.
    engine = _bare()
    engine.defcon = 5
    engine.hands["USSR"] = ["Duck_and_Cover"]
    _play_card_for(engine, Side.USSR, "Duck_and_Cover", "ops")

    order = engine.pending_decision
    assert order.kind is DecisionKind.EVENT_OPS_ORDER
    assert {a.payload["order"] for a in order.options} == {"event_first", "ops_first"}

    # event_first: the event resolves immediately, then Ops are offered.
    engine.step(Action(DecisionKind.EVENT_OPS_ORDER, {"order": "event_first"}))
    assert engine.defcon == 4  # Duck and Cover fired
    assert engine.pending_decision.kind is DecisionKind.OPS_TYPE
    assert engine.pending_decision.actor is Side.USSR
    # Ops reflect nothing unusual here, but the card is already filed once.
    assert "Duck_and_Cover" not in engine.hands["USSR"]


def test_opponent_card_ops_first_defers_the_event_until_after_ops():
    engine = _bare()
    engine.defcon = 5
    engine.hands["USSR"] = ["Duck_and_Cover"]
    _play_card_for(engine, Side.USSR, "Duck_and_Cover", "ops")
    engine.step(Action(DecisionKind.EVENT_OPS_ORDER, {"order": "ops_first"}))
    # Ops come first: the event has NOT fired yet, and a resume marker waits
    # underneath the Ops decision to fire it once operations finish.
    assert engine.defcon == 5
    assert engine.pending_decision.kind is DecisionKind.OPS_TYPE
    resume = engine._decision_stack[0]
    assert resume.kind is DecisionKind.EVENT_RESUME
    assert resume.context["what"] == "event"


def test_neutral_card_for_ops_never_triggers_an_event():
    # Captured Nazi Scientist is NEUTRAL, so playing it for Ops must not fire it.
    engine = _bare()
    engine.hands["US"] = ["Captured_Nazi_Scientist"]
    _play_card_for(engine, Side.US, "Captured_Nazi_Scientist", "ops")
    assert engine.space_race["US"] == 0  # event did not fire
    assert engine.pending_decision.kind is DecisionKind.OPS_TYPE


# -- headline events fire (with interrupt ordering) -------------------------


def _headline_setup(engine: Engine, ussr_card: str, us_card: str) -> None:
    """Put a controlled headline in front of a bare, events-on engine."""
    engine.phase = "headline"
    engine.hands = {"USSR": [ussr_card], "US": [us_card]}
    engine._advance()  # pushes the USSR headline choice


def test_headline_fires_both_events_high_ops_first():
    engine = _bare(seed=1)
    engine.defcon = 5
    engine.board.influence["Cuba"] = {"US": 1, "USSR": 0}
    # USSR: Fidel (2 ops); US: Duck and Cover (3 ops) -> Duck resolves first.
    _headline_setup(engine, "Fidel", "Duck_and_Cover")
    engine.step(Action(DecisionKind.HEADLINE_PLAY, {"card": "Fidel"}))
    engine.step(Action(DecisionKind.HEADLINE_PLAY, {"card": "Duck_and_Cover"}))
    assert engine.defcon == 4  # Duck and Cover fired
    assert engine.board.control("Cuba") is Side.USSR  # Fidel fired
    assert engine.phase == "action_rounds"  # headline complete
    assert engine._headline_pending == [] and engine._headline_resolving is False


def test_headline_event_interrupt_drains_before_the_second_card():
    # USSR Korean War (2 ops) outranks US Captured Nazi Scientist (1 op), so the
    # war resolves first and enqueues its CHANCE roll; the second headline card
    # must not fire until that roll is stepped.
    engine = _bare(seed=4)
    engine.board.influence["South_Korea"] = {"US": 0, "USSR": 0}
    _headline_setup(engine, "Korean_War", "Captured_Nazi_Scientist")
    engine.step(Action(DecisionKind.HEADLINE_PLAY, {"card": "Korean_War"}))
    engine.step(Action(DecisionKind.HEADLINE_PLAY, {"card": "Captured_Nazi_Scientist"}))

    pending = engine.pending_decision
    assert pending.kind is DecisionKind.WAR_ROLL and pending.actor is Side.CHANCE
    assert engine.space_race["US"] == 0  # the second card has NOT fired yet

    engine.step(pending.options[0])  # resolve the war's roll
    assert engine.space_race["US"] == 1  # now Captured Nazi Scientist fires
    assert engine.phase == "action_rounds"


def test_headline_non_event_card_is_still_a_no_op_discard():
    engine = _bare(seed=2)
    # Quagmire and Defectors have no implemented event yet: headlining them must
    # be a plain discard, exactly as in M2, even with events on.
    _headline_setup(engine, "Quagmire", "Defectors")
    engine.step(Action(DecisionKind.HEADLINE_PLAY, {"card": "Quagmire"}))
    engine.step(Action(DecisionKind.HEADLINE_PLAY, {"card": "Defectors"}))
    assert "Quagmire" in engine.discard_pile
    assert "Defectors" in engine.discard_pile
    assert engine.phase == "action_rounds"


# -- events off (M2) is untouched -------------------------------------------


def test_events_disabled_never_fires_an_event_on_ops_play():
    engine = Engine(seed=0)  # events_enabled defaults to False
    engine.defcon = 5
    engine.hands["USSR"] = ["Duck_and_Cover"]
    _play_card_for(engine, Side.USSR, "Duck_and_Cover", "ops")
    assert engine.pending_decision.kind is DecisionKind.OPS_TYPE  # no order choice
    assert engine.defcon == 5


# -- full-game invariants with events on ------------------------------------


def _cards_in_play(engine: Engine) -> Counter:
    c: Counter = Counter()
    for cards in engine.hands.values():
        c.update(cards)
    c.update(engine.draw_pile)
    c.update(engine.discard_pile)
    c.update(engine.removed_cards)
    for cid in engine._headline.values():
        if cid is not None:
            c.update([cid])
    # A headlined card whose event is mid-resolution (its sub-decisions still
    # draining) lives here until it is filed to a pile.
    for _side, cid in engine._headline_pending:
        c.update([cid])
    return c


def _expected_in_play(engine: Engine) -> set[str]:
    ids = set(cards_entering(engine.cards, Period.EARLY_WAR, engine.include_optional))
    if engine.turn >= 4:
        ids |= set(cards_entering(engine.cards, Period.MID_WAR, engine.include_optional))
    if engine.turn >= 8:
        ids |= set(cards_entering(engine.cards, Period.LATE_WAR, engine.include_optional))
    return ids


def _assert_invariants(engine: Engine) -> None:
    assert 1 <= engine.defcon <= 5
    for values in engine.board.influence.values():
        assert values["US"] >= 0 and values["USSR"] >= 0
    if not engine.is_terminal:
        assert engine.pending_decision is not None
        assert len(engine.legal_actions()) > 0
    in_play = _cards_in_play(engine)
    assert all(count == 1 for count in in_play.values())
    assert CHINA_CARD_ID not in in_play
    assert set(in_play) == _expected_in_play(engine)


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(min_value=0, max_value=MAX_INT32),
       driver_seed=st.integers(min_value=0, max_value=MAX_INT32))
def test_random_full_game_with_events_terminates_with_invariants(seed, driver_seed):
    engine = Engine.new_game(seed=seed, events=True)
    driver = random.Random(driver_seed)
    steps = 0
    while not engine.is_terminal:
        _assert_invariants(engine)
        engine.step(driver.choice(engine.legal_actions()))
        steps += 1
        assert steps < 20000, "a full game should terminate well before this"
    assert engine.pending_decision is None
    assert engine.winner in (Side.US, Side.USSR, None)


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=MAX_INT32),
       driver_seed=st.integers(min_value=0, max_value=MAX_INT32))
def test_events_game_serializes_and_never_leaks(seed, driver_seed):
    engine = Engine.new_game(seed=seed, events=True)
    driver = random.Random(driver_seed)
    steps = 0
    while not engine.is_terminal and steps < 300:
        for player in (Side.US, Side.USSR):
            obs = engine.observe(player)
            opponent_hand = set(engine.hands[player.opponent.value])
            assert set(obs.hand).isdisjoint(opponent_hand)
            assert obs.opponent_hand_size == len(opponent_hand)
        data = engine.serialize()
        json.dumps(data)
        assert Engine.deserialize(data).serialize() == data
        engine.step(driver.choice(engine.legal_actions()))
        steps += 1


# -- M3 tail cards ----------------------------------------------------------
#
# The most idiosyncratic events: taking cards from a hand/discard and playing
# them, a conditional-repeat coup, deferred per-turn conditions, and
# scoring-time modifiers.


def test_missile_envy_passes_to_opponent_and_takes_top_ops_card():
    engine = _bare()
    engine.defcon = 5
    engine.hands["US"] = ["Missile_Envy"]
    engine.hands["USSR"] = ["Fidel", "Nasser"]  # Fidel (2) outranks Nasser (1)
    engine.board.influence["Cuba"] = {"US": 2, "USSR": 0}
    _play_card_for(engine, Side.US, "Missile_Envy", "event")
    assert "Missile_Envy" in engine.hands["USSR"]  # Missile Envy changes hands
    assert "Fidel" not in engine.hands["USSR"]      # the top-Ops card was taken
    # Fidel is the giver's own (USSR) event, so the US must use it for Ops only.
    d = engine.pending_decision
    assert d.kind is DecisionKind.OPS_TYPE and d.actor is Side.US
    assert d.context["ops"] == engine.cards["Fidel"].ops


def test_missile_envy_neutral_card_offers_ops_or_event():
    engine = _bare()
    engine.hands["US"] = []
    engine.hands["USSR"] = ["Captured_Nazi_Scientist"]  # NEUTRAL -> taker may choose
    engine._fire_event(Side.US, "Missile_Envy")
    d = engine.pending_decision
    assert d.kind is DecisionKind.EVENT_CHOICE and d.actor is Side.US
    assert {a.payload["choice"] for a in d.options} == {"ops", "event"}
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "event"}))
    assert engine.space_race["US"] == 1  # the taken event fired for the US


def test_missile_envy_opponent_breaks_a_tie():
    engine = _bare()
    engine.hands["US"] = []
    engine.hands["USSR"] = ["COMECON", "Socialist_Governments"]  # both 3 Ops
    engine._fire_event(Side.US, "Missile_Envy")
    d = engine.pending_decision
    assert d.actor is Side.USSR  # the giver decides which tied card to hand over
    assert {a.payload["choice"] for a in d.options} == {"COMECON", "Socialist_Governments"}
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "COMECON"}))
    assert "COMECON" not in engine.hands["USSR"]
    assert engine.pending_decision.kind is DecisionKind.OPS_TYPE  # giver's event -> Ops


def test_missile_envy_no_op_on_an_empty_opponent_hand():
    engine = _bare()
    engine.hands["US"] = []
    engine.hands["USSR"] = []
    engine._fire_event(Side.US, "Missile_Envy")
    assert engine.pending_decision is None


def test_star_wars_requires_us_space_race_lead():
    engine = _bare()
    engine.discard_pile = ["Fidel"]
    engine.space_race["US"] = engine.space_race["USSR"] = 1  # not ahead
    engine._fire_event(Side.US, "Star_Wars")
    assert engine.pending_decision is None
    engine.space_race["US"] = 2  # now ahead
    engine._fire_event(Side.US, "Star_Wars")
    assert engine.pending_decision.kind is DecisionKind.EVENT_CHOICE


def test_star_wars_plays_a_discard_card_immediately():
    engine = _bare()
    engine.space_race["US"] = 3
    engine.discard_pile = ["Fidel", "Asia_Scoring"]  # scoring is not offered
    engine.board.influence["Cuba"] = {"US": 2, "USSR": 0}
    engine._fire_event(Side.US, "Star_Wars")
    offered = {a.payload["choice"] for a in engine.pending_decision.options}
    assert "Asia_Scoring" not in offered and {"Fidel", "none"} <= offered
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "Fidel"}))
    assert engine.board.control("Cuba") is Side.USSR  # Fidel fired from the discard
    assert "Fidel" not in engine.discard_pile


def _americas_africa_non_bg(engine):
    return [
        cid
        for cid, info in engine.board.countries.items()
        if not info.battleground
        and info.region in (Region.CENTRAL_AMERICA, Region.SOUTH_AMERICA, Region.AFRICA)
    ]


def test_che_offers_a_free_coup_in_the_americas_and_africa():
    engine = _bare(seed=1)
    engine.defcon = 5
    engine._fire_event(Side.USSR, "Che")
    d = engine.pending_decision
    assert d.kind is DecisionKind.EVENT_CHOICE and d.actor is Side.USSR
    choices = {a.payload["choice"] for a in d.options}
    assert "none" in choices  # the coup is optional
    for cid in choices - {"none"}:
        info = engine.board.countries[cid]
        assert not info.battleground
        assert info.region in (Region.CENTRAL_AMERICA, Region.SOUTH_AMERICA, Region.AFRICA)


def test_che_second_coup_after_removing_us_influence_excludes_the_first():
    engine = _bare(seed=3)
    engine.defcon = 5
    engine.board.influence["Nicaragua"] = {"US": 2, "USSR": 0}  # stability 1, non-bg
    engine._fire_event(Side.USSR, "Che")
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "Nicaragua"}))
    roll = engine.pending_decision  # COUP_ROLL, che state attached
    assert roll.kind is DecisionKind.COUP_ROLL and "che" in roll.context
    assert engine.military_ops["USSR"] == 3  # a free coup still counts as military Ops
    # Nicaragua has stability 1, so even the seeded roll here removes US Influence.
    engine.step(roll.options[0])
    assert engine.board.influence["Nicaragua"]["US"] == 0
    second = engine.pending_decision
    assert second.kind is DecisionKind.EVENT_CHOICE
    assert "Nicaragua" not in {a.payload["choice"] for a in second.options}


def test_che_serializes_with_its_repeat_state_on_the_stack():
    engine = _bare(seed=3)
    engine.defcon = 5
    engine.board.influence["Nicaragua"] = {"US": 2, "USSR": 0}
    engine._fire_event(Side.USSR, "Che")
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "Nicaragua"}))
    data = engine.serialize()
    json.dumps(data)  # the nested "che" context is JSON-native (mandate #5)
    assert Engine.deserialize(data).serialize() == data


def test_cuban_missile_crisis_sets_defcon_two_and_can_be_defused():
    engine = _bare(seed=1)
    engine.defcon = 5
    engine.board.influence["Cuba"] = {"US": 0, "USSR": 3}  # USSR can afford to defuse
    engine._fire_event(Side.US, "Cuban_Missile_Crisis")  # US plays it -> USSR at risk
    assert engine.defcon == 2
    assert engine.turn_effects.get("cuban_missile_crisis") == "USSR"
    d = engine.pending_decision
    assert d.actor is Side.USSR and {a.payload["choice"] for a in d.options} == {"defuse", "keep"}
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "defuse"}))
    assert engine.board.influence["Cuba"]["USSR"] == 1  # 2 removed
    assert "cuban_missile_crisis" not in engine.turn_effects  # threat gone


def test_cuban_missile_crisis_coup_by_the_flagged_side_loses_the_game():
    engine = _bare(seed=1)
    engine.defcon = 5
    engine.turn_effects["cuban_missile_crisis"] = "USSR"
    engine.board.influence["Nicaragua"] = {"US": 2, "USSR": 0}
    _resolve_coup_roll(engine, Side.USSR, "Nicaragua", ops=3, value=6)
    assert engine.is_terminal and engine.winner is Side.US


def test_we_will_bury_you_degrades_defcon_and_scores_at_end_of_turn():
    engine = Engine.new_game(seed=2, events=True)
    engine.defcon = 5
    engine._fire_event(Side.USSR, "We_Will_Bury_You")
    assert engine.defcon == 4
    assert engine.turn_effects.get("we_will_bury_you") is True
    engine.military_ops = {"US": 9, "USSR": 9}  # silence the required-military-Ops VP
    vp0 = engine.vp
    engine._end_of_turn()
    assert engine.vp == vp0 - 3  # 3 VP to the USSR (negative on the US-positive track)


def test_we_will_bury_you_defused_by_us_un_intervention():
    engine = _bare()
    engine.defcon = 5
    engine.turn_effects["we_will_bury_you"] = True
    engine.hands["US"] = ["Fidel", "UN_Intervention"]  # Fidel is a USSR (opponent) event
    _play_card_for(engine, Side.US, "Fidel", "un_intervention")
    assert "we_will_bury_you" not in engine.turn_effects


def test_formosan_makes_taiwan_a_battleground_for_asia_scoring():
    engine = _bare()
    engine._fire_event(Side.US, "Formosan_Resolution")
    engine.board.influence["Taiwan"] = {"US": 4, "USSR": 0}  # US controls (stability 3)
    extra_bg, _ = engine._scoring_overrides(Region.ASIA)
    assert "Taiwan" in extra_bg
    engine.board.influence["Taiwan"] = {"US": 0, "USSR": 0}  # US no longer controls
    extra_bg2, _ = engine._scoring_overrides(Region.ASIA)
    assert "Taiwan" not in extra_bg2


def test_formosan_resolution_nullified_by_the_china_card():
    engine = _bare()
    engine._fire_event(Side.US, "Formosan_Resolution")
    assert engine.game_effects.get("formosan_resolution") is True
    engine._file_card(Side.USSR, CHINA_CARD_ID, fired=False)  # the China Card is played
    assert "formosan_resolution" not in engine.game_effects


def test_shuttle_diplomacy_drops_one_ussr_battleground_then_expires():
    engine = _bare()
    engine.game_effects["shuttle_diplomacy"] = True
    target = next(
        cid for cid, info in engine.board.countries.items()
        if info.region is Region.MIDDLE_EAST and info.battleground
    )
    engine.board.influence[target] = {"US": 0, "USSR": 9}  # USSR-controlled battleground
    _, ignored = engine._scoring_overrides(Region.MIDDLE_EAST)
    assert target in ignored
    assert "shuttle_diplomacy" not in engine.game_effects  # consumed at first scoring
    _, ignored_again = engine._scoring_overrides(Region.ASIA)
    assert ignored_again == frozenset()  # gone for later scorings


def test_north_sea_oil_blocks_opec_and_grants_us_an_extra_action_round():
    engine = Engine.new_game(seed=2, events=True)
    base = 2 * action_rounds(engine.turn)
    engine._fire_event(Side.US, "North_Sea_Oil")
    assert engine.game_effects.get("north_sea_oil") is True
    assert not EVENTS["OPEC"].eligible(engine, Side.USSR)  # OPEC no longer playable
    assert engine._total_action_rounds() == base + 1
    assert engine._side_for_play_index(base) is Side.US  # the extra round is the US's


# -- further batch (existing primitives + a relocate flow) -------------------


def test_east_european_unrest_removes_one_early_two_late():
    early = _bare()
    early.turn = 3
    for cid in ("Poland", "East_Germany", "Czechoslovakia"):
        early.board.influence[cid] = {"US": 0, "USSR": 3}
    early._fire_event(Side.US, "East_European_Unrest")
    assert _drain_event_influence(early) == 3  # one step per country
    assert early.board.influence["Poland"]["USSR"] == 2  # removed 1

    late = _bare()
    late.turn = 8
    late.board.influence["Poland"] = {"US": 0, "USSR": 3}
    late._fire_event(Side.US, "East_European_Unrest")
    late.step(Action(DecisionKind.EVENT_INFLUENCE, {"country": "Poland"}))
    assert late.board.influence["Poland"]["USSR"] == 1  # removed 2 in the Late War


def test_south_african_unrest_adjacent_branch():
    engine = _bare()
    engine._fire_event(Side.USSR, "South_African_Unrest")
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "and_adjacent"}))
    assert engine.board.influence["South_Africa"]["USSR"] == 1
    d = engine.pending_decision
    assert {a.payload["choice"] for a in d.options} == {"Angola", "Botswana"}
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "Angola"}))
    assert engine.board.influence["Angola"]["USSR"] == 2


def test_south_african_unrest_south_africa_only_branch():
    engine = _bare()
    engine._fire_event(Side.USSR, "South_African_Unrest")
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "south_africa_only"}))
    assert engine.board.influence["South_Africa"]["USSR"] == 2
    assert engine.pending_decision is None


def test_blockade_removes_west_germany_without_a_payable_card():
    engine = _bare()
    engine.board.influence["West_Germany"] = {"US": 4, "USSR": 0}
    engine.hands["US"] = ["Nasser"]  # only a 1-Op card: cannot pay
    engine._fire_event(Side.USSR, "Blockade")
    assert engine.board.influence["West_Germany"]["US"] == 0
    assert engine.pending_decision is None


def test_blockade_can_be_paid_with_a_three_ops_card():
    engine = _bare()
    engine.board.influence["West_Germany"] = {"US": 4, "USSR": 0}
    engine.hands["US"] = ["Duck_and_Cover"]  # 3 Ops
    engine._fire_event(Side.USSR, "Blockade")
    assert engine.pending_decision.actor is Side.US
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "Duck_and_Cover"}))
    assert engine.board.influence["West_Germany"]["US"] == 4  # kept
    assert "Duck_and_Cover" in engine.discard_pile


def test_arms_race_scores_three_when_leading_and_meeting_the_requirement():
    engine = _bare()
    engine.defcon = 3
    engine.military_ops = {"US": 4, "USSR": 1}
    engine._fire_event(Side.US, "Arms_Race")
    assert engine.vp == 3  # US leads and 4 >= DEFCON 3
    engine2 = _bare()
    engine2.defcon = 5
    engine2.military_ops = {"US": 2, "USSR": 1}
    engine2._fire_event(Side.US, "Arms_Race")
    assert engine2.vp == 1  # leads but 2 < DEFCON 5


def test_de_stalinization_relocates_influence():
    engine = _bare()
    engine.board.influence["Angola"] = {"US": 0, "USSR": 2}
    engine.board.influence["Cuba"] = {"US": 0, "USSR": 1}  # keeps a target so "done" is offered
    engine._fire_event(Side.USSR, "De_Stalinization")
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "Angola"}))
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "Angola"}))
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "done"}))  # stop after 2
    assert engine.board.influence["Angola"]["USSR"] == 0  # 2 removed
    d = engine.pending_decision
    assert d.kind is DecisionKind.EVENT_INFLUENCE  # placement phase
    assert engine.board.control(d.options[0].payload["country"]) is not Side.US
    assert _drain_event_influence(engine) == 2  # exactly the 2 removed are placed


def test_latin_american_debt_crisis_doubles_two_south_america_countries():
    engine = _bare()
    engine.hands["US"] = ["Nasser"]  # cannot pay
    engine.board.influence["Brazil"] = {"US": 0, "USSR": 2}
    engine.board.influence["Argentina"] = {"US": 0, "USSR": 3}
    engine._fire_event(Side.USSR, "Latin_American_Debt_Crisis")
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "Brazil"}))
    assert engine.board.influence["Brazil"]["USSR"] == 4
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "Argentina"}))
    assert engine.board.influence["Argentina"]["USSR"] == 6
    assert engine.pending_decision is None


def test_latin_american_debt_crisis_cancelled_by_us_payment():
    engine = _bare()
    engine.hands["US"] = ["Duck_and_Cover"]  # 3 Ops
    engine.board.influence["Brazil"] = {"US": 0, "USSR": 2}
    engine._fire_event(Side.USSR, "Latin_American_Debt_Crisis")
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "Duck_and_Cover"}))
    assert engine.board.influence["Brazil"]["USSR"] == 2  # not doubled
    assert engine.pending_decision is None


def test_soviets_shoot_down_kal_007_conducts_ops_only_with_south_korea():
    engine = _bare()
    engine.defcon = 5
    engine.board.influence["South_Korea"] = {"US": 4, "USSR": 0}  # US-controlled
    engine.board.influence["Japan"] = {"US": 1, "USSR": 0}  # a reachable foothold
    engine._fire_event(Side.US, "Soviets_Shoot_Down_KAL_007")
    assert engine.defcon == 4 and engine.vp == 2
    assert engine.pending_decision.kind is DecisionKind.OPS_TYPE
    assert engine.pending_decision.context["ops"] == 4

    engine2 = _bare()
    engine2.defcon = 5
    engine2.board.influence["South_Korea"] = {"US": 0, "USSR": 0}  # not US-controlled
    engine2._fire_event(Side.US, "Soviets_Shoot_Down_KAL_007")
    assert engine2.vp == 2 and engine2.pending_decision is None  # no Operations


def test_ussuri_river_skirmish_takes_the_china_card_or_places_in_asia():
    taking = _bare()
    taking.china_card_owner = "USSR"
    taking.china_card_available = False
    taking._fire_event(Side.US, "Ussuri_River_Skirmish")
    assert taking.china_card_owner == "US" and taking.china_card_available is True

    placing = _bare()
    placing.china_card_owner = "US"
    placing._fire_event(Side.US, "Ussuri_River_Skirmish")
    assert _drain_event_influence(placing) == 4  # 4 Influence into Asia (cap 2/country)


def test_glasnost_scores_and_grants_ops_only_after_the_reformer():
    plain = _bare()
    plain.defcon = 3
    plain._fire_event(Side.USSR, "Glasnost")
    assert plain.vp == -2 and plain.defcon == 4
    assert plain.pending_decision is None  # no free Operations without The Reformer

    reformer = _bare()
    reformer.defcon = 3
    reformer.game_effects["reformer"] = True
    reformer.board.influence["France"] = {"US": 0, "USSR": 1}
    reformer._fire_event(Side.USSR, "Glasnost")
    assert reformer.pending_decision.kind is DecisionKind.OPS_TYPE
    assert reformer.pending_decision.context["ops"] == 4


# -- golden replay -----------------------------------------------------------


def test_golden_events_replay_matches_checkpoints():
    with (REPLAY_DIR / "m3_events.json").open(encoding="utf-8") as f:
        log = json.load(f)
    assert log.get("events") is True
    recorded = run_with_checkpoints(log)
    assert len(recorded) == len(log["checkpoints"])
    for rec, checkpoint in zip(recorded, log["checkpoints"]):
        assert rec["after_step"] == checkpoint["after_step"]
        assert rec["state"] == checkpoint["state"]  # exact, diffable equality


def test_golden_events_replay_actually_fires_events():
    # Guard against a regression where the log stops exercising the event layer.
    # A fired event shows up as any of: an opponent-Ops order choice, an
    # event-mode action-round play, or a headline of an implemented-event card
    # (every id in EVENTS is a non-scoring event card).
    with (REPLAY_DIR / "m3_events.json").open(encoding="utf-8") as f:
        log = json.load(f)
    fired = any(
        a["kind"] == DecisionKind.EVENT_OPS_ORDER.value
        or (a["kind"] == DecisionKind.PLAY_MODE.value
            and a["payload"].get("mode") == "event")
        or (a["kind"] == DecisionKind.HEADLINE_PLAY.value
            and a["payload"].get("card") in EVENTS)
        for a in log["actions"]
    )
    assert fired
