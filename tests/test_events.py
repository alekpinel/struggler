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

from struggler.cards import cards_entering
from struggler.engine import CHINA_CARD_ID, Engine
from struggler.events import EVENTS
from struggler.replay import run_with_checkpoints
from struggler.types import Action, DecisionKind, Period, Side

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
    # Socialist Governments has no implemented event yet: headlining it must be
    # a plain discard, exactly as in M2, even with events on.
    _headline_setup(engine, "Socialist_Governments", "Olympic_Games")
    engine.step(Action(DecisionKind.HEADLINE_PLAY, {"card": "Socialist_Governments"}))
    engine.step(Action(DecisionKind.HEADLINE_PLAY, {"card": "Olympic_Games"}))
    assert "Socialist_Governments" in engine.discard_pile
    assert "Olympic_Games" in engine.discard_pile
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
