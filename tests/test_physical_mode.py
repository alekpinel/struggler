"""Physical/external-player mode: hidden hand, manual dealing, manual dice.

See docs/BOTS.md's "Physical mode" section for the design.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import assert_invariants as _assert_invariants
from struggler.engine import Action, DecisionKind, Engine, Side
from struggler.engine.cards import hand_limit
from struggler.engine.core import HIDDEN_CARD
from struggler.engine.events import _payable_cards

MAX_INT32 = 2**31 - 1


def _bare_physical(physical_side: Side, seed: int = 0) -> Engine:
    """A minimal physical-mode engine, event layer on, no turn loop running
    (mirrors conftest.bare_engine, plus the physical-mode fields)."""
    engine = Engine(seed=seed)
    engine.events_enabled = True
    engine.physical_mode = True
    engine.physical_side = physical_side
    return engine


# -- dealing from a single shared physical deck ------------------------------


def test_new_game_starts_with_deal_card_decision_for_the_bot():
    engine = Engine.new_game(seed=1, physical_mode=True, physical_side=Side.USSR, events=False)
    decision = engine.pending_decision
    assert decision.kind is DecisionKind.DEAL_CARD
    assert decision.actor is Side.CHANCE
    assert decision.context["side"] == "US"
    assert len(decision.options) == len(engine.hidden_pool)


def test_dealing_grows_bot_hand_shrinks_pool_and_rejects_repeats():
    engine = Engine.new_game(seed=1, physical_mode=True, physical_side=Side.USSR, events=False)
    before_hand = len(engine.hands["US"])
    before_pool = len(engine.hidden_pool)
    action = engine.pending_decision.options[0]
    engine.step(action)
    assert len(engine.hands["US"]) == before_hand + 1
    assert len(engine.hidden_pool) == before_pool - 1
    assert action.payload["card"] in engine.hands["US"]
    with pytest.raises(ValueError):
        engine.step(action)  # already dealt: no longer a legal option


def test_full_opening_deal_completes_into_opening_setup():
    engine = Engine.new_game(seed=2, physical_mode=True, physical_side=Side.US, events=False)
    steps = 0
    while engine.pending_decision.kind is DecisionKind.DEAL_CARD:
        engine.step(engine.pending_decision.options[0])
        steps += 1
        assert steps < 200
    # Both hands full: opening setup placement follows (USSR always first,
    # regardless of which side is physical -- setup doesn't touch hands).
    assert engine.pending_decision.kind is DecisionKind.PLACE_INFLUENCE
    assert engine.pending_decision.actor is Side.USSR
    assert len(engine.hands["US"]) == hand_limit(1)
    assert len(engine.hands["USSR"]) == hand_limit(1)


# -- manual dice --------------------------------------------------------------


def test_manual_coup_roll_offers_six_options_and_resolves_by_formula():
    engine = Engine(seed=5)
    engine.physical_mode = True
    engine.physical_side = Side.USSR
    engine.board.influence["Guatemala"]["USSR"] = 3
    engine.begin_coup(Side.US, ops=3)
    target = next(a for a in engine.legal_actions() if a.payload["country"] == "Guatemala")
    engine.step(target)

    decision = engine.pending_decision
    assert decision.kind is DecisionKind.COUP_ROLL
    assert decision.actor is Side.CHANCE
    assert [a.payload["value"] for a in decision.options] == [1, 2, 3, 4, 5, 6]

    roll = 4
    chosen = next(a for a in decision.options if a.payload["value"] == roll)
    engine.step(chosen)
    assert engine.pending_decision is None

    stability = engine.board.countries["Guatemala"].stability
    margin = roll + 3 - 2 * stability
    if margin > 0:
        removed = min(margin, 3)
        assert engine.board.influence["Guatemala"]["USSR"] == 3 - removed
        assert engine.board.influence["Guatemala"]["US"] == margin - removed
    else:
        assert engine.board.influence["Guatemala"] == {"US": 0, "USSR": 3}


def test_manual_dice_never_consumes_the_seeded_rng():
    engine = Engine(seed=5)
    engine.physical_mode = True
    engine.physical_side = Side.USSR
    engine.board.influence["Guatemala"]["USSR"] = 3
    state_before = engine._rng.getstate()
    engine.begin_coup(Side.US, ops=3)
    target = next(a for a in engine.legal_actions() if a.payload["country"] == "Guatemala")
    engine.step(target)
    assert engine._rng.getstate() == state_before  # no die drawn from the seeded RNG


# -- random forced discard from a hidden hand ---------------------------------


def test_push_random_discard_bot_owner_offers_real_hand_no_rng_consumed():
    engine = _bare_physical(Side.USSR, seed=3)
    engine.hands["US"] = ["Duck_and_Cover", "Fidel"]
    state_before = engine._rng.getstate()
    engine.push_random_discard(Side.US, "terrorism", count=1)
    assert engine._rng.getstate() == state_before
    decision = engine.pending_decision
    assert decision.actor is Side.CHANCE
    assert {a.payload["card"] for a in decision.options} == {"Duck_and_Cover", "Fidel"}


def test_push_random_discard_physical_owner_offers_hidden_pool():
    engine = _bare_physical(Side.USSR, seed=3)
    engine.hidden_pool = ["Duck_and_Cover", "Fidel"]
    engine.hands["USSR"] = [HIDDEN_CARD, HIDDEN_CARD]
    engine.push_random_discard(Side.USSR, "terrorism", count=1)
    decision = engine.pending_decision
    assert {a.payload["card"] for a in decision.options} == {"Duck_and_Cover", "Fidel"}
    chosen = decision.options[0]
    engine.step(chosen)
    cid = chosen.payload["card"]
    assert cid not in engine.hidden_pool
    assert cid in engine.discard_pile
    assert engine.hands["USSR"].count(HIDDEN_CARD) == 1


# -- own-hand choice sourced from hidden_pool ---------------------------------


def test_payable_cards_sources_from_hidden_pool_for_physical_side():
    engine = _bare_physical(Side.US, seed=1)
    payable_real = next(cid for cid, c in engine.cards.items() if not c.scoring and c.ops >= 3)
    engine.hidden_pool = [payable_real]
    engine.hands["US"] = [HIDDEN_CARD]
    assert _payable_cards(engine, Side.US) == [payable_real]


def test_blockade_end_to_end_with_a_physical_us_hand():
    engine = _bare_physical(Side.US, seed=1)
    engine.board.influence["West_Germany"] = {"US": 4, "USSR": 0}
    engine.hidden_pool = ["Duck_and_Cover"]  # 3 Ops
    engine.hands["US"] = [HIDDEN_CARD]
    engine._fire_event(Side.USSR, "Blockade")
    decision = engine.pending_decision
    assert decision.actor is Side.US
    assert {a.payload["choice"] for a in decision.options} == {"Duck_and_Cover", "refuse"}
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "Duck_and_Cover"}))
    assert engine.board.influence["West_Germany"]["US"] == 4  # kept
    assert "Duck_and_Cover" in engine.discard_pile
    assert "Duck_and_Cover" not in engine.hidden_pool
    assert engine.hands["US"] == []  # the one placeholder was consumed


def test_un_intervention_offered_and_resolved_for_a_physical_hand():
    # The engine can't verify a physical hand's contents, so UN Intervention's
    # combo mode is offered on trust -- same as the must-play-a-scoring-card
    # rule (see docs/LIMITATIONS.md) -- as long as it's still a plausible
    # `hidden_pool` candidate for that hand.
    engine = _bare_physical(Side.US, seed=1)
    engine.hidden_pool = ["Fidel", "UN_Intervention"]  # Fidel is a USSR (opponent) event
    engine.hands["US"] = [HIDDEN_CARD, HIDDEN_CARD]
    assert "un_intervention" in engine._play_modes(Side.US, "Fidel")

    engine._push(
        Side.US,
        DecisionKind.PLAY_MODE,
        (Action(DecisionKind.PLAY_MODE, {"mode": "un_intervention"}),),
        {"card": "Fidel"},
    )
    engine.step(Action(DecisionKind.PLAY_MODE, {"mode": "un_intervention"}))

    assert engine.board.control("Cuba") is not Side.USSR  # Fidel's event did NOT fire
    assert "UN_Intervention" in engine.discard_pile  # spent
    assert "UN_Intervention" not in engine.hidden_pool
    assert "Fidel" in engine.discard_pile
    assert engine.hands["US"] == []  # both placeholders consumed
    assert engine.pending_decision.kind is DecisionKind.OPS_TYPE  # used for Ops


def test_un_intervention_not_offered_once_resolved_elsewhere_in_a_physical_hand():
    # Once UN Intervention itself has been revealed and filed away (played or
    # discarded), it's no longer a `hidden_pool` candidate, so the mode must
    # stop being offered even though the physical hand still has open slots.
    engine = _bare_physical(Side.US, seed=1)
    engine.hidden_pool = ["Fidel"]
    engine.hands["US"] = [HIDDEN_CARD]
    engine.discard_pile.append("UN_Intervention")  # already played/discarded earlier
    assert "un_intervention" not in engine._play_modes(Side.US, "Fidel")


def test_ask_not_with_a_physical_hand_discards_and_redraws():
    engine = _bare_physical(Side.US, seed=5)
    engine.draw_pile = [HIDDEN_CARD, HIDDEN_CARD, HIDDEN_CARD]
    engine.hidden_pool = ["Containment", "NATO", "Blockade", "Defectors", "Quagmire"]
    engine.hands["US"] = [HIDDEN_CARD, HIDDEN_CARD]
    engine._fire_event(Side.US, "Ask_Not_What_Your_Country_Can_Do_For_You")
    decision = engine.pending_decision
    assert decision.actor is Side.US
    choice_values = {a.payload["choice"] for a in decision.options}
    assert "stop" in choice_values
    picked = next(v for v in choice_values if v != "stop")
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": picked}))
    assert picked in engine.discard_pile
    assert picked not in engine.hidden_pool
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "stop"}))
    assert len(engine.hands["US"]) == 2  # one discarded, one redealt


# -- cross-hand events routed to the operator ---------------------------------


def test_aldrich_ames_routes_to_operator_when_us_hand_is_physical():
    engine = _bare_physical(Side.US, seed=1)
    engine.hidden_pool = ["Duck_and_Cover", "Fidel"]
    engine.hands["US"] = [HIDDEN_CARD, HIDDEN_CARD]
    engine._fire_event(Side.USSR, "Aldrich_Ames_Remix")
    decision = engine.pending_decision
    # Not USSR: the USSR bot cannot see the physical US hand, so the choice
    # is routed to the operator (CHANCE), same as manual dice / DEAL_CARD.
    assert decision.actor is Side.CHANCE
    assert {a.payload["choice"] for a in decision.options} == {"Duck_and_Cover", "Fidel"}
    chosen = decision.options[0]
    engine.step(chosen)
    cid = chosen.payload["choice"]
    assert cid in engine.discard_pile
    assert cid not in engine.hidden_pool
    assert engine.hands["US"].count(HIDDEN_CARD) == 1


def test_aldrich_ames_unaffected_when_us_hand_is_not_physical():
    engine = _bare_physical(Side.USSR, seed=1)  # USSR physical, US is the bot
    engine.hands["US"] = ["Duck_and_Cover", "Fidel"]
    engine._fire_event(Side.USSR, "Aldrich_Ames_Remix")
    decision = engine.pending_decision
    assert decision.actor is Side.USSR
    assert {a.payload["choice"] for a in decision.options} == {"Duck_and_Cover", "Fidel"}


def test_cambridge_five_queries_operator_per_scoring_card_when_us_is_physical():
    engine = _bare_physical(Side.US, seed=1)
    engine.hidden_pool = ["Asia_Scoring", "Europe_Scoring", "Duck_and_Cover"]
    engine.hands["US"] = [HIDDEN_CARD, HIDDEN_CARD, HIDDEN_CARD]
    engine._fire_event(Side.USSR, "The_Cambridge_Five")

    decision = engine.pending_decision
    assert decision.kind is DecisionKind.EVENT_CHOICE
    assert decision.actor is Side.CHANCE
    assert decision.context["scoring_id"] == "Asia_Scoring"
    assert {a.payload["choice"] for a in decision.options} == {"yes", "no"}
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "yes"}))
    # Answering "yes" reveals the card in hand (declared, no longer hidden).
    assert "Asia_Scoring" in engine.hands["US"]
    assert "Asia_Scoring" not in engine.hidden_pool

    decision = engine.pending_decision
    assert decision.context["scoring_id"] == "Europe_Scoring"
    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "no"}))

    # Only Asia applies: USSR's EVENT_INFLUENCE candidates are Asian countries.
    decision = engine.pending_decision
    assert decision.kind is DecisionKind.EVENT_INFLUENCE
    assert decision.actor is Side.USSR
    assert all(
        engine.board.countries[a.payload["country"]].region.value == "ASIA"
        for a in decision.options
    )


def test_cambridge_five_no_op_when_us_physical_hand_has_no_scoring_card():
    engine = _bare_physical(Side.US, seed=1)
    engine.hidden_pool = ["Duck_and_Cover", "Fidel"]
    engine.hands["US"] = [HIDDEN_CARD, HIDDEN_CARD]
    engine._fire_event(Side.USSR, "The_Cambridge_Five")
    assert engine.pending_decision is None


def test_missile_envy_giver_physical_routes_the_pick_to_operator():
    engine = _bare_physical(Side.USSR, seed=1)  # USSR (giver) is physical
    engine.hidden_pool = ["Duck_and_Cover"]  # US-aligned, 3 Ops: not ops_only
    engine.hands["USSR"] = [HIDDEN_CARD]
    engine._fire_event(Side.US, "Missile_Envy")  # US plays it (taker)

    decision = engine.pending_decision
    assert decision.kind is DecisionKind.EVENT_CHOICE
    assert decision.actor is Side.CHANCE
    assert {a.payload["choice"] for a in decision.options} == {"Duck_and_Cover"}
    engine.step(decision.options[0])

    # The picked card leaves the physical giver's hand; Missile Envy itself
    # (a known, public card) moves into it, forcing USSR's next play.
    assert "Duck_and_Cover" not in engine.hidden_pool
    assert engine.hands["USSR"].count(HIDDEN_CARD) == 0
    assert "Missile_Envy" in engine.hands["USSR"]
    assert engine.game_effects["missile_envy_forced"] == "USSR"

    # The US taker isn't physical here, so it gets the ordinary ops/event
    # choice for the card it took.
    decision = engine.pending_decision
    assert decision.kind is DecisionKind.EVENT_CHOICE
    assert decision.actor is Side.US
    assert set(a.payload["choice"] for a in decision.options) == {"ops", "event"}


def test_missile_envy_taker_physical_does_not_strip_a_stray_placeholder():
    engine = _bare_physical(Side.US, seed=1)  # US (taker) is physical
    engine.hands["USSR"] = ["Duck_and_Cover"]  # the bot giver's real hand
    engine.hidden_pool = ["Fidel"]
    engine.hands["US"] = [HIDDEN_CARD]  # one unrelated card, must stay untouched
    engine._fire_event(Side.US, "Missile_Envy")  # US (physical) plays it

    # Non-physical giver path: resolves directly, no operator query needed
    # for the pick itself; US (taker, physical) gets the ops/event choice.
    decision = engine.pending_decision
    assert decision.kind is DecisionKind.EVENT_CHOICE
    assert decision.actor is Side.US
    assert engine.game_effects["missile_envy_forced"] == "USSR"

    engine.step(Action(DecisionKind.EVENT_CHOICE, {"choice": "event"}))

    # The card left the (bot) giver's real hand once actually used...
    assert "Duck_and_Cover" not in engine.hands["USSR"]
    # ...but was never really in the physical US hand -- the unrelated
    # placeholder must survive untouched (the bug this guards against:
    # _file_card mistaking this for a real hand departure).
    assert engine.hands["US"] == [HIDDEN_CARD]
    assert "Fidel" in engine.hidden_pool


# -- serialization --------------------------------------------------------


def test_physical_mode_serialize_deserialize_round_trips():
    engine = Engine.new_game(seed=7, physical_mode=True, physical_side=Side.US, events=False)
    for _ in range(2):
        if engine.pending_decision.kind is DecisionKind.DEAL_CARD:
            engine.step(engine.pending_decision.options[0])
    data = engine.serialize()
    assert data["physical_mode"] is True
    assert data["physical_side"] == "US"
    assert HIDDEN_CARD in data["hands"]["US"] or HIDDEN_CARD in data["draw_pile"]
    restored = Engine.deserialize(data)
    assert restored.serialize() == data
    assert restored.hidden_pool == engine.hidden_pool


# -- full-game property test ------------------------------------------------


@settings(max_examples=6, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=MAX_INT32),
    driver_seed=st.integers(min_value=0, max_value=MAX_INT32),
    physical_side=st.sampled_from([Side.US, Side.USSR]),
)
def test_random_physical_mode_game_terminates_with_invariants(seed, driver_seed, physical_side):
    engine = Engine.new_game(
        seed=seed, physical_mode=True, physical_side=physical_side, events=True
    )
    bot_side = physical_side.opponent
    driver = random.Random(driver_seed)
    steps = 0
    while not engine.is_terminal:
        _assert_invariants(engine)
        obs_bot = engine.observe(bot_side)
        assert HIDDEN_CARD not in obs_bot.hand
        obs_phys = engine.observe(physical_side)
        assert HIDDEN_CARD not in obs_phys.discard_pile
        assert HIDDEN_CARD not in obs_phys.removed_cards
        engine.step(driver.choice(engine.legal_actions()))
        steps += 1
        assert steps < 20000, "a full game should terminate well before this"
    assert engine.pending_decision is None
    assert engine.winner in (Side.US, Side.USSR, None)
