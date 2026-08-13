"""M2: full game through the public API — cards played for Ops, no events fire.

These tests prove the milestone's headline claim: a complete game is playable
start-to-finish via Engine.new_game / legal_actions / step, with the mandated
invariants holding and hidden information never leaking.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from struggler.engine import DecisionKind, Engine, Period, Side
from struggler.engine.cards import cards_entering
from struggler.engine.replay import run_with_checkpoints
from struggler.engine.rules import CHINA_CARD_ID

MAX_INT32 = 2**31 - 1
REPLAY_DIR = Path(__file__).parent / "replays"


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
        assert len(engine.legal_actions()) > 0  # never deadlock on a live decision

    # No card is ever in two places at once, and The China Card is tracked
    # separately (never in a hand or pile).
    in_play = _cards_in_play(engine)
    assert all(count == 1 for count in in_play.values())
    assert CHINA_CARD_ID not in in_play
    assert set(in_play) == _expected_in_play(engine)


def _no_coup(actions):
    kept = [
        a
        for a in actions
        if not (a.kind is DecisionKind.OPS_TYPE and a.payload.get("type") == "coup")
    ]
    return kept or list(actions)


def test_new_game_opens_with_setup_and_full_hands():
    engine = Engine.new_game(seed=1)
    decision = engine.pending_decision
    assert decision is not None
    # Opening choice is the USSR's additional Eastern Europe setup placement.
    assert decision.kind is DecisionKind.PLACE_INFLUENCE
    assert decision.actor is Side.USSR
    assert decision.context.get("setup") is True
    # Printed at-start influence is already on the board.
    assert engine.board.influence["North_Korea"]["USSR"] == 3
    assert engine.board.influence["UK"]["US"] == 5
    # Both players were dealt to the Early War hand limit; The China Card is
    # not dealt into a hand.
    assert len(engine.hands["USSR"]) == 8
    assert len(engine.hands["US"]) == 8
    assert CHINA_CARD_ID not in engine.hands["USSR"]


def test_setup_places_the_additional_influence_then_reaches_headline():
    engine = Engine.new_game(seed=1)
    # Base printed totals before the additional placement.
    base_ussr = sum(v["USSR"] for v in engine.board.influence.values())
    base_us = sum(v["US"] for v in engine.board.influence.values())
    assert (base_ussr, base_us) == (9, 18)  # printed at-start sums

    # Resolve the whole setup by always taking the first legal placement.
    while engine.pending_decision.context.get("setup"):
        engine.step(engine.legal_actions()[0])

    total_ussr = sum(v["USSR"] for v in engine.board.influence.values())
    total_us = sum(v["US"] for v in engine.board.influence.values())
    assert total_ussr == base_ussr + 6  # USSR added 6 in Eastern Europe
    assert total_us == base_us + 7       # US added 7 in Western Europe
    # Setup done -> the turn-1 headline begins with the USSR.
    assert engine.pending_decision.kind is DecisionKind.HEADLINE_PLAY
    assert engine.pending_decision.actor is Side.USSR


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=MAX_INT32),
       driver_seed=st.integers(min_value=0, max_value=MAX_INT32))
def test_random_full_game_terminates_with_invariants(seed, driver_seed):
    engine = Engine.new_game(seed=seed)
    driver = random.Random(driver_seed)
    steps = 0
    while not engine.is_terminal:
        _assert_invariants(engine)
        engine.step(driver.choice(engine.legal_actions()))
        steps += 1
        assert steps < 20000, "a full game should terminate well before this"
    # Terminal state: no pending decision, and a definite outcome.
    assert engine.pending_decision is None
    assert engine.winner in (Side.US, Side.USSR, None)


def test_observe_exposes_public_track_state():
    # Military ops, phase, and the M3 modifier maps are all public board
    # state; a player needs them to reason about the game, not just the
    # bare minimum required to stay legal.
    engine = Engine.new_game(seed=1)
    engine.military_ops["US"] = 3
    engine.turn_effects["containment"] = True
    engine.game_effects["nato"] = True

    obs = engine.observe(Side.US)

    assert obs.phase == engine.phase
    assert obs.military_ops == {"US": 3, "USSR": 0}
    assert obs.turn_effects == {"containment": True}
    assert obs.game_effects == {"nato": True}
    # Mutating the engine's live dicts after the fact must not retroactively
    # change an already-taken snapshot (same discipline as `influence`).
    engine.military_ops["US"] = 99
    assert obs.military_ops == {"US": 3, "USSR": 0}


def test_observe_does_not_leak_in_progress_secret_headline_pick():
    # Headline is a simultaneous, secret reveal: while USSR has picked but
    # US hasn't, US's Observation must not carry USSR's pick anywhere.
    engine = Engine.new_game(seed=1)
    while engine.pending_decision.context.get("setup"):
        engine.step(engine.legal_actions()[0])
    assert engine.pending_decision.kind is DecisionKind.HEADLINE_PLAY
    assert engine.pending_decision.actor is Side.USSR
    ussr_pick = engine.pending_decision.options[0]
    engine.step(ussr_pick)  # USSR has now secretly picked; US has not
    assert engine._headline["USSR"] is not None
    assert engine._headline["US"] is None

    obs = engine.observe(Side.US)

    assert not hasattr(obs, "headline")
    picked_card = ussr_pick.payload["card"]
    assert picked_card not in obs.turn_effects.values()
    assert picked_card not in obs.game_effects.values()
    assert picked_card not in obs.hand
    assert picked_card not in obs.discard_pile


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(min_value=0, max_value=MAX_INT32),
       driver_seed=st.integers(min_value=0, max_value=MAX_INT32))
def test_observe_never_reveals_opponent_hand(seed, driver_seed):
    engine = Engine.new_game(seed=seed)
    driver = random.Random(driver_seed)
    steps = 0
    while not engine.is_terminal and steps < 400:
        for player in (Side.US, Side.USSR):
            obs = engine.observe(player)
            opponent_hand = set(engine.hands[player.opponent.value])
            # The opponent's actual cards never appear in the view, only a count.
            assert set(obs.hand).isdisjoint(opponent_hand)
            assert obs.opponent_hand_size == len(opponent_hand)
            assert obs.hand == tuple(engine.hands[player.value])
        engine.step(driver.choice(engine.legal_actions()))
        steps += 1


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=MAX_INT32),
       driver_seed=st.integers(min_value=0, max_value=MAX_INT32))
def test_serialize_round_trips_after_every_step_of_a_full_game(seed, driver_seed):
    engine = Engine.new_game(seed=seed)
    driver = random.Random(driver_seed)
    steps = 0
    while not engine.is_terminal and steps < 300:
        engine.step(driver.choice(engine.legal_actions()))
        data = engine.serialize()
        json.dumps(data)  # JSON-native, no custom encoder (mandate #5)
        assert Engine.deserialize(data).serialize() == data
        steps += 1


def test_scoring_card_can_only_be_played_as_its_event():
    # Drive to a state where a scoring card is the card being played and check
    # the play-mode options offered for it.
    engine = Engine.new_game(seed=3)
    scoring_ids = {cid for cid, c in engine.cards.items() if c.scoring}
    modes = engine._play_modes(Side.US, next(iter(scoring_ids)))
    assert modes == ("event",)  # never Ops, never Space Race


def test_non_scoring_card_offers_the_event_vs_ops_choice():
    engine = Engine.new_game(seed=3)
    # A plain 3-Ops card: Ops and Event are both enumerated (event is a no-op
    # in M2, but the choice must exist per the milestone).
    modes = engine._play_modes(Side.US, "Duck_and_Cover")
    assert "ops" in modes and "event" in modes


def test_china_card_passes_to_the_opponent_when_played():
    engine = Engine.new_game(seed=5)
    assert engine.china_card_owner == "USSR"
    engine._file_card(Side.USSR, CHINA_CARD_ID, fired=False)
    assert engine.china_card_owner == "US"
    assert engine.china_card_available is False  # face-down until next turn


def test_golden_full_game_replay_matches_checkpoints():
    with (REPLAY_DIR / "m2_full_game.json").open(encoding="utf-8") as f:
        log = json.load(f)
    recorded = run_with_checkpoints(log)
    assert len(recorded) == len(log["checkpoints"])
    for rec, checkpoint in zip(recorded, log["checkpoints"]):
        assert rec["after_step"] == checkpoint["after_step"]
        assert rec["state"] == checkpoint["state"]  # exact, diffable equality


def test_peaceful_game_never_touches_the_defcon_loss_track():
    # With no coups, DEFCON is never degraded, so it stays pinned at 5 and the
    # game can only end on VP (a scoring swing) or turn-10 final scoring —
    # never on defcon_1. Exercises multi-turn end-of-turn processing.
    engine = Engine.new_game(seed=20260811)
    driver = random.Random(1)
    while not engine.is_terminal:
        assert engine.defcon == 5
        engine.step(driver.choice(_no_coup(engine.legal_actions())))
    assert engine._game_over_reason in ("vp", "final_vp", "europe_control")
    assert engine.turn > 1  # several turns were processed


def test_last_action_round_forces_a_held_scoring_card():
    # A scoring card cannot be carried out of a turn: when a side has as many
    # scoring cards as action rounds left, those rounds must spend them.
    engine = Engine.new_game(seed=2)
    engine.phase = "action_rounds"
    engine.turn = 1  # 6 action rounds/side -> 12 plays total
    engine._decision_stack = []

    # Last play of the turn (index 11 -> US), holding one scoring card.
    engine._ars_played = 12
    engine.hands["US"] = ["Asia_Scoring", "Duck_and_Cover"]
    engine._push_action_round_play(Side.US)
    options = engine.legal_actions()
    assert [a.payload["card"] for a in options] == ["Asia_Scoring"]

    # Early in the turn (index 1 -> US, five rounds still to come) the same
    # single scoring card imposes no restriction.
    engine._decision_stack = []
    engine._ars_played = 2
    engine.hands["US"] = ["Asia_Scoring", "Duck_and_Cover"]
    engine._push_action_round_play(Side.US)
    cards = {a.payload["card"] for a in engine.legal_actions()}
    assert "Duck_and_Cover" in cards and "Asia_Scoring" in cards
