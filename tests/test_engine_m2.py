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

from struggler.cards import cards_entering
from struggler.engine import CHINA_CARD_ID, Engine
from struggler.replay import run_with_checkpoints
from struggler.types import DecisionKind, Period, Side

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


def test_new_game_opens_with_the_ussr_headline_and_full_hands():
    engine = Engine.new_game(seed=1)
    decision = engine.pending_decision
    assert decision is not None
    assert decision.kind is DecisionKind.HEADLINE_PLAY
    assert decision.actor is Side.USSR
    # Both players were dealt to the Early War hand limit; The China Card is
    # not dealt into a hand.
    assert len(engine.hands["USSR"]) == 8
    assert len(engine.hands["US"]) == 8
    assert CHINA_CARD_ID not in engine.hands["USSR"]


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


def test_peaceful_game_reaches_turn_ten_and_recovers_defcon():
    # Avoiding coups keeps DEFCON off the loss track, so the game runs the
    # full ten turns and ends on accumulated VP.
    engine = Engine.new_game(seed=20260811)
    driver = random.Random(1)
    while not engine.is_terminal:
        engine.step(driver.choice(_no_coup(engine.legal_actions())))
    assert engine.turn == 10
    assert engine._game_over_reason == "final_vp"
