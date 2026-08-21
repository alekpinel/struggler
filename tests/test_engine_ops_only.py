"""Full game through the public API — cards played for Ops, no events fire.

These tests prove the milestone's headline claim: a complete game is playable
start-to-finish via Engine.new_game / legal_actions / step, with the mandated
invariants holding and hidden information never leaking.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import assert_invariants
from struggler.engine import DecisionKind, Engine, Region, ScoringTier, Side
from struggler.engine.replay import run_with_checkpoints
from struggler.engine.rules import RULES

MAX_INT32 = 2**31 - 1
REPLAY_DIR = Path(__file__).parent / "replays"


def _no_coup(actions):
    kept = [
        a
        for a in actions
        if not (a.kind is DecisionKind.OPS_TYPE and a.payload.get("type") == "coup")
    ]
    return kept or list(actions)


def test_new_game_opens_with_setup_and_full_hands():
    engine = Engine.new_game(seed=1, events=False)
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
    assert RULES["china_card_id"] not in engine.hands["USSR"]


def test_setup_places_the_additional_influence_then_reaches_headline():
    engine = Engine.new_game(seed=1, events=False)
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


def test_action_round_resets_to_1_at_the_start_of_a_new_turns_headline():
    # Regression: `action_round` used to be written only by
    # `_begin_action_rounds`, which doesn't run until headline resolution
    # finishes. So every decision made during a new turn's headline phase --
    # including a Player's very first look at that turn -- still reported the
    # PREVIOUS turn's last action round (e.g. 6). A turn plan built from that
    # stale number (LLMPlayer's remaining-action-rounds calculation) badly
    # undercounts how many rounds are actually left and can wrongly mark most
    # of the hand 'hold'.
    engine = Engine.new_game(seed=1, events=False)
    while engine.turn == 1:
        engine.step(engine.legal_actions()[0])
    assert engine.turn == 2
    assert engine.phase == "headline"  # before _begin_action_rounds runs
    assert engine.action_round == 1
    assert engine.observe(engine.pending_decision.actor).action_round == 1


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=MAX_INT32),
       driver_seed=st.integers(min_value=0, max_value=MAX_INT32))
def test_random_full_game_terminates_with_invariants(seed, driver_seed):
    engine = Engine.new_game(seed=seed, events=False)
    driver = random.Random(driver_seed)
    steps = 0
    while not engine.is_terminal:
        assert_invariants(engine)
        engine.step(driver.choice(engine.legal_actions()))
        steps += 1
        assert steps < 20000, "a full game should terminate well before this"
    # Terminal state: no pending decision, and a definite outcome.
    assert engine.pending_decision is None
    assert engine.winner in (Side.US, Side.USSR, None)


def test_observe_exposes_public_track_state():
    # Military ops, phase, and the event modifier maps are all public board
    # state; a player needs them to reason about the game, not just the
    # bare minimum required to stay legal.
    engine = Engine.new_game(seed=1, events=False)
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
    engine = Engine.new_game(seed=1, events=False)
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
    engine = Engine.new_game(seed=seed, events=False)
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
    engine = Engine.new_game(seed=seed, events=False)
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
    engine = Engine.new_game(seed=3, events=False)
    scoring_ids = {cid for cid, c in engine.cards.items() if c.scoring}
    modes = engine._play_modes(Side.US, next(iter(scoring_ids)))
    assert modes == ("event",)  # never Ops, never Space Race


def test_non_scoring_card_offers_the_event_vs_ops_choice():
    engine = Engine.new_game(seed=3, events=False)
    # A plain 3-Ops card: Ops and Event are both enumerated (event is a no-op
    # with events off, but the choice must still be offered).
    modes = engine._play_modes(Side.US, "Duck_and_Cover")
    assert "ops" in modes and "event" in modes


def test_china_card_passes_to_the_opponent_when_played():
    engine = Engine.new_game(seed=5, events=False)
    assert engine.china_card_owner == "USSR"
    engine._file_card(Side.USSR, RULES["china_card_id"], fired=False)
    assert engine.china_card_owner == "US"
    assert engine.china_card_available is False  # face-down until next turn


def test_golden_full_game_replay_matches_checkpoints():
    with (REPLAY_DIR / "full_game_ops_only.json").open(encoding="utf-8") as f:
        log = json.load(f)
    recorded = run_with_checkpoints(log)
    assert len(recorded) == len(log["checkpoints"])
    for rec, checkpoint in zip(recorded, log["checkpoints"]):
        assert rec["after_step"] == checkpoint["after_step"]
        assert rec["state"] == checkpoint["state"]  # exact, diffable equality


def test_last_action_round_forces_a_held_scoring_card():
    # A scoring card cannot be carried out of a turn: when a side has as many
    # scoring cards as action rounds left, those rounds must spend them.
    engine = Engine.new_game(seed=2, events=False)
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


def _give_europe_control_tier(engine: Engine, holder: Side) -> None:
    """Every European Battleground plus Austria and Finland: the Control
    *tier* of 10.1.1, well short of every country in the region."""
    for cid in engine.board.countries_in(Region.EUROPE):
        if engine.board.countries[cid].battleground or cid in ("Austria", "Finland"):
            influence = engine.board.influence[cid]
            influence[holder.value] = influence[holder.opponent.value] + engine.board.countries[cid].stability
    assert engine.board.region_tier(holder, Region.EUROPE) is ScoringTier.CONTROL
    assert engine.board.controls_all_of_europe() is None


def test_scoring_europe_at_control_tier_is_an_automatic_victory():
    """Europe Scoring reads "Control: automatic victory" (10.1.3), Control
    being the scoring tier -- every Battleground and more countries than the
    opponent -- not every country on the map. Holding it merely on the board
    wins nothing; scoring Europe while holding it ends the game."""
    engine = Engine.new_game(seed=1, events=False)
    _give_europe_control_tier(engine, Side.USSR)
    assert not engine.is_terminal

    engine._resolve_scoring_card("Europe_Scoring")

    assert engine.is_terminal and engine.winner is Side.USSR
    assert engine._game_over_reason == "europe_control"
    assert engine.pending_decision is None  # mandate #1: nothing pending on a finished game
    assert engine.vp == 0  # the win, not a VP swing


def test_final_scoring_of_europe_at_control_tier_also_wins():
    engine = Engine.new_game(seed=1, events=False)
    _give_europe_control_tier(engine, Side.US)
    engine.vp = -15  # far behind on points: the Europe win still takes precedence

    engine._finish_game()

    assert engine.winner is Side.US and engine._game_over_reason == "europe_control"


def test_scoring_europe_at_domination_scores_normally():
    """Short of Control, Europe scores like any region: Domination 7 plus
    the 10.1.2 bonuses, and the game goes on."""
    engine = Engine.new_game(seed=1, events=False)
    board = engine.board
    for cid in ("Poland", "East_Germany", "Austria", "Finland"):
        board.influence[cid]["USSR"] = board.influence[cid]["US"] + board.countries[cid].stability
    board.influence["Italy"]["US"] = board.influence["Italy"]["USSR"] + board.countries["Italy"].stability
    assert board.region_tier(Side.USSR, Region.EUROPE) is ScoringTier.DOMINATION
    vp_before = engine.vp

    engine._resolve_scoring_card("Europe_Scoring")

    assert not engine.is_terminal
    assert engine.vp == vp_before + board.score_region(Region.EUROPE)
