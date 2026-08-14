"""Tests for GreedyPlayer: the board evaluator, DEFCON-safety heuristic,
fallback behavior, and a win-rate sanity check against RandomPlayer."""

from __future__ import annotations

import dataclasses

import pytest

from struggler.bots.greedy import GreedyPlayer, GreedyWeights, board_value
from struggler.bots.naive import RandomPlayer
from struggler.engine import Action, Decision, DecisionKind, Engine, Side
from struggler.engine.board import Board
from struggler.engine.player_registry import available, build_player
from struggler.runner import play_game


def test_board_value_zero_with_no_influence_anywhere():
    board = Board()
    assert board_value(GreedyWeights(), board, Side.US) == 0.0


def test_board_value_rewards_controlling_a_battleground_over_a_non_battleground():
    weights = GreedyWeights()
    battleground = next(cid for cid, info in Board().countries.items() if info.battleground)
    non_battleground = next(cid for cid, info in Board().countries.items() if not info.battleground)

    bg_board = Board()
    bg_board.influence[battleground]["US"] = bg_board.countries[battleground].stability
    bg_value = board_value(weights, bg_board, Side.US)

    plain_board = Board()
    plain_board.influence[non_battleground]["US"] = plain_board.countries[non_battleground].stability
    plain_value = board_value(weights, plain_board, Side.US)

    assert bg_value > plain_value > 0.0


def test_greedy_avoids_coup_as_an_ops_type_at_defcon_2():
    """DEFCON 2 -> 1 loses the game for whoever caused the drop (mandate:
    CLAUDE.md's worked example, priority #1: never die to DEFCON). Even
    with a juicy Coup target on offer, GreedyPlayer must pick something
    else at the OPS_TYPE decision."""
    engine = Engine(seed=1)
    engine.board.influence["Guatemala"]["USSR"] = 3
    engine._change_defcon(-3, caused_by=Side.US)  # 5 -> 2
    assert engine.defcon == 2

    engine._push_ops_type(Side.US, ops=3)
    observation = engine.observe(Side.US)
    assert observation.pending_decision.kind is DecisionKind.OPS_TYPE
    offered = {a.payload["type"] for a in observation.pending_decision.options}
    assert "coup" in offered  # the engine itself doesn't forbid the suicidal option

    action = GreedyPlayer().choose_action(observation, [])
    assert action.payload["type"] != "coup"


def test_greedy_falls_back_to_first_option_for_unmapped_decision_kinds():
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    fallback_decision = Decision(
        id=999,
        actor=observation.side,
        kind=DecisionKind.EVENT_CHOICE,
        options=(
            Action(DecisionKind.EVENT_CHOICE, {"choice": "a"}),
            Action(DecisionKind.EVENT_CHOICE, {"choice": "b"}),
        ),
    )
    observation = dataclasses.replace(observation, pending_decision=fallback_decision)

    action = GreedyPlayer().choose_action(observation, [])

    assert action == fallback_decision.options[0]


def test_registry_has_all_baseline_bots_and_greedy():
    # "llm" only appears once struggler.bots.llm.player has been imported --
    # which some test module in this session may or may not have done yet,
    # since player_registry._FACTORIES is a process-global dict.
    expected = {"human", "random", "first", "greedy"}
    assert expected <= set(available())
    assert set(available()) - expected <= {"llm"}
    assert isinstance(build_player("greedy"), GreedyPlayer)


def test_registry_rejects_unknown_bot_name():
    with pytest.raises(ValueError):
        build_player("nonexistent")


@pytest.mark.parametrize("greedy_side", [Side.US, Side.USSR])
def test_greedy_beats_random_over_many_seeds(greedy_side: Side):
    total = 12
    wins = 0
    for seed in range(total):
        engine = Engine.new_game(seed=seed)
        greedy, random_player = GreedyPlayer(), RandomPlayer(seed=seed + 1000)
        players = (
            {Side.US: greedy, Side.USSR: random_player}
            if greedy_side is Side.US
            else {Side.US: random_player, Side.USSR: greedy}
        )
        if play_game(engine, players) is greedy_side:
            wins += 1
    assert wins >= total * 0.7
