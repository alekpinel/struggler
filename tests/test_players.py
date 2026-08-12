"""Tests for the Player protocol, baseline bots, HumanPlayer, and play_game()."""

from __future__ import annotations

import pytest

from struggler.engine import Engine
from struggler.players import FirstLegalPlayer, HumanPlayer, RandomPlayer
from struggler.runner import play_game
from struggler.types import Action, Observation, Side

MAX_STEPS = 20_000


class _SpyPlayer:
    """Wraps another player and records every side it was consulted for."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.sides_seen: list[Side] = []

    def choose_action(self, observation: Observation) -> Action:
        self.sides_seen.append(observation.side)
        return self._inner.choose_action(observation)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_first_legal_vs_first_legal_terminates(seed: int) -> None:
    engine = Engine.new_game(seed=seed)
    us = _SpyPlayer(FirstLegalPlayer())
    ussr = _SpyPlayer(FirstLegalPlayer())

    steps = 0
    while not engine.is_terminal and steps < MAX_STEPS:
        decision = engine.pending_decision
        if decision.actor is Side.CHANCE:
            engine.step(decision.options[0])
        else:
            player = us if decision.actor is Side.US else ussr
            engine.step(player.choose_action(engine.observe(decision.actor)))
        steps += 1

    assert engine.is_terminal
    assert Side.CHANCE not in us.sides_seen
    assert Side.CHANCE not in ussr.sides_seen


@pytest.mark.parametrize("seed", [1, 2])
def test_play_game_random_vs_random_terminates(seed: int) -> None:
    engine = Engine.new_game(seed=seed)
    players = {Side.US: RandomPlayer(seed=seed + 1), Side.USSR: RandomPlayer(seed=seed + 2)}

    winner = play_game(engine, players)

    assert engine.is_terminal
    assert winner in (Side.US, Side.USSR, None)


def test_play_game_never_consults_a_player_for_chance() -> None:
    engine = Engine.new_game(seed=1)
    us = _SpyPlayer(FirstLegalPlayer())
    ussr = _SpyPlayer(FirstLegalPlayer())

    play_game(engine, {Side.US: us, Side.USSR: ussr})

    assert Side.CHANCE not in us.sides_seen
    assert Side.CHANCE not in ussr.sides_seen


def test_human_player_returns_selected_option(monkeypatch) -> None:
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    monkeypatch.setattr("builtins.input", lambda _: "0")

    action = HumanPlayer().choose_action(observation)

    assert action == observation.pending_decision.options[0]


def test_human_player_reprompts_on_invalid_input(monkeypatch) -> None:
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    responses = iter(["not-a-number", "999", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    action = HumanPlayer().choose_action(observation)

    assert action == observation.pending_decision.options[0]
