"""Tests for the Player protocol, baseline bots, HumanPlayer, and play_game()."""

from __future__ import annotations

from typing import Sequence

import pytest

from struggler.bots.naive import FirstLegalPlayer, RandomPlayer
from struggler.engine import Action, Engine, Observation, Side
from struggler.engine.human import HumanPlayer
from struggler.engine.player import Event
from struggler.runner import play_game


class _SpyPlayer:
    """Wraps another player and records every side it was consulted for."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.sides_seen: list[Side] = []

    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        self.sides_seen.append(observation.side)
        return self._inner.choose_action(observation, history)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_first_legal_vs_first_legal_terminates(seed: int) -> None:
    engine = Engine.new_game(seed=seed)
    us = _SpyPlayer(FirstLegalPlayer())
    ussr = _SpyPlayer(FirstLegalPlayer())

    play_game(engine, {Side.US: us, Side.USSR: ussr})

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


def test_play_game_builds_history_of_every_resolved_decision() -> None:
    engine = Engine.new_game(seed=1)
    recorded: list[Sequence[Event]] = []

    class _RecordingPlayer:
        def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
            recorded.append(history)
            return FirstLegalPlayer().choose_action(observation, history)

    play_game(engine, {Side.US: _RecordingPlayer(), Side.USSR: _RecordingPlayer()})

    # Each consultation sees strictly more history than the previous one for
    # that same player (opponent moves and chance rolls accumulate in between).
    lengths = [len(h) for h in recorded]
    assert lengths == sorted(lengths)
    assert lengths[-1] > 0


def test_play_game_records_space_race_and_military_ops_results() -> None:
    engine = Engine.new_game(seed=1)
    history: list[Event] = []

    class _RecordingPlayer:
        def choose_action(self, observation: Observation, hist: Sequence[Event]) -> Action:
            history[:] = hist
            return FirstLegalPlayer().choose_action(observation, hist)

    play_game(engine, {Side.US: _RecordingPlayer(), Side.USSR: _RecordingPlayer()})

    assert history
    for event in history:
        assert set(event.space_race) == {"US", "USSR"}
        assert set(event.military_ops) == {"US", "USSR"}


def test_human_player_returns_selected_option(monkeypatch) -> None:
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    monkeypatch.setattr("builtins.input", lambda _: "0")

    action = HumanPlayer().choose_action(observation, [])

    assert action == observation.pending_decision.options[0]


def test_human_player_reprompts_on_invalid_input(monkeypatch) -> None:
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    responses = iter(["not-a-number", "999", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    action = HumanPlayer().choose_action(observation, [])

    assert action == observation.pending_decision.options[0]


def test_human_player_board_and_history_commands_reprompt(monkeypatch) -> None:
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    responses = iter(["b", "h", "0"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    action = HumanPlayer().choose_action(observation, [])

    assert action == observation.pending_decision.options[0]


def test_human_player_only_shows_events_since_its_last_turn(monkeypatch, capsys) -> None:
    engine = Engine.new_game(seed=1)
    decision = engine.pending_decision
    observation = engine.observe(decision.actor)
    fake_event = Event(
        actor=decision.actor.opponent,
        decision=decision,
        action=decision.options[0],
        defcon=5,
        vp=0,
        turn=1,
        action_round=1,
    )
    monkeypatch.setattr("builtins.input", lambda _: "0")
    player = HumanPlayer()

    player.choose_action(observation, [fake_event])
    capsys.readouterr()  # first call already consumed the event
    player.choose_action(observation, [fake_event])
    output = capsys.readouterr().out

    assert "Since your last turn" not in output
