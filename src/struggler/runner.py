"""Drives a game to completion by delegating each decision to a `Player`.

The one piece of orchestration logic needed on top of `Engine`: it makes
human-vs-human, human-vs-bot, and bot-vs-bot all the same code path,
distinguished only by which `Player` is registered for which `Side`.
"""

from __future__ import annotations

from typing import Mapping

from struggler.engine import Engine
from struggler.players.base import Player
from struggler.types import Side


def play_game(engine: Engine, players: Mapping[Side, Player]) -> Side | None:
    """Run `engine` to completion, returning the winner (or None on a draw)."""
    while not engine.is_terminal:
        decision = engine.pending_decision
        if decision.actor is Side.CHANCE:
            # Chance decisions carry exactly one pre-rolled option — nothing
            # for a Player to decide, so the runner resolves it directly.
            engine.step(decision.options[0])
            continue
        observation = engine.observe(decision.actor)
        action = players[decision.actor].choose_action(observation)
        engine.step(action)
    return engine.winner
