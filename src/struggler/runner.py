"""Drives a game to completion by delegating each decision to a `Player`.

The one piece of orchestration logic needed on top of `Engine`: it makes
human-vs-human, human-vs-bot, and bot-vs-bot all the same code path,
distinguished only by which `Player` is registered for which `Side`.
"""

from __future__ import annotations

from typing import Mapping

from struggler.engine import Engine, Side
from struggler.engine.player import Event, Player


def play_game(engine: Engine, players: Mapping[Side, Player]) -> Side | None:
    """Run `engine` to completion, returning the winner (or None on a draw)."""
    history: list[Event] = []
    while not engine.is_terminal:
        decision = engine.pending_decision
        if decision.actor is Side.CHANCE:
            # Chance decisions carry exactly one pre-rolled option — nothing
            # for a Player to decide, so the runner resolves it directly.
            action = decision.options[0]
        else:
            observation = engine.observe(decision.actor)
            action = players[decision.actor].choose_action(observation, history)
        engine.step(action)
        history.append(
            Event(
                actor=decision.actor,
                decision=decision,
                action=action,
                defcon=engine.defcon,
                vp=engine.vp,
                turn=engine.turn,
                action_round=engine.action_round,
                space_race=dict(engine.space_race),
                military_ops=dict(engine.military_ops),
            )
        )
    return engine.winner
