"""Drives a game to completion by delegating each decision to a `Player`.

The one piece of orchestration logic needed on top of `Engine`: it makes
human-vs-human, human-vs-bot, and bot-vs-bot all the same code path,
distinguished only by which `Player` is registered for which `Side`.
"""

from __future__ import annotations

from typing import Mapping

from struggler.engine import DecisionKind, Engine, Side
from struggler.engine.player import Event, Player


def play_game(engine: Engine, players: Mapping[Side, Player]) -> Side | None:
    """Run `engine` to completion, returning the winner (or None on a draw)."""
    history: list[Event] = []
    # Headline cards are picked secretly (USSR then US) and only revealed
    # once both are chosen. Buffering both HEADLINE_PLAY events here — instead
    # of appending each immediately — keeps the second picker's `history` from
    # leaking the first picker's card before their own choice is locked in.
    pending_headline: list[Event] = []
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

        country = action.payload.get("country")
        country_influence: Mapping[str, int] = {}
        country_control: str | None = None
        if country is not None and country in engine.board.influence:
            country_influence = dict(engine.board.influence[country])
            control = engine.board.control(country)
            country_control = control.value if control is not None else None

        event = Event(
            actor=decision.actor,
            decision=decision,
            action=action,
            defcon=engine.defcon,
            vp=engine.vp,
            turn=engine.turn,
            action_round=engine.action_round,
            space_race=dict(engine.space_race),
            military_ops=dict(engine.military_ops),
            country=country,
            country_influence=country_influence,
            country_control=country_control,
        )
        if decision.kind is DecisionKind.HEADLINE_PLAY:
            pending_headline.append(event)
            if len(pending_headline) == 2:
                history.extend(pending_headline)
                pending_headline = []
        else:
            history.append(event)
    history.extend(pending_headline)
    return engine.winner
