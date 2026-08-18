"""Drives a game to completion by delegating each decision to a `Player`.

The one piece of orchestration logic needed on top of `Engine`: it makes
human-vs-human, human-vs-bot, and bot-vs-bot all the same code path,
distinguished only by which `Player` is registered for which `Side`.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from struggler.engine import Engine, Side
from struggler.engine.player import Player
from struggler.engine.replay import GameLogWriter, HistoryBuilder


def play_game(
    engine: Engine,
    players: Mapping[Side, Player],
    *,
    log_path: str | None = None,
    history_builder: HistoryBuilder | None = None,
    initial_actions: Sequence[Mapping[str, Any]] | None = None,
) -> Side | None:
    """Run `engine` to completion, returning the winner (or None on a draw).

    If `log_path` is given, every step is also recorded to that path as a
    replay-log (see `engine.replay.GameLogWriter`) — the full game record,
    distinct from and independent of any LLM player's own reasoning log.

    `engine` is normally fresh (`Engine.new_game(...)`), starting `history`
    from empty. To resume a game already in progress instead, pass a
    `history_builder` from `engine.replay.replay_history` (built from `engine`
    itself already advanced to that point) — the buffering it applies to
    secret `HEADLINE_PLAY` pairs (see below) then continues seamlessly
    across the resume boundary — and `initial_actions` (that same log's
    `"actions"`) so `log_path`'s on-disk record continues instead of
    restarting.
    """
    builder = history_builder if history_builder is not None else HistoryBuilder()
    log_writer = (
        GameLogWriter(log_path, engine, initial_actions=initial_actions) if log_path is not None else None
    )
    while not engine.is_terminal:
        decision = engine.pending_decision
        if decision.actor is Side.CHANCE and Side.CHANCE not in players:
            # Chance decisions normally carry exactly one pre-rolled option —
            # nothing for a Player to decide, so the runner resolves it
            # directly. Physical-mode games register a `Side.CHANCE` player
            # (the operator console): see the branch below.
            action = decision.options[0]
        else:
            # In physical mode, `players[Side.CHANCE]` is the operator
            # console, which also answers the physical side's own decisions
            # (registered under `players[physical_side]` too) — it's the
            # single source of truth for every dice roll, card deal, and
            # physical move, regardless of which side or CHANCE they're
            # nominally attributed to.
            obs_side = decision.actor if decision.actor in (Side.US, Side.USSR) else engine.physical_side
            observation = engine.observe(obs_side)
            responder = decision.actor if decision.actor in players else Side.CHANCE
            action = players[responder].choose_action(observation, builder.history)
        engine.step(action)

        # Headline cards are picked secretly (USSR then US) and only revealed
        # once both are chosen; `HistoryBuilder` buffers both HEADLINE_PLAY
        # events instead of exposing each immediately, keeping the second
        # picker's `history` from leaking the first picker's card before
        # their own choice is locked in. The on-disk game log below is NOT
        # buffered the same way: it isn't consulted by any Player (only the
        # engine API is), so that secrecy mandate doesn't apply to it —
        # though a human reading the file mid-game could see a still-secret
        # headline pick before it's revealed in-game.
        event = builder.record(decision, action, engine)
        if log_writer is not None:
            log_writer.record_step(event)
    builder.finalize()
    if log_writer is not None:
        log_writer.finalize(engine.winner)
    return engine.winner
