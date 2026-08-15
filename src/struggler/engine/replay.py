"""Deterministic replay: the primary testing strategy per CLAUDE.md.

A replay log is {seed, actions, checkpoints} plus a start descriptor:

- M2 full games set ``"new_game": true`` (optionally ``"include_optional"``);
  the whole game — headline picks, card plays, and dice — lives in
  ``actions``, since chance is a logged CHANCE decision (mandate #3).
  Adding ``"events": true`` turns on the M3 event layer (see events.py).
- M1 sandbox logs instead carry a ``setup`` object naming one of the
  begin_* Ops-only entry points, a scaffold for the pre-card milestone.

Because chance is replayed from the same seeded RNG, either kind of log is
byte-for-byte reproducible with no separate RNG trace to keep in sync.

Golden fixtures under tests/replays/ are hand/script-built and read-only,
and pin a ``checkpoints`` list of full ``engine.serialize()`` snapshots so
``run_with_checkpoints`` can assert byte-for-byte equality against a fixed
expectation. ``GameLogWriter`` is the write direction for a *live* game
instead: it lets ``runner.play_game`` record an actually-played game to
disk, in the same ``new_game``/``actions`` shape (each entry enriched via
``encode_event`` with who acted and what changed) but without
``checkpoints`` — a played game isn't being checked against a pinned
snapshot, and ``seed + actions`` alone already reproduces it exactly, so
the file stays readable instead of embedding raw internal state. It's
still fully replayable via ``run_replay``.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

from struggler.engine.core import Engine
from struggler.engine.player import Event
from struggler.engine.types import Action, DecisionKind, Side

_SETUP_KINDS = {
    "begin_influence_operations": Engine.begin_influence_operations,
    "begin_coup": Engine.begin_coup,
    "begin_realignment_operations": Engine.begin_realignment_operations,
}


def apply_setup(engine: Engine, setup: dict) -> None:
    method = _SETUP_KINDS[setup["kind"]]
    method(engine, Side(setup["side"]), setup["ops"])


def make_engine(log: dict[str, Any]) -> Engine:
    """Create and prime the engine for a log, before any `actions` replay.

    M2 logs (`new_game`) build a full game; M1 logs run their `setup`.
    """
    if log.get("new_game"):
        return Engine.new_game(
            seed=log["seed"],
            include_optional=log.get("include_optional", False),
            events=log.get("events", False),
        )
    engine = Engine(seed=log["seed"])
    apply_setup(engine, log["setup"])
    return engine


def decode_action(data: dict) -> Action:
    return Action(kind=DecisionKind(data["kind"]), payload=data["payload"])


def encode_action(action: Action) -> dict:
    return {"kind": action.kind.value, "payload": dict(action.payload)}


def encode_event(event: Event) -> dict:
    """Encode a resolved `Event` for the live game log.

    Carries the same fields `engine.human._format_event` shows a human
    player between prompts (actor, the action, and — only when it targets
    a country — that country's resulting influence/control, plus
    DEFCON/VP/turn/round). Still decodable by `decode_action` (which only
    reads "kind"/"payload"), so replay compatibility is unaffected.
    """
    data = {"actor": event.actor.value, **encode_action(event.action)}
    if event.country is not None:
        data["country"] = event.country
        data["country_influence"] = dict(event.country_influence)
        data["country_control"] = event.country_control
    data["defcon"] = event.defcon
    data["vp"] = event.vp
    data["turn"] = event.turn
    data["action_round"] = event.action_round
    return data


def run_replay(log: dict[str, Any]) -> Engine:
    """Replay a log from scratch, returning the resulting Engine.

    Does not check checkpoints itself — callers compare
    engine.serialize() against log["checkpoints"] as needed.
    """
    engine = make_engine(log)
    for action_data in log["actions"]:
        engine.step(decode_action(action_data))
    return engine


def run_with_checkpoints(log: dict[str, Any]) -> list[dict]:
    """Replay a log, recording engine.serialize() after each checkpointed step.

    Golden-fixture-only: needs `log["checkpoints"]`, which live logs written
    by `GameLogWriter` don't carry (see its docstring). Use `run_replay` for
    those.
    """
    engine = make_engine(log)
    checkpoint_steps = {c["after_step"] for c in log["checkpoints"]}
    recorded = []
    for i, action_data in enumerate(log["actions"], start=1):
        engine.step(decode_action(action_data))
        if i in checkpoint_steps:
            recorded.append({"after_step": i, "state": engine.serialize()})
    return recorded


class GameLogWriter:
    """Records a live `play_game` run to `path`, as a lean, human-readable
    `new_game` replay log: `{seed, new_game, include_optional, events,
    actions, winner}`, no `checkpoints`.

    Each entry in `actions` is `encode_event`'s output — actor, the action,
    and the same DEFCON/VP/turn/country context `HumanPlayer` shows a human
    between prompts — still decodable by `decode_action`/replayable via
    `run_replay`. `seed`/`include_optional`/`events_enabled` are read once
    at construction (they're static for the whole game), so no write needs
    `engine.serialize()`'s full internal dump (rng state, hands, pile
    order, ...) — that's only meaningful for pinning a golden fixture
    (`run_with_checkpoints`), not for a live game's own record, since
    `seed + actions` alone is already enough to reproduce it exactly.

    Assumes `engine` was built via `Engine.new_game(...)` — true for every
    game driven through `runner.play_game`. Call `record_step(event)` right
    after each resolved decision, and `finalize(winner)` once when the game
    ends. The file is atomically rewritten on every call (same
    tempfile+`os.replace` pattern as `bots/llm/conversation_log.save`), so a
    crash mid-game still leaves a replayable, if truncated, log on disk.

    A logging failure never raises: it's turned into a `RuntimeWarning`, the
    same contract `conversation_log.save` uses, so a broken log path can
    never break the actual game.
    """

    def __init__(self, path: str | Path, engine: Engine) -> None:
        self._path = Path(path)
        state = engine.serialize()
        self._seed = state["seed"]
        self._include_optional = engine.include_optional
        self._events_enabled = engine.events_enabled
        self._actions: list[dict[str, Any]] = []
        self._winner: str | None = None

    def record_step(self, event: Event) -> None:
        self._actions.append(encode_event(event))
        self._write()

    def finalize(self, winner: Side | None) -> None:
        """Call once after the game ends, to stamp the result."""
        self._winner = winner.value if winner is not None else None
        self._write()

    def _write(self) -> None:
        log = {
            "seed": self._seed,
            "new_game": True,
            "include_optional": self._include_optional,
            "events": self._events_enabled,
            "actions": self._actions,
            "winner": self._winner,
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(log, f, indent=2)
                os.replace(tmp_name, self._path)
            finally:
                if os.path.exists(tmp_name):
                    os.remove(tmp_name)
        except Exception as exc:  # logging must never break the caller
            warnings.warn(f"GameLogWriter._write({self._path!r}) failed: {exc}", RuntimeWarning, stacklevel=2)
