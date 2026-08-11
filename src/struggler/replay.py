"""Deterministic replay: the primary testing strategy per CLAUDE.md.

A replay log is {seed, actions, checkpoints} plus a start descriptor:

- M2 full games set ``"new_game": true`` (optionally ``"include_optional"``);
  the whole game — headline picks, card plays, and dice — lives in
  ``actions``, since chance is a logged CHANCE decision (mandate #3).
- M1 sandbox logs instead carry a ``setup`` object naming one of the
  begin_* Ops-only entry points, a scaffold for the pre-card milestone.

Because chance is replayed from the same seeded RNG, either kind of log is
byte-for-byte reproducible with no separate RNG trace to keep in sync.
"""

from __future__ import annotations

from typing import Any

from struggler.engine import Engine
from struggler.types import Action, DecisionKind, Side

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
            seed=log["seed"], include_optional=log.get("include_optional", False)
        )
    engine = Engine(seed=log["seed"])
    apply_setup(engine, log["setup"])
    return engine


def decode_action(data: dict) -> Action:
    return Action(kind=DecisionKind(data["kind"]), payload=data["payload"])


def encode_action(action: Action) -> dict:
    return {"kind": action.kind.value, "payload": dict(action.payload)}


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
    """Replay a log, recording engine.serialize() after each checkpointed step."""
    engine = make_engine(log)
    checkpoint_steps = {c["after_step"] for c in log["checkpoints"]}
    recorded = []
    for i, action_data in enumerate(log["actions"], start=1):
        engine.step(decode_action(action_data))
        if i in checkpoint_steps:
            recorded.append({"after_step": i, "state": engine.serialize()})
    return recorded
