"""Deterministic replay: the primary testing strategy per CLAUDE.md.

A replay log is {seed, setup, actions, checkpoints}. `setup` is an M1-only
affordance standing in for the "Ops-only actions driven directly for
testing" scaffolding (begin_influence_operations / begin_coup /
begin_realignment_operations) — M2 will replace it with a PLAY_CARD action
that belongs in `actions` like everything else, and this field goes away.
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


def decode_action(data: dict) -> Action:
    return Action(kind=DecisionKind(data["kind"]), payload=data["payload"])


def encode_action(action: Action) -> dict:
    return {"kind": action.kind.value, "payload": dict(action.payload)}


def run_replay(log: dict[str, Any]) -> Engine:
    """Replay a log from scratch, returning the resulting Engine.

    Does not check checkpoints itself — callers compare
    engine.serialize() against log["checkpoints"] as needed.
    """
    engine = Engine(seed=log["seed"])
    apply_setup(engine, log["setup"])
    for action_data in log["actions"]:
        engine.step(decode_action(action_data))
    return engine


def run_with_checkpoints(log: dict[str, Any]) -> list[dict]:
    """Replay a log, recording engine.serialize() after each checkpointed step."""
    engine = Engine(seed=log["seed"])
    apply_setup(engine, log["setup"])
    checkpoint_steps = {c["after_step"] for c in log["checkpoints"]}
    recorded = []
    for i, action_data in enumerate(log["actions"], start=1):
        engine.step(decode_action(action_data))
        if i in checkpoint_steps:
            recorded.append({"after_step": i, "state": engine.serialize()})
    return recorded
