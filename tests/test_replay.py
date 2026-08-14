"""Deterministic replay logs: the primary testing strategy (CLAUDE.md)."""

import json
from pathlib import Path

from struggler.engine import Engine, Side
from struggler.engine.replay import run_replay, run_with_checkpoints

REPLAY_DIR = Path(__file__).parent / "replays"


def _load(name: str) -> dict:
    with (REPLAY_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


def test_golden_replay_matches_recorded_checkpoints():
    log = _load("m1_influence_basic.json")
    recorded = run_with_checkpoints(log)
    assert len(recorded) == len(log["checkpoints"])
    for rec, checkpoint in zip(recorded, log["checkpoints"]):
        assert rec["after_step"] == checkpoint["after_step"]
        assert rec["state"] == checkpoint["state"]


def test_replay_is_deterministic_across_independent_runs():
    log = _load("m1_influence_basic.json")
    engine_a = run_replay(log)
    engine_b = run_replay(log)
    assert engine_a.serialize() == engine_b.serialize()


def test_dice_driven_replay_is_deterministic():
    def play(seed: int) -> tuple[Engine, list]:
        engine = Engine(seed=seed)
        engine.begin_coup(Side.US, 3)
        actions = []
        while engine.pending_decision is not None:
            action = engine.legal_actions()[0]
            engine.step(action)
            actions.append(action)
        return engine, actions

    engine_a, actions_a = play(999)
    engine_b, actions_b = play(999)
    assert actions_a == actions_b
    assert engine_a.serialize() == engine_b.serialize()
