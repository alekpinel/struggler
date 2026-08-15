"""Deterministic replay logs: the primary testing strategy (CLAUDE.md)."""

import json
from pathlib import Path

from struggler.bots.naive import FirstLegalPlayer, RandomPlayer
from struggler.engine import Engine, Side
from struggler.engine.replay import GameLogWriter, run_replay, run_with_checkpoints
from struggler.runner import play_game

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


def test_game_log_writer_finalize_with_no_actions(tmp_path):
    engine = Engine.new_game(seed=7)
    writer = GameLogWriter(tmp_path / "game.json", engine)

    writer.finalize(Side.US)

    log = json.loads((tmp_path / "game.json").read_text(encoding="utf-8"))
    assert log["actions"] == []
    assert log["winner"] == "US"
    assert "checkpoints" not in log


def test_play_game_log_path_produces_a_readable_replayable_log(tmp_path):
    log_path = tmp_path / "full_game.json"
    engine = Engine.new_game(seed=3)
    players = {Side.US: FirstLegalPlayer(), Side.USSR: RandomPlayer(seed=4)}

    winner = play_game(engine, players, log_path=str(log_path))

    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert "checkpoints" not in log
    assert log["winner"] == (winner.value if winner is not None else None)
    assert log["actions"]
    for entry in log["actions"]:
        assert set(entry) >= {"actor", "kind", "payload", "defcon", "vp", "turn", "action_round"}
        assert entry["actor"] in {"US", "USSR", "CHANCE"}

    country_entries = [e for e in log["actions"] if "country" in e]
    assert country_entries
    assert set(country_entries[0]["country_influence"]) <= {"US", "USSR"}

    # seed + actions alone is enough to reproduce the exact final state,
    # even without an embedded checkpoint (mandate #3).
    replayed = run_replay(log)
    assert replayed.is_terminal
    assert replayed.winner == winner
    assert replayed.serialize() == engine.serialize()
