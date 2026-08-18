"""Deterministic replay logs: the primary testing strategy (docs/TESTING.md)."""

import json
from pathlib import Path

from struggler.bots.naive import FirstLegalPlayer, RandomPlayer
from struggler.engine import Engine, Side
from struggler.engine.replay import GameLogWriter, HistoryBuilder, replay_history, run_replay, run_with_checkpoints
from struggler.runner import play_game

REPLAY_DIR = Path(__file__).parent / "replays"


def _load(name: str) -> dict:
    with (REPLAY_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


def test_golden_replay_matches_recorded_checkpoints():
    log = _load("influence_basic.json")
    recorded = run_with_checkpoints(log)
    assert len(recorded) == len(log["checkpoints"])
    for rec, checkpoint in zip(recorded, log["checkpoints"]):
        assert rec["after_step"] == checkpoint["after_step"]
        assert rec["state"] == checkpoint["state"]


def test_golden_physical_replay_matches_recorded_checkpoints():
    log = _load("physical_basic.json")
    recorded = run_with_checkpoints(log)
    assert len(recorded) == len(log["checkpoints"])
    for rec, checkpoint in zip(recorded, log["checkpoints"]):
        assert rec["after_step"] == checkpoint["after_step"]
        assert rec["state"] == checkpoint["state"]
    final_state = recorded[-1]["state"]
    assert final_state["physical_mode"] is True
    assert final_state["physical_side"] == "USSR"


def test_replay_is_deterministic_across_independent_runs():
    log = _load("influence_basic.json")
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


def test_replay_history_matches_run_replay_and_the_logged_actions(tmp_path):
    log_path = tmp_path / "full_game.json"
    engine = Engine.new_game(seed=3)
    players = {Side.US: FirstLegalPlayer(), Side.USSR: RandomPlayer(seed=4)}
    play_game(engine, players, log_path=str(log_path))
    log = json.loads(log_path.read_text(encoding="utf-8"))

    replayed_engine, history = replay_history(log)

    assert replayed_engine.serialize() == run_replay(log).serialize()
    # The game ended, so any secretly-buffered headline pick has been
    # flushed -- history covers every logged action, in order.
    assert [e.action.payload for e in history] == [a["payload"] for a in log["actions"]]


def test_game_log_writer_continues_from_initial_actions(tmp_path):
    engine = Engine.new_game(seed=7)
    prior_action = {
        "actor": "USSR",
        "kind": "place_influence",
        "payload": {"country": "Poland"},
        "defcon": 5,
        "vp": 0,
        "turn": 1,
        "action_round": 1,
    }
    writer = GameLogWriter(tmp_path / "game.json", engine, initial_actions=[prior_action])

    writer.finalize(None)

    log = json.loads((tmp_path / "game.json").read_text(encoding="utf-8"))
    assert log["actions"] == [prior_action]


def test_resuming_from_a_trimmed_log_continues_the_same_on_disk_record(tmp_path):
    # Play a full game once to get a real, non-trivial action log.
    full_log_path = tmp_path / "full.json"
    engine = Engine.new_game(seed=3)
    players = {Side.US: FirstLegalPlayer(), Side.USSR: RandomPlayer(seed=4)}
    play_game(engine, players, log_path=str(full_log_path))
    full_log = json.loads(full_log_path.read_text(encoding="utf-8"))
    full_actions = full_log["actions"]
    assert len(full_actions) > 10  # enough steps to cut mid-game

    # Simulate hand-trimming the log (e.g. to undo a bad play) and resuming
    # from that earlier point instead of letting the game finish.
    cut = len(full_actions) // 2
    trimmed_log = {**full_log, "actions": full_actions[:cut], "winner": None}
    resumed_engine, history = replay_history(trimmed_log)
    assert not resumed_engine.is_terminal

    resume_path = tmp_path / "resumed.json"
    winner = play_game(
        resumed_engine,
        {Side.US: FirstLegalPlayer(), Side.USSR: RandomPlayer(seed=4)},
        log_path=str(resume_path),
        history_builder=HistoryBuilder(initial_history=history),
        initial_actions=trimmed_log["actions"],
    )

    resumed_log = json.loads(resume_path.read_text(encoding="utf-8"))
    assert resumed_log["actions"][:cut] == full_actions[:cut]  # trimmed prefix preserved verbatim
    assert len(resumed_log["actions"]) > cut  # new steps were appended after resuming
    assert resumed_log["winner"] == (winner.value if winner is not None else None)

    # The continued log is itself a complete, independently replayable record.
    replayed = run_replay(resumed_log)
    assert replayed.is_terminal
    assert replayed.serialize() == resumed_engine.serialize()
