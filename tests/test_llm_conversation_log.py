"""Tests for conversation_log.py: save/load round-tripping, atomicity, and
the "never raise" contract, all against the local filesystem only."""

from __future__ import annotations

import dataclasses
import json
import os

import pytest

from struggler.bots.llm import conversation_log
from struggler.bots.llm.client import LLMMessage
from struggler.bots.llm.conversation_log import ConversationSnapshot, JournalEntry
from struggler.bots.llm.schema import PlannedStep, TurnPlan
from struggler.engine import DecisionKind


def _sample_snapshot() -> ConversationSnapshot:
    return ConversationSnapshot(
        seed=42,
        provider="fake",
        model="fake-model",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        last_seen=3,
        cumulative_usage={"input_tokens": 10, "output_tokens": 5},
        messages=(
            LLMMessage(role="user", content="hello"),
            LLMMessage(role="assistant", content='{"justification": "hi", "steps": []}'),
        ),
        plan=(PlannedStep(kind=DecisionKind.PLACE_INFLUENCE, payload={"country": "Poland"}),),
        journal=(
            JournalEntry(
                decision_id=1,
                justification="because",
                fallback_used=False,
                fallback_reason=None,
                usage={"input_tokens": 10, "output_tokens": 5},
                timestamp="2026-01-01T00:00:00+00:00",
                raw_responses=('{"justification": "because", "steps": []}',),
            ),
        ),
    )


def test_save_creates_parent_dirs_and_round_trips_via_load(tmp_path):
    path = tmp_path / "a" / "b" / "log.json"
    snapshot = _sample_snapshot()

    conversation_log.save(path, snapshot)
    loaded = conversation_log.load(path)

    assert loaded is not None
    assert loaded.seed == snapshot.seed
    assert loaded.provider == snapshot.provider
    assert loaded.model == snapshot.model
    assert loaded.created_at == snapshot.created_at
    assert loaded.last_seen == snapshot.last_seen
    assert loaded.cumulative_usage == snapshot.cumulative_usage
    assert loaded.messages == snapshot.messages
    assert loaded.plan == snapshot.plan
    assert loaded.journal == snapshot.journal
    # updated_at is stamped by save() with the current time, not copied verbatim.
    assert loaded.updated_at != ""


def test_load_returns_none_when_file_missing(tmp_path):
    assert conversation_log.load(tmp_path / "does_not_exist.json") is None


def test_load_returns_none_and_warns_on_corrupted_json(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("not json")

    with pytest.warns(RuntimeWarning):
        result = conversation_log.load(path)

    assert result is None


def test_save_is_atomic_leaves_prior_file_untouched_on_replace_failure(tmp_path, monkeypatch):
    path = tmp_path / "log.json"
    first = _sample_snapshot()
    conversation_log.save(path, first)
    first_bytes = path.read_bytes()

    def _broken_replace(*args, **kwargs):
        raise OSError("simulated failure")

    monkeypatch.setattr(os, "replace", _broken_replace)

    second = ConversationSnapshot(
        seed=999,
        provider="fake",
        model="fake-model",
        created_at="2026-02-02T00:00:00+00:00",
        updated_at="2026-02-02T00:00:00+00:00",
        last_seen=0,
        cumulative_usage={},
        messages=(),
        plan=(),
        journal=(),
    )
    with pytest.warns(RuntimeWarning):
        conversation_log.save(path, second)

    assert path.read_bytes() == first_bytes  # untouched by the failed second write
    loaded = conversation_log.load(path)
    assert loaded.seed == 42  # still the first snapshot


def test_save_swallows_write_failure_and_warns(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a directory")
    path = blocked / "log.json"

    with pytest.warns(RuntimeWarning):
        conversation_log.save(path, _sample_snapshot())

    assert not path.exists()


def test_load_defaults_raw_responses_when_missing_from_older_snapshot(tmp_path):
    """`raw_responses` was added after `JournalEntry` shipped -- a snapshot
    written by an older version of this module won't have the key. Loading
    it must default to an empty tuple rather than raising."""
    path = tmp_path / "log.json"
    data = conversation_log._snapshot_to_dict(_sample_snapshot())
    del data["journal"][0]["raw_responses"]
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded = conversation_log.load(path)

    assert loaded is not None
    assert loaded.journal[0].raw_responses == ()


def test_now_iso_format():
    from datetime import datetime, timedelta, timezone

    before = datetime.now(timezone.utc)
    text = conversation_log.now_iso()
    after = datetime.now(timezone.utc)

    parsed = datetime.fromisoformat(text)
    assert parsed.utcoffset() == timedelta(0)  # UTC, as the docstring promises
    assert before <= parsed <= after


# -- turn plan (snapshot version 2) --------------------------------------------


def _sample_turn_plan() -> TurnPlan:
    return TurnPlan(
        turn=2,
        assessment="Asia is open.",
        objective="Take Thailand before Asia Scoring.",
        scoring_cards=({"card": "Asia_Scoring", "when": "AR6", "preparation": "take Thailand"},),
        card_plan=({"card": "Korean_War", "intended_use": "event", "purpose": "military ops"},),
        influence_targets=({"country": "Thailand", "why": "Battleground"},),
        military_ops_plan="Korean War pays 2.",
        defend=("East_Germany",),
        contingencies=({"trigger": "US coups Vietnam", "response": "retake it"},),
    )


def test_turn_plan_round_trips_through_save_and_load(tmp_path):
    path = tmp_path / "log.json"
    snapshot = dataclasses.replace(
        _sample_snapshot(), turn_plan=_sample_turn_plan(), planned_turn=2
    )

    conversation_log.save(path, snapshot)
    loaded = conversation_log.load(path)

    assert loaded is not None
    assert loaded.planned_turn == 2
    assert loaded.turn_plan == snapshot.turn_plan


def test_a_version_1_snapshot_still_loads_without_a_turn_plan(tmp_path):
    # Snapshots written before turn planning existed carry neither key; they
    # must resume as "no plan for this turn yet", not fail to load.
    path = tmp_path / "log.json"
    conversation_log.save(path, _sample_snapshot())
    data = json.loads(path.read_text())
    data["version"] = 1
    del data["turn_plan"]
    del data["planned_turn"]
    for entry in data["journal"]:
        del entry["kind"]
    path.write_text(json.dumps(data))

    loaded = conversation_log.load(path)

    assert loaded is not None
    assert loaded.turn_plan is None
    assert loaded.planned_turn is None
    assert loaded.journal[0].kind == "decision"


def test_journal_entry_kind_round_trips(tmp_path):
    path = tmp_path / "log.json"
    snapshot = dataclasses.replace(
        _sample_snapshot(),
        journal=(
            JournalEntry(
                decision_id=7,
                justification="plan the turn",
                fallback_used=False,
                kind="turn_plan",
            ),
        ),
    )

    conversation_log.save(path, snapshot)
    loaded = conversation_log.load(path)

    assert loaded is not None
    assert loaded.journal[0].kind == "turn_plan"
