"""Persists an `LLMPlayer`'s growing conversation, pending plan, and
journal to a JSON snapshot file -- passed to `LLMPlayer`'s constructor as
`log_path` -- so a game can be resumed in a later process. Written
atomically (temp file + `os.replace`) after every real LLM-consulting
call, so the file on disk always reflects a complete, loadable state.

Resumption contract: this module makes `LLMPlayer`'s OWN state resumable
(the conversation, pending plan, `last_seen` index, journal, cumulative
token usage) -- it does not orchestrate resuming an actual game. The
caller is independently responsible for reconstructing a matching `Engine`
(via `Engine.serialize()`/`deserialize()` or a replay log, see CLAUDE.md's
Testing strategy) and for building a `history: Sequence[Event]` of at
least `last_seen` entries, in the same order as before persistence, to
pass into `choose_action`. If `history` is shorter than the restored
`last_seen`, `LLMPlayer.choose_action` raises `ValueError` rather than
silently losing context.

This file is deliberately separate from the Engine's own serialize()/
replay-log machinery -- CLAUDE.md mandate #5's "flat, serializable state"
is about `GameState`, not `Player` state; resuming a game means loading
BOTH the Engine and each `LLMPlayer`, independently.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from struggler.bots.llm.client import LLMMessage
from struggler.bots.llm.schema import PlannedStep
from struggler.engine.types import DecisionKind


@dataclass(frozen=True)
class JournalEntry:
    """One LLM-consulting call's outcome. Kept entirely outside the Engine
    and the replay-log format -- the bot's own bookkeeping, for debugging,
    explainability, and (via `save`/`load` below) persistence."""

    decision_id: int
    justification: str | None
    fallback_used: bool
    fallback_reason: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    timestamp: str = ""


@dataclass(frozen=True)
class ConversationSnapshot:
    seed: int
    provider: str
    model: str
    created_at: str
    updated_at: str
    last_seen: int
    cumulative_usage: Mapping[str, int]
    messages: tuple[LLMMessage, ...]
    plan: tuple[PlannedStep, ...]
    journal: tuple[JournalEntry, ...]


def now_iso() -> str:
    """ISO-8601 UTC timestamp -- the one place this format is chosen, so
    every timestamp in a snapshot agrees."""
    return datetime.now(timezone.utc).isoformat()


def _snapshot_to_dict(snapshot: ConversationSnapshot) -> dict[str, Any]:
    return {
        "version": 1,
        "seed": snapshot.seed,
        "provider": snapshot.provider,
        "model": snapshot.model,
        "created_at": snapshot.created_at,
        "updated_at": snapshot.updated_at,
        "last_seen": snapshot.last_seen,
        "cumulative_usage": dict(snapshot.cumulative_usage),
        "messages": [{"role": m.role, "content": m.content} for m in snapshot.messages],
        "plan": [
            {"kind": step.kind.value, "payload": dict(step.payload)} for step in snapshot.plan
        ],
        "journal": [
            {
                "decision_id": entry.decision_id,
                "justification": entry.justification,
                "fallback_used": entry.fallback_used,
                "fallback_reason": entry.fallback_reason,
                "usage": dict(entry.usage),
                "timestamp": entry.timestamp,
            }
            for entry in snapshot.journal
        ],
    }


def _dict_to_snapshot(data: Mapping[str, Any]) -> ConversationSnapshot:
    return ConversationSnapshot(
        seed=data["seed"],
        provider=data["provider"],
        model=data["model"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        last_seen=data["last_seen"],
        cumulative_usage=dict(data["cumulative_usage"]),
        messages=tuple(
            LLMMessage(role=m["role"], content=m["content"]) for m in data["messages"]
        ),
        plan=tuple(
            PlannedStep(kind=DecisionKind(step["kind"]), payload=dict(step["payload"]))
            for step in data["plan"]
        ),
        journal=tuple(
            JournalEntry(
                decision_id=e["decision_id"],
                justification=e["justification"],
                fallback_used=e["fallback_used"],
                fallback_reason=e.get("fallback_reason"),
                usage=dict(e.get("usage", {})),
                timestamp=e.get("timestamp", ""),
            )
            for e in data["journal"]
        ),
    )


def save(path: str | Path, snapshot: ConversationSnapshot) -> None:
    """Atomically write `snapshot` to `path` (temp file in the same
    directory, then `os.replace`), stamping `updated_at` with the current
    time regardless of what `snapshot.updated_at` was.

    Never raises: any failure (permission error, disk full, a path
    component that isn't a directory, `os.replace` itself failing, ...) is
    caught and reported as a `RuntimeWarning` -- the engine-facing contract
    is that `choose_action` always returns a legal `Action`, and a logging
    failure must never be allowed to break that.
    """
    try:
        stamped = dataclasses.replace(snapshot, updated_at=now_iso())
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(_snapshot_to_dict(stamped), f, indent=2)
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
    except Exception as exc:  # logging must never break the caller
        warnings.warn(f"conversation_log.save({path!r}) failed: {exc}", RuntimeWarning, stacklevel=2)


def load(path: str | Path) -> ConversationSnapshot | None:
    """`None` if `path` doesn't exist yet (fresh start) or fails to
    parse/validate (a corrupted or foreign file -- also warns in that
    case); either way the caller should start fresh rather than crash."""
    target = Path(path)
    if not target.exists():
        return None
    try:
        with target.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return _dict_to_snapshot(data)
    except Exception as exc:
        warnings.warn(f"conversation_log.load({path!r}) failed: {exc}", RuntimeWarning, stacklevel=2)
        return None
