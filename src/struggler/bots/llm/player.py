"""The LLM reasoning-layer bot (roadmap tier 3): a `Player` backed by an
injected `LLMClient`.

Design:

- A `pending_decision` with exactly one legal option is returned directly,
  with no LLM call at all.
- Otherwise the bot asks the LLM to predict a short *batch* of its own
  upcoming decisions in one call (mandate #2's atomic action space keeps
  each individual decision small, and CHANCE decisions never reach a
  Player at all -- runner.py resolves them directly -- so "the next
  several decisions with no intervening uncertainty" is a real, useful
  unit to plan in one shot). Each predicted step is validated against the
  *live* `Decision.options` only when actually reached; a mismatch
  discards the rest of the stale plan and triggers a fresh call.
- Memory is a single, ever-growing conversation (`self._messages`) per
  `LLMPlayer` instance, resent in full on every call -- relying on the
  provider's prompt caching for cost, not on client-side summarization.
  No compaction/safety-valve in v1 (see "Known limitations" below).
- Every real LLM-consulting `choose_action` call commits exactly one
  "user" turn and one "assistant" turn to `self._messages`, even when an
  internal retry happened first -- retry noise is resolved locally before
  anything is committed, so the persisted conversation always alternates
  cleanly (a strict requirement of the Messages/Chat Completions APIs) and
  never carries transient error chatter forward into the game's memory.
- Reasoning/justification is recorded in `self.journal`, entirely outside
  the `Engine` and the replay-log format -- a `Player`'s own bookkeeping,
  never engine state.
- If `log_path` is given, every real LLM-consulting call also persists a
  full JSON snapshot (conversation, pending plan, journal, cumulative
  token usage) via `conversation_log.py`. Passing the same `log_path` to a
  fresh `LLMPlayer` in a later process resumes from it automatically --
  see `conversation_log.py`'s module docstring for the exact resumption
  contract (this only covers the bot's own state; the caller is
  responsible for the `Engine` and `history` halves of resuming a game).

Known limitations (v1, documented rather than engineered around):
- Resuming from `log_path` does not restore `self._rng`'s exact internal
  sequence position -- a fresh instance's `random.Random(seed)` starts
  over. Acceptable because `_rng` is only ever consulted on the fallback
  path (picking *a* legal action after total LLM failure); mandate #3's
  determinism guarantee is about the engine's own RNG, not a bot's
  internal fallback RNG.
- No context-budget safety valve: the conversation resends in full every
  call; a pathological game could in theory approach the model's context
  limit.
- No full card event/flavor text is available to the model, only the
  mechanical facts in `cards.json` plus `event_summaries.py`'s hand-
  maintained short mechanical summaries (which can drift from `events.py`
  as M3 evolves -- no automated sync check in v1).
- Two concrete provider adapters ship (`anthropic_client.py`,
  `openai_client.py`), though `LLMClient` itself is provider-agnostic and
  a third is just another adapter module away.
- The plan-queue matcher checks legality only, not optimality: a step
  predicted past a CHANCE roll can be silently consumed if it happens to
  remain legal regardless of the roll's actual outcome -- the system
  prompt asks the model to avoid this, the mechanism itself doesn't
  enforce it. The same class of imperfection `GreedyPlayer` already has
  from having no lookahead.
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Mapping, Sequence

from struggler.bots.llm import conversation_log
from struggler.bots.llm.client import LLMClient, LLMClientError, LLMMessage, LLMRequest
from struggler.bots.llm.conversation_log import ConversationSnapshot, JournalEntry
from struggler.bots.llm.prompt import build_system_prompt, build_user_turn
from struggler.bots.llm.schema import (
    OUTPUT_SPEC,
    DecisionPlan,
    PlannedStep,
    PlanParseError,
    parse_plan_response,
)
from struggler.engine import Action, Decision, Observation
from struggler.engine.player import Event

_MAX_RETRIES = 1  # one retry after an invalid response, then fall back to RNG


def _add_usage(a: Mapping[str, int], b: Mapping[str, int]) -> dict[str, int]:
    return {
        "input_tokens": a.get("input_tokens", 0) + b.get("input_tokens", 0),
        "output_tokens": a.get("output_tokens", 0) + b.get("output_tokens", 0),
    }


def _find_matching_option(options: Sequence[Action], payload: dict) -> Action | None:
    """The first live option whose payload agrees with `payload` on every
    key `payload` specifies -- a subset match, since `payload` only ever
    carries the one key relevant to its decision kind (see schema.py)."""
    for action in options:
        if all(action.payload.get(k) == v for k, v in payload.items()):
            return action
    return None


class LLMPlayer:
    def __init__(
        self,
        client: LLMClient,
        *,
        seed: int = 0,
        max_plan_steps: int = 8,
        log_path: str | Path | None = None,
    ) -> None:
        self._client = client
        self._rng = random.Random(seed)  # own seeded RNG -- never the engine's (RandomPlayer convention)
        self._max_plan_steps = max_plan_steps
        self._seed = seed
        self._log_path = Path(log_path) if log_path is not None else None
        self._messages: list[LLMMessage] = []  # the one growing conversation = memory
        self._plan: deque[PlannedStep] = deque()
        self._last_seen = 0  # index into `history`; advances only on a real LLM call
        self.journal: list[JournalEntry] = []
        self.cumulative_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        self._created_at = conversation_log.now_iso()

        snapshot = conversation_log.load(self._log_path) if self._log_path is not None else None
        if snapshot is not None:
            self._messages = list(snapshot.messages)
            self._plan = deque(snapshot.plan)
            self._last_seen = snapshot.last_seen
            self.journal = list(snapshot.journal)
            self.cumulative_usage = dict(snapshot.cumulative_usage)
            self._created_at = snapshot.created_at  # preserve the original creation time across resumes
        # else: fresh start. If log_path is set, the file is created on the
        # first real LLM call (_persist_if_configured), never eagerly here.

    def choose_action(self, observation: Observation, history: Sequence[Event]) -> Action:
        if len(history) < self._last_seen:
            raise ValueError(
                f"history has {len(history)} entries but this player's resumed "
                f"last_seen is {self._last_seen}; the caller must reconstruct a "
                f"history at least this long, in the same order, per the "
                f"resumption contract documented in conversation_log.py"
            )

        decision = observation.pending_decision

        if len(decision.options) == 1:
            return decision.options[0]

        action = self._try_consume_plan(decision)
        if action is not None:
            return action

        self._plan.clear()
        new_events = history[self._last_seen :]
        self._last_seen = len(history)
        return self._call_llm_and_consume(observation, decision, new_events)

    def _try_consume_plan(self, decision: Decision) -> Action | None:
        if not self._plan:
            return None
        step = self._plan[0]
        if step.kind is not decision.kind:
            return None
        action = _find_matching_option(decision.options, dict(step.payload))
        if action is None:
            return None
        self._plan.popleft()
        return action

    def _call_llm_and_consume(
        self, observation: Observation, decision: Decision, new_events: Sequence[Event]
    ) -> Action:
        user_text = build_user_turn(observation, decision, new_events)
        plan, first_action, assistant_text, error, usage = self._request_plan_with_retry(
            user_text, decision
        )

        # Exactly one user + one assistant turn is committed per call, no
        # matter how many internal retries happened, so the persisted
        # conversation always alternates cleanly.
        self._messages.append(LLMMessage(role="user", content=user_text))
        self._messages.append(LLMMessage(role="assistant", content=assistant_text))
        self.cumulative_usage = _add_usage(self.cumulative_usage, usage)
        timestamp = conversation_log.now_iso()

        if plan is not None and first_action is not None:
            self._plan = deque(plan.steps[1 : 1 + self._max_plan_steps])
            self.journal.append(
                JournalEntry(
                    decision_id=decision.id,
                    justification=plan.justification,
                    fallback_used=False,
                    usage=usage,
                    timestamp=timestamp,
                )
            )
            action = first_action
        else:
            action = self._rng.choice(decision.options)
            self.journal.append(
                JournalEntry(
                    decision_id=decision.id,
                    justification=None,
                    fallback_used=True,
                    fallback_reason=error or "unknown",
                    usage=usage,
                    timestamp=timestamp,
                )
            )

        self._persist_if_configured()
        return action

    def _request_plan_with_retry(
        self, user_text: str, decision: Decision
    ) -> tuple[DecisionPlan | None, Action | None, str, str | None, dict[str, int]]:
        """Attempts the LLM call up to `_MAX_RETRIES + 1` times, using a
        local scratch message list so failed attempts never pollute the
        persisted conversation. A response is retried both when it fails to
        parse at all and when it parses but its first step doesn't match
        the live `decision` (wrong kind, or an option that isn't actually
        legal right now) -- either way the model gets one chance to correct
        itself with the reason appended, before `LLMPlayer` gives up and
        falls back to a legal choice via its own seeded RNG.

        Returns `(plan, first_action, assistant_text, error, usage)`: on
        success `error` is `None`, `first_action` is the live option the
        plan's first step resolved to, and `assistant_text` is the model's
        real raw response; on total failure `plan`/`first_action` are
        `None` and `assistant_text` is a synthetic note recording the
        fallback, so the one committed conversation turn still makes sense
        on replay. `usage` is the summed token usage across every attempt
        that actually received a response (a `LLMClientError` raised
        before any response arrives contributes nothing, since none was
        received).
        """
        attempt_messages = list(self._messages) + [LLMMessage(role="user", content=user_text)]
        last_error: str | None = None
        total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        for _ in range(_MAX_RETRIES + 1):
            request = LLMRequest(
                system=build_system_prompt(),
                messages=tuple(attempt_messages),
                output=OUTPUT_SPEC,
            )
            try:
                response = self._client.complete(request)
            except LLMClientError as exc:
                last_error = str(exc)
                attempt_messages.append(
                    LLMMessage(role="assistant", content=f"[invalid response: {last_error}]")
                )
                attempt_messages.append(
                    LLMMessage(
                        role="user",
                        content="That response was invalid. Please try again, following the schema exactly.",
                    )
                )
                continue  # no response received -- nothing to add to total_usage

            total_usage = _add_usage(total_usage, response.usage)

            try:
                plan = parse_plan_response(response.structured)
            except PlanParseError as exc:
                last_error = str(exc)
                attempt_messages.append(
                    LLMMessage(role="assistant", content=f"[invalid response: {last_error}]")
                )
                attempt_messages.append(
                    LLMMessage(
                        role="user",
                        content="That response was invalid. Please try again, following the schema exactly.",
                    )
                )
                continue

            first_step = plan.steps[0]
            first_action = (
                _find_matching_option(decision.options, dict(first_step.payload))
                if first_step.kind is decision.kind
                else None
            )
            if first_action is None:
                last_error = "first planned step did not match the live decision"
                attempt_messages.append(LLMMessage(role="assistant", content=response.raw_text))
                attempt_messages.append(
                    LLMMessage(
                        role="user",
                        content=(
                            "Your first step didn't match the current live decision (wrong "
                            "kind, or that option isn't actually legal right now). Please try "
                            "again using one of the live options shown."
                        ),
                    )
                )
                continue

            return plan, first_action, response.raw_text, None, total_usage

        note = f"[fallback: no valid plan after {_MAX_RETRIES + 1} attempt(s): {last_error}]"
        return None, None, note, last_error, total_usage

    def _persist_if_configured(self) -> None:
        if self._log_path is None:
            return
        snapshot = ConversationSnapshot(
            seed=self._seed,
            provider=getattr(self._client, "provider_name", "unknown"),
            model=getattr(self._client, "model_name", "unknown"),
            created_at=self._created_at,
            updated_at=self._created_at,  # conversation_log.save() overwrites this with now_iso()
            last_seen=self._last_seen,
            cumulative_usage=dict(self.cumulative_usage),
            messages=tuple(self._messages),
            plan=tuple(self._plan),
            journal=tuple(self.journal),
        )
        conversation_log.save(self._log_path, snapshot)
