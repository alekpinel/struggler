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
- At the first real decision of each game turn, one extra call produces a
  *turn plan* instead of an action (`schema.TURN_PLAN_SCHEMA`): what each
  card in hand is for, when a Scoring card gets played and what must change
  in its region first, how the Military Operations requirement gets met,
  which Battlegrounds to hold or retake, and contingencies. It is then
  re-injected into every user turn for the rest of that turn, so each
  individual decision is made against a standing intent rather than from
  scratch -- `justification` is explainability, deliberately not memory.
  Planning failure never blocks the game (the turn just plays without a
  plan), and `plan_turns=False` skips the call entirely.
- Memory is a single, ever-growing conversation (`self._messages`) per
  `LLMPlayer` instance, resent in full on every call. What gets persisted
  per turn is deliberately thinner than what the live call was sent with,
  though: `prompt.build_history_entry` keeps only the event delta ("since
  your last request, X happened"), never the board report/hand dossier/
  cards-in-play a call was actually answered against -- those are a
  snapshot of one instant, true only for the live call that carried them,
  so resending them on every later call would just be paying tokens for a
  stale copy of state the next live call recomputes fresh anyway. The live
  call itself (`_attempt_with_retry`) still gets the full picture; only what
  gets remembered afterward is trimmed. This still relies on the provider's
  prompt caching for cost on the parts that do persist (event history,
  the model's own past reasoning), not on client-side summarization of
  those -- no compaction/safety-valve for them in v1 (see "Known
  limitations" below).
- Every real LLM-consulting call -- a decision or a turn plan -- commits
  exactly one "user" turn and one "assistant" turn to `self._messages`,
  even when an internal retry happened first -- retry noise is resolved locally before
  anything is committed, so the persisted conversation always alternates
  cleanly (a strict requirement of the Messages/Chat Completions APIs) and
  never carries transient error chatter forward into the game's memory.
- Reasoning/justification is recorded in `self.journal`, entirely outside
  the `Engine` and the replay-log format; `JournalEntry.kind` separates a
  turn plan's entry from a decision's -- a `Player`'s own bookkeeping,
  never engine state.
- If `log_path` is given, every real LLM-consulting call also persists a
  full JSON snapshot (conversation, pending plan, journal, cumulative
  token usage) via `conversation_log.py`. This never loads anything by
  itself: a fresh `LLMPlayer` always starts with empty memory, even if a
  snapshot already exists at `log_path` (a stale file from an earlier,
  unrelated game must never be picked up silently -- the engine itself
  always starts a new game unless the caller explicitly reconstructs one,
  see mandate #5 in docs/ARCHITECTURE.md, and this bot's memory follows the same rule).
  Pass `resume=True` to load the existing snapshot at construction time
  instead -- see `conversation_log.py`'s module docstring for the exact
  resumption contract (this only covers the bot's own state; the caller is
  responsible for the `Engine` and `history` halves of resuming a game).

Known limitations (v1, documented rather than engineered around):
- Resuming from `log_path` does not restore `self._rng`'s exact internal
  sequence position -- a fresh instance's `random.Random(seed)` starts
  over. Acceptable because `_rng` is only ever consulted on the fallback
  path (picking *a* legal action after total LLM failure); mandate #3's
  determinism guarantee is about the engine's own RNG, not a bot's
  internal fallback RNG.
- No context-budget safety valve: the conversation resends in full every
  call, and the event-delta history plus the model's own past responses
  still grow without bound turn over turn (only the per-instant board
  report/hand dossier/cards-in-play are kept from re-accumulating). A long
  enough game can still in theory approach the model's context limit or a
  provider's tokens-per-minute rate limit.
- No full card event/flavor text is available to the model, only the
  mechanical facts in `cards.json`, including its hand-maintained
  `event_summary` field per card (which can drift from `events.py` as it
  evolves -- no automated sync check in v1).
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from struggler.bots.llm import conversation_log
from struggler.bots.llm.client import LLMClient, LLMClientError, LLMMessage, LLMRequest
from struggler.bots.llm.conversation_log import ConversationSnapshot, JournalEntry
from struggler.bots.llm.prompt import (
    build_history_entry,
    build_system_prompt,
    build_turn_plan_request,
    build_user_turn,
)
from struggler.bots.llm.schema import (
    OUTPUT_SPEC,
    TURN_PLAN_OUTPUT_SPEC,
    DecisionPlan,
    PlannedStep,
    PlanParseError,
    TurnPlan,
    parse_plan_response,
    parse_turn_plan_response,
    render_turn_plan,
)
from struggler.bots.llm.client import StructuredOutputSpec
from struggler.engine import Action, Decision, Observation, Side
from struggler.engine.player import Event

_MAX_RETRIES = 1  # one retry after an invalid response, then fall back to RNG

_RETRY_NUDGE = "That response was invalid. Please try again, following the schema exactly."


class _PlanMismatch(Exception):
    """A response that parsed but can't be used against the live decision.

    Distinct from `PlanParseError` (which means the JSON itself was wrong) so
    the retry can tell the model *which* of the two it got wrong; the message
    is the nudge sent back verbatim.
    """


@dataclass(frozen=True)
class _Attempt:
    """One completed request/response round trip's outcome, whatever the
    output spec was: the interpreted result (None if every attempt failed),
    the text to commit as the single assistant turn, the last error, summed
    usage, and every raw attempt for the journal."""

    result: object | None
    assistant_text: str
    error: str | None
    usage: dict[str, int]
    raw_responses: tuple[str, ...]


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
        plan_turns: bool = True,
        log_path: str | Path | None = None,
        resume: bool = False,
    ) -> None:
        self._client = client
        # One extra LLM call per game turn, producing intent rather than an
        # action (see `_make_turn_plan`). Off makes the bot decide each
        # decision on its own again, which is what the pre-turn-plan version
        # did -- useful for measuring what the plan is worth, and for tests
        # that are about decision mechanics rather than planning.
        self._plan_turns = plan_turns
        self._rng = random.Random(seed)  # own seeded RNG -- never the engine's (RandomPlayer convention)
        self._max_plan_steps = max_plan_steps
        self._seed = seed
        self._log_path = Path(log_path) if log_path is not None else None
        self._messages: list[LLMMessage] = []  # the one growing conversation = memory
        self._plan: deque[PlannedStep] = deque()
        # The once-per-game-turn intent (see schema.TURN_PLAN_SCHEMA), and the
        # turn it was written for. Every decision in that turn is prompted with
        # it, so a card held for a later action round is still held *for*
        # something.
        self._turn_plan: TurnPlan | None = None
        self._planned_turn: int | None = None
        # Every turn plan the game has produced so far, oldest first -- kept
        # purely as a record (nothing re-injects from here; that's still
        # `self._turn_plan`, the current one). A turn whose planning call
        # failed contributes no entry, same as `self._turn_plan` going `None`
        # for that turn.
        self._turn_plan_history: list[TurnPlan] = []
        self._last_seen = 0  # index into `history`; advances only on a real LLM call
        self.journal: list[JournalEntry] = []
        self.cumulative_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        self._created_at = conversation_log.now_iso()

        snapshot = conversation_log.load(self._log_path) if resume and self._log_path is not None else None
        if snapshot is not None:
            self._messages = list(snapshot.messages)
            self._plan = deque(snapshot.plan)
            self._turn_plan = snapshot.turn_plan
            self._planned_turn = snapshot.planned_turn
            self._turn_plan_history = list(snapshot.turn_plan_history)
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

        if self._plan_turns and observation.turn != self._planned_turn:
            # First real decision of a new game turn: plan the turn before
            # taking any of it. A stale step plan can never survive a turn
            # boundary, so it goes too.
            self._plan.clear()
            new_events = history[self._last_seen :]
            self._last_seen = len(history)
            self._make_turn_plan(observation, decision, new_events)
            return self._call_llm_and_consume(observation, decision, ())

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

    def _make_turn_plan(
        self, observation: Observation, decision: Decision, new_events: Sequence[Event]
    ) -> None:
        """One extra LLM call at the start of each game turn, producing intent
        rather than an action (see schema.TURN_PLAN_SCHEMA).

        Failure is not fatal and never blocks the game: `_planned_turn` is
        stamped either way, so a turn whose planning call failed simply plays
        without a plan instead of retrying the planning call at every decision.
        """
        self._planned_turn = observation.turn
        user_text = build_turn_plan_request(observation, new_events)
        attempt = self._attempt_with_retry(
            observation.side,
            user_text,
            TURN_PLAN_OUTPUT_SPEC,
            lambda structured: parse_turn_plan_response(structured, turn=observation.turn),
        )
        self._commit_turn(build_history_entry(new_events), attempt)
        plan = attempt.result
        if isinstance(plan, TurnPlan):
            self._turn_plan = plan
            self._turn_plan_history.append(plan)
        else:
            self._turn_plan = None
        self.journal.append(
            JournalEntry(
                decision_id=decision.id,
                kind="turn_plan",
                justification=plan.objective if isinstance(plan, TurnPlan) else None,
                fallback_used=not isinstance(plan, TurnPlan),
                fallback_reason=None if isinstance(plan, TurnPlan) else (attempt.error or "unknown"),
                usage=attempt.usage,
                timestamp=conversation_log.now_iso(),
                raw_responses=attempt.raw_responses,
            )
        )
        self._persist_if_configured()

    def _commit_turn(self, history_text: str, attempt: _Attempt) -> None:
        """Commit exactly one user + one assistant turn to the persisted
        conversation, and bank the call's token usage -- however many internal
        retries produced it.

        `history_text` is deliberately not the full text the live call was
        sent with: it's `build_history_entry`'s trimmed, event-only view, so
        the board report/hand dossier/cards-in-play a call was actually
        answered against never gets resent on every later call -- only the
        event delta that is still true for the rest of the game does. The
        live call itself (`_attempt_with_retry`) always saw the full text;
        only what gets remembered afterward is thinner."""
        self._messages.append(LLMMessage(role="user", content=history_text))
        self._messages.append(LLMMessage(role="assistant", content=attempt.assistant_text))
        self.cumulative_usage = _add_usage(self.cumulative_usage, attempt.usage)

    def _call_llm_and_consume(
        self, observation: Observation, decision: Decision, new_events: Sequence[Event]
    ) -> Action:
        turn_plan_text = render_turn_plan(self._turn_plan) if self._turn_plan is not None else None
        user_text = build_user_turn(observation, decision, new_events, turn_plan_text)

        def interpret(structured: Mapping[str, Any]) -> tuple[DecisionPlan, Action]:
            plan = parse_plan_response(structured)
            first_step = plan.steps[0]
            first_action = (
                _find_matching_option(decision.options, dict(first_step.payload))
                if first_step.kind is decision.kind
                else None
            )
            if first_action is None:
                raise _PlanMismatch(
                    "Your first step didn't match the current live decision (wrong "
                    "kind, or that option isn't actually legal right now). Please try "
                    "again using one of the live options shown."
                )
            return plan, first_action

        attempt = self._attempt_with_retry(observation.side, user_text, OUTPUT_SPEC, interpret)

        # Exactly one user + one assistant turn is committed per call, no
        # matter how many internal retries happened, so the persisted
        # conversation always alternates cleanly.
        self._commit_turn(build_history_entry(new_events), attempt)
        timestamp = conversation_log.now_iso()

        if attempt.result is not None:
            plan, action = attempt.result  # type: ignore[misc]
            self._plan = deque(plan.steps[1 : 1 + self._max_plan_steps])
            self.journal.append(
                JournalEntry(
                    decision_id=decision.id,
                    justification=plan.justification,
                    fallback_used=False,
                    usage=attempt.usage,
                    timestamp=timestamp,
                    raw_responses=attempt.raw_responses,
                )
            )
        else:
            action = self._rng.choice(decision.options)
            self.journal.append(
                JournalEntry(
                    decision_id=decision.id,
                    justification=None,
                    fallback_used=True,
                    fallback_reason=attempt.error or "unknown",
                    usage=attempt.usage,
                    timestamp=timestamp,
                    raw_responses=attempt.raw_responses,
                )
            )

        self._persist_if_configured()
        return action

    def _attempt_with_retry(
        self,
        side: Side,
        user_text: str,
        output: StructuredOutputSpec,
        interpret: Callable[[Mapping[str, Any]], Any],
    ) -> _Attempt:
        """Attempts one LLM call up to `_MAX_RETRIES + 1` times, using a local
        scratch message list so failed attempts never pollute the persisted
        conversation.

        `interpret` turns a structured response into whatever this call wanted
        back, raising `PlanParseError` (malformed) or `_PlanMismatch` (parsed,
        but not usable against the live decision) to ask for one more attempt
        with the reason appended. Both output specs this bot uses -- a decision
        plan and a turn plan -- go through here, so the retry/usage/raw-response
        bookkeeping exists once rather than once per kind of call.

        On total failure `_Attempt.result` is `None` and `assistant_text` is a
        synthetic note recording it, so the one committed conversation turn
        still makes sense on replay. `usage` sums every attempt that actually
        received a response (an `LLMClientError` raised before any response
        arrives contributes nothing).
        """
        attempt_messages = list(self._messages) + [LLMMessage(role="user", content=user_text)]
        last_error: str | None = None
        total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        raw_responses: list[str] = []

        def retry_after(assistant_text: str, nudge: str) -> None:
            attempt_messages.append(LLMMessage(role="assistant", content=assistant_text))
            attempt_messages.append(LLMMessage(role="user", content=nudge))

        for _ in range(_MAX_RETRIES + 1):
            request = LLMRequest(
                system=build_system_prompt(side),
                messages=tuple(attempt_messages),
                output=output,
            )
            try:
                response = self._client.complete(request)
            except LLMClientError as exc:
                last_error = str(exc)
                raw_responses.append(f"[client error: {last_error}]")
                retry_after(f"[invalid response: {last_error}]", _RETRY_NUDGE)
                continue  # no response received -- nothing to add to total_usage

            total_usage = _add_usage(total_usage, response.usage)
            raw_responses.append(response.raw_text)

            try:
                result = interpret(response.structured)
            except PlanParseError as exc:
                last_error = str(exc)
                retry_after(f"[invalid response: {last_error}]", _RETRY_NUDGE)
                continue
            except _PlanMismatch as exc:
                last_error = "first planned step did not match the live decision"
                retry_after(response.raw_text, str(exc))
                continue

            return _Attempt(
                result=result,
                assistant_text=response.raw_text,
                error=None,
                usage=total_usage,
                raw_responses=tuple(raw_responses),
            )

        note = f"[fallback: no valid response after {_MAX_RETRIES + 1} attempt(s): {last_error}]"
        return _Attempt(
            result=None,
            assistant_text=note,
            error=last_error,
            usage=total_usage,
            raw_responses=tuple(raw_responses),
        )

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
            turn_plan=self._turn_plan,
            planned_turn=self._planned_turn,
            turn_plan_history=tuple(self._turn_plan_history),
            journal=tuple(self.journal),
        )
        conversation_log.save(self._log_path, snapshot)
