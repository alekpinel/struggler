"""The structured "decision plan" output the LLM must produce each call:
a prediction of the acting side's own upcoming decisions (mandate #2's
atomic action space is what keeps each individual decision small, and
what makes a short predicted batch tractable), plus a justification string
for explainability.

Steps describe *what* the model intends semantically -- a country, a card
id, a mode/type/order/choice string -- never an index into a not-yet-
existing future `Decision.options`. `player.py` matches each step against
the *live* options only once that step is actually reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from struggler.bots.llm.client import StructuredOutputSpec
from struggler.engine import DecisionKind

# Every DecisionKind that can ever reach a Player. Side.CHANCE decisions
# (the *_ROLL / RANDOM_DISCARD / CONTEST_ROLL kinds) are resolved directly by
# struggler.runner.play_game from their single pre-rolled option and never
# reach a Player at all, so they're excluded here.
PLAYER_FACING_KINDS: tuple[DecisionKind, ...] = (
    DecisionKind.PLACE_INFLUENCE,
    DecisionKind.COUP_TARGET,
    DecisionKind.REALIGNMENT_TARGET,
    DecisionKind.HEADLINE_PLAY,
    DecisionKind.ACTION_ROUND_PLAY,
    DecisionKind.PLAY_MODE,
    DecisionKind.OPS_TYPE,
    DecisionKind.EVENT_OPS_ORDER,
    DecisionKind.EVENT_RESUME,
    DecisionKind.WAR_TARGET,
    DecisionKind.EVENT_INFLUENCE,
    DecisionKind.EVENT_CHOICE,
    DecisionKind.QUAGMIRE_DISCARD,
    DecisionKind.HELD_CARD_DISCARD,
)

# The payload key each kind is carried under (ground truth: every
# `Action(DecisionKind.X, {...})` construction site in engine/core.py).
# EVENT_RESUME always has exactly one option (a forced continuation
# marker), so `choose_action`'s single-option shortcut fires before it
# would ever need a payload -- it never actually reaches the LLM.
PAYLOAD_KEY_BY_KIND: Mapping[DecisionKind, str] = {
    DecisionKind.PLACE_INFLUENCE: "country",
    DecisionKind.COUP_TARGET: "country",
    DecisionKind.REALIGNMENT_TARGET: "country",
    DecisionKind.HEADLINE_PLAY: "card",
    DecisionKind.ACTION_ROUND_PLAY: "card",
    DecisionKind.PLAY_MODE: "mode",
    DecisionKind.OPS_TYPE: "type",
    DecisionKind.EVENT_OPS_ORDER: "order",
    DecisionKind.WAR_TARGET: "country",
    DecisionKind.EVENT_INFLUENCE: "country",
    DecisionKind.EVENT_CHOICE: "choice",
    DecisionKind.QUAGMIRE_DISCARD: "card",
    DecisionKind.HELD_CARD_DISCARD: "card",
}

_VALID_PAYLOAD_KEYS = {"country", "card", "mode", "type", "order", "choice"}

# Anthropic's structured-output JSON Schema subset doesn't support a
# kind-discriminated union (payload shape varying by `kind`). Instead
# `payload` is one closed object with every possible key optional; exactly
# one is populated per step, matching PAYLOAD_KEY_BY_KIND for that step's
# `kind` (EVENT_CHOICE's `choice` is a free string -- its legal values are
# event-specific and only ever known from the live decision's options).
PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["justification", "steps"],
    "properties": {
        "justification": {
            "type": "string",
            "description": (
                "Brief reasoning for this plan. Explainability output only -- "
                "not a persistent memory field, the growing conversation itself "
                "is the memory."
            ),
        },
        "steps": {
            "type": "array",
            "minItems": 1,
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["kind", "payload"],
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": [k.value for k in PLAYER_FACING_KINDS],
                    },
                    "payload": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "country": {"type": "string"},
                            "card": {"type": "string"},
                            "mode": {
                                "type": "string",
                                "enum": ["ops", "event", "space_race", "un_intervention"],
                            },
                            "type": {
                                "type": "string",
                                "enum": ["influence", "coup", "realignment"],
                            },
                            "order": {
                                "type": "string",
                                "enum": ["event_first", "ops_first"],
                            },
                            "choice": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
}

OUTPUT_SPEC = StructuredOutputSpec(
    name="decision_plan",
    description=(
        "A batch of the acting side's own upcoming decisions, predicted as far "
        "ahead as safely possible, plus a brief justification."
    ),
    schema=PLAN_SCHEMA,
)


@dataclass(frozen=True)
class PlannedStep:
    kind: DecisionKind
    payload: Mapping[str, Any]  # only the key(s) populated for this step


@dataclass(frozen=True)
class DecisionPlan:
    justification: str
    steps: tuple[PlannedStep, ...]


class PlanParseError(ValueError):
    """The model's structured response doesn't parse into a `DecisionPlan`.

    Raised for anything structurally wrong -- an unknown `kind`, a missing
    or malformed `steps` list, an unknown payload key -- so `LLMPlayer` can
    retry/fall back instead of crashing.
    """


_VALID_KIND_VALUES = {k.value for k in PLAYER_FACING_KINDS}


def parse_plan_response(payload: Mapping[str, Any]) -> DecisionPlan:
    """Validate and convert a raw structured-output dict (already parsed
    from JSON by the `LLMClient`) into a `DecisionPlan`."""
    try:
        justification = payload["justification"]
        raw_steps = payload["steps"]
    except (KeyError, TypeError) as exc:
        raise PlanParseError(f"missing required key: {exc}") from exc
    if not isinstance(justification, str):
        raise PlanParseError("'justification' must be a string")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise PlanParseError("'steps' must be a non-empty list")

    steps: list[PlannedStep] = []
    for i, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise PlanParseError(f"step {i}: must be an object")
        try:
            kind_value = raw_step["kind"]
            raw_payload = raw_step["payload"]
        except KeyError as exc:
            raise PlanParseError(f"step {i}: missing required key: {exc}") from exc
        if kind_value not in _VALID_KIND_VALUES:
            raise PlanParseError(f"step {i}: unknown decision kind {kind_value!r}")
        if not isinstance(raw_payload, dict):
            raise PlanParseError(f"step {i}: 'payload' must be an object")
        unknown_keys = set(raw_payload) - _VALID_PAYLOAD_KEYS
        if unknown_keys:
            raise PlanParseError(f"step {i}: unknown payload key(s) {unknown_keys}")
        kind = DecisionKind(kind_value)
        # A provider's strict-mode schema (see openai_client.py's
        # `_to_openai_strict_schema`) forces every payload key to be
        # present, nullable, on every step -- so the model sometimes fills
        # irrelevant keys with a plausible-looking non-null value instead
        # of `null` (e.g. a PLACE_INFLUENCE step also carrying
        # `"mode": "ops"`). `_find_matching_option` in player.py does an
        # exact subset match against the live Action's payload, which only
        # ever carries this kind's one real key -- so any such stray key
        # makes an otherwise-correct, live-legal step (e.g. a real country)
        # silently fail to match anything. Drop everything except the key
        # this kind actually uses.
        expected_key = PAYLOAD_KEY_BY_KIND.get(kind)
        populated = {k: v for k, v in raw_payload.items() if v is not None}
        if expected_key is not None:
            populated = {k: v for k, v in populated.items() if k == expected_key}
        steps.append(PlannedStep(kind=kind, payload=populated))

    return DecisionPlan(justification=justification, steps=tuple(steps))
