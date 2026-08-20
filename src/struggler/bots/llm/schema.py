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
from struggler.engine import DecisionKind, Region

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


# -- turn plan ----------------------------------------------------------------
#
# A second, separate structured output, produced once per game turn (at the
# first decision of that turn) and never executed directly: it carries no
# actions, only the intent the turn's individual decisions are then made
# against. The reviewed game's failure mode was not illegal play -- every
# response was legal -- but the absence of any standing intent: a Scoring
# card sat in hand for six action rounds while the bot optimized each
# decision on its own, and the region it was obliged to score was never
# invested in. `justification` deliberately isn't memory (see PLAN_SCHEMA
# above); this is.

TURN_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "assessment",
        "objective",
        "region_focus",
        "scoring_cards",
        "card_plan",
        "influence_targets",
        "military_ops_plan",
        "defend",
        "contingencies",
    ],
    "properties": {
        "assessment": {
            "type": "string",
            "description": (
                "Where the game stands right now: who leads and by how much, "
                "which regions are winnable, what the opponent did last turn "
                "that still needs answering."
            ),
        },
        "objective": {
            "type": "string",
            "description": "The one thing this turn has to achieve. Concrete, checkable.",
        },
        "region_focus": {
            "type": "array",
            "description": (
                "Which region(s) this turn's Ops should concentrate on, in "
                "priority order. A region whose Scoring card you hold always "
                "comes first (it must be played this turn). With no Scoring "
                "card in hand, prioritize the region(s) that haven't been "
                "scored in the longest time -- 'never scored' outranks every "
                "turn number. Only list a region you're actually spending Ops "
                "or an Event's Influence change in this turn -- a card whose "
                "Event merely touches a region while you HOLD that card (not "
                "playing it) doesn't count. On turns 1-3, never list a Mid War "
                "region (Central America, South America, Africa) "
                "-- it has no Scoring card in the deck yet, so it cannot "
                "be this turn's focus regardless of priority rank."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["region", "why"],
                "properties": {
                    "region": {"type": "string", "enum": [r.value for r in Region]},
                    "why": {"type": "string"},
                },
            },
        },
        "scoring_cards": {
            "type": "array",
            "description": (
                "One entry per Scoring card in your hand -- these are "
                "obligations, not options: each must be played this turn."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["card", "when", "preparation"],
                "properties": {
                    "card": {"type": "string"},
                    "when": {
                        "type": "string",
                        "description": "Which action round you intend to play it in, and why then.",
                    },
                    "preparation": {
                        "type": "string",
                        "description": (
                            "What has to change in that region before you play it, "
                            "in Ops and countries -- or 'none' if the region already "
                            "scores in your favour."
                        ),
                    },
                },
            },
        },
        "card_plan": {
            "type": "array",
            "description": "One entry per card in hand, including the China Card if you hold it.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["card", "intended_use", "purpose", "order"],
                "properties": {
                    "card": {"type": "string"},
                    "intended_use": {
                        "type": "string",
                        "enum": ["headline", "event", "ops", "space_race", "un_intervention", "hold"],
                    },
                    "purpose": {
                        "type": "string",
                        "description": "What that use is meant to accomplish, in one line.",
                    },
                    "order": {
                        "type": "integer",
                        "description": (
                            "This card's position in the sequence you intend to play "
                            "cards this turn. -1 for a card you intend to hold rather "
                            "than play now. 0 for a card you intend to headline. 1, 2, "
                            "3... for the action rounds you'll spend, in order, never "
                            "exceeding the number of action rounds you actually have "
                            "left this turn. No two cards played this turn may share "
                            "the same order except -1."
                        ),
                    },
                },
            },
        },
        "influence_targets": {
            "type": "array",
            "description": (
                "Countries this turn's Ops are meant to take or hold, in priority "
                "order, with the points needed. Battlegrounds first."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["country", "why"],
                "properties": {
                    "country": {"type": "string"},
                    "why": {"type": "string"},
                },
            },
        },
        "military_ops_plan": {
            "type": "string",
            "description": (
                "How you will reach the Military Operations requirement (equal to "
                "DEFCON) this turn, naming the card and target -- every point short "
                "is 1 VP to your opponent."
            ),
        },
        "defend": {
            "type": "array",
            "description": (
                "Countries you must hold or retake this turn -- Battlegrounds you "
                "control by a thin margin, and Battlegrounds the opponent has just "
                "broken."
            ),
            "items": {"type": "string"},
        },
        "contingencies": {
            "type": "array",
            "description": "Opponent plays that would force a change of plan, and the answer to each.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["trigger", "response"],
                "properties": {
                    "trigger": {"type": "string"},
                    "response": {"type": "string"},
                },
            },
        },
    },
}

TURN_PLAN_OUTPUT_SPEC = StructuredOutputSpec(
    name="turn_plan",
    description=(
        "A plan for the whole game turn: what each card in hand is for, where "
        "Influence is going, when Scoring cards get played, how the Military "
        "Operations requirement is met, and what to do if the opponent "
        "interferes. No actions are taken from this -- it is the intent every "
        "individual decision this turn is then made against."
    ),
    schema=TURN_PLAN_SCHEMA,
)


@dataclass(frozen=True)
class TurnPlan:
    turn: int
    assessment: str
    objective: str
    region_focus: tuple[Mapping[str, str], ...]
    scoring_cards: tuple[Mapping[str, str], ...]
    card_plan: tuple[Mapping[str, str], ...]
    influence_targets: tuple[Mapping[str, str], ...]
    military_ops_plan: str
    defend: tuple[str, ...]
    contingencies: tuple[Mapping[str, str], ...]


def _str_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise PlanParseError(f"'{field}' must be a list")
    return tuple(str(item) for item in value)


def _obj_list(value: Any, field: str, keys: tuple[str, ...]) -> tuple[Mapping[str, str], ...]:
    if not isinstance(value, list):
        raise PlanParseError(f"'{field}' must be a list")
    out = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            raise PlanParseError(f"{field}[{i}]: must be an object")
        out.append({key: str(item.get(key, "")) for key in keys})
    return tuple(out)


def parse_turn_plan_response(payload: Mapping[str, Any], *, turn: int) -> TurnPlan:
    """Validate a raw structured turn-plan response into a `TurnPlan`.

    Raises `PlanParseError` on anything structurally wrong, so `LLMPlayer`
    can treat a bad turn plan exactly like a bad decision plan: retry once,
    then carry on without one. A turn plan is guidance, never an action --
    a game must still be playable when planning fails outright.
    """
    try:
        assessment = payload["assessment"]
        objective = payload["objective"]
        military_ops_plan = payload["military_ops_plan"]
    except (KeyError, TypeError) as exc:
        raise PlanParseError(f"missing required key: {exc}") from exc
    for name, value in (
        ("assessment", assessment),
        ("objective", objective),
        ("military_ops_plan", military_ops_plan),
    ):
        if not isinstance(value, str):
            raise PlanParseError(f"'{name}' must be a string")
    return TurnPlan(
        turn=turn,
        assessment=assessment,
        objective=objective,
        region_focus=_obj_list(
            payload.get("region_focus", []), "region_focus", ("region", "why")
        ),
        scoring_cards=_obj_list(
            payload.get("scoring_cards", []), "scoring_cards", ("card", "when", "preparation")
        ),
        card_plan=_obj_list(
            payload.get("card_plan", []), "card_plan", ("card", "intended_use", "purpose", "order")
        ),
        influence_targets=_obj_list(
            payload.get("influence_targets", []), "influence_targets", ("country", "why")
        ),
        military_ops_plan=military_ops_plan,
        defend=_str_list(payload.get("defend", []), "defend"),
        contingencies=_obj_list(
            payload.get("contingencies", []), "contingencies", ("trigger", "response")
        ),
    )


def render_turn_plan(plan: TurnPlan) -> str:
    """The plan as prompt text, re-injected into every user turn of the turn
    it belongs to, so each individual decision is taken against a standing
    intent rather than from scratch."""
    lines = [
        f"YOUR PLAN FOR TURN {plan.turn} (you wrote this at the start of the turn; "
        "follow it unless the board has changed, and say so in your justification "
        "when you deliberately depart from it):",
        f"  Assessment: {plan.assessment}",
        f"  Objective: {plan.objective}",
        f"  Military Operations: {plan.military_ops_plan}",
    ]
    if plan.region_focus:
        lines.append("  Region focus this turn, in priority order:")
        lines += [f"    - {item['region']}: {item['why']}" for item in plan.region_focus]
    if plan.scoring_cards:
        lines.append("  Scoring cards to play this turn:")
        lines += [
            f"    - {item['card']}: when={item['when']} | prepare={item['preparation']}"
            for item in plan.scoring_cards
        ]
    if plan.card_plan:
        def _order_key(item: Mapping[str, str]) -> tuple[int, int]:
            try:
                order = int(item.get("order", "-1") or "-1")
            except ValueError:
                order = -1
            # Held cards (order -1) sort after every played card; headline
            # (order 0) plays before any action round (order 1, 2, 3...).
            return (1, 0) if order < 0 else (0, order)

        lines.append(
            "  Cards, in the order you intend to play them "
            "(order -1 = held, not played this turn; order 0 = headline):"
        )
        lines += [
            f"    - [{item.get('order', '-1')}] {item['card']} -> {item['intended_use']}: {item['purpose']}"
            for item in sorted(plan.card_plan, key=_order_key)
        ]
    if plan.influence_targets:
        lines.append("  Influence targets, in priority order:")
        lines += [f"    - {item['country']}: {item['why']}" for item in plan.influence_targets]
    if plan.defend:
        lines.append(f"  Hold or retake: {', '.join(plan.defend)}")
    if plan.contingencies:
        lines.append("  Contingencies:")
        lines += [
            f"    - if {item['trigger']} -> {item['response']}" for item in plan.contingencies
        ]
    return "\n".join(lines)
