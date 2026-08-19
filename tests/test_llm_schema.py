"""Tests for schema.parse_plan_response, in particular the strict-mode
payload-noise bug: a provider whose structured-output schema forces every
payload key to be present (nullable) can fill irrelevant keys with a
non-null value instead of `null` (see openai_client.py's
`_to_openai_strict_schema`). `parse_plan_response` must strip those stray
keys down to the one the step's `kind` actually uses, or an otherwise
correct, live-legal step will silently fail to match anything downstream
in `player.py`'s `_find_matching_option`.
"""

from __future__ import annotations

import pytest

from struggler.bots.llm.schema import (
    TURN_PLAN_SCHEMA,
    PlanParseError,
    parse_plan_response,
    parse_turn_plan_response,
    render_turn_plan,
)
from struggler.engine import DecisionKind


def test_strips_irrelevant_non_null_payload_keys():
    """Reproduces the real failure recorded in logs/12347.json: the model
    named a live-legal country ("Poland") but also populated `mode`,
    `order`, and `type` (irrelevant to PLACE_INFLUENCE) with non-null
    values instead of `null`, because the openai strict-mode schema marks
    every payload key as required-but-nullable."""
    raw = {
        "justification": "Place influence in Poland.",
        "steps": [
            {
                "kind": "place_influence",
                "payload": {
                    "card": None,
                    "choice": None,
                    "country": "Poland",
                    "mode": "ops",
                    "order": "event_first",
                    "type": "influence",
                },
            }
        ],
    }

    plan = parse_plan_response(raw)

    assert plan.steps[0].kind is DecisionKind.PLACE_INFLUENCE
    assert dict(plan.steps[0].payload) == {"country": "Poland"}


def test_keeps_the_relevant_key_for_each_kind():
    raw = {
        "justification": "ok",
        "steps": [
            {
                "kind": "ops_type",
                "payload": {
                    "country": None,
                    "card": None,
                    "mode": None,
                    "type": "coup",
                    "order": None,
                    "choice": None,
                },
            }
        ],
    }

    plan = parse_plan_response(raw)

    assert dict(plan.steps[0].payload) == {"type": "coup"}


def test_a_genuinely_clean_payload_is_unaffected():
    raw = {
        "justification": "ok",
        "steps": [{"kind": "coup_target", "payload": {"country": "Angola"}}],
    }

    plan = parse_plan_response(raw)

    assert dict(plan.steps[0].payload) == {"country": "Angola"}


# -- turn plan -----------------------------------------------------------------


def _turn_plan_payload(**overrides):
    payload = {
        "assessment": "US leads Europe; Asia is still open.",
        "objective": "Control Poland and put 2 Influence into Asia.",
        "region_focus": [{"region": "asia", "why": "Asia_Scoring is in hand."}],
        "scoring_cards": [
            {"card": "Asia_Scoring", "when": "last action round", "preparation": "take Thailand"}
        ],
        "card_plan": [{"card": "Fidel", "intended_use": "event", "purpose": "Cuba"}],
        "influence_targets": [{"country": "Poland", "why": "Battleground, adjacent to the US"}],
        "military_ops_plan": "Coup Syria with De Gaulle.",
        "defend": ["East_Germany"],
        "contingencies": [{"trigger": "US coups Iran", "response": "retake with the China Card"}],
    }
    payload.update(overrides)
    return payload


def test_parse_turn_plan_keeps_every_section():
    plan = parse_turn_plan_response(_turn_plan_payload(), turn=3)

    assert plan.turn == 3
    assert plan.objective.startswith("Control Poland")
    assert plan.region_focus[0]["region"] == "asia"
    assert plan.scoring_cards[0]["card"] == "Asia_Scoring"
    assert plan.card_plan[0]["intended_use"] == "event"
    assert plan.influence_targets[0]["country"] == "Poland"
    assert plan.defend == ("East_Germany",)
    assert plan.contingencies[0]["trigger"] == "US coups Iran"


def test_parse_turn_plan_tolerates_missing_lists():
    plan = parse_turn_plan_response(
        {"assessment": "a", "objective": "o", "military_ops_plan": "m"}, turn=1
    )

    assert plan.region_focus == ()
    assert plan.scoring_cards == ()
    assert plan.card_plan == ()
    assert plan.defend == ()


def test_parse_turn_plan_rejects_a_missing_objective():
    payload = _turn_plan_payload()
    del payload["objective"]

    with pytest.raises(PlanParseError):
        parse_turn_plan_response(payload, turn=1)


def test_parse_turn_plan_rejects_a_non_list_section():
    with pytest.raises(PlanParseError):
        parse_turn_plan_response(_turn_plan_payload(defend="East_Germany"), turn=1)


def test_render_turn_plan_states_the_turn_and_every_section():
    text = render_turn_plan(parse_turn_plan_response(_turn_plan_payload(), turn=4))

    assert "YOUR PLAN FOR TURN 4" in text
    assert "Coup Syria with De Gaulle." in text
    assert "asia: Asia_Scoring is in hand." in text
    assert "Asia_Scoring: when=last action round" in text
    assert "Fidel -> event: Cuba" in text
    assert "Poland: Battleground, adjacent to the US" in text
    assert "Hold or retake: East_Germany" in text
    assert "if US coups Iran -> retake with the China Card" in text


def test_turn_plan_schema_requires_every_top_level_section():
    # OpenAI strict mode makes any property not in `required` nullable, which
    # would let a model answer a whole section with null instead of an empty
    # list -- the sections are the plan, so all of them stay required.
    assert set(TURN_PLAN_SCHEMA["required"]) == set(TURN_PLAN_SCHEMA["properties"])
