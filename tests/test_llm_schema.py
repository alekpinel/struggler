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

from struggler.bots.llm.schema import parse_plan_response
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
