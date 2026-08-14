"""A deterministic, network-free `LLMClient` test double."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from struggler.bots.llm.client import LLMRequest, LLMResponse
from struggler.engine import DecisionKind


class FakeLLMClient:
    """Consumes a scripted queue of `LLMResponse | Exception`, in order.

    Records every `LLMRequest` it received in `self.requests` so tests can
    assert on what `LLMPlayer` actually sent (e.g. how many real calls were
    made, or what the growing conversation looked like at call time).
    """

    def __init__(
        self,
        responses: Sequence[LLMResponse | Exception],
        *,
        provider_name: str = "fake",
        model_name: str = "fake-model",
    ) -> None:
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []
        self.provider_name = provider_name
        self.model_name = model_name

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("FakeLLMClient: response script exhausted")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_plan_response(
    justification: str,
    steps: Sequence[tuple[DecisionKind, Mapping[str, object]]],
    *,
    usage: Mapping[str, int] | None = None,
) -> LLMResponse:
    """Build a well-formed `LLMResponse` for `steps` -- test convenience
    matching exactly the shape `schema.parse_plan_response` expects back."""
    payload = {
        "justification": justification,
        "steps": [
            {"kind": kind.value, "payload": dict(step_payload)} for kind, step_payload in steps
        ],
    }
    return LLMResponse(structured=payload, raw_text=json.dumps(payload), usage=dict(usage or {}))
