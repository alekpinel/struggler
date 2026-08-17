"""OpenAI adapter for `LLMClient`.

Same conventions as `anthropic_client.py`: the only module that imports
`openai`, done lazily inside `__init__` -- so `import
struggler.bots.llm.player` never requires the optional `openai` package;
only actually constructing a client does.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from struggler.bots.llm.client import (
    DEFAULT_MAX_TOKENS,
    LLMClientError,
    LLMRequest,
    LLMResponse,
)


class OpenAIClient:
    """Uses OpenAI Chat Completions' strict structured-output mode.

    NOTE: verify the exact SDK call shape (`response_format` vs. a newer
    dedicated "responses" API, `max_tokens` vs. `max_completion_tokens` for
    newer model families, the minimum `openai` package version, and strict
    mode's precise schema constraints -- see `_to_openai_strict_schema`
    below) against the `openai` package's current documentation before this
    is first exercised for real -- the same caveat `anthropic_client.py`
    already carries for its own SDK. Nothing else in `struggler.bots.llm`
    depends on this detail.
    """

    def __init__(
        self, *, model: str, api_key: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "OpenAIClient requires the 'openai' package: "
                "pip install 'struggler[llm-openai]'"
            ) from exc
        self._client = openai.OpenAI(**({"api_key": api_key} if api_key else {}))
        self._model = model
        self._max_tokens = max_tokens
        self.provider_name = "openai"
        self.model_name = model

    def complete(self, request: LLMRequest) -> LLMResponse:
        messages = [{"role": "system", "content": request.system}]
        messages += [{"role": m.role, "content": m.content} for m in request.messages]
        schema = _to_openai_strict_schema(dict(request.output.schema))

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_completion_tokens=request.max_tokens or self._max_tokens,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": request.output.name,
                        "description": request.output.description,
                        "schema": schema,
                        "strict": True,
                    },
                },
            )
        except Exception as exc:  # network/HTTP/SDK failure
            raise LLMClientError(str(exc)) from exc

        text = response.choices[0].message.content if response.choices else None
        if text is None:
            raise LLMClientError("OpenAI response carried no message content")
        try:
            structured = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"unparseable structured output: {exc}") from exc

        usage: dict[str, int] = {}
        try:
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        except AttributeError:
            pass  # older SDK / shape drift -- usage is enrichment, never fatal
        return LLMResponse(structured=structured, raw_text=text, usage=usage)


def _to_openai_strict_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Derive an OpenAI-strict-mode-compatible schema from a logical
    `schema.PLAN_SCHEMA`-shaped JSON Schema, without modifying `schema.py`.

    OpenAI's strict structured-output mode requires every property of
    every object to be listed in `required`; `PLAN_SCHEMA`'s `payload`
    object has 6 genuinely-optional properties instead (only one is
    populated per step, see `schema.py`). This closes that gap the way
    OpenAI's own docs describe: every optional property is (a) added to
    `required` and (b) has its `type` unioned with `"null"`, so the model
    emits an explicit `null` for fields it isn't populating this step.
    `schema.parse_plan_response` already drops null-valued payload keys
    before building a `PlannedStep`, so this is fully transparent
    downstream -- nothing else needs to change to support it.
    """
    if not isinstance(schema, Mapping):
        return schema
    result = dict(schema)
    if result.get("type") == "object" and "properties" in result:
        original_required = set(result.get("required", ()))
        properties = {}
        for key, subschema in result["properties"].items():
            converted = _to_openai_strict_schema(subschema)
            properties[key] = converted if key in original_required else _make_nullable(converted)
        result["properties"] = properties
        result["required"] = list(properties)
    if result.get("type") == "array" and "items" in result:
        result["items"] = _to_openai_strict_schema(result["items"])
    return result


def _make_nullable(subschema: Mapping[str, Any]) -> dict[str, Any]:
    subschema = dict(subschema)
    t = subschema.get("type")
    if isinstance(t, str):
        subschema["type"] = [t, "null"]
    elif isinstance(t, list) and "null" not in t:
        subschema["type"] = [*t, "null"]
    return subschema
