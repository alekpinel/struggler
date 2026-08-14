"""Anthropic adapter for `LLMClient`.

The only module in this package that imports `anthropic`, and it does so
lazily inside `__init__` -- so `import struggler.bots.llm.player` (what
registers `"llm"` with the bot registry) never requires the optional
`anthropic` package to be installed; only actually constructing a client
(building an `"llm"` player for real) does.
"""

from __future__ import annotations

import json

from struggler.bots.llm.client import LLMClientError, LLMRequest, LLMResponse


class AnthropicClient:
    """Uses Anthropic's structured-output mode: the response is a single
    JSON text block matching `request.output.schema`, which is exactly what
    needs to be resent verbatim as the conversation's next "assistant" turn
    -- no `tool_use`/`tool_result` bookkeeping to maintain across calls for
    a feature that never actually executes a tool.

    NOTE: the exact SDK call shape (the structured-output parameter name,
    the minimum package version that supports it) should be verified
    against the `anthropic` package's current documentation before this is
    first exercised against the real API -- it is a newer SDK feature and
    may have moved since this was written. The rest of this class (and
    everything else in `struggler.bots.llm`) does not depend on that detail.
    """

    def __init__(self, *, model: str, api_key: str | None = None, max_tokens: int = 4096) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "AnthropicClient requires the 'anthropic' package: "
                "pip install 'struggler[llm]'"
            ) from exc
        self._client = anthropic.Anthropic(**({"api_key": api_key} if api_key else {}))
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=request.max_tokens or self._max_tokens,
                system=request.system,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                output_config={
                    "format": {"type": "json_schema", "schema": dict(request.output.schema)}
                },
            )
        except Exception as exc:  # network/HTTP/SDK failure
            raise LLMClientError(str(exc)) from exc

        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise LLMClientError("Anthropic response carried no text content block")
        try:
            structured = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"unparseable structured output: {exc}") from exc
        return LLMResponse(structured=structured, raw_text=text)
