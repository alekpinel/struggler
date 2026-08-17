"""Provider-agnostic contract for talking to an LLM.

`LLMClient` mirrors how `struggler.engine.player.Player` is a structural
`Protocol` rather than a base class: any object with a matching `complete`
method is a valid client, no inheritance required. `LLMPlayer` (see
`player.py`) only ever depends on this module -- never on a specific
vendor SDK -- so adding a new provider means writing a new adapter module
(see `anthropic_client.py`), not touching the bot's decision logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class LLMMessage:
    """One turn of the growing conversation `LLMPlayer` maintains as its
    memory (see player.py's module docstring)."""

    role: str  # "user" | "assistant"
    content: str


@dataclass(frozen=True)
class StructuredOutputSpec:
    """Provider-agnostic "I want JSON matching this shape back."

    `name`/`description` double as a tool's name/description for an
    adapter that has to go through tool-use rather than a native
    structured-output mode. `schema` is a JSON Schema object (closed:
    `additionalProperties: false` throughout, per the subset most
    providers' structured-output modes accept).
    """

    name: str
    description: str
    schema: Mapping[str, Any]


# Output-token budget for one call. Shared by `LLMRequest` and every adapter's
# own constructor default, so the two can never silently disagree about what an
# unconfigured client actually asks for.
DEFAULT_MAX_TOKENS = 16000


@dataclass(frozen=True)
class LLMRequest:
    system: str
    messages: Sequence[LLMMessage]
    output: StructuredOutputSpec
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass(frozen=True)
class LLMResponse:
    structured: Mapping[str, Any]  # the response, already parsed from JSON
    raw_text: str  # verbatim text -- becomes the next LLMMessage(role="assistant", ...)
    # Normalized token usage for this one call: {"input_tokens": int,
    # "output_tokens": int}. Empty if the provider didn't report it --
    # usage is enrichment for conversation_log.py's metadata, never
    # correctness-critical, so a missing value is never an error.
    usage: Mapping[str, int] = field(default_factory=dict)


class LLMClientError(Exception):
    """A network/HTTP/unparseable-response failure.

    Never raised for a well-formed but semantically illegal plan (e.g. a
    step naming an action that isn't actually legal right now) -- that is
    `LLMPlayer`'s own retry/fallback responsibility, not the client's.
    """


class LLMClient(Protocol):
    # Set by every adapter's __init__ (e.g. "anthropic"/"claude-sonnet-5") --
    # the client is the one object that actually knows what it's talking to,
    # so conversation_log.py's metadata reads these rather than LLMPlayer
    # having to be told separately. LLMPlayer reads them defensively
    # (`getattr(client, "provider_name", "unknown")`) so a minimal hand-rolled
    # test double that doesn't set them still can't crash `choose_action`.
    provider_name: str
    model_name: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        ...
