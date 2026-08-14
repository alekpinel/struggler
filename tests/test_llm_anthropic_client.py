"""Tests for the Anthropic `LLMClient` adapter: request building and
response parsing only. The `anthropic` SDK's own HTTP call is monkeypatched
out -- no real network access.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("anthropic")

from struggler.bots.llm.anthropic_client import AnthropicClient
from struggler.bots.llm.client import LLMClientError, LLMMessage, LLMRequest, StructuredOutputSpec


def _client() -> AnthropicClient:
    return AnthropicClient(model="claude-sonnet-5", api_key="test-key-not-a-real-key")


def test_complete_builds_request_and_parses_response(monkeypatch):
    client = _client()
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        payload = {"justification": "because", "steps": []}
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload))])

    monkeypatch.setattr(client._client.messages, "create", fake_create)

    request = LLMRequest(
        system="system prompt",
        messages=(LLMMessage(role="user", content="hello"),),
        output=StructuredOutputSpec(name="x", description="y", schema={"type": "object"}),
    )
    response = client.complete(request)

    assert captured["system"] == "system prompt"
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["output_config"]["format"]["schema"] == {"type": "object"}
    assert response.structured == {"justification": "because", "steps": []}
    assert json.loads(response.raw_text) == {"justification": "because", "steps": []}


def test_complete_raises_llm_client_error_on_unparseable_text(monkeypatch):
    client = _client()

    def fake_create(**kwargs):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="not json")])

    monkeypatch.setattr(client._client.messages, "create", fake_create)

    request = LLMRequest(
        system="s",
        messages=(),
        output=StructuredOutputSpec(name="x", description="y", schema={"type": "object"}),
    )
    with pytest.raises(LLMClientError):
        client.complete(request)


def test_complete_raises_llm_client_error_on_missing_text_block(monkeypatch):
    client = _client()

    def fake_create(**kwargs):
        return SimpleNamespace(content=[])

    monkeypatch.setattr(client._client.messages, "create", fake_create)

    request = LLMRequest(
        system="s",
        messages=(),
        output=StructuredOutputSpec(name="x", description="y", schema={"type": "object"}),
    )
    with pytest.raises(LLMClientError):
        client.complete(request)


def test_complete_raises_llm_client_error_on_sdk_exception(monkeypatch):
    client = _client()

    def fake_create(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(client._client.messages, "create", fake_create)

    request = LLMRequest(
        system="s",
        messages=(),
        output=StructuredOutputSpec(name="x", description="y", schema={"type": "object"}),
    )
    with pytest.raises(LLMClientError):
        client.complete(request)
