"""Tests for the OpenAI `LLMClient` adapter: request building, response
parsing, usage normalization, and the strict-mode schema transform. The
`openai` SDK's own HTTP call is monkeypatched out -- no real network access.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")

from struggler.bots.llm.client import LLMClientError, LLMMessage, LLMRequest, StructuredOutputSpec
from struggler.bots.llm.openai_client import OpenAIClient, _to_openai_strict_schema
from struggler.bots.llm.schema import PLAN_SCHEMA


def _client() -> OpenAIClient:
    return OpenAIClient(model="gpt-5", api_key="test-key-not-a-real-key")


def test_complete_builds_request_and_parses_response(monkeypatch):
    client = _client()
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        payload = {"justification": "because", "steps": []}
        message = SimpleNamespace(content=json.dumps(payload))
        usage = SimpleNamespace(prompt_tokens=12, completion_tokens=34)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    request = LLMRequest(
        system="system prompt",
        messages=(LLMMessage(role="user", content="hello"),),
        output=StructuredOutputSpec(name="x", description="y", schema={"type": "object"}),
    )
    response = client.complete(request)

    assert captured["messages"][0] == {"role": "system", "content": "system prompt"}
    assert captured["messages"][1] == {"role": "user", "content": "hello"}
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert response.structured == {"justification": "because", "steps": []}
    assert json.loads(response.raw_text) == {"justification": "because", "steps": []}
    assert response.usage == {"input_tokens": 12, "output_tokens": 34}


def test_complete_raises_llm_client_error_on_unparseable_content(monkeypatch):
    client = _client()

    def fake_create(**kwargs):
        message = SimpleNamespace(content="not json")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    request = LLMRequest(
        system="s", messages=(), output=StructuredOutputSpec(name="x", description="y", schema={"type": "object"})
    )
    with pytest.raises(LLMClientError):
        client.complete(request)


def test_complete_raises_llm_client_error_on_missing_content(monkeypatch):
    client = _client()

    def fake_create(**kwargs):
        message = SimpleNamespace(content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    request = LLMRequest(
        system="s", messages=(), output=StructuredOutputSpec(name="x", description="y", schema={"type": "object"})
    )
    with pytest.raises(LLMClientError):
        client.complete(request)


def test_complete_raises_llm_client_error_on_sdk_exception(monkeypatch):
    client = _client()

    def fake_create(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(client._client.chat.completions, "create", fake_create)

    request = LLMRequest(
        system="s", messages=(), output=StructuredOutputSpec(name="x", description="y", schema={"type": "object"})
    )
    with pytest.raises(LLMClientError):
        client.complete(request)


def test_to_openai_strict_schema_marks_optional_payload_keys_nullable_and_required():
    transformed = _to_openai_strict_schema(PLAN_SCHEMA)

    top_required = set(transformed["required"])
    assert top_required == {"justification", "steps"}

    step_schema = transformed["properties"]["steps"]["items"]
    assert set(step_schema["required"]) == {"kind", "payload"}

    payload_schema = step_schema["properties"]["payload"]
    payload_keys = set(payload_schema["properties"])
    assert payload_keys == {"country", "card", "mode", "type", "order", "choice"}
    assert set(payload_schema["required"]) == payload_keys  # every optional key is now required
    for key in payload_keys:
        prop_type = payload_schema["properties"][key]["type"]
        assert "null" in prop_type
