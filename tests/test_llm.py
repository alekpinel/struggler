"""Tests for LLMPlayer: the single-option shortcut, plan batching/replanning,
the parse-failure/illegal-action fallback path, the journal, and registry
wiring -- all against a `FakeLLMClient`, no network access.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from struggler.bots.llm import conversation_log
from struggler.bots.llm.client import LLMClientError, LLMResponse
from struggler.bots.llm.fake_client import FakeLLMClient, make_plan_response
from struggler.bots.llm.player import LLMPlayer
from struggler.bots.llm.schema import PAYLOAD_KEY_BY_KIND
from struggler.engine import DecisionKind, Engine, Side
from struggler.engine.player_registry import available, build_player


def test_single_option_auto_resolves_without_llm_call():
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    decision = observation.pending_decision
    single_option_decision = dataclasses.replace(decision, options=(decision.options[0],))
    observation = dataclasses.replace(observation, pending_decision=single_option_decision)

    client = FakeLLMClient([])
    player = LLMPlayer(client=client, seed=0)

    action = player.choose_action(observation, [])

    assert action == single_option_decision.options[0]
    assert client.requests == []


def test_multi_step_plan_consumed_across_calls_with_one_llm_call():
    engine = Engine(seed=1)
    engine._push_ops_type(Side.US, ops=3)
    observation = engine.observe(Side.US)
    influence_action = next(
        a for a in observation.pending_decision.options if a.payload["type"] == "influence"
    )
    engine.step(influence_action)

    observation = engine.observe(Side.US)
    assert observation.pending_decision.kind is DecisionKind.PLACE_INFLUENCE
    country = observation.pending_decision.options[0].payload["country"]

    response = make_plan_response(
        "Spend all 3 Ops of Influence in the same country.",
        [(DecisionKind.PLACE_INFLUENCE, {"country": country})] * 3,
    )
    client = FakeLLMClient([response])
    player = LLMPlayer(client=client, seed=0)

    chosen = []
    for _ in range(3):
        observation = engine.observe(Side.US)
        action = player.choose_action(observation, [])
        chosen.append(action.payload["country"])
        engine.step(action)

    assert chosen == [country, country, country]
    assert len(client.requests) == 1
    assert player.journal[-1].fallback_used is False


def test_queue_mismatch_triggers_fresh_llm_call():
    engine = Engine(seed=1)
    engine._push_ops_type(Side.US, ops=2)
    observation = engine.observe(Side.US)
    influence_action = next(
        a for a in observation.pending_decision.options if a.payload["type"] == "influence"
    )
    engine.step(influence_action)

    observation = engine.observe(Side.US)
    country = observation.pending_decision.options[0].payload["country"]

    # A 2-step plan whose second step names the wrong DecisionKind -- it can
    # never match the live PLACE_INFLUENCE decision that actually follows.
    first_response = make_plan_response(
        "First point is fine; (deliberately) mispredict the second.",
        [
            (DecisionKind.PLACE_INFLUENCE, {"country": country}),
            (DecisionKind.COUP_TARGET, {"country": country}),
        ],
    )
    second_response = make_plan_response(
        "Fresh plan for the second point after the mismatch.",
        [(DecisionKind.PLACE_INFLUENCE, {"country": country})],
    )
    client = FakeLLMClient([first_response, second_response])
    player = LLMPlayer(client=client, seed=0)

    action1 = player.choose_action(observation, [])
    engine.step(action1)
    observation2 = engine.observe(Side.US)
    action2 = player.choose_action(observation2, [])
    engine.step(action2)

    assert action1.payload["country"] == country
    assert action2.payload["country"] == country
    assert len(client.requests) == 2


def test_parse_failure_falls_back_to_rng_and_never_crashes():
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    malformed = LLMResponse(structured={}, raw_text="{}")
    client = FakeLLMClient([malformed, malformed])
    player = LLMPlayer(client=client, seed=0)

    action = player.choose_action(observation, [])

    assert action in observation.pending_decision.options
    assert len(client.requests) == 2  # the one retry was used
    assert player.journal[-1].fallback_used is True
    assert player.journal[-1].justification is None


def test_illegal_first_step_after_retry_falls_back():
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    decision = observation.pending_decision
    payload_key = PAYLOAD_KEY_BY_KIND[decision.kind]

    illegal = make_plan_response(
        "Confidently wrong.", [(decision.kind, {payload_key: "__nonexistent__"})]
    )
    client = FakeLLMClient([illegal, illegal])
    player = LLMPlayer(client=client, seed=0)

    action = player.choose_action(observation, [])

    assert action in decision.options
    assert len(client.requests) == 2
    assert player.journal[-1].fallback_used is True


def test_journal_records_justification_on_success():
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    decision = observation.pending_decision
    payload_key = PAYLOAD_KEY_BY_KIND[decision.kind]
    correct_value = decision.options[0].payload[payload_key]

    response = make_plan_response(
        "Because it's the best option available.",
        [(decision.kind, {payload_key: correct_value})],
    )
    client = FakeLLMClient([response])
    player = LLMPlayer(client=client, seed=0)

    action = player.choose_action(observation, [])

    assert action == decision.options[0]
    assert player.journal[-1].justification == "Because it's the best option available."
    assert player.journal[-1].fallback_used is False


def test_conversation_alternates_cleanly_even_after_a_retry():
    """The persisted conversation must always alternate user/assistant --
    a hard requirement of the Messages API -- even when an internal retry
    happened first (see player.py's `_request_plan_with_retry`)."""
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    malformed = LLMResponse(structured={}, raw_text="{}")
    client = FakeLLMClient([malformed, malformed])
    player = LLMPlayer(client=client, seed=0)

    player.choose_action(observation, [])

    roles = [m.role for m in player._messages]
    assert roles == ["user", "assistant"]


def test_registry_has_llm_after_import():
    import struggler.bots.llm.player  # noqa: F401

    assert "llm" in available()


def test_llm_player_constructs_with_anthropic_client_given_an_api_key():
    pytest.importorskip("anthropic")
    from struggler.bots.llm.anthropic_client import AnthropicClient

    client = AnthropicClient(model="claude-sonnet-5", api_key="test-key-not-a-real-key")
    player = LLMPlayer(client=client, seed=0)

    assert isinstance(player, LLMPlayer)


def _single_step_response_and_decision(engine_seed: int = 1):
    engine = Engine.new_game(seed=engine_seed)
    observation = engine.observe(engine.pending_decision.actor)
    decision = observation.pending_decision
    payload_key = PAYLOAD_KEY_BY_KIND[decision.kind]
    correct_value = decision.options[0].payload[payload_key]
    return observation, decision, payload_key, correct_value


def test_usage_accumulates_across_a_normal_call():
    observation, decision, payload_key, correct_value = _single_step_response_and_decision()
    response = make_plan_response(
        "ok",
        [(decision.kind, {payload_key: correct_value})],
        usage={"input_tokens": 100, "output_tokens": 20},
    )
    client = FakeLLMClient([response])
    player = LLMPlayer(client=client, seed=0)

    player.choose_action(observation, [])

    assert player.journal[-1].usage == {"input_tokens": 100, "output_tokens": 20}
    assert player.cumulative_usage == {"input_tokens": 100, "output_tokens": 20}


def test_usage_accumulates_across_a_retried_call():
    observation, decision, payload_key, correct_value = _single_step_response_and_decision()
    bad = LLMResponse(structured={}, raw_text="{}", usage={"input_tokens": 50, "output_tokens": 10})
    good = make_plan_response(
        "ok",
        [(decision.kind, {payload_key: correct_value})],
        usage={"input_tokens": 80, "output_tokens": 15},
    )
    client = FakeLLMClient([bad, good])
    player = LLMPlayer(client=client, seed=0)

    player.choose_action(observation, [])

    # The failed-but-token-consuming first attempt's usage is still counted.
    assert player.journal[-1].usage == {"input_tokens": 130, "output_tokens": 25}


def test_llm_client_error_before_response_contributes_no_usage():
    observation, decision, payload_key, correct_value = _single_step_response_and_decision()
    good = make_plan_response(
        "ok",
        [(decision.kind, {payload_key: correct_value})],
        usage={"input_tokens": 40, "output_tokens": 8},
    )
    client = FakeLLMClient([LLMClientError("boom"), good])
    player = LLMPlayer(client=client, seed=0)

    player.choose_action(observation, [])

    assert player.journal[-1].usage == {"input_tokens": 40, "output_tokens": 8}


def test_log_path_none_never_touches_disk(monkeypatch):
    observation, decision, payload_key, correct_value = _single_step_response_and_decision()
    response = make_plan_response("ok", [(decision.kind, {payload_key: correct_value})])
    client = FakeLLMClient([response])
    player = LLMPlayer(client=client, seed=0, log_path=None)

    def _fail_save(*args, **kwargs):
        raise AssertionError("save should not be called when log_path is None")

    monkeypatch.setattr(conversation_log, "save", _fail_save)

    action = player.choose_action(observation, [])

    assert action == decision.options[0]


def test_save_failure_does_not_raise_out_of_choose_action(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a directory")
    log_path = blocked / "log.json"

    observation, decision, payload_key, correct_value = _single_step_response_and_decision()
    response = make_plan_response("ok", [(decision.kind, {payload_key: correct_value})])
    client = FakeLLMClient([response])
    player = LLMPlayer(client=client, seed=0, log_path=log_path)

    with pytest.warns(RuntimeWarning):
        action = player.choose_action(observation, [])

    assert action == decision.options[0]


def test_llm_player_resumes_pending_plan_from_log_path(tmp_path):
    log_path = tmp_path / "log.json"
    engine = Engine(seed=1)
    engine._push_ops_type(Side.US, ops=3)
    observation = engine.observe(Side.US)
    influence_action = next(
        a for a in observation.pending_decision.options if a.payload["type"] == "influence"
    )
    engine.step(influence_action)

    observation = engine.observe(Side.US)
    country = observation.pending_decision.options[0].payload["country"]

    response = make_plan_response(
        "Spend all 3 Ops of Influence in the same country.",
        [(DecisionKind.PLACE_INFLUENCE, {"country": country})] * 3,
    )
    first_client = FakeLLMClient([response])
    first_player = LLMPlayer(client=first_client, seed=0, log_path=log_path)

    observation = engine.observe(Side.US)
    action = first_player.choose_action(observation, [])
    engine.step(action)
    assert len(first_client.requests) == 1

    # Simulate a fresh process: a brand-new LLMPlayer with a brand-new
    # client whose response script is empty, resuming purely from disk.
    second_client = FakeLLMClient([])
    second_player = LLMPlayer(client=second_client, seed=0, log_path=log_path)

    chosen = [action.payload["country"]]
    for _ in range(2):
        observation = engine.observe(Side.US)
        action = second_player.choose_action(observation, [])
        chosen.append(action.payload["country"])
        engine.step(action)

    assert chosen == [country, country, country]
    assert second_client.requests == []  # entirely served from the resumed plan
    assert second_player.journal == first_player.journal
    assert second_player.cumulative_usage == first_player.cumulative_usage


def test_choose_action_raises_when_history_shorter_than_resumed_last_seen(tmp_path):
    log_path = tmp_path / "log.json"
    snapshot = conversation_log.ConversationSnapshot(
        seed=0,
        provider="fake",
        model="fake-model",
        created_at=conversation_log.now_iso(),
        updated_at=conversation_log.now_iso(),
        last_seen=5,
        cumulative_usage={},
        messages=(),
        plan=(),
        journal=(),
    )
    conversation_log.save(log_path, snapshot)

    player = LLMPlayer(client=FakeLLMClient([]), seed=0, log_path=log_path)
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)

    with pytest.raises(ValueError):
        player.choose_action(observation, [])


def test_registry_selects_openai_provider_via_env(monkeypatch):
    monkeypatch.setenv("STRUGGLER_LLM_PROVIDER", "openai")
    captured = {}

    class _StubOpenAIClient:
        provider_name = "openai"
        model_name = "stub-model"

        def __init__(self, *, model, api_key=None, max_tokens=4096):
            captured["model"] = model

        def complete(self, request):
            raise AssertionError("should not be called")

    monkeypatch.setattr("struggler.bots.llm.openai_client.OpenAIClient", _StubOpenAIClient)

    player = build_player("llm", seed=0)

    assert isinstance(player, LLMPlayer)
    assert captured["model"] == "gpt-5"


def test_registry_folds_seed_into_log_path(monkeypatch, tmp_path):
    base = str(tmp_path / "game")
    monkeypatch.setenv("STRUGGLER_LLM_LOG_PATH", base)

    class _StubAnthropicClient:
        provider_name = "anthropic"
        model_name = "stub-model"

        def __init__(self, *, model, api_key=None, max_tokens=4096):
            pass

        def complete(self, request):
            raise AssertionError("should not be called")

    monkeypatch.setattr("struggler.bots.llm.anthropic_client.AnthropicClient", _StubAnthropicClient)

    player = build_player("llm", seed=7)

    assert player._log_path == Path(f"{base}.7.json")


def test_registry_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("STRUGGLER_LLM_PROVIDER", "bogus")
    with pytest.raises(ValueError):
        build_player("llm", seed=0)
