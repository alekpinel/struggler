"""Tests for LLMPlayer: the single-option shortcut, plan batching/replanning,
the parse-failure/illegal-action fallback path, the journal -- all against a `FakeLLMClient`, no
network access.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import pytest

from struggler.bots.llm import conversation_log
from struggler.bots.llm.client import LLMClientError, LLMResponse
from struggler.bots.llm.fake_client import (
    FakeLLMClient,
    make_plan_response,
    make_turn_plan_response,
)
from struggler.bots.llm.player import LLMPlayer
from struggler.bots.llm.schema import PAYLOAD_KEY_BY_KIND
from struggler.engine import Action, DecisionKind, Engine, Side
from struggler.engine.player import Event


def test_single_option_auto_resolves_without_llm_call():
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    decision = observation.pending_decision
    single_option_decision = dataclasses.replace(decision, options=(decision.options[0],))
    observation = dataclasses.replace(observation, pending_decision=single_option_decision)

    client = FakeLLMClient([])
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

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
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

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
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

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
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

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
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

    action = player.choose_action(observation, [])

    assert action in decision.options
    assert len(client.requests) == 2
    assert player.journal[-1].fallback_used is True


def test_strict_mode_payload_noise_still_matches_the_live_option():
    """Regression test for logs/12347.json: a real OpenAI strict-mode
    response filled PLACE_INFLUENCE's irrelevant payload keys (`mode`,
    `order`, `type`) with non-null values instead of `null`, which used to
    make `_find_matching_option`'s exact subset match reject an otherwise
    correct, live-legal country every single time."""
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    decision = observation.pending_decision
    payload_key = PAYLOAD_KEY_BY_KIND[decision.kind]
    correct_value = decision.options[0].payload[payload_key]

    noisy_payload = {"card": None, "choice": None, "country": None, "mode": "ops", "order": "event_first", "type": "influence"}
    noisy_payload[payload_key] = correct_value
    noisy = LLMResponse(
        structured={
            "justification": "noisy but correct",
            "steps": [{"kind": decision.kind.value, "payload": noisy_payload}],
        },
        raw_text="noisy but correct payload",
    )
    client = FakeLLMClient([noisy])
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

    action = player.choose_action(observation, [])

    assert action == decision.options[0]
    assert player.journal[-1].fallback_used is False


def test_fallback_journal_entry_records_raw_responses_for_debugging():
    """A fallback used to record only `fallback_reason`, never what the
    model actually said -- making a systematic mismatch (e.g. the model
    consistently naming an option that never matches) undiagnosable from
    the log alone. Every attempt's raw text must survive onto the journal
    entry even though none of it is committed to the persisted conversation."""
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    decision = observation.pending_decision
    payload_key = PAYLOAD_KEY_BY_KIND[decision.kind]

    illegal = make_plan_response(
        "Confidently wrong.", [(decision.kind, {payload_key: "__nonexistent__"})]
    )
    malformed = LLMResponse(structured={}, raw_text="not json shaped correctly")
    client = FakeLLMClient([illegal, malformed])
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

    player.choose_action(observation, [])

    entry = player.journal[-1]
    assert len(entry.raw_responses) == 2
    assert "__nonexistent__" in entry.raw_responses[0]
    assert entry.raw_responses[1] == "not json shaped correctly"


def test_successful_journal_entry_also_records_raw_responses():
    observation, decision, payload_key, correct_value = _single_step_response_and_decision()
    response = make_plan_response("ok", [(decision.kind, {payload_key: correct_value})])
    client = FakeLLMClient([response])
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

    player.choose_action(observation, [])

    assert player.journal[-1].raw_responses == (response.raw_text,)


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
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

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
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

    player.choose_action(observation, [])

    roles = [m.role for m in player._messages]
    assert roles == ["user", "assistant"]


def test_persisted_history_drops_stale_board_snapshot_but_keeps_event_deltas():
    """Only the event delta of a user turn should survive into the
    persisted conversation -- the board report/hand dossier/cards-in-play it
    was answered against are a snapshot of that instant, true only for the
    live call, and must not be resent turn after turn as the game goes on
    (see prompt.build_history_entry)."""
    engine = Engine.new_game(seed=1)
    observation, country = _setup_placement(engine)
    decision = observation.pending_decision

    client = FakeLLMClient(
        [
            make_plan_response("First.", [(decision.kind, {"country": country})]),
            make_plan_response("Second.", [(decision.kind, {"country": country})]),
        ]
    )
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

    first = player.choose_action(observation, [])
    engine.step(first)
    history = [_dummy_event(decision)]
    player.choose_action(engine.observe(engine.pending_decision.actor), history)

    # The live call for the second decision still gets the full picture.
    assert "BOARD BY REGION" in client.requests[1].messages[-1].content

    # But what's left behind in memory for both turns is thin.
    persisted_user_messages = [m.content for m in player._messages if m.role == "user"]
    assert len(persisted_user_messages) == 2
    assert "BOARD BY REGION" not in persisted_user_messages[0]
    assert "YOUR HAND" not in persisted_user_messages[0]
    assert persisted_user_messages[0] == "(no new events since your last request)"
    assert "Since your last request (1 event(s))" in persisted_user_messages[1]
    assert "BOARD BY REGION" not in persisted_user_messages[1]


def test_region_last_scored_reaches_the_prompt_from_full_history_not_just_new_events():
    """`choose_action` receives the whole game's `history`, not just the
    delta since this player's last real call -- `build_board_report` needs
    the full history to answer "how long ago was this region scored",
    which can reach further back than one call's worth of new events."""
    engine = Engine.new_game(seed=1)
    observation, country = _setup_placement(engine)
    decision = observation.pending_decision

    client = FakeLLMClient(
        [
            make_plan_response("First.", [(decision.kind, {"country": country})]),
            make_plan_response("Second.", [(decision.kind, {"country": country})]),
        ]
    )
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

    scoring_event = Event(
        actor=Side.USSR,
        decision=dataclasses.replace(decision, kind=DecisionKind.ACTION_ROUND_PLAY),
        action=Action(DecisionKind.ACTION_ROUND_PLAY, {"card": "Asia_Scoring"}),
        defcon=5,
        vp=0,
        turn=1,
        action_round=1,
    )
    first = player.choose_action(observation, [scoring_event])
    engine.step(first)

    second_observation = engine.observe(engine.pending_decision.actor)
    later_event = _dummy_event(decision)
    player.choose_action(second_observation, [scoring_event, later_event])

    second_request_text = client.requests[1].messages[-1].content
    assert "last scored: turn 1" in second_request_text


def test_llm_player_constructs_with_anthropic_client_given_an_api_key():
    pytest.importorskip("anthropic")
    from struggler.bots.llm.anthropic_client import AnthropicClient

    client = AnthropicClient(model="claude-sonnet-5", api_key="test-key-not-a-real-key")
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

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
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

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
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

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
    player = LLMPlayer(client=client, seed=0, plan_turns=False)

    player.choose_action(observation, [])

    assert player.journal[-1].usage == {"input_tokens": 40, "output_tokens": 8}


def test_log_path_none_never_touches_disk(monkeypatch):
    observation, decision, payload_key, correct_value = _single_step_response_and_decision()
    response = make_plan_response("ok", [(decision.kind, {payload_key: correct_value})])
    client = FakeLLMClient([response])
    player = LLMPlayer(client=client, seed=0, plan_turns=False, log_path=None)

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
    player = LLMPlayer(client=client, seed=0, plan_turns=False, log_path=log_path)

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
    first_player = LLMPlayer(client=first_client, seed=0, plan_turns=False, log_path=log_path)

    observation = engine.observe(Side.US)
    action = first_player.choose_action(observation, [])
    engine.step(action)
    assert len(first_client.requests) == 1

    # Simulate a fresh process: a brand-new LLMPlayer with a brand-new
    # client whose response script is empty, resuming purely from disk.
    second_client = FakeLLMClient([])
    second_player = LLMPlayer(client=second_client, seed=0, plan_turns=False, log_path=log_path, resume=True)

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


def test_llm_player_does_not_auto_resume_without_explicit_flag(tmp_path):
    """A snapshot already sitting at `log_path` (e.g. left over from an
    earlier, unrelated game that happened to reuse the same seed/path) must
    never be picked up unless the caller explicitly passes `resume=True`."""
    log_path = tmp_path / "log.json"
    snapshot = conversation_log.ConversationSnapshot(
        seed=0,
        provider="fake",
        model="fake-model",
        created_at=conversation_log.now_iso(),
        updated_at=conversation_log.now_iso(),
        last_seen=5,
        cumulative_usage={"input_tokens": 99, "output_tokens": 99},
        messages=(),
        plan=(),
        journal=(),
    )
    conversation_log.save(log_path, snapshot)

    player = LLMPlayer(client=FakeLLMClient([]), seed=0, plan_turns=False, log_path=log_path)

    assert player._last_seen == 0
    assert player.journal == []
    assert player.cumulative_usage == {"input_tokens": 0, "output_tokens": 0}


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

    player = LLMPlayer(client=FakeLLMClient([]), seed=0, plan_turns=False, log_path=log_path, resume=True)
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)

    with pytest.raises(ValueError):
        player.choose_action(observation, [])


# -- turn planning -------------------------------------------------------------


def _setup_placement(engine: Engine) -> tuple[object, str]:
    """A live PLACE_INFLUENCE decision plus one of its legal countries."""
    observation = engine.observe(engine.pending_decision.actor)
    return observation, observation.pending_decision.options[0].payload["country"]


def test_turn_plan_is_requested_once_per_turn_and_precedes_the_decision():
    engine = Engine.new_game(seed=1)
    observation, country = _setup_placement(engine)
    decision = observation.pending_decision

    client = FakeLLMClient(
        [
            make_turn_plan_response("Take Poland and meet Military Operations."),
            make_plan_response("First point.", [(decision.kind, {"country": country})]),
            make_plan_response("Second point.", [(decision.kind, {"country": country})]),
        ]
    )
    player = LLMPlayer(client=client, seed=0)

    first = player.choose_action(observation, [])
    engine.step(first)
    second = player.choose_action(engine.observe(engine.pending_decision.actor), [])

    assert first.payload["country"] == country
    assert second.payload["country"] == country
    # Three calls: one plan for the turn, then one per decision -- the second
    # decision in the same turn must NOT trigger another planning call.
    assert len(client.requests) == 3
    assert client.requests[0].output.name == "turn_plan"
    assert client.requests[1].output.name == "decision_plan"
    assert client.requests[2].output.name == "decision_plan"


def test_turn_plan_is_injected_into_every_later_decision_prompt():
    engine = Engine.new_game(seed=1)
    observation, country = _setup_placement(engine)
    decision = observation.pending_decision

    client = FakeLLMClient(
        [
            make_turn_plan_response(
                "Finish Poland before playing Asia Scoring.",
                defend=["East_Germany"],
                scoring_cards=[
                    {"card": "Asia_Scoring", "when": "AR6", "preparation": "take Thailand"}
                ],
            ),
            make_plan_response("Point one.", [(decision.kind, {"country": country})]),
        ]
    )
    player = LLMPlayer(client=client, seed=0)

    player.choose_action(observation, [])

    decision_prompt = client.requests[1].messages[-1].content
    assert "YOUR PLAN FOR TURN 1" in decision_prompt
    assert "Finish Poland before playing Asia Scoring." in decision_prompt
    assert "Hold or retake: East_Germany" in decision_prompt
    assert "Asia_Scoring" in decision_prompt


def test_a_new_turn_triggers_a_fresh_turn_plan():
    engine = Engine.new_game(seed=1)
    observation, country = _setup_placement(engine)
    decision = observation.pending_decision
    turn_two = dataclasses.replace(observation, turn=2)

    client = FakeLLMClient(
        [
            make_turn_plan_response("Turn 1 plan."),
            make_plan_response("Turn 1 move.", [(decision.kind, {"country": country})]),
            make_turn_plan_response("Turn 2 plan."),
            make_plan_response("Turn 2 move.", [(decision.kind, {"country": country})]),
        ]
    )
    player = LLMPlayer(client=client, seed=0)

    player.choose_action(observation, [])
    player.choose_action(turn_two, [])

    assert [r.output.name for r in client.requests] == [
        "turn_plan",
        "decision_plan",
        "turn_plan",
        "decision_plan",
    ]
    assert "Turn 2 plan." in client.requests[3].messages[-1].content
    assert "Turn 1 plan." not in client.requests[3].messages[-1].content.split("YOUR PLAN")[-1]


def test_failed_turn_planning_still_plays_the_turn_without_replanning():
    engine = Engine.new_game(seed=1)
    observation, country = _setup_placement(engine)
    decision = observation.pending_decision
    malformed = LLMResponse(structured={}, raw_text="{}")

    client = FakeLLMClient(
        [
            malformed,  # planning attempt
            malformed,  # its one retry
            make_plan_response("Play on regardless.", [(decision.kind, {"country": country})]),
            make_plan_response("Still no plan.", [(decision.kind, {"country": country})]),
        ]
    )
    player = LLMPlayer(client=client, seed=0)

    first = player.choose_action(observation, [])
    engine.step(first)
    second = player.choose_action(engine.observe(engine.pending_decision.actor), [])

    assert first.payload["country"] == country
    assert second.payload["country"] == country
    assert len(client.requests) == 4  # no second planning attempt this turn
    plan_entry = next(e for e in player.journal if e.kind == "turn_plan")
    assert plan_entry.fallback_used is True
    assert "YOUR PLAN FOR TURN" not in client.requests[2].messages[-1].content


def test_turn_plan_journal_entry_records_the_objective():
    engine = Engine.new_game(seed=1)
    observation, country = _setup_placement(engine)
    decision = observation.pending_decision

    client = FakeLLMClient(
        [
            make_turn_plan_response("Control Poland and Czechoslovakia."),
            make_plan_response("Point one.", [(decision.kind, {"country": country})]),
        ]
    )
    player = LLMPlayer(client=client, seed=0)

    player.choose_action(observation, [])

    assert [e.kind for e in player.journal] == ["turn_plan", "decision"]
    assert player.journal[0].justification == "Control Poland and Czechoslovakia."
    assert player.journal[0].fallback_used is False


def test_turn_plan_survives_a_resume_from_the_log():
    engine = Engine.new_game(seed=1)
    observation, country = _setup_placement(engine)
    decision = observation.pending_decision

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "ussr.json"
        first_client = FakeLLMClient(
            [
                make_turn_plan_response("Hold East Germany.", defend=["East_Germany"]),
                make_plan_response("Point one.", [(decision.kind, {"country": country})]),
            ]
        )
        first_player = LLMPlayer(client=first_client, seed=0, log_path=log_path)
        first_player.choose_action(observation, [])
        history = [_dummy_event(decision)] * first_player._last_seen

        second_client = FakeLLMClient(
            [make_plan_response("Point two.", [(decision.kind, {"country": country})])]
        )
        second_player = LLMPlayer(client=second_client, seed=0, log_path=log_path, resume=True)
        second_player.choose_action(observation, history)

        # Same turn, so the resumed player must not re-plan -- and must still
        # be prompted with the plan it wrote before the process ended.
        assert [r.output.name for r in second_client.requests] == ["decision_plan"]
        assert "Hold East Germany." in second_client.requests[0].messages[-1].content


def _dummy_event(decision):
    return Event(
        actor=Side.USSR,
        decision=decision,
        action=decision.options[0],
        defcon=5,
        vp=0,
        turn=1,
        action_round=1,
    )
