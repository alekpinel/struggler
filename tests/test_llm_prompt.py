"""Tests for bots/llm/prompt.py: the rules primer, the curated map dump,
and the DecisionKind catalog that make up the LLM system prompt, plus the
per-call user turn builder.
"""

from __future__ import annotations

import dataclasses

from struggler.bots.llm.prompt import (
    build_history_entry,
    build_system_prompt,
    build_turn_plan_request,
    build_user_turn,
)
from struggler.bots.llm.schema import PLAYER_FACING_KINDS
from struggler.engine import Action, DecisionKind, Engine, Side
from struggler.engine.player import Event


def test_system_prompt_explains_control_formula():
    prompt = build_system_prompt(Side.US)
    assert "stability" in prompt
    assert "Influence there) - (opponent's Influence there) >= " in prompt


def test_system_prompt_explains_regional_scoring_tiers():
    prompt = build_system_prompt(Side.US)
    for tier in ("Presence", "Domination", "Control"):
        assert tier in prompt
    assert "Battleground countries there" in prompt  # Domination definition
    assert "EVERY Battleground country" in prompt  # Control definition


def test_system_prompt_explains_defcon_coup_lock_table():
    prompt = build_system_prompt(Side.US)
    assert "Europe needs DEFCON >= 5" in prompt
    assert "Asia >= 4" in prompt
    assert "Middle East >= 3" in prompt


def test_system_prompt_explains_coup_formula():
    prompt = build_system_prompt(Side.US)
    assert "die_roll + ops_spent - 2*country.stability" in prompt


def test_system_prompt_explains_opponent_event_firing_on_ops_play():
    prompt = build_system_prompt(Side.US)
    assert "EVENT_OPS_ORDER" in prompt
    assert "Event still fires" in prompt


def test_system_prompt_explains_china_card_bonus():
    prompt = build_system_prompt(Side.US)
    assert "China Card" in prompt
    assert "spent inside Asia" in prompt


def test_system_prompt_explains_victory_conditions():
    prompt = build_system_prompt(Side.US)
    assert "+/-20" in prompt
    assert "DEFCON hits" in prompt
    assert "Controls every" in prompt and "Europe" in prompt


def test_system_prompt_strips_internal_provenance_metadata():
    prompt = build_system_prompt(Side.US)
    for internal_key in (
        "_disclaimer",
        "_confirmed_against_physical_board",
        "_uncertain",
        "_setup_influence_note",
    ):
        assert internal_key not in prompt


def test_system_prompt_still_carries_country_map_data():
    prompt = build_system_prompt(Side.US)
    # A representative Battleground country and a representative
    # adjacency entry must still be present in the (curated) dump.
    assert '"France"' in prompt
    assert '"battleground": true' in prompt
    assert '"adjacent_to"' in prompt


def test_payload_catalog_has_a_meaning_for_every_player_facing_kind():
    prompt = build_system_prompt(Side.US)
    for kind in PLAYER_FACING_KINDS:
        if kind is DecisionKind.EVENT_RESUME:
            continue  # single-option marker, never actually reaches the LLM
        assert f"- {kind.value}:" in prompt, f"missing catalog line for {kind}"
        # The line must carry more than just the payload key -- a
        # semantic description before the "-- payload.X" suffix.
        line = next(line for line in prompt.splitlines() if line.strip().startswith(f"- {kind.value}:"))
        assert " -- payload." in line
        description = line.split(":", 1)[1].split(" -- payload.")[0].strip()
        assert len(description) > 10


def test_build_user_turn_reports_current_decision_and_board_reading():
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    decision = observation.pending_decision

    text = build_user_turn(observation, decision, [])

    assert f"id={decision.id}" in text
    assert f"kind={decision.kind.value}" in text
    assert "STATUS: turn 1" in text
    assert "REGIONAL SCORING STATUS" in text
    assert "BOARD BY REGION" in text
    assert "decision_plan" in text


def test_build_user_turn_includes_untouched_countries():
    # A country neither side has ever placed Influence in (0-0) must still
    # appear in the board reading -- an earlier version filtered these out,
    # which made empty-but-reachable Battlegrounds (e.g. West Germany)
    # invisible to the model for the whole game.
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    decision = observation.pending_decision

    text = build_user_turn(observation, decision, [])
    lines = {line.split()[0]: line for line in (l.strip() for l in text.splitlines()) if line}

    for empty_battleground in ("West_Germany", "France", "Italy"):
        assert empty_battleground in lines
        assert "US0/SU0" in lines[empty_battleground]
        assert "BG" in lines[empty_battleground]


def test_build_user_turn_states_control_needs_and_reachability():
    # The derived reading is the point of the board section: raw influence
    # numbers are what the model already failed to turn into Control
    # decisions on its own.
    engine = Engine.new_game(seed=1)
    observation = engine.observe(Side.USSR)
    decision = observation.pending_decision

    text = build_user_turn(observation, decision, [])
    poland = next(l.strip() for l in text.splitlines() if l.strip().startswith("Poland "))

    assert "need:+3" in poland  # stability 3, empty at setup
    assert poland.endswith("R")  # adjacent to the USSR home space


def test_build_user_turn_reports_battleground_priorities_and_scoring_net():
    engine = Engine.new_game(seed=1)
    observation = engine.observe(Side.USSR)

    text = build_user_turn(observation, observation.pending_decision, [])

    assert "BATTLEGROUND PRIORITIES" in text
    assert "AT RISK" in text  # East Germany is held at exactly its margin at setup
    assert "MILITARY OPERATIONS" in text
    assert "SPACE RACE" in text


def test_build_user_turn_hand_dossier_carries_facts_and_advice():
    engine = Engine.new_game(seed=1, events=True)
    observation = engine.observe(Side.USSR)

    text = build_user_turn(observation, observation.pending_decision, [])
    hand_section = text.split("YOUR HAND")[1]

    for card in observation.hand:
        assert card in hand_section
    assert "effective Ops" in hand_section
    assert "space race:" in hand_section


def test_turn_plan_request_asks_for_intent_not_an_action():
    engine = Engine.new_game(seed=1, events=True)
    observation = engine.observe(Side.USSR)

    text = build_turn_plan_request(observation, [])

    assert "turn_plan" in text
    assert "BOARD BY REGION" in text
    assert "decision_plan" not in text
    assert "You are not choosing an action now." in text


def test_turn_plan_request_asks_for_a_region_focus():
    engine = Engine.new_game(seed=1, events=True)
    observation = engine.observe(Side.USSR)

    text = build_turn_plan_request(observation, [])

    assert "region_focus" in text
    assert "Scoring card is in your hand comes first" in text


def test_turn_plan_request_carries_last_scored_from_full_history_not_just_deltas():
    # `history` is the whole game so far, distinct from `new_events` (the
    # delta since the last call) -- the region-focus decision needs to look
    # back further than one call's worth of events to know how long ago a
    # region was last scored.
    engine = Engine.new_game(seed=1, events=True)
    observation = engine.observe(Side.USSR)
    decision = observation.pending_decision
    old_event = Event(
        actor=Side.USSR,
        decision=dataclasses.replace(decision, kind=DecisionKind.ACTION_ROUND_PLAY),
        action=Action(DecisionKind.ACTION_ROUND_PLAY, {"card": "Asia_Scoring"}),
        defcon=5,
        vp=0,
        turn=1,
        action_round=1,
    )

    text = build_turn_plan_request(observation, [], history=[old_event])

    assert "last scored: turn 1" in text


def test_build_history_entry_carries_only_the_event_delta():
    # The persisted-conversation view of a user turn must never carry the
    # board report/hand dossier/cards-in-play `build_user_turn` sends live --
    # those are a snapshot of one instant and would just be stale token cost
    # on every later call.
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    decision = observation.pending_decision
    event = Event(
        actor=Side.USSR,
        decision=decision,
        action=decision.options[0],
        defcon=5,
        vp=0,
        turn=1,
        action_round=1,
    )

    assert build_history_entry([]) == "(no new events since your last request)"

    entry = build_history_entry([event])
    assert "Since your last request (1 event(s))" in entry
    assert "BOARD BY REGION" not in entry
    assert "YOUR HAND" not in entry
