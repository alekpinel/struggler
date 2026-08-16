"""Tests for bots/llm/prompt.py: the rules primer, the curated map dump,
and the DecisionKind catalog that make up the LLM system prompt, plus the
per-call user turn builder.
"""

from __future__ import annotations

from struggler.bots.llm.prompt import build_system_prompt, build_user_turn
from struggler.bots.llm.schema import PLAYER_FACING_KINDS
from struggler.engine import DecisionKind, Engine


def test_system_prompt_explains_control_formula():
    prompt = build_system_prompt()
    assert "stability" in prompt
    assert "Influence there) - (opponent's Influence there) >= " in prompt


def test_system_prompt_explains_regional_scoring_tiers():
    prompt = build_system_prompt()
    for tier in ("Presence", "Domination", "Control"):
        assert tier in prompt
    assert "Battleground countries there" in prompt  # Domination definition
    assert "EVERY Battleground country" in prompt  # Control definition


def test_system_prompt_explains_defcon_coup_lock_table():
    prompt = build_system_prompt()
    assert "Europe needs DEFCON >= 5" in prompt
    assert "Asia >= 4" in prompt
    assert "Middle East >= 3" in prompt


def test_system_prompt_explains_coup_formula():
    prompt = build_system_prompt()
    assert "die_roll + ops_spent - 2*country.stability" in prompt


def test_system_prompt_explains_opponent_event_firing_on_ops_play():
    prompt = build_system_prompt()
    assert "EVENT_OPS_ORDER" in prompt
    assert "Event still fires" in prompt


def test_system_prompt_explains_china_card_bonus():
    prompt = build_system_prompt()
    assert "China Card" in prompt
    assert "spent inside Asia" in prompt


def test_system_prompt_explains_victory_conditions():
    prompt = build_system_prompt()
    assert "+/-20" in prompt
    assert "DEFCON hits" in prompt
    assert "Controls every" in prompt and "Europe" in prompt


def test_system_prompt_strips_internal_provenance_metadata():
    prompt = build_system_prompt()
    for internal_key in (
        "_disclaimer",
        "_confirmed_against_physical_board",
        "_uncertain",
        "_setup_influence_note",
    ):
        assert internal_key not in prompt


def test_system_prompt_still_carries_country_map_data():
    prompt = build_system_prompt()
    # A representative Battleground country and a representative
    # adjacency entry must still be present in the (curated) dump.
    assert '"France"' in prompt
    assert '"battleground": true' in prompt
    assert '"adjacent_to"' in prompt


def test_payload_catalog_has_a_meaning_for_every_player_facing_kind():
    prompt = build_system_prompt()
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


def test_build_user_turn_reports_current_decision_and_observation():
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    decision = observation.pending_decision

    text = build_user_turn(observation, decision, [])

    assert f"id={decision.id}" in text
    assert f"kind={decision.kind.value}" in text
    assert "Current observation:" in text
    assert "decision_plan" in text


def test_build_user_turn_includes_untouched_countries():
    # A country neither side has ever placed Influence in (0-0) must still
    # appear in the observation dump -- an earlier version of
    # `_observation_to_text` filtered these out, which made empty-but-
    # reachable Battlegrounds (e.g. West Germany) invisible to the model
    # for the whole game.
    engine = Engine.new_game(seed=1)
    observation = engine.observe(engine.pending_decision.actor)
    decision = observation.pending_decision

    text = build_user_turn(observation, decision, [])

    assert '"West_Germany"' in text
    assert '"France"' in text
    assert '"Italy"' in text
