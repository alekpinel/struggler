"""Tests for bots/llm/board_report.py: the derived board reading the LLM
prompt is built from.

Every number here is a rules-defined fact the raw `Observation` only
implies, so these assert on the arithmetic (Control margins, placement
cost, region net VP, Space Race attempts left), not on wording.
"""

from __future__ import annotations

import dataclasses

from struggler.bots.llm.board_report import (
    battleground_alerts,
    board_from_observation,
    build_board_report,
    coup_targets_text,
    military_ops_line,
    opponent_activity,
    points_to_break,
    points_to_control,
    possible_coup_targets,
    region_status,
    space_race_line,
)
from struggler.engine import Action, DecisionKind, Engine, Side
from struggler.engine.player import Event
from struggler.engine.types import Region, ScoringTier


def _observation(engine: Engine, side: Side = Side.USSR):
    return engine.observe(side)


def test_points_to_control_accounts_for_opponent_influence():
    engine = Engine.new_game(seed=1)
    board = board_from_observation(_observation(engine))
    board.influence["Poland"] = {"US": 2, "USSR": 1}

    # Stability 3: USSR needs 3 - (1 - 2) = 4 more points.
    assert points_to_control(board, Side.USSR, "Poland") == 4
    assert points_to_control(board, Side.US, "Poland") == 2


def test_points_to_control_is_zero_once_controlled():
    engine = Engine.new_game(seed=1)
    board = board_from_observation(_observation(engine))
    board.influence["Poland"] = {"US": 0, "USSR": 4}

    assert points_to_control(board, Side.USSR, "Poland") == 0
    assert points_to_break(board, Side.USSR, "Poland") == 2  # 4 - 3 + 1
    assert points_to_break(board, Side.US, "Poland") == 0  # not theirs to lose


def test_region_status_signs_net_vp_for_the_asking_side():
    engine = Engine.new_game(seed=1)
    observation = _observation(engine)
    board = board_from_observation(observation)

    us_view = region_status(board, Side.US, Region.EUROPE)
    ussr_view = region_status(board, Side.USSR, Region.EUROPE)

    assert us_view.net_vp_for_side == -ussr_view.net_vp_for_side
    assert us_view.own_tier is ussr_view.opp_tier


def test_region_status_separates_controlled_battlegrounds():
    engine = Engine.new_game(seed=1)
    observation = _observation(engine)
    board = board_from_observation(observation)
    board.influence["Poland"] = {"US": 0, "USSR": 3}
    board.influence["West_Germany"] = {"US": 4, "USSR": 0}

    status = region_status(board, Side.USSR, Region.EUROPE)

    assert "Poland" in status.own_bg
    assert "West_Germany" in status.opp_bg
    assert "France" in status.free_bg
    assert status.own_tier is not ScoringTier.NONE


def test_battleground_alerts_flag_retake_at_risk_and_unclaimed():
    engine = Engine.new_game(seed=1)
    board = board_from_observation(_observation(engine))
    for cid in board.influence:
        board.influence[cid] = {"US": 0, "USSR": 0}
    board.influence["Poland"] = {"US": 1, "USSR": 2}  # contested, USSR reachable, not controlled
    board.influence["East_Germany"] = {"US": 0, "USSR": 3}  # controlled at exactly its margin

    alerts = "\n".join(battleground_alerts(board, Side.USSR))

    assert "RETAKE" in alerts and "Poland" in alerts
    assert "AT RISK" in alerts and "East_Germany" in alerts
    assert "UNCLAIMED" in alerts  # e.g. West Germany, adjacent to East Germany


def test_battleground_alerts_price_an_opponent_controlled_country_at_double():
    engine = Engine.new_game(seed=1)
    board = board_from_observation(_observation(engine))
    board.influence["Poland"] = {"US": 4, "USSR": 1}  # US-controlled, USSR still present

    alerts = "\n".join(battleground_alerts(board, Side.USSR))

    # Needs 6 points at 2 Ops each: the report must state the Ops cost, not
    # just the point count -- the doubling rule is what makes retaking late
    # so expensive.
    assert "Poland (you hold 1, need +6 = 12 Ops)" in alerts


def test_opponent_activity_reports_only_the_opponent_and_nets_per_country():
    engine = Engine.new_game(seed=1)
    decision = engine.pending_decision

    def event(actor: Side, country: str, us: int, ussr: int) -> Event:
        return Event(
            actor=actor,
            decision=decision,
            action=Action(DecisionKind.PLACE_INFLUENCE, {"country": country}),
            defcon=5,
            vp=0,
            turn=2,
            action_round=1,
            country=country,
            country_influence={"US": us, "USSR": ussr},
            country_control=None,
        )

    activity = opponent_activity(
        [
            event(Side.US, "Poland", 1, 3),
            event(Side.US, "Poland", 2, 3),  # same country twice -> one line, latest state
            event(Side.USSR, "Romania", 0, 1),  # our own move -> not activity
        ],
        Side.USSR,
    )

    assert len(activity) == 1
    assert "Poland" in activity[0]
    assert "US2/USSR3" in activity[0]


def test_military_ops_line_states_the_shortfall_as_vp():
    engine = Engine.new_game(seed=1)
    observation = dataclasses.replace(
        _observation(engine), defcon=4, military_ops={"US": 4, "USSR": 1}
    )

    line = military_ops_line(observation)

    assert "1/4" in line
    assert "3 short" in line
    assert "3 VP" in line


def test_military_ops_line_says_so_when_the_requirement_is_met():
    engine = Engine.new_game(seed=1)
    observation = dataclasses.replace(
        _observation(engine), defcon=3, military_ops={"US": 0, "USSR": 5}
    )

    assert "already met" in military_ops_line(observation)


def test_space_race_line_reports_attempts_left():
    engine = Engine.new_game(seed=1)
    base = _observation(engine)

    fresh = space_race_line(base)
    spent = space_race_line(dataclasses.replace(base, space_race_attempts={"US": 0, "USSR": 1}))

    assert "attempts left this turn 1/1" in fresh
    assert "attempts left this turn 0/1" in spent


def test_space_race_line_reports_the_second_attempt_from_the_qualifying_box():
    engine = Engine.new_game(seed=1)
    observation = dataclasses.replace(
        _observation(engine),
        space_race={"US": 0, "USSR": 2},
        space_race_attempts={"US": 0, "USSR": 1},
    )

    assert "attempts left this turn 1/2" in space_race_line(observation)


def test_possible_coup_targets_filters_by_opponent_influence_and_defcon_lock():
    engine = Engine.new_game(seed=1)
    observation = dataclasses.replace(_observation(engine, Side.USSR), defcon=4)
    board = board_from_observation(observation)
    for cid in board.influence:
        board.influence[cid] = {"US": 0, "USSR": 0}
    board.influence["Guatemala"] = {"US": 2, "USSR": 0}  # Central America: no DEFCON lock
    board.influence["Poland"] = {"US": 2, "USSR": 0}  # Europe: needs DEFCON 5+

    targets = {cid for cid, _, _ in possible_coup_targets(board, observation)}

    assert "Guatemala" in targets
    assert "Poland" not in targets  # DEFCON 4 locks Europe out


def test_possible_coup_targets_flags_battleground_and_defcon_drop():
    engine = Engine.new_game(seed=1)
    observation = _observation(engine, Side.USSR)
    board = board_from_observation(observation)
    for cid in board.influence:
        board.influence[cid] = {"US": 0, "USSR": 0}
    board.influence["Cuba"] = {"US": 1, "USSR": 0}  # Central America, Battleground
    board.influence["Guatemala"] = {"US": 1, "USSR": 0}  # Central America, not a Battleground

    targets = {cid: (is_bg, would_drop) for cid, is_bg, would_drop in possible_coup_targets(board, observation)}

    assert targets["Cuba"] == (True, True)
    assert targets["Guatemala"] == (False, False)


def test_possible_coup_targets_nuclear_subs_exempts_only_the_us():
    engine = Engine.new_game(seed=1)
    observation = dataclasses.replace(
        _observation(engine, Side.US), turn_effects={"nuclear_subs": True}
    )
    board = board_from_observation(observation)
    for cid in board.influence:
        board.influence[cid] = {"US": 0, "USSR": 0}
    board.influence["Cuba"] = {"US": 0, "USSR": 1}  # USSR-held Battleground; US is the actor

    targets = {cid: would_drop for cid, _, would_drop in possible_coup_targets(board, observation)}

    assert targets["Cuba"] is False


def test_coup_targets_text_reports_none_when_nothing_is_legal():
    engine = Engine.new_game(seed=1)
    observation = _observation(engine, Side.USSR)
    board = board_from_observation(observation)
    for cid in board.influence:
        board.influence[cid] = {"US": 0, "USSR": 0}

    assert "COUP TARGETS: none" in coup_targets_text(board, observation)


def test_coup_targets_text_warns_when_every_target_is_a_battleground_at_defcon_2():
    engine = Engine.new_game(seed=1)
    observation = dataclasses.replace(_observation(engine, Side.USSR), defcon=2)
    board = board_from_observation(observation)
    for cid in board.influence:
        board.influence[cid] = {"US": 0, "USSR": 0}
    board.influence["Cuba"] = {"US": 1, "USSR": 0}  # only target, and it's a Battleground

    text = coup_targets_text(board, observation)

    assert "WARNING" in text
    assert "immediate loss" in text


def test_coup_targets_text_does_not_warn_when_a_non_battleground_target_exists():
    engine = Engine.new_game(seed=1)
    observation = dataclasses.replace(_observation(engine, Side.USSR), defcon=2)
    board = board_from_observation(observation)
    for cid in board.influence:
        board.influence[cid] = {"US": 0, "USSR": 0}
    board.influence["Cuba"] = {"US": 1, "USSR": 0}
    board.influence["Guatemala"] = {"US": 1, "USSR": 0}

    assert "WARNING" not in coup_targets_text(board, observation)


def test_report_covers_every_country_including_empty_ones():
    engine = Engine.new_game(seed=1)
    observation = _observation(engine)
    board = board_from_observation(observation)

    text = build_board_report(observation)

    for cid in board.countries:
        assert cid in text
