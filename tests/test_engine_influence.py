"""Engine: PLACE_INFLUENCE decision flow (atomic Ops spending)."""

import pytest

from struggler.engine import DecisionKind, Engine, Side


def test_begin_influence_operations_pushes_one_decision_per_point():
    engine = Engine(seed=1)
    engine.begin_influence_operations(Side.USSR, 4)
    for ops_left in (4, 3, 2, 1):
        decision = engine.pending_decision
        assert decision is not None
        assert decision.kind is DecisionKind.PLACE_INFLUENCE
        assert decision.actor is Side.USSR
        assert decision.context["ops_remaining"] == ops_left
        engine.step(decision.options[0])
    assert engine.pending_decision is None


def test_legal_options_stay_in_the_tens_not_thousands():
    # Mandate #2: atomic action space, never in the thousands.
    engine = Engine(seed=1)
    engine.begin_influence_operations(Side.USSR, 4)
    assert len(engine.legal_actions()) < 200


def test_step_rejects_illegal_action():
    from struggler.engine import Action

    engine = Engine(seed=1)
    engine.begin_influence_operations(Side.USSR, 4)
    bogus = Action(DecisionKind.PLACE_INFLUENCE, {"country": "Cuba"})  # not reachable yet
    assert bogus not in engine.legal_actions()
    with pytest.raises(ValueError):
        engine.step(bogus)


def test_placing_in_opponent_controlled_country_spends_double():
    engine = Engine(seed=1)
    engine.board.influence["Finland"]["US"] = 4  # US controls Finland (stability 4)
    engine.begin_influence_operations(Side.USSR, 3)
    decision = engine.pending_decision
    finland_action = next(a for a in decision.options if a.payload["country"] == "Finland")
    engine.step(finland_action)
    # cost 2 for placing in opponent-controlled Finland, leaving 1 ops
    assert engine.pending_decision.context["ops_remaining"] == 1
    assert engine.board.influence["Finland"]["USSR"] == 1


def test_leftover_ops_are_wasted_when_no_legal_targets_remain():
    engine = Engine(seed=1)
    # 1 leftover Op, but the only reachable move left costs 2 (opponent-controlled).
    engine.board.influence["Finland"]["US"] = 4
    engine.begin_influence_operations(Side.USSR, 1)
    decision = engine.pending_decision
    assert all(a.payload["country"] != "Finland" for a in decision.options)


def test_observe_reflects_influence_and_pending_decision():
    engine = Engine(seed=1)
    engine.begin_influence_operations(Side.USSR, 1)
    obs = engine.observe(Side.USSR)
    assert obs.side is Side.USSR
    assert obs.defcon == 5
    assert obs.pending_decision is engine.pending_decision

    with pytest.raises(ValueError):
        engine.observe(Side.CHANCE)
