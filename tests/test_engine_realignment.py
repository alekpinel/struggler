"""Engine: realignment mechanics."""

from struggler.engine import Engine
from struggler.types import DecisionKind, Side


def test_realignment_pushes_actor_then_opponent_chance_rolls():
    engine = Engine(seed=9)
    engine.board.influence["Guatemala"]["USSR"] = 5
    engine.begin_realignment_operations(Side.US, ops=1)

    target = next(a for a in engine.legal_actions() if a.payload["country"] == "Guatemala")
    engine.step(target)
    assert engine.pending_decision.kind is DecisionKind.REALIGNMENT_ACTOR_ROLL
    assert engine.pending_decision.actor is Side.CHANCE
    actor_roll = engine.pending_decision.options[0].payload["value"]
    engine.step(engine.pending_decision.options[0])

    assert engine.pending_decision.kind is DecisionKind.REALIGNMENT_OPPONENT_ROLL
    assert engine.pending_decision.actor is Side.CHANCE
    opp_roll = engine.pending_decision.options[0].payload["value"]

    actor_bonus = engine._realignment_bonus(Side.US, "Guatemala")
    opp_bonus = engine._realignment_bonus(Side.USSR, "Guatemala")
    engine.step(engine.pending_decision.options[0])

    margin = (actor_roll + 1 + actor_bonus) - (opp_roll + opp_bonus)
    expected = max(0, 5 - margin) if margin > 0 else 5
    assert engine.board.influence["Guatemala"]["USSR"] == expected


def test_realignment_never_adds_actor_influence():
    engine = Engine(seed=9)
    engine.begin_realignment_operations(Side.US, ops=1)
    target = next(a for a in engine.legal_actions() if a.payload["country"] == "Guatemala")
    engine.step(target)
    engine.step(engine.pending_decision.options[0])
    engine.step(engine.pending_decision.options[0])
    assert engine.board.influence["Guatemala"]["US"] == 0


def test_realignment_chains_attempts_until_exhausted():
    engine = Engine(seed=2)
    engine.begin_realignment_operations(Side.USSR, ops=2)
    for expected_attempts_left in (2, 1):
        assert engine.pending_decision.kind is DecisionKind.REALIGNMENT_TARGET
        assert engine.pending_decision.context["attempts_remaining"] == expected_attempts_left
        engine.step(engine.pending_decision.options[0])  # target
        engine.step(engine.pending_decision.options[0])  # actor roll
        engine.step(engine.pending_decision.options[0])  # opponent roll
    assert engine.pending_decision is None
