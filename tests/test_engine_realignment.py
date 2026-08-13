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

    # 6.2.2: the die roll is not modified by the card's Operations value.
    margin = (actor_roll + actor_bonus) - (opp_roll + opp_bonus)
    expected = max(0, 5 - margin) if margin > 0 else 5
    assert engine.board.influence["Guatemala"]["USSR"] == expected


def test_realignment_negative_margin_reduces_actors_own_influence():
    # Confirmed rule: losing the realignment roll costs the acting side
    # their own influence in the target country (not just a wasted Op).
    engine = Engine(seed=4)
    engine.board.influence["Guatemala"]["US"] = 5
    engine.board.influence["Guatemala"]["USSR"] = 1  # opponent Influence required to realign
    engine.begin_realignment_operations(Side.US, ops=1)

    target = next(a for a in engine.legal_actions() if a.payload["country"] == "Guatemala")
    engine.step(target)
    actor_roll = engine.pending_decision.options[0].payload["value"]
    engine.step(engine.pending_decision.options[0])
    opp_roll = engine.pending_decision.options[0].payload["value"]

    actor_bonus = engine._realignment_bonus(Side.US, "Guatemala")
    opp_bonus = engine._realignment_bonus(Side.USSR, "Guatemala")
    engine.step(engine.pending_decision.options[0])

    margin = (actor_roll + actor_bonus) - (opp_roll + opp_bonus)
    if margin < 0:
        assert engine.board.influence["Guatemala"]["US"] == max(0, 5 + margin)
    else:
        assert engine.board.influence["Guatemala"]["US"] == 5


def test_realignment_never_adds_actor_influence():
    engine = Engine(seed=9)
    engine.board.influence["Guatemala"]["USSR"] = 1  # opponent Influence required to realign
    engine.begin_realignment_operations(Side.US, ops=1)
    target = next(a for a in engine.legal_actions() if a.payload["country"] == "Guatemala")
    engine.step(target)
    engine.step(engine.pending_decision.options[0])
    engine.step(engine.pending_decision.options[0])
    assert engine.board.influence["Guatemala"]["US"] == 0


def test_realignment_chains_attempts_until_exhausted():
    engine = Engine(seed=2)
    engine.board.influence["Guatemala"]["US"] = 1  # opponent Influence required to realign
    engine.begin_realignment_operations(Side.USSR, ops=2)
    for expected_attempts_left in (2, 1):
        assert engine.pending_decision.kind is DecisionKind.REALIGNMENT_TARGET
        assert engine.pending_decision.context["attempts_remaining"] == expected_attempts_left
        engine.step(engine.pending_decision.options[0])  # target
        engine.step(engine.pending_decision.options[0])  # actor roll
        engine.step(engine.pending_decision.options[0])  # opponent roll
    assert engine.pending_decision is None


def test_realignment_die_roll_ignores_card_ops_value():
    # 6.2.2: Operations only buy attempts (1 per point); they never modify
    # the die roll the way a Coup's Ops value does (6.3.2).
    low_ops = Engine(seed=5)
    low_ops.board.influence["Guatemala"]["USSR"] = 5
    low_ops.begin_realignment_operations(Side.US, ops=1)
    low_ops.step(next(a for a in low_ops.legal_actions() if a.payload["country"] == "Guatemala"))
    low_ops.step(low_ops.pending_decision.options[0])
    low_ops.step(low_ops.pending_decision.options[0])
    low_ops_result = low_ops.board.influence["Guatemala"]["USSR"]

    high_ops = Engine(seed=5)
    high_ops.board.influence["Guatemala"]["USSR"] = 5
    high_ops.begin_realignment_operations(Side.US, ops=4)
    high_ops.step(
        next(a for a in high_ops.legal_actions() if a.payload["country"] == "Guatemala")
    )
    high_ops.step(high_ops.pending_decision.options[0])
    high_ops.step(high_ops.pending_decision.options[0])
    high_ops_first_attempt_result = high_ops.board.influence["Guatemala"]["USSR"]

    # Same seed -> same dice draws for the first attempt regardless of the
    # card's Operations value, since Ops no longer feeds the roll.
    assert low_ops_result == high_ops_first_attempt_result


def test_realignment_requires_opponent_influence_in_target():
    # Rule 6.2.1: a Realignment roll may only be attempted where the
    # opponent holds at least 1 Influence.
    engine = Engine(seed=1)
    assert engine.board.influence["Guatemala"]["USSR"] == 0
    offered = {a.payload["country"] for a in engine._realignment_target_options(Side.US)}
    assert "Guatemala" not in offered

    engine.board.influence["Guatemala"]["USSR"] = 1
    offered = {a.payload["country"] for a in engine._realignment_target_options(Side.US)}
    assert "Guatemala" in offered


def test_realignment_blocked_by_defcon_same_as_coup():
    # Rule 8.1.5 restricts "Coup or Realignment rolls" identically:
    # Europe needs DEFCON 5, Asia needs DEFCON 4, Middle East needs DEFCON 3.
    engine = Engine(seed=1)
    for country in ("France", "Japan", "Egypt", "Guatemala"):
        engine.board.influence[country]["USSR"] = 1

    engine.defcon = 4
    offered = {a.payload["country"] for a in engine._realignment_target_options(Side.US)}
    assert "France" not in offered  # Europe: requires DEFCON 5
    assert "Japan" in offered  # Asia: requires DEFCON 4, satisfied
    assert "Egypt" in offered  # Middle East: requires DEFCON 3, satisfied

    engine.defcon = 3
    offered = {a.payload["country"] for a in engine._realignment_target_options(Side.US)}
    assert "Japan" not in offered  # Asia now below its threshold
    assert "Egypt" in offered

    engine.defcon = 2
    offered = {a.payload["country"] for a in engine._realignment_target_options(Side.US)}
    assert "Egypt" not in offered  # Middle East now below its threshold
    assert "Guatemala" in offered  # Central America stays unrestricted throughout
