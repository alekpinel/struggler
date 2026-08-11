"""Engine: coup mechanics, DEFCON interaction, region restrictions."""

from struggler.engine import Engine
from struggler.types import DecisionKind, Region, Side


def test_coup_pushes_chance_decision_then_resolves_by_formula():
    engine = Engine(seed=5)
    engine.board.influence["Guatemala"]["USSR"] = 3
    engine.begin_coup(Side.US, ops=3)

    target = next(a for a in engine.legal_actions() if a.payload["country"] == "Guatemala")
    engine.step(target)

    roll_decision = engine.pending_decision
    assert roll_decision.kind is DecisionKind.COUP_ROLL
    assert roll_decision.actor is Side.CHANCE
    assert len(roll_decision.options) == 1  # pre-drawn: only one legal outcome
    roll = roll_decision.options[0].payload["value"]

    engine.step(roll_decision.options[0])
    assert engine.pending_decision is None

    stability = engine.board.countries["Guatemala"].stability
    margin = roll + 3 - 2 * stability
    if margin > 0:
        removed = min(margin, 3)
        assert engine.board.influence["Guatemala"]["USSR"] == 3 - removed
        assert engine.board.influence["Guatemala"]["US"] == margin - removed
    else:
        assert engine.board.influence["Guatemala"] == {"US": 0, "USSR": 3}


def test_coup_in_europe_degrades_defcon_regardless_of_success():
    engine = Engine(seed=3)
    before = engine.defcon
    engine.begin_coup(Side.US, ops=5)
    target = next(a for a in engine.legal_actions() if a.payload["country"] == "France")
    engine.step(target)
    engine.step(engine.legal_actions()[0])
    assert engine.defcon == before - 1


def test_coup_outside_europe_does_not_change_defcon():
    engine = Engine(seed=3)
    before = engine.defcon
    engine.begin_coup(Side.US, ops=5)
    target = next(a for a in engine.legal_actions() if a.payload["country"] == "Guatemala")
    engine.step(target)
    engine.step(engine.legal_actions()[0])
    assert engine.defcon == before


def test_coup_region_restriction_excludes_europe_below_threshold():
    engine = Engine(seed=1)
    engine.defcon = 2
    engine.begin_coup(Side.US, ops=1)
    offered = {a.payload["country"] for a in engine.legal_actions()}
    europe_ids = set(engine.board.countries_in(Region.EUROPE))
    assert not (offered & europe_ids)
    assert "Guatemala" in offered  # Central America stays unrestricted


def test_change_defcon_clamps_and_ends_game_at_defcon_one():
    engine = Engine(seed=1)
    engine.defcon = 2
    engine._change_defcon(-1, caused_by=Side.US)
    assert engine.defcon == 1
    assert engine.is_terminal
    assert engine.winner is Side.USSR  # the side that did NOT cause DEFCON 1 wins


def test_change_defcon_clamps_at_five():
    engine = Engine(seed=1)
    engine.defcon = 5
    engine._change_defcon(1, caused_by=Side.US)
    assert engine.defcon == 5


def test_auto_win_on_full_control_of_europe_stops_further_decisions():
    engine = Engine(seed=1)
    europe = engine.board.countries_in(Region.EUROPE)
    for cid in europe[:-1]:
        engine.board.influence[cid]["US"] = engine.board.countries[cid].stability
    last = europe[-1]
    engine.board.influence[last]["US"] = engine.board.countries[last].stability - 1

    engine.begin_influence_operations(Side.US, 5)
    action = next(a for a in engine.legal_actions() if a.payload["country"] == last)
    engine.step(action)

    assert engine.is_terminal
    assert engine.winner is Side.US
    assert engine.pending_decision is None
