"""Engine: coup mechanics, DEFCON interaction, region restrictions."""

from struggler.engine import DecisionKind, Engine, Region, Side


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


def test_every_coup_degrades_defcon_regardless_of_region_or_success():
    for country in ("France", "Guatemala"):  # Europe and non-Europe
        engine = Engine(seed=3)
        engine.board.influence[country]["USSR"] = 1  # opponent Influence required to coup
        before = engine.defcon
        engine.begin_coup(Side.US, ops=5)
        target = next(a for a in engine.legal_actions() if a.payload["country"] == country)
        engine.step(target)
        engine.step(engine.legal_actions()[0])
        assert engine.defcon == before - 1, country


def test_coup_region_restrictions_by_defcon_threshold():
    # Europe needs DEFCON 5, Asia needs DEFCON 4, Middle East needs DEFCON 3.
    engine = Engine(seed=1)
    for country in ("France", "Japan", "Egypt", "Guatemala"):
        engine.board.influence[country]["USSR"] = 1  # opponent Influence required to coup
    engine.defcon = 4
    offered = {a.payload["country"] for a in engine._coup_target_options(Side.US)}
    assert "France" not in offered  # Europe: requires DEFCON 5
    assert "Japan" in offered  # Asia: requires DEFCON 4, satisfied
    assert "Egypt" in offered  # Middle East: requires DEFCON 3, satisfied

    engine.defcon = 3
    offered = {a.payload["country"] for a in engine._coup_target_options(Side.US)}
    assert "Japan" not in offered  # Asia now below its threshold
    assert "Egypt" in offered  # Middle East still satisfied

    engine.defcon = 2
    offered = {a.payload["country"] for a in engine._coup_target_options(Side.US)}
    assert "Egypt" not in offered  # Middle East now below its threshold
    assert "Guatemala" in offered  # Central America stays unrestricted throughout


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


def test_full_control_of_europe_does_not_auto_win():
    # Confirmed rule: controlling all of Europe wins only when the Europe
    # Scoring card is played (out of scope until M2/M3) — it must NOT end
    # the game immediately in M1.
    engine = Engine(seed=1)
    europe = engine.board.countries_in(Region.EUROPE)
    for cid in europe[:-1]:
        engine.board.influence[cid]["US"] = engine.board.countries[cid].stability
    last = europe[-1]
    engine.board.influence[last]["US"] = engine.board.countries[last].stability - 1

    engine.begin_influence_operations(Side.US, 5)
    action = next(a for a in engine.legal_actions() if a.payload["country"] == last)
    engine.step(action)

    assert engine.board.controls_all_of_europe() is Side.US


def test_coup_requires_opponent_influence_in_target():
    # Rule 6.3.1: a Coup may only be attempted where the opponent holds at
    # least 1 Influence.
    engine = Engine(seed=1)
    assert engine.board.influence["Guatemala"]["USSR"] == 0
    offered = {a.payload["country"] for a in engine._coup_target_options(Side.US)}
    assert "Guatemala" not in offered

    engine.board.influence["Guatemala"]["USSR"] = 1
    offered = {a.payload["country"] for a in engine._coup_target_options(Side.US)}
    assert "Guatemala" in offered
    assert not engine.is_terminal
    assert engine.winner is None
