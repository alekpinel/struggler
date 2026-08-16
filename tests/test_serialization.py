"""Flat, JSON-native serialize()/deserialize() (mandate #5)."""

import json

from struggler.engine import Action, DecisionKind, Engine, Side


def test_serialize_is_json_native():
    engine = Engine(seed=1)
    engine.board.influence["Poland"]["USSR"] = 3  # a coup needs opponent Influence there
    engine.begin_coup(Side.US, 2)
    engine.step(engine.legal_actions()[0])
    data = engine.serialize()
    json.dumps(data)  # must not raise: no custom encoder needed


def test_round_trip_preserves_full_state_including_rng():
    engine = Engine(seed=123)
    engine.begin_influence_operations(Side.USSR, 4)
    engine.step(engine.legal_actions()[0])

    data = engine.serialize()
    restored = Engine.deserialize(data)
    assert restored.serialize() == data

    # Continuing play from the restored engine must match continuing the
    # original exactly, including any future dice draws (RNG state carried).
    a1 = engine.legal_actions()[0]
    a2 = restored.legal_actions()[0]
    assert a1 == a2
    engine.step(a1)
    restored.step(a2)
    assert engine.serialize() == restored.serialize()


def test_ops_round_snapshot_survives_a_round_trip_mid_placement():
    # The start-of-Action-Round snapshot (rule 6.1.1) must itself round-trip,
    # or a deserialized-and-resumed game would re-freeze from the *current*
    # (already-mutated) board instead of the real start state, silently
    # reopening the chaining bug across a save/load boundary.
    engine = Engine(seed=1)
    engine.begin_influence_operations(Side.USSR, 2)
    engine.step(Action(DecisionKind.PLACE_INFLUENCE, {"country": "Finland"}))
    assert engine._ops_round_snapshot is not None

    restored = Engine.deserialize(engine.serialize())
    assert restored._ops_round_snapshot == engine._ops_round_snapshot
    # Sweden only became reachable via the point just placed in Finland; the
    # restored engine must still refuse it, exactly like the original.
    assert all(
        a.payload["country"] != "Sweden" for a in restored.pending_decision.options
    )


def test_round_trip_through_a_chance_decision_preserves_rng_state():
    engine = Engine(seed=7)
    engine.board.influence["Poland"]["USSR"] = 3  # opponent Influence for the US coup below
    engine.board.influence["Guatemala"]["US"] = 3  # opponent Influence for the USSR coup below
    engine.begin_coup(Side.US, ops=2)
    engine.step(engine.legal_actions()[0])  # target -> pushes COUP_ROLL (draws from RNG)

    restored = Engine.deserialize(engine.serialize())
    assert restored.pending_decision == engine.pending_decision

    # Advance both past this game and start a fresh coup: RNG continuation
    # must match, proving the RNG's internal state (not just its seed) round-trips.
    engine.step(engine.legal_actions()[0])
    restored.step(restored.legal_actions()[0])
    engine.begin_coup(Side.USSR, ops=2)
    restored.begin_coup(Side.USSR, ops=2)
    engine.step(engine.legal_actions()[0])
    restored.step(restored.legal_actions()[0])
    assert engine.serialize() == restored.serialize()
