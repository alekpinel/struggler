"""Flat, JSON-native serialize()/deserialize() (mandate #5)."""

import json

from struggler.engine import Engine
from struggler.types import Side


def test_serialize_is_json_native():
    engine = Engine(seed=1)
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


def test_round_trip_through_a_chance_decision_preserves_rng_state():
    engine = Engine(seed=7)
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
