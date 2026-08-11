"""Core enums and dataclasses shared across the engine.

These types are the vocabulary the rest of the engine is built on:
Side/Region are fixed facts about the game; DecisionKind/Action/Decision
are the pending-decision-stack primitives mandated by CLAUDE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Side(Enum):
    US = "US"
    USSR = "USSR"
    CHANCE = "CHANCE"

    @property
    def opponent(self) -> "Side":
        if self is Side.US:
            return Side.USSR
        if self is Side.USSR:
            return Side.US
        raise ValueError("CHANCE has no opponent")


class Region(Enum):
    EUROPE = "EUROPE"
    ASIA = "ASIA"
    MIDDLE_EAST = "MIDDLE_EAST"
    AFRICA = "AFRICA"
    CENTRAL_AMERICA = "CENTRAL_AMERICA"
    SOUTH_AMERICA = "SOUTH_AMERICA"


class Subregion(Enum):
    WESTERN_EUROPE = "WESTERN_EUROPE"
    EASTERN_EUROPE = "EASTERN_EUROPE"
    SOUTHEAST_ASIA = "SOUTHEAST_ASIA"


class DecisionKind(Enum):
    PLACE_INFLUENCE = "place_influence"
    COUP_TARGET = "coup_target"
    COUP_ROLL = "coup_roll"
    REALIGNMENT_TARGET = "realignment_target"
    REALIGNMENT_ACTOR_ROLL = "realignment_actor_roll"
    REALIGNMENT_OPPONENT_ROLL = "realignment_opponent_roll"


class ScoringTier(Enum):
    NONE = "none"
    PRESENCE = "presence"
    DOMINATION = "domination"
    CONTROL = "control"


@dataclass(frozen=True)
class Action:
    kind: DecisionKind
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    id: int
    actor: Side
    kind: DecisionKind
    options: tuple[Action, ...]
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """Player-scoped view of the game (mandate #4).

    In M1 there is no hidden information at all (no cards => no hands,
    no deck order to hide), so observe(US) and observe(USSR) are
    identical except for `side`. The shape is deliberately ready for
    M2 to add a `hand` field without changing this contract.
    """

    side: Side
    defcon: int
    vp: int
    turn: int
    action_round: int
    influence: Mapping[str, Mapping[str, int]]
    pending_decision: Decision | None
