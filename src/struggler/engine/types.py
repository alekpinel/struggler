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


class CardSide(Enum):
    """A card's allegiance (which superpower its event favors).

    Distinct from `Side`: a card can be NEUTRAL (playable by either
    superpower), which has no analogue on `Side`. NEUTRAL is deliberately
    NOT `Side.CHANCE` — conflating "no owning superpower" with "the dice"
    would be a category error.
    """

    US = "US"
    USSR = "USSR"
    NEUTRAL = "NEUTRAL"


class Period(Enum):
    """When a card enters the draw deck over the course of a game."""

    EARLY_WAR = "EARLY_WAR"
    MID_WAR = "MID_WAR"
    LATE_WAR = "LATE_WAR"


class DecisionKind(Enum):
    PLACE_INFLUENCE = "place_influence"
    COUP_TARGET = "coup_target"
    COUP_ROLL = "coup_roll"
    REALIGNMENT_TARGET = "realignment_target"
    REALIGNMENT_ACTOR_ROLL = "realignment_actor_roll"
    REALIGNMENT_OPPONENT_ROLL = "realignment_opponent_roll"
    # -- M2: cards & the full game loop --
    HEADLINE_PLAY = "headline_play"        # pick a card from hand for the headline
    ACTION_ROUND_PLAY = "action_round_play"  # pick which card to play this action round
    PLAY_MODE = "play_mode"                # use the chosen card for ops / event / space race
    OPS_TYPE = "ops_type"                  # spend the ops on influence / coup / realignment
    SPACE_RACE_ROLL = "space_race_roll"    # CHANCE: the space-race attempt die
    # -- M3: card events fire --
    EVENT_OPS_ORDER = "event_ops_order"    # opponent's card played for Ops: event- or ops-first
    EVENT_RESUME = "event_resume"          # forced continuation after the first half resolves
    WAR_ROLL = "war_roll"                  # CHANCE: a "war" event's success die
    WAR_TARGET = "war_target"              # a "war" event where the attacker picks the target
    EVENT_INFLUENCE = "event_influence"    # a player-choice event's targeted place/remove step
    EVENT_CHOICE = "event_choice"          # a player-choice event's branch (pick a sub-option)
    RANDOM_DISCARD = "random_discard"      # CHANCE: a forced random discard from a hand
    CONTEST_ROLL = "contest_roll"          # CHANCE: a two-die "both roll, higher wins" contest
    QUAGMIRE_DISCARD = "quagmire_discard"  # a trapped player discards an Ops card to try to break free
    QUAGMIRE_ROLL = "quagmire_roll"        # CHANCE: the die that may free a trapped player
    HELD_CARD_DISCARD = "held_card_discard"  # Space Race box 6: may discard the Held Card at end of turn


class ScoringTier(Enum):
    NONE = "none"
    PRESENCE = "presence"
    DOMINATION = "domination"
    CONTROL = "control"


@dataclass(frozen=True)
class Card:
    """A single card as data only (mandate: M2 models no event mechanics).

    `side` is the event's allegiance; `ops` is 0 for scoring cards, which
    cannot be played for operations. `in_deck` is False only for The China
    Card, which is tracked separately and never shuffled into the draw pile.
    `event_summary` is None iff the card has no implemented event yet (see
    CLAUDE.md's Card data policy); when present it is a hand-maintained
    paraphrase of `events.py`'s mechanics, not the physical card's text.
    """

    id: str
    number: int
    name: str
    ops: int
    side: CardSide
    period: Period
    scoring: bool
    remove_after_event: bool
    optional: bool
    in_deck: bool
    event_summary: str | None


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

    Hidden information is *absent*, never masked: `hand` is only this
    player's cards, and the opponent's hand appears solely as a count
    (`opponent_hand_size`). The draw pile is a count too — its order and
    the identity of undrawn cards never appear. The discard and removed
    piles are public in Twilight Struggle, so they appear in full.
    `observe(US)` and `observe(USSR)` are therefore genuinely different
    objects, not one object with a redaction flag.

    `military_ops`, `turn_effects`, and `game_effects` are public board
    state (the Military Operations track, and the M3 event modifiers
    currently in force, e.g. NATO or Containment) — every value ever
    stored in them is a fact both players already know once the event
    that set it has resolved, so surfacing them here is not a leak. The
    one exception, `Engine._headline` (each side's secret headline pick
    before both are revealed), is deliberately never exposed and stays
    off this dataclass.
    """

    side: Side
    phase: str
    defcon: int
    vp: int
    turn: int
    action_round: int
    influence: Mapping[str, Mapping[str, int]]
    pending_decision: Decision | None
    hand: tuple[str, ...]
    opponent_hand_size: int
    draw_pile_size: int
    discard_pile: tuple[str, ...]
    removed_cards: tuple[str, ...]
    china_card_owner: Side
    china_card_available: bool
    space_race: Mapping[str, int]
    military_ops: Mapping[str, int]
    turn_effects: Mapping[str, Any]
    game_effects: Mapping[str, Any]
