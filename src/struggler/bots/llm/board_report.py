"""Turns a raw `Observation` into the board *reading* a player actually
reasons with, for the LLM prompt (see prompt.py).

The engine hands a player 85 countries as two integers each. Everything
that actually decides a game of Twilight Struggle -- who Controls what,
which Battlegrounds are contested, what a region would score if its card
were played right now, how many Influence points a country still needs --
is a derivation off that raw table, and asking a model to redo those
derivations from a JSON dump on every single decision is where a
reviewed game went wrong: influence spread without ever crossing a
Control threshold, Battlegrounds abandoned the round after they were
taken, and Scoring cards played into regions the bot had never invested
in.

So this module computes them once, deterministically, from the same
public data the player is already entitled to (`Board` + `Observation`),
and renders them as text. Nothing here is a heuristic or a
recommendation -- every number is a fact the rules define and a human
player reads straight off the board.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from struggler.engine import Observation, Side
from struggler.engine.board import Board
from struggler.engine.player import Event
from struggler.engine.rules import RULES
from struggler.engine.types import Region, ScoringTier

# The Scoring card that pays out each region, so a region's line can say
# whether the card for it is in your hand, already spent, or still live in
# the deck. Ground truth is cards.json's `scoring` flag plus the card's
# name; kept explicit here because the region is not a field on the card.
SCORING_CARD_BY_REGION: dict[Region, str] = {
    Region.EUROPE: "Europe_Scoring",
    Region.ASIA: "Asia_Scoring",
    Region.MIDDLE_EAST: "Middle_East_Scoring",
    Region.AFRICA: "Africa_Scoring",
    Region.CENTRAL_AMERICA: "Central_America_Scoring",
    Region.SOUTH_AMERICA: "South_America_Scoring",
}

_TIER_LABEL = {
    ScoringTier.NONE: "none",
    ScoringTier.PRESENCE: "presence",
    ScoringTier.DOMINATION: "domination",
    ScoringTier.CONTROL: "control",
}


def board_from_observation(observation: Observation) -> Board:
    """A `Board` loaded with the observation's influence -- the same trick
    `GreedyPlayer._sync_board` uses. Country metadata and adjacency are
    static public data, so building one costs nothing hidden."""
    board = Board()
    for cid, values in observation.influence.items():
        board.influence[cid]["US"] = values.get("US", 0)
        board.influence[cid]["USSR"] = values.get("USSR", 0)
    return board


def points_to_control(board: Board, side: Side, country: str) -> int:
    """Influence points `side` must add in `country` to Control it (0 if it
    already does). Points, not Ops -- see `Board.influence_cost` for the
    doubling rule that turns points into Ops."""
    info = board.countries[country]
    own = board.influence[country][side.value]
    opp = board.influence[country][side.opponent.value]
    return max(0, info.stability - (own - opp))


def points_to_break(board: Board, side: Side, country: str) -> int:
    """Influence points the opponent must add in `country` to break
    `side`'s Control of it. 0 if `side` doesn't Control it."""
    if board.control(country) is not side:
        return 0
    info = board.countries[country]
    own = board.influence[country][side.value]
    opp = board.influence[country][side.opponent.value]
    return (own - opp) - info.stability + 1


@dataclass(frozen=True)
class RegionStatus:
    region: Region
    net_vp_for_side: int  # signed FOR the acting side: positive = scoring now helps you
    own_tier: ScoringTier
    opp_tier: ScoringTier
    own_countries: tuple[str, ...]
    opp_countries: tuple[str, ...]
    own_bg: tuple[str, ...]
    opp_bg: tuple[str, ...]
    free_bg: tuple[str, ...]  # Battlegrounds nobody Controls


def region_status(board: Board, side: Side, region: Region) -> RegionStatus:
    net_us = board.score_region(region)  # positive favors US
    countries = board.countries_in(region)
    own = tuple(c for c in countries if board.control(c) is side)
    opp = tuple(c for c in countries if board.control(c) is side.opponent)
    bgs = tuple(c for c in countries if board.countries[c].battleground)
    return RegionStatus(
        region=region,
        net_vp_for_side=net_us if side is Side.US else -net_us,
        own_tier=board.region_tier(side, region),
        opp_tier=board.region_tier(side.opponent, region),
        own_countries=own,
        opp_countries=opp,
        own_bg=tuple(c for c in bgs if board.control(c) is side),
        opp_bg=tuple(c for c in bgs if board.control(c) is side.opponent),
        free_bg=tuple(c for c in bgs if board.control(c) is None),
    )


def _scoring_card_state(observation: Observation, region: Region) -> str:
    card = SCORING_CARD_BY_REGION.get(region)
    if card is None:
        return "?"
    if card in observation.hand:
        return "YOU HOLD IT -- it must be played this turn"
    if card in observation.discard_pile:
        return "already played (in discard)"
    return "not in your hand"


def _regional_scoring_text(board: Board, observation: Observation) -> str:
    side = observation.side
    lines = [
        "REGIONAL SCORING STATUS (what each region would pay RIGHT NOW; "
        "'net' is signed for YOU -- positive means playing that Scoring "
        "card now gains you VP, negative means it gains your opponent VP):"
    ]
    for region in Region:
        st = region_status(board, side, region)
        lines.append(
            f"  {region.value}: net {st.net_vp_for_side:+d} VP for you | "
            f"you={_TIER_LABEL[st.own_tier]} ({len(st.own_countries)} countries, "
            f"{len(st.own_bg)} BG) opponent={_TIER_LABEL[st.opp_tier]} "
            f"({len(st.opp_countries)} countries, {len(st.opp_bg)} BG) | "
            f"uncontrolled BG: {', '.join(st.free_bg) or 'none'} | "
            f"scoring card: {_scoring_card_state(observation, region)}"
        )
    return "\n".join(lines)


def _country_line(board: Board, side: Side, cid: str) -> str:
    info = board.countries[cid]
    us = board.influence[cid]["US"]
    ussr = board.influence[cid]["USSR"]
    controller = board.control(cid)
    ctl = {Side.US: "US", Side.USSR: "SU", None: "--"}[controller]
    bg = "BG" if info.battleground else "  "
    if controller is side:
        status = f"brk:+{points_to_break(board, side, cid)}"
    else:
        need = points_to_control(board, side, cid)
        cost = board.influence_cost(side, cid)
        status = f"need:+{need}" + ("(x2 Ops)" if cost == 2 else "")
    reach = "R" if board.is_reachable(side, cid) else "-"
    return f"    {cid:<22}{bg} s{info.stability} US{us}/SU{ussr} ctl:{ctl} {status:<12} {reach}"


def _map_text(board: Board, side: Side) -> str:
    lines = [
        "BOARD BY REGION -- every country, one line each. Legend: BG = "
        "Battleground | s<n> = stability | US<n>/SU<n> = Influence | ctl = who "
        "Controls it (-- = nobody) | need:+n = Influence points YOU must add to "
        "take Control, '(x2 Ops)' when the opponent Controls it and each point "
        "therefore costs 2 Ops | brk:+n = points the OPPONENT must add to break "
        "your Control | R = you may place Influence there right now, - = you "
        "cannot reach it yet:"
    ]
    for region in Region:
        lines.append(f"  {region.value}:")
        for cid in sorted(board.countries_in(region)):
            lines.append(_country_line(board, side, cid))
    return "\n".join(lines)


def battleground_alerts(board: Board, side: Side) -> list[str]:
    """The Battleground bookkeeping a player does out loud every round:
    what you just lost and should retake, what you hold but barely, and
    what is sitting there uncontrolled and cheap."""
    recover: list[str] = []
    at_risk: list[str] = []
    free: list[str] = []
    for cid, info in board.countries.items():
        if not info.battleground:
            continue
        controller = board.control(cid)
        need = points_to_control(board, side, cid)
        cost = need * board.influence_cost(side, cid)
        if controller is side:
            margin = points_to_break(board, side, cid)
            if margin <= 2:
                at_risk.append(f"{cid} (opponent breaks Control with +{margin})")
            continue
        if not board.is_reachable(side, cid):
            continue
        if board.influence[cid][side.value] > 0:
            recover.append(f"{cid} (you hold {board.influence[cid][side.value]}, need +{need} = {cost} Ops)")
        elif controller is None:
            free.append(f"{cid} (empty of Control, need +{need} = {cost} Ops)")

    alerts: list[str] = []
    if recover:
        alerts.append(
            "  RETAKE -- Battlegrounds you already have Influence in but do NOT "
            "Control. Contested Battlegrounds are where VP actually moves; "
            "finishing one of these beats opening a new non-Battleground: "
            + "; ".join(sorted(recover))
        )
    if at_risk:
        alerts.append(
            "  AT RISK -- Battlegrounds you Control by a thin margin. One "
            "opponent Ops play or Coup flips them: " + "; ".join(sorted(at_risk))
        )
    if free:
        alerts.append(
            "  UNCLAIMED -- reachable Battlegrounds nobody Controls: "
            + "; ".join(sorted(free))
        )
    return alerts


def opponent_activity(new_events: Sequence[Event], side: Side) -> list[str]:
    """Which countries the opponent touched since you last acted, netted per
    country. This is the same information the raw event list already
    carries -- restated as 'here is what is being taken from you', because
    in the reviewed game the bot read past three consecutive opponent
    build-ups in Poland and lost the country to Truman Doctrine."""
    touched: dict[str, str] = {}
    for event in new_events:
        if event.actor is not side.opponent or event.country is None:
            continue
        inf = dict(event.country_influence)
        touched[event.country] = (
            f"{event.country} (now US{inf.get('US', 0)}/USSR{inf.get('USSR', 0)}"
            f", control={event.country_control or '-'})"
        )
    return sorted(touched.values())


def military_ops_line(observation: Observation) -> str:
    """The end-of-turn Military Operations bill, stated as VP rather than as
    a track position -- a shortfall is a guaranteed VP payment to the
    opponent, not an abstract requirement."""
    side = observation.side
    own = observation.military_ops.get(side.value, 0)
    required = observation.defcon
    short = max(0, required - own)
    if short == 0:
        return (
            f"MILITARY OPERATIONS: {own}/{required} required -- requirement already met."
        )
    return (
        f"MILITARY OPERATIONS: {own}/{required} required. You are {short} short; "
        f"if the turn ended now that is {short} VP handed to your opponent. "
        "Coups and war Events count, Realignments do not."
    )


def space_race_line(observation: Observation) -> str:
    """How many Space Race attempts are actually left this turn. Without
    this, a plan to 'discard this card to the Space Race' is a coin flip on
    a rule the player cannot see -- the reviewed game committed a card to a
    Space Race play three separate times with no attempt left."""
    side = observation.side
    pos = observation.space_race.get(side.value, 0)
    used = observation.space_race_attempts.get(side.value, 0)
    allowed = 2 if pos >= RULES["space_race_two_attempts_from_box"] else 1
    left = max(0, allowed - used)
    if pos >= RULES["space_race_max_box"]:
        return "SPACE RACE: you are on the last box -- no further attempts possible."
    next_box = RULES["space_race_boxes"][str(pos + 1)]
    first = observation.space_race.get(side.opponent.value, 0) < pos + 1
    vp = next_box["vp_first"] if first else next_box["vp_second"]
    return (
        f"SPACE RACE: box {pos}, attempts left this turn {left}/{allowed}. "
        f"Next box needs a card of {next_box['ops']}+ effective Ops, succeeds on "
        f"a roll of 1-{next_box['roll_max']} ({next_box['roll_max']}/6), and pays "
        f"{vp} VP."
    )


def build_board_report(observation: Observation, new_events: Sequence[Event] = ()) -> str:
    """The full derived reading: headline numbers, per-region scoring
    status, Battleground alerts, opponent activity, and the country table."""
    board = board_from_observation(observation)
    side = observation.side
    vp = observation.vp
    lead = "US" if vp > 0 else ("USSR" if vp < 0 else "level")
    parts = [
        f"STATUS: turn {observation.turn}, action round {observation.action_round}, "
        f"phase {observation.phase}, DEFCON {observation.defcon}, VP {vp:+d} "
        f"({lead} lead; you are {side.value}). Hand {len(observation.hand)} cards, "
        f"opponent holds {observation.opponent_hand_size}, draw pile "
        f"{observation.draw_pile_size}.",
        military_ops_line(observation),
        space_race_line(observation),
        _regional_scoring_text(board, observation),
    ]

    alerts = battleground_alerts(board, side)
    if alerts:
        parts.append("BATTLEGROUND PRIORITIES:\n" + "\n".join(alerts))

    activity = opponent_activity(new_events, side)
    if activity:
        parts.append(
            "OPPONENT ACTIVITY since your last decision -- treat each of these "
            "as a claim on a country you may have to answer now, before it "
            "becomes permanent:\n  " + "\n  ".join(activity)
        )

    effects = []
    if observation.turn_effects:
        effects.append(f"turn effects: {dict(observation.turn_effects)}")
    if observation.game_effects:
        effects.append(f"game effects: {dict(observation.game_effects)}")
    if effects:
        parts.append("ACTIVE EFFECTS -- " + "; ".join(effects))

    parts.append(_map_text(board, side))
    return "\n\n".join(parts)
