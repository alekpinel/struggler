"""M3 card-event mechanics.

This module owns the *event text* of cards — deliberately kept out of the M1/M2
data layer (see CLAUDE.md's card data policy and the M3 milestone). Each event is
a small function that mutates game state through the same primitives the board
mechanics already expose (influence, control, DEFCON, VP, Space Race, the seeded
dice-as-CHANCE decisions), never by reaching around the decision stack.

Design (mandates #1-#2):

- An event `resolve(engine, side)` may complete immediately (pure state change,
  tier 1), enqueue a fresh CHANCE roll as an explicit decision (the "war" family;
  the roll is a logged `WAR_ROLL` decision, never a silent `random` call), or set
  a persistent per-turn modifier the engine consults later (tier 3). Player-choice
  events (tier 2) enqueue their own player decisions and are added incrementally.
- `EVENTS` maps a card id to its `Event`. A card absent from this map has no
  implemented event yet: in events mode it is a no-op discard, exactly as in M2.
  This is what lets M3 grow card-by-card without ever regressing the M2 loop.

Every numeric effect below is taken from the physical card text (GMT Games,
2005 / 2009 Deluxe), not from any reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from struggler.types import Region, Side, Subregion

if TYPE_CHECKING:  # avoid a circular import at module load; engine imports us.
    from struggler.engine import Engine


@dataclass(frozen=True)
class Event:
    """One card's event: how it resolves, and whether it can fire at all.

    `resolve` applies the effect for the given phasing side. `eligible` gates
    whether the event is allowed to happen; it defaults to "always" and exists
    for events (and future rule-modifiers) whose text has a precondition.
    """

    resolve: Callable[["Engine", Side], None]
    eligible: Callable[["Engine", Side], bool] = lambda engine, side: True


EVENTS: dict[str, Event] = {}


def event(
    card_id: str,
    *,
    eligible: Callable[["Engine", Side], bool] | None = None,
) -> Callable[[Callable[["Engine", Side], None]], Callable[["Engine", Side], None]]:
    """Register the decorated function as `card_id`'s event resolver."""

    def register(fn: Callable[["Engine", Side], None]) -> Callable[["Engine", Side], None]:
        EVENTS[card_id] = Event(resolve=fn, eligible=eligible or Event.eligible)
        return fn

    return register


# ---------------------------------------------------------------------------
# Tier 1 — pure/immediate state change
# ---------------------------------------------------------------------------


@event("Duck_and_Cover")
def _duck_and_cover(engine: "Engine", side: Side) -> None:
    # Degrade DEFCON by 1, then the US gains VP equal to 5 minus the new DEFCON.
    engine._change_defcon(-1, caused_by=Side.USSR)
    if not engine.is_terminal:
        engine._award_vp(Side.US, 5 - engine.defcon)


@event("Fidel")
def _fidel(engine: "Engine", side: Side) -> None:
    # Remove all US Influence in Cuba; USSR adds enough for Control.
    engine.gain_control("Cuba", Side.USSR)


@event("Romanian_Abdication")
def _romanian_abdication(engine: "Engine", side: Side) -> None:
    engine.gain_control("Romania", Side.USSR)


@event("Nasser")
def _nasser(engine: "Engine", side: Side) -> None:
    # USSR +2 in Egypt; remove half (rounded up) of the US Influence there.
    us_here = engine.board.influence["Egypt"]["US"]
    engine.remove_influence("Egypt", Side.US, (us_here + 1) // 2)
    engine.add_influence("Egypt", Side.USSR, 2)


@event("De_Gaulle_Leads_France")
def _de_gaulle(engine: "Engine", side: Side) -> None:
    # Remove 2 US Influence in France, add 1 USSR Influence there.
    # (The printed "cancels NATO for France" clause is inert until NATO exists;
    # tracked as remaining M3 work in CLAUDE.md.)
    engine.remove_influence("France", Side.US, 2)
    engine.add_influence("France", Side.USSR, 1)


@event("Captured_Nazi_Scientist")
def _captured_nazi_scientist(engine: "Engine", side: Side) -> None:
    # Advance the phasing player's Space Race marker one box (awarding that
    # box's VP through the same path a successful roll uses).
    engine.advance_space_race_box(side)


@event("Nuclear_Test_Ban")
def _nuclear_test_ban(engine: "Engine", side: Side) -> None:
    # The player gains VP equal to the current DEFCON minus 2, then DEFCON
    # improves two levels.
    engine._award_vp(side, engine.defcon - 2)
    if not engine.is_terminal:
        engine._change_defcon(+2, caused_by=side)


# ---------------------------------------------------------------------------
# Tier 1 — the "war" family (fixed target, resolved by a seeded CHANCE roll)
# ---------------------------------------------------------------------------


@event("Korean_War")
def _korean_war(engine: "Engine", side: Side) -> None:
    # North Korea invades South Korea: USSR war, target South Korea, win on a
    # modified 4-6, +2 VP and seize the target on success, +2 military ops.
    engine.begin_war(
        card_id="Korean_War",
        attacker=Side.USSR,
        target="South_Korea",
        win_from=4,
        vp=2,
        military_ops=2,
        count_target_control=False,
    )


@event("Arab_Israeli_War")
def _arab_israeli_war(engine: "Engine", side: Side) -> None:
    # Pan-Arab coalition attacks Israel: USSR war, target Israel, penalty also
    # counts Israel itself if US-controlled, win on a modified 4-6.
    engine.begin_war(
        card_id="Arab_Israeli_War",
        attacker=Side.USSR,
        target="Israel",
        win_from=4,
        vp=2,
        military_ops=2,
        count_target_control=True,
    )


# ---------------------------------------------------------------------------
# Tier 3 — persistent per-turn modifiers (consulted by the engine, expire at
# end of turn)
# ---------------------------------------------------------------------------


@event("Containment")
def _containment(engine: "Engine", side: Side) -> None:
    # All US Operations are +1 for the remainder of the turn.
    engine.turn_effects["containment"] = True


@event("Brezhnev_Doctrine")
def _brezhnev_doctrine(engine: "Engine", side: Side) -> None:
    # All USSR Operations are +1 for the remainder of the turn.
    engine.turn_effects["brezhnev"] = True


@event("Red_Scare_Purge")
def _red_scare_purge(engine: "Engine", side: Side) -> None:
    # The opponent's Operations are -1 (min 1) for the remainder of the turn.
    engine.turn_effects["red_scare"] = side.opponent.value


# ---------------------------------------------------------------------------
# Tier 2 — player-choice events (the event enqueues its own decisions)
#
# Each of these hands a set of countries to a player and lets them distribute
# the effect, through the engine's generic EVENT_INFLUENCE / EVENT_CHOICE
# steps. Candidate country sets are derived from the live board, not hard-coded
# lists, so they stay correct if the board data changes.
# ---------------------------------------------------------------------------


def _in_subregion(engine: "Engine", subregion: Subregion) -> list[str]:
    return [
        cid for cid, info in engine.board.countries.items() if info.subregion is subregion
    ]


def _in_region(engine: "Engine", region: Region) -> list[str]:
    return [cid for cid, info in engine.board.countries.items() if info.region is region]


@event("COMECON")
def _comecon(engine: "Engine", side: Side) -> None:
    # Add 1 USSR Influence to each of 4 non-US-controlled Eastern Europe
    # countries.
    engine.push_event_influence(
        event="COMECON", op="place", choose_side=Side.USSR, inf_side=Side.USSR,
        remaining=4, candidates=_in_subregion(engine, Subregion.EASTERN_EUROPE),
        cap=1, exclude_controlled_by=Side.US,
    )


@event("Marshall_Plan")
def _marshall_plan(engine: "Engine", side: Side) -> None:
    # Add 1 US Influence to each of 7 non-USSR-controlled Western Europe
    # countries.
    engine.push_event_influence(
        event="Marshall_Plan", op="place", choose_side=Side.US, inf_side=Side.US,
        remaining=7, candidates=_in_subregion(engine, Subregion.WESTERN_EUROPE),
        cap=1, exclude_controlled_by=Side.USSR,
    )


@event("Decolonization")
def _decolonization(engine: "Engine", side: Side) -> None:
    # Add 1 USSR Influence to each of any 4 countries in Africa and/or Southeast
    # Asia.
    candidates = _in_region(engine, Region.AFRICA) + _in_subregion(
        engine, Subregion.SOUTHEAST_ASIA
    )
    engine.push_event_influence(
        event="Decolonization", op="place", choose_side=Side.USSR, inf_side=Side.USSR,
        remaining=4, candidates=candidates, cap=1,
    )


@event("Suez_Crisis")
def _suez_crisis(engine: "Engine", side: Side) -> None:
    # Remove a total of 4 US Influence from France, the UK and Israel, no more
    # than 2 from any one country.
    engine.push_event_influence(
        event="Suez_Crisis", op="remove", choose_side=Side.USSR, inf_side=Side.US,
        remaining=4, candidates=["France", "UK", "Israel"], cap=2,
    )


@event("Truman_Doctrine")
def _truman_doctrine(engine: "Engine", side: Side) -> None:
    # Remove all USSR Influence from a single uncontrolled country in Europe.
    engine.push_event_influence(
        event="Truman_Doctrine", op="remove", choose_side=Side.US, inf_side=Side.USSR,
        remaining=1, candidates=_in_region(engine, Region.EUROPE),
        whole=True, requires_uncontrolled=True,
    )


@event("Warsaw_Pact_Formed")
def _warsaw_pact_formed(engine: "Engine", side: Side) -> None:
    # Either remove all US Influence from 4 Eastern Europe countries, or add 5
    # USSR Influence to Eastern Europe (no more than 2 per country).
    engine.push_event_choice("Warsaw_Pact_Formed", Side.USSR, ("remove", "add"))


def _warsaw_pact_choice(engine: "Engine", side: Side, choice: str) -> None:
    eastern = _in_subregion(engine, Subregion.EASTERN_EUROPE)
    if choice == "remove":
        engine.push_event_influence(
            event="Warsaw_Pact_Formed", op="remove", choose_side=Side.USSR,
            inf_side=Side.US, remaining=4, candidates=eastern, whole=True,
        )
    else:  # "add"
        engine.push_event_influence(
            event="Warsaw_Pact_Formed", op="place", choose_side=Side.USSR,
            inf_side=Side.USSR, remaining=5, candidates=eastern, cap=2,
        )


# Routers for EVENT_CHOICE branches, looked up by the engine at handle time
# (the decision stack stays serializable — only the event id and the chosen
# option are stored, never a function).
CHOICE_ROUTERS: dict[str, Callable[["Engine", Side, str], None]] = {
    "Warsaw_Pact_Formed": _warsaw_pact_choice,
}
