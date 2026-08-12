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
    # Remove 2 US Influence in France, add 1 USSR Influence there, and cancel
    # NATO's protection for France (persistent, see _usable_coup_realign_target).
    engine.remove_influence("France", Side.US, 2)
    engine.add_influence("France", Side.USSR, 1)
    engine.game_effects["degaulle_france"] = True


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


@event(
    "Arab_Israeli_War",
    eligible=lambda engine, side: not engine.game_effects.get("camp_david"),
)
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
    # countries. Also a precondition for NATO.
    engine.game_effects["marshall_or_warsaw"] = True
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
    # USSR Influence to Eastern Europe (no more than 2 per country). Also a
    # precondition for NATO.
    engine.game_effects["marshall_or_warsaw"] = True
    engine.push_event_choice("Warsaw_Pact_Formed", Side.USSR, ("remove", "add"))


def _warsaw_pact_choice(engine: "Engine", side: Side, choice: str, context: dict) -> None:
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


# ---------------------------------------------------------------------------
# Tier 3 — persistent game-long effects (change future legality)
#
# These set flags in engine.game_effects (never cleared at end of turn); the
# engine consults them in _usable_coup_realign_target when enumerating USSR
# coup/realignment targets.
# ---------------------------------------------------------------------------


def _marshall_or_warsaw_played(engine: "Engine", side: Side) -> bool:
    return bool(engine.game_effects.get("marshall_or_warsaw"))


@event("NATO", eligible=_marshall_or_warsaw_played)
def _nato(engine: "Engine", side: Side) -> None:
    # The USSR may no longer Coup or make Realignment rolls against any
    # US-controlled country in Europe. Playable only after Marshall Plan or
    # Warsaw Pact Formed (enforced by the eligible predicate above).
    engine.game_effects["nato"] = True


@event("US_Japan_Mutual_Defense_Pact")
def _us_japan_pact(engine: "Engine", side: Side) -> None:
    # US gains enough Influence to Control Japan; the USSR may never Coup or
    # make Realignment rolls against Japan for the rest of the game.
    engine.gain_control("Japan", Side.US)
    engine.game_effects["us_japan_pact"] = True


@event("Willy_Brandt")
def _willy_brandt(engine: "Engine", side: Side) -> None:
    # USSR gains 1 VP and 1 Influence in West Germany, and NATO no longer
    # protects West Germany (persistent).
    engine._award_vp(Side.USSR, 1)
    if not engine.is_terminal:
        engine.add_influence("West_Germany", Side.USSR, 1)
        engine.game_effects["willy_brandt"] = True


# ---------------------------------------------------------------------------
# More cards, grouped by the primitive they reuse. Numeric effects are from the
# physical card text. Cards whose text needs a subsystem we do not model yet
# (random discard from a hidden hand, revealing/taking hand cards, per-turn
# regional Ops bonuses, taking cards from the discard pile, DEFCON-status
# restrictions) are intentionally left unregistered — see CLAUDE.md.
# ---------------------------------------------------------------------------

# -- immediate, fixed board/VP/DEFCON/space effects -------------------------


@event("Allende")
def _allende(engine: "Engine", side: Side) -> None:
    engine.add_influence("Chile", Side.USSR, 2)


@event("Portuguese_Empire_Crumbles")
def _portuguese_empire(engine: "Engine", side: Side) -> None:
    engine.add_influence("Angola", Side.USSR, 2)
    engine.add_influence("SE_African_States", Side.USSR, 2)


@event("Panama_Canal_Returned")
def _panama_canal(engine: "Engine", side: Side) -> None:
    for cid in ("Panama", "Costa_Rica", "Venezuela"):
        engine.add_influence(cid, Side.US, 1)


@event("Sadat_Expels_Soviets")
def _sadat(engine: "Engine", side: Side) -> None:
    engine.remove_all_influence("Egypt", Side.USSR)
    engine.add_influence("Egypt", Side.US, 1)


@event("John_Paul_II_Elected_Pope")
def _john_paul(engine: "Engine", side: Side) -> None:
    engine.remove_influence("Poland", Side.USSR, 2)
    engine.add_influence("Poland", Side.US, 1)
    engine.game_effects["john_paul"] = True  # precondition for Solidarity


@event("Camp_David_Accords")
def _camp_david(engine: "Engine", side: Side) -> None:
    engine._award_vp(Side.US, 1)
    for cid in ("Israel", "Jordan", "Egypt"):
        engine.add_influence(cid, Side.US, 1)
    engine.game_effects["camp_david"] = True  # blocks Arab-Israeli War


@event("Iranian_Hostage_Crisis")
def _iranian_hostage(engine: "Engine", side: Side) -> None:
    engine.remove_all_influence("Iran", Side.US)
    engine.add_influence("Iran", Side.USSR, 2)
    engine.game_effects["iranian_hostage"] = True  # makes Terrorism hit the US twice


@event("The_Iron_Lady")
def _iron_lady(engine: "Engine", side: Side) -> None:
    engine._award_vp(Side.US, 1)
    engine.add_influence("Argentina", Side.USSR, 1)
    engine.remove_all_influence("UK", Side.USSR)
    engine.game_effects["iron_lady"] = True  # blocks Socialist Governments


@event("An_Evil_Empire")
def _evil_empire(engine: "Engine", side: Side) -> None:
    engine._award_vp(Side.US, 1)
    engine.game_effects.pop("flower_power", None)  # cancels Flower Power
    engine.game_effects["evil_empire"] = True


@event("U2_Incident")
def _u2_incident(engine: "Engine", side: Side) -> None:
    # (The extra VP if UN Intervention is later played this turn is not modeled.)
    engine._award_vp(Side.USSR, 1)


@event("Cultural_Revolution")
def _cultural_revolution(engine: "Engine", side: Side) -> None:
    if engine.china_card_owner == "US":
        engine.china_card_owner = "USSR"
        engine.china_card_available = True  # taken face up
    else:
        engine._award_vp(Side.USSR, 1)


@event("Ortega_Elected_in_Nicaragua")
def _ortega(engine: "Engine", side: Side) -> None:
    # (The optional free coup against a Nicaragua-adjacent country is not
    # modeled.)
    engine.remove_all_influence("Nicaragua", Side.US)


@event("Tear_Down_This_Wall")
def _tear_down_wall(engine: "Engine", side: Side) -> None:
    # (The optional US Operations/coup in Europe is not modeled.)
    engine.game_effects.pop("willy_brandt", None)  # cancels Willy Brandt
    engine.add_influence("East_Germany", Side.US, 3)


@event("Kitchen_Debates")
def _kitchen_debates(engine: "Engine", side: Side) -> None:
    us_bg = _controlled_battlegrounds(engine, Side.US)
    ussr_bg = _controlled_battlegrounds(engine, Side.USSR)
    if us_bg > ussr_bg:
        engine._award_vp(Side.US, 2)


@event(
    "OPEC",
    eligible=lambda engine, side: not engine.game_effects.get("north_sea_oil"),
)
def _opec(engine: "Engine", side: Side) -> None:
    fields = ["Egypt", "Iran", "Libya", "Saudi_Arabia", "Iraq", "Gulf_States",
              "Venezuela", "Nigeria"]
    vp = sum(1 for cid in fields if engine.board.control(cid) is Side.USSR)
    engine._award_vp(Side.USSR, vp)


@event("Alliance_for_Progress")
def _alliance_for_progress(engine: "Engine", side: Side) -> None:
    regions = (Region.CENTRAL_AMERICA, Region.SOUTH_AMERICA)
    vp = sum(
        1
        for cid, info in engine.board.countries.items()
        if info.battleground
        and info.region in regions
        and engine.board.control(cid) is Side.US
    )
    engine._award_vp(Side.US, vp)


@event("Reagan_Bombs_Libya")
def _reagan_bombs_libya(engine: "Engine", side: Side) -> None:
    engine._award_vp(Side.US, engine.board.influence["Libya"]["USSR"] // 2)


@event("One_Small_Step")
def _one_small_step(engine: "Engine", side: Side) -> None:
    # If you are behind on the Space Race, jump two boxes.
    if engine.space_race[side.value] < engine.space_race[side.opponent.value]:
        engine.advance_space_race_box(side)
        engine.advance_space_race_box(side)


@event("AWACS_Sale_to_Saudis")
def _awacs(engine: "Engine", side: Side) -> None:
    engine.add_influence("Saudi_Arabia", Side.US, 2)
    engine.game_effects["awacs"] = True  # blocks Muslim Revolution


# -- events that conduct Operations -----------------------------------------


@event("CIA_Created")
def _cia_created(engine: "Engine", side: Side) -> None:
    # (The "USSR reveals hand" is information only; the mechanical effect is the
    # 1 Op of US Operations.)
    engine.push_event_operations(Side.US, 1)


@event("Lone_Gunman")
def _lone_gunman(engine: "Engine", side: Side) -> None:
    engine.push_event_operations(Side.USSR, 1)


@event("ABM_Treaty")
def _abm_treaty(engine: "Engine", side: Side) -> None:
    engine._change_defcon(+1, caused_by=side)
    if not engine.is_terminal:
        engine.push_event_operations(side, 4)


# -- player-choice influence (EVENT_INFLUENCE) ------------------------------


@event(
    "Socialist_Governments",
    eligible=lambda engine, side: not engine.game_effects.get("iron_lady"),
)
def _socialist_governments(engine: "Engine", side: Side) -> None:
    engine.push_event_influence(
        event="Socialist_Governments", op="remove", choose_side=Side.USSR,
        inf_side=Side.US, remaining=3,
        candidates=_in_subregion(engine, Subregion.WESTERN_EUROPE), cap=2,
    )


@event("Muslim_Revolution")
def _muslim_revolution(engine: "Engine", side: Side) -> None:
    countries = ["Sudan", "Iran", "Iraq", "Egypt", "Libya", "Saudi_Arabia",
                 "Syria", "Jordan"]
    engine.push_event_influence(
        event="Muslim_Revolution", op="remove", choose_side=Side.USSR,
        inf_side=Side.US, remaining=2, candidates=countries, whole=True,
    )


@event("Colonial_Rear_Guards")
def _colonial_rear_guards(engine: "Engine", side: Side) -> None:
    candidates = _in_region(engine, Region.AFRICA) + _in_subregion(
        engine, Subregion.SOUTHEAST_ASIA
    )
    engine.push_event_influence(
        event="Colonial_Rear_Guards", op="place", choose_side=Side.US,
        inf_side=Side.US, remaining=4, candidates=candidates, cap=1,
    )


@event("Liberation_Theology")
def _liberation_theology(engine: "Engine", side: Side) -> None:
    engine.push_event_influence(
        event="Liberation_Theology", op="place", choose_side=Side.USSR,
        inf_side=Side.USSR, remaining=3,
        candidates=_in_region(engine, Region.CENTRAL_AMERICA), cap=2,
    )


@event("The_Voice_Of_America")
def _voice_of_america(engine: "Engine", side: Side) -> None:
    candidates = [
        cid for cid, info in engine.board.countries.items()
        if info.region is not Region.EUROPE
    ]
    engine.push_event_influence(
        event="The_Voice_Of_America", op="remove", choose_side=Side.US,
        inf_side=Side.USSR, remaining=4, candidates=candidates, cap=2,
    )


@event("Puppet_Governments")
def _puppet_governments(engine: "Engine", side: Side) -> None:
    empty = [
        cid for cid in engine.board.countries
        if engine.board.influence[cid]["US"] == 0
        and engine.board.influence[cid]["USSR"] == 0
    ]
    engine.push_event_influence(
        event="Puppet_Governments", op="place", choose_side=Side.US,
        inf_side=Side.US, remaining=3, candidates=empty, cap=1,
    )


@event("OAS_Founded")
def _oas_founded(engine: "Engine", side: Side) -> None:
    candidates = _in_region(engine, Region.CENTRAL_AMERICA) + _in_region(
        engine, Region.SOUTH_AMERICA
    )
    engine.push_event_influence(
        event="OAS_Founded", op="place", choose_side=Side.US, inf_side=Side.US,
        remaining=2, candidates=candidates,
    )


@event("Pershing_II_Deployed")
def _pershing_ii(engine: "Engine", side: Side) -> None:
    engine._award_vp(Side.USSR, 1)
    engine.push_event_influence(
        event="Pershing_II_Deployed", op="remove", choose_side=Side.USSR,
        inf_side=Side.US, remaining=3,
        candidates=_in_subregion(engine, Subregion.WESTERN_EUROPE), cap=1,
    )


@event("The_Reformer")
def _the_reformer(engine: "Engine", side: Side) -> None:
    # 6 Influence if the USSR is ahead on VP (US-positive track, so vp < 0),
    # else 4; the USSR may no longer coup in Europe.
    amount = 6 if engine.vp < 0 else 4
    engine.game_effects["reformer"] = True
    engine.push_event_influence(
        event="The_Reformer", op="place", choose_side=Side.USSR,
        inf_side=Side.USSR, remaining=amount,
        candidates=_in_region(engine, Region.EUROPE), cap=2,
    )


@event(
    "Solidarity",
    eligible=lambda engine, side: bool(engine.game_effects.get("john_paul")),
)
def _solidarity(engine: "Engine", side: Side) -> None:
    engine.add_influence("Poland", Side.US, 3)


# -- an immediate effect that then hands off a player-choice removal ---------


@event("Marine_Barracks_Bombing")
def _marine_barracks(engine: "Engine", side: Side) -> None:
    engine.remove_all_influence("Lebanon", Side.US)
    middle_east = [
        cid for cid, info in engine.board.countries.items()
        if info.region is Region.MIDDLE_EAST and cid != "Lebanon"
    ]
    engine.push_event_influence(
        event="Marine_Barracks_Bombing", op="remove", choose_side=Side.USSR,
        inf_side=Side.US, remaining=2, candidates=middle_east, whole=True,
    )


# -- wars where the attacker picks the target -------------------------------


@event("Indo_Pakistani_War")
def _indo_pakistani_war(engine: "Engine", side: Side) -> None:
    engine.push_war_target_choice(
        card_id="Indo_Pakistani_War", attacker=side,
        candidates=["India", "Pakistan"], win_from=4, vp=2, military_ops=2,
    )


@event("Iran_Iraq_War")
def _iran_iraq_war(engine: "Engine", side: Side) -> None:
    engine.push_war_target_choice(
        card_id="Iran_Iraq_War", attacker=side,
        candidates=["Iran", "Iraq"], win_from=4, vp=2, military_ops=2,
    )


@event("Brush_War")
def _brush_war(engine: "Engine", side: Side) -> None:
    # Attack any country with stability 1 or 2; success on a modified 3-6.
    candidates = [
        cid for cid, info in engine.board.countries.items() if info.stability <= 2
    ]
    engine.push_war_target_choice(
        card_id="Brush_War", attacker=side, candidates=candidates,
        win_from=3, vp=1, military_ops=3,
    )


# -- match-influence branch (Independent Reds) ------------------------------


@event("Independent_Reds")
def _independent_reds(engine: "Engine", side: Side) -> None:
    # Add US Influence in one of these to equal the USSR Influence there.
    engine.push_event_choice(
        "Independent_Reds", Side.US,
        ("Yugoslavia", "Romania", "Bulgaria", "Hungary", "Czechoslovakia"),
    )


def _independent_reds_choice(engine: "Engine", side: Side, choice: str, context: dict) -> None:
    ussr = engine.board.influence[choice]["USSR"]
    if engine.board.influence[choice]["US"] < ussr:
        engine.board.influence[choice]["US"] = ussr


# -- forced-random-discard events (a seeded CHANCE decision) ----------------


@event("Five_Year_Plan")
def _five_year_plan(engine: "Engine", side: Side) -> None:
    # The USSR randomly discards a card; if it is a USSR event, that event fires.
    engine.push_random_discard(Side.USSR, "five_year_plan")


@event("Terrorism")
def _terrorism(engine: "Engine", side: Side) -> None:
    # The opponent randomly discards a card (two if the USSR plays it after the
    # Iranian Hostage Crisis).
    count = 2 if (side is Side.USSR and engine.game_effects.get("iranian_hostage")) else 1
    engine.push_random_discard(side.opponent, "terrorism", count)


# -- per-turn coup modifiers (turn_effects, cleared at end of turn) ----------


@event("Nuclear_Subs")
def _nuclear_subs(engine: "Engine", side: Side) -> None:
    # US coups in Battleground countries do not degrade DEFCON this turn.
    engine.turn_effects["nuclear_subs"] = True


@event("Vietnam_Revolts")
def _vietnam_revolts(engine: "Engine", side: Side) -> None:
    # Add 2 USSR Influence to Vietnam; for the rest of the turn the USSR gets
    # +1 Op on any play whose Ops are all used in Southeast Asia (handled by the
    # region-bonus machinery via turn_effects).
    engine.add_influence("Vietnam", Side.USSR, 2)
    engine.turn_effects["vietnam_revolts"] = True


@event("Latin_American_Death_Squads")
def _latin_american_death_squads(engine: "Engine", side: Side) -> None:
    # +1 to the player's coup rolls (and -1 to the opponent's) in Central and
    # South America for the rest of the turn.
    engine.turn_effects["la_death_squads"] = side.value


@event("Iran_Contra_Scandal")
def _iran_contra(engine: "Engine", side: Side) -> None:
    # US Realignment rolls are -1 for the remainder of the turn.
    engine.turn_effects["iran_contra"] = True


@event("Chernobyl")
def _chernobyl(engine: "Engine", side: Side) -> None:
    # The US designates a region; the USSR may not add Influence there via
    # Operations for the rest of the turn.
    engine.push_event_choice(
        "Chernobyl", side,
        ("EUROPE", "ASIA", "MIDDLE_EAST", "AFRICA", "CENTRAL_AMERICA", "SOUTH_AMERICA"),
    )


def _chernobyl_choice(engine: "Engine", side: Side, choice: str, context: dict) -> None:
    engine.turn_effects["chernobyl"] = choice


# -- persistent game-long triggers -------------------------------------------


@event("Flower_Power")
def _flower_power(engine: "Engine", side: Side) -> None:
    # The USSR scores 2 VP each time the US plays a war card, until An Evil
    # Empire is played (checked in the engine at play time).
    if not engine.game_effects.get("evil_empire"):
        engine.game_effects["flower_power"] = True


@event("Yuri_and_Samantha")
def _yuri_and_samantha(engine: "Engine", side: Side) -> None:
    # The USSR scores 1 VP for every US coup attempt for the rest of the game.
    engine.game_effects["yuri_samantha"] = True


# -- set-DEFCON branch -------------------------------------------------------


@event("How_I_Learned_to_Stop_Worrying")
def _how_i_learned(engine: "Engine", side: Side) -> None:
    # Set DEFCON to any level, then add 5 to the Military Operations track.
    engine.push_event_choice(
        "How_I_Learned_to_Stop_Worrying", side, ("1", "2", "3", "4", "5")
    )


def _how_i_learned_choice(engine: "Engine", side: Side, choice: str, context: dict) -> None:
    engine.set_defcon(int(choice), caused_by=side)
    if not engine.is_terminal:
        engine.military_ops[side.value] += 5


# -- influence then an optional free operation (Junta) ----------------------


def _central_and_south_america(engine: "Engine") -> list[str]:
    return [
        cid for cid, info in engine.board.countries.items()
        if info.region in (Region.CENTRAL_AMERICA, Region.SOUTH_AMERICA)
    ]


@event("Junta")
def _junta(engine: "Engine", side: Side) -> None:
    # Place 2 Influence in Central/South America, then optionally make a free
    # Coup or Realignment there. The free-op choice is pushed first so it
    # resolves *after* the placement drains. (The card says a single country;
    # here the 2 points may be split within the region — a minor deviation.)
    americas = _central_and_south_america(engine)
    engine.push_free_coup_or_realign(side, "Junta", ops=2, countries=americas)
    engine.push_event_influence(
        event="Junta", op="place", choose_side=side, inf_side=side,
        remaining=2, candidates=americas, cap=2,
    )


def _junta_choice(engine: "Engine", side: Side, choice: str, context: dict) -> None:
    engine.resolve_free_op_choice(side, choice, 2, _central_and_south_america(engine))


# -- reclaim-from-discard -----------------------------------------------------


@event("Salt_Negotiations")
def _salt_negotiations(engine: "Engine", side: Side) -> None:
    # Improve DEFCON two levels; both sides get -1 to coups for the rest of the
    # turn; the player may reclaim one non-scoring card from the discard pile.
    engine._change_defcon(+2, caused_by=side)
    if engine.is_terminal:
        return
    engine.turn_effects["salt"] = True
    engine.push_take_from_discard(side, "Salt_Negotiations")


def _salt_reclaim_choice(engine: "Engine", side: Side, choice: str, context: dict) -> None:
    if choice != "none" and choice in engine.discard_pile:
        engine.discard_pile.remove(choice)
        engine.hands[side.value].append(choice)


# -- dice-contest / branch events -------------------------------------------


@event("Olympic_Games")
def _olympic_games(engine: "Engine", side: Side) -> None:
    # The sponsor is the phasing player; the opponent chooses to participate
    # (a die contest, sponsor +2, winner +2 VP) or boycott (DEFCON -1 and the
    # sponsor conducts 4 Ops).
    engine.push_event_choice("Olympic_Games", side.opponent, ("participate", "boycott"))


def _olympic_choice(engine: "Engine", chooser: Side, choice: str, context: dict) -> None:
    sponsor = chooser.opponent
    if choice == "participate":
        engine.push_dice_contest("Olympic_Games", sponsor, sponsor_mod=2, defender_mod=0, vp=2)
    else:  # boycott
        engine._change_defcon(-1, caused_by=sponsor)
        if not engine.is_terminal:
            engine.push_event_operations(sponsor, 4)


@event("Summit")
def _summit(engine: "Engine", side: Side) -> None:
    # Both roll, +1 per region Dominated/Controlled; winner +2 VP and may raise
    # or lower DEFCON one level.
    engine.push_dice_contest(
        "Summit", side,
        sponsor_mod=engine._regions_dominated(side),
        defender_mod=engine._regions_dominated(side.opponent),
        vp=2,
    )


def _summit_result(engine: "Engine", sponsor: Side, winner: Side) -> None:
    engine.push_event_choice("Summit_defcon", winner, ("raise", "lower", "none"))


def _summit_defcon_choice(engine: "Engine", side: Side, choice: str, context: dict) -> None:
    if choice == "raise":
        engine._change_defcon(+1, caused_by=side)
    elif choice == "lower":
        engine._change_defcon(-1, caused_by=side)


@event("Wargames", eligible=lambda engine, side: engine.defcon <= 2)
def _wargames(engine: "Engine", side: Side) -> None:
    # Only at DEFCON 2: the player may give the opponent 6 VP and end the game
    # (final scoring), or decline.
    engine.push_event_choice("Wargames", side, ("end_game", "decline"))


def _wargames_choice(engine: "Engine", side: Side, choice: str, context: dict) -> None:
    if choice == "end_game":
        engine._award_vp(side.opponent, 6)
        if not engine.is_terminal:
            engine._finish_game()


# Per-event follow-ups after a dice contest resolves (see push_dice_contest).
CONTEST_RESOLVERS: dict[str, Callable[["Engine", Side, Side], None]] = {
    "Summit": _summit_result,
}


# -- revealing / taking cards from the opponent's hand ----------------------
#
# These cards let the phasing player see (part of) the opponent's hand — a
# reveal the card text sanctions, so surfacing the involved cards as decision
# options is correct, not a leak: the only other observer is the hand's owner,
# who already knows it.


@event("Aldrich_Ames_Remix")
def _aldrich_ames(engine: "Engine", side: Side) -> None:
    # The USSR sees the US hand and chooses one card the US must discard. (The
    # remix's ongoing "sees the hand for the turn" reveal is not modeled.)
    us_hand = engine.hands["US"]
    if not us_hand:
        return
    engine.push_event_choice("Aldrich_Ames_Remix", Side.USSR, tuple(us_hand))


def _aldrich_ames_choice(engine: "Engine", side: Side, choice: str, context: dict) -> None:
    engine._file_card(Side.US, choice, fired=False)  # the US discards the chosen card


@event("Grain_Sales_to_Soviets")
def _grain_sales(engine: "Engine", side: Side) -> None:
    # Randomly reveal one USSR card to the US, who then takes it or returns it.
    engine.push_random_discard(Side.USSR, "grain_sales")


def _grain_sales_choice(engine: "Engine", side: Side, choice: str, context: dict) -> None:
    card = context["card"]
    if choice == "take":
        # The US takes the card, uses its Ops, then discards it.
        if card in engine.hands["USSR"]:
            engine.hands["USSR"].remove(card)
        engine.discard_pile.append(card)
        engine.push_event_operations(Side.US, engine.cards[card].ops)
    else:  # return: the card stays with the USSR; the US uses Grain Sales' 2 Ops
        engine.push_event_operations(Side.US, 2)


@event("Ask_Not_What_Your_Country_Can_Do_For_You")
def _ask_not(engine: "Engine", side: Side) -> None:
    # The player may discard any number of cards from hand and draw that many
    # replacements.
    _push_ask_not(engine, side, 0)


def _push_ask_not(engine: "Engine", side: Side, discarded: int) -> None:
    hand = engine.hands[side.value]
    choices = tuple(cid for cid in hand if not engine.cards[cid].scoring) + ("stop",)
    if len(choices) == 1:  # nothing left to discard -> draw and finish
        engine.draw_cards_to_hand(side, discarded)
        return
    engine.push_event_choice(
        "Ask_Not_What_Your_Country_Can_Do_For_You", side, choices,
        extra={"discarded": discarded},
    )


def _ask_not_choice(engine: "Engine", side: Side, choice: str, context: dict) -> None:
    if choice == "stop":
        engine.draw_cards_to_hand(side, context["discarded"])
    else:
        engine._file_card(side, choice, fired=False)
        _push_ask_not(engine, side, context["discarded"] + 1)


def _scoring_card_countries(engine: "Engine", scoring_id: str) -> list[str]:
    from struggler.engine import SCORING_CARD_REGION

    if scoring_id == "Southeast_Asia_Scoring":
        return _in_subregion(engine, Subregion.SOUTHEAST_ASIA)
    region = SCORING_CARD_REGION.get(scoring_id)
    if region is None:
        return []
    return _in_region(engine, region)


@event("The_Cambridge_Five")
def _cambridge_five(engine: "Engine", side: Side) -> None:
    # The US reveals its scoring cards; the USSR adds 1 Influence to a country
    # in one of those regions.
    candidates: list[str] = []
    for cid in engine.hands["US"]:
        if engine.cards[cid].scoring:
            candidates += _scoring_card_countries(engine, cid)
    candidates = list(dict.fromkeys(candidates))  # dedupe, keep order
    if not candidates:
        return
    engine.push_event_influence(
        event="The_Cambridge_Five", op="place", choose_side=Side.USSR,
        inf_side=Side.USSR, remaining=1, candidates=candidates,
    )


# ---------------------------------------------------------------------------
# M3 tail cards — the most idiosyncratic events, each reusing/extending the
# primitives above. Documented simplifications are noted per card and mirrored
# in CLAUDE.md's "Known limitations" list.
# ---------------------------------------------------------------------------

# -- Missile Envy: take the opponent's top-Ops card, then use it -------------


@event("Missile_Envy")
def _missile_envy(engine: "Engine", side: Side) -> None:
    # Exchange Missile Envy for the highest-Ops card in the opponent's hand
    # (opponent chooses among ties); Missile Envy passes to the opponent. The
    # taker then uses the card for Ops, or its Event when allowed. (The card's
    # "opponent must play Missile Envy next action round" rider is not modeled —
    # the opponent simply gains it in hand.)
    opp = side.opponent
    hand = engine.hands[opp.value]
    if not hand:
        return  # nothing to exchange: a no-op discard (Missile Envy stays filed)
    max_ops = max(engine.cards[c].ops for c in hand)
    candidates = [c for c in hand if engine.cards[c].ops == max_ops]
    # Missile Envy was just filed to the discard pile by the play; move it to the
    # opponent's hand instead (computed the candidates first, so it never counts).
    if "Missile_Envy" in engine.discard_pile:
        engine.discard_pile.remove("Missile_Envy")
    engine.hands[opp.value].append("Missile_Envy")
    if len(candidates) == 1:
        engine.missile_envy_take(side, candidates[0])
    else:
        engine.push_event_choice("Missile_Envy_pick", opp, tuple(candidates))


def _missile_envy_pick_choice(engine: "Engine", giver: Side, choice: str, context: dict) -> None:
    engine.missile_envy_take(giver.opponent, choice)


def _missile_envy_use_choice(engine: "Engine", taker: Side, choice: str, context: dict) -> None:
    engine.missile_envy_use(taker, context["card"], choice)


# -- Star Wars: take a card from the discard pile and play it immediately -----


@event(
    "Star_Wars",
    eligible=lambda engine, side: engine.space_race["US"] > engine.space_race["USSR"],
)
def _star_wars(engine: "Engine", side: Side) -> None:
    # Eligible only while the US leads the Space Race. The US takes any
    # non-scoring card from the discard pile and plays its event immediately.
    choices = tuple(cid for cid in engine.discard_pile if not engine.cards[cid].scoring)
    if not choices:
        return
    engine.push_event_choice("Star_Wars", Side.US, choices + ("none",))


def _star_wars_choice(engine: "Engine", side: Side, choice: str, context: dict) -> None:
    if choice == "none":
        return
    if choice in engine.discard_pile:
        engine.discard_pile.remove(choice)
        engine.play_card_from_discard(Side.US, choice)


# -- Che: a free USSR coup in the Americas/Africa, with a conditional repeat ---


@event("Che")
def _che(engine: "Engine", side: Side) -> None:
    # The USSR makes a free Coup against a non-Battleground country in Central
    # America, South America, or Africa; if it removes any US Influence, a second
    # free Coup against a different such country. The coup is always the USSR's,
    # even when the US plays Che for its Operations.
    candidates = [
        cid
        for cid, info in engine.board.countries.items()
        if not info.battleground
        and info.region in (Region.CENTRAL_AMERICA, Region.SOUTH_AMERICA, Region.AFRICA)
    ]
    engine.push_che_coup(Side.USSR, ops=3, candidates=candidates)


def _che_choice(engine: "Engine", side: Side, choice: str, context: dict) -> None:
    if choice == "none":
        return
    engine.begin_che_coup(
        side, choice, context["che_ops"], context["che_candidates"], context["che_used"]
    )


# -- Cuban Missile Crisis: coup = loss this turn, unless defused --------------


@event("Cuban_Missile_Crisis")
def _cuban_missile_crisis(engine: "Engine", side: Side) -> None:
    # Set DEFCON to 2. For the rest of the turn any Coup attempt by the opponent
    # loses them the game (checked in _handle_coup_roll). The opponent may defuse
    # at once by removing 2 Influence from Cuba (USSR) or West Germany (US).
    # (Faithful simplification: the real defuse can be taken at any later point in
    # the turn; here it is offered immediately.)
    engine.set_defcon(2, caused_by=side)
    if engine.is_terminal:
        return
    opp = side.opponent
    engine.turn_effects["cuban_missile_crisis"] = opp.value
    country = "Cuba" if opp is Side.USSR else "West_Germany"
    choices = ["defuse", "keep"] if engine.board.influence[country][opp.value] >= 2 else ["keep"]
    if len(choices) > 1:
        engine.push_event_choice(
            "Cuban_Missile_Crisis", opp, tuple(choices), extra={"country": country}
        )


def _cuban_missile_crisis_choice(engine: "Engine", side: Side, choice: str, context: dict) -> None:
    if choice == "defuse":
        engine.remove_influence(context["country"], side, 2)
        engine.turn_effects.pop("cuban_missile_crisis", None)


# -- We Will Bury You: end-of-turn VP unless UN Intervention defuses it -------


@event("We_Will_Bury_You")
def _we_will_bury_you(engine: "Engine", side: Side) -> None:
    # Degrade DEFCON one level; the USSR scores 3 VP at end of turn unless the US
    # plays UN Intervention (which clears the flag, see _handle_play_mode).
    engine._change_defcon(-1, caused_by=Side.USSR)
    if not engine.is_terminal:
        engine.turn_effects["we_will_bury_you"] = True


# -- Formosan Resolution: Taiwan scores as a Battleground for the US ----------


@event("Formosan_Resolution")
def _formosan_resolution(engine: "Engine", side: Side) -> None:
    # While active and the US controls Taiwan, Taiwan scores as a Battleground in
    # Asia (see _scoring_overrides). Nullified once the China Card is played.
    engine.game_effects["formosan_resolution"] = True


# -- Shuttle Diplomacy: drop one USSR Battleground at the next ME/Asia score --


@event("Shuttle_Diplomacy")
def _shuttle_diplomacy(engine: "Engine", side: Side) -> None:
    # At the next scoring of the Middle East or Asia, one USSR-controlled
    # Battleground is not counted (consumed there; see _scoring_overrides). (The
    # card is filed to the discard now rather than kept "in front of you" — a
    # cosmetic simplification, since the effect flag is what matters.)
    engine.game_effects["shuttle_diplomacy"] = True


# -- North Sea Oil: block OPEC and grant the US an extra action round ----------


@event("North_Sea_Oil")
def _north_sea_oil(engine: "Engine", side: Side) -> None:
    # OPEC may no longer be played as an event (game-long), and the US plays one
    # extra action round this turn (see _total_action_rounds / _side_for_play_index).
    engine.game_effects["north_sea_oil"] = True
    engine.turn_effects["north_sea_oil_extra"] = True


# -- shared helpers ---------------------------------------------------------


def _controlled_battlegrounds(engine: "Engine", side: Side) -> int:
    return sum(
        1
        for cid, info in engine.board.countries.items()
        if info.battleground and engine.board.control(cid) is side
    )


# Routers for EVENT_CHOICE branches, looked up by the engine at handle time
# (the decision stack stays serializable — only the event id and the chosen
# option are stored, never a function).
CHOICE_ROUTERS: dict[str, Callable[["Engine", Side, str], None]] = {
    "Warsaw_Pact_Formed": _warsaw_pact_choice,
    "Independent_Reds": _independent_reds_choice,
    "How_I_Learned_to_Stop_Worrying": _how_i_learned_choice,
    "Salt_Negotiations": _salt_reclaim_choice,
    "Junta": _junta_choice,
    "Chernobyl": _chernobyl_choice,
    "Olympic_Games": _olympic_choice,
    "Wargames": _wargames_choice,
    "Summit_defcon": _summit_defcon_choice,
    "Aldrich_Ames_Remix": _aldrich_ames_choice,
    "Grain_Sales_to_Soviets": _grain_sales_choice,
    "Ask_Not_What_Your_Country_Can_Do_For_You": _ask_not_choice,
    "Missile_Envy_pick": _missile_envy_pick_choice,
    "Missile_Envy_use": _missile_envy_use_choice,
    "Star_Wars": _star_wars_choice,
    "Che": _che_choice,
    "Cuban_Missile_Crisis": _cuban_missile_crisis_choice,
}
