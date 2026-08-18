"""Board: country data, influence, control, adjacency, and region scoring."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from struggler.engine.data_loader import load_json
from struggler.engine.rules import RULES
from struggler.engine.types import Region, ScoringTier, Side, Subregion


@dataclass(frozen=True)
class CountryInfo:
    id: str
    name: str
    region: Region
    # Almost every country belongs to zero or one subregion; Austria and
    # Finland are the exception (Twilight Struggle FAQ ruling: both are
    # historically-neutral countries counted as part of both Western Europe
    # and Eastern Europe for every card/rule that references either).
    subregions: frozenset[Subregion]
    stability: int
    battleground: bool


def _load_subregions(raw: str | list[str] | None) -> frozenset[Subregion]:
    """Country data stores `subregion` as null, a single name, or (for
    Austria/Finland) a list of names -- normalize all three to a set."""
    if raw is None:
        return frozenset()
    names = [raw] if isinstance(raw, str) else raw
    return frozenset(Subregion[name] for name in names)


class Board:
    """Owns country metadata, the adjacency graph, and influence markers.

    Adjacency is loaded exactly as declared in data/countries.json (no
    auto-symmetrization) and then validated for reciprocity, so a
    one-directional edge in the data file is a hard error, not a silent
    bug.
    """

    def __init__(self) -> None:
        raw = load_json("countries.json")

        self.countries: dict[str, CountryInfo] = {}
        self._adjacency: dict[str, set[str]] = {"US": set(), "USSR": set()}

        for cid, entry in raw["countries"].items():
            self.countries[cid] = CountryInfo(
                id=cid,
                name=entry["name"],
                region=Region[entry["region"]],
                subregions=_load_subregions(entry.get("subregion")),
                stability=entry["stability"],
                battleground=entry["battleground"],
            )
            self._adjacency.setdefault(cid, set())

        for cid, entry in raw["countries"].items():
            for neighbor in entry["adjacent_to"]:
                self._adjacency[cid].add(neighbor)
        for side_id, entry in raw["superpowers"].items():
            for neighbor in entry["adjacent_to"]:
                self._adjacency[side_id].add(neighbor)

        self._validate_symmetric()

        self.influence: dict[str, dict[str, int]] = {
            cid: {"US": 0, "USSR": 0} for cid in self.countries
        }

        # Printed at-start influence for the standard game (the additional
        # player-chosen Eastern/Western Europe points are placed by the engine
        # as decisions, not here). Absent in minimal test data -> empty.
        self.setup_influence: dict[str, dict[str, int]] = raw.get("setup_influence", {})

    def _validate_symmetric(self) -> None:
        broken = []
        for node, neighbors in self._adjacency.items():
            for neighbor in neighbors:
                if node not in self._adjacency.get(neighbor, set()):
                    broken.append((node, neighbor))
        if broken:
            pairs = ", ".join(f"{a}->{b}" for a, b in broken)
            raise ValueError(f"Asymmetric adjacency in board data: {pairs}")

    # -- adjacency / reachability -------------------------------------------------

    def is_adjacent(self, a: str, b: str) -> bool:
        return b in self._adjacency.get(a, set())

    def neighbors(self, country_id: str) -> frozenset[str]:
        return frozenset(self._adjacency.get(country_id, set()))

    def is_reachable(
        self,
        side: Side,
        country_id: str,
        influence: dict[str, dict[str, int]] | None = None,
    ) -> bool:
        """Whether `side` may add influence to `country_id` at all.

        A side can place influence in a country that's adjacent to its own
        superpower, that it already has influence in, or that's adjacent to
        another country it already has influence in.

        `influence` optionally overrides which influence snapshot to consult
        (rule 6.1.1: within one Operations spend, every point must be
        adjacent to friendly markers that were in place at the *start* of
        the phasing player's Action Round, not markers placed earlier in the
        same spend). Defaults to the board's live influence for callers that
        don't need that distinction (e.g. Event-driven placement, which rule
        6.1.1's own exception exempts).
        """
        inf = influence if influence is not None else self.influence
        if country_id in self._adjacency[side.value]:
            return True
        if inf[country_id][side.value] > 0:
            return True
        return any(
            inf[n][side.value] > 0
            for n in self._adjacency.get(country_id, set())
            if n in inf
        )

    def influence_cost(self, side: Side, country_id: str) -> int:
        """Ops cost to add 1 influence point to `country_id` for `side`.

        Doubled if the opponent controls the country (the "doubling rule").
        """
        return 2 if self.control(country_id) is side.opponent else 1

    # -- control --------------------------------------------------------------

    def control(self, country_id: str) -> Side | None:
        """A side controls a country when its influence exceeds the
        opponent's by at least the country's stability number.

        Returns None for anything that isn't a real country (e.g. the "US"/
        "USSR" superpower nodes in the adjacency graph aren't controllable).
        """
        info = self.countries.get(country_id)
        if info is None:
            return None
        us = self.influence[country_id]["US"]
        ussr = self.influence[country_id]["USSR"]
        if us - ussr >= info.stability:
            return Side.US
        if ussr - us >= info.stability:
            return Side.USSR
        return None

    def countries_in(self, region: Region) -> tuple[str, ...]:
        return tuple(cid for cid, info in self.countries.items() if info.region == region)

    def controls_all_of_europe(self) -> Side | None:
        """Whether one side currently controls every country in Europe.
        """
        europe = self.countries_in(Region.EUROPE)
        if all(self.control(cid) is Side.US for cid in europe):
            return Side.US
        if all(self.control(cid) is Side.USSR for cid in europe):
            return Side.USSR
        return None

    # -- region scoring ---------------------------------------------------------

    def region_tier(
        self,
        side: Side,
        region: Region,
        extra_battlegrounds: frozenset[str] = frozenset(),
        ignored: frozenset[str] = frozenset(),
    ) -> ScoringTier:
        """The Presence/Domination/Control tier `side` holds in `region`.

        `extra_battlegrounds` treats the named countries as Battlegrounds for
        this scoring only (Formosan Resolution promotes Taiwan); `ignored`
        treats the named countries as controlled by neither side (Shuttle
        Diplomacy drops one USSR Battleground from the tally). Both default to
        empty, so every existing caller is unaffected."""
        country_ids = self.countries_in(region)
        bg_ids = [
            cid
            for cid in country_ids
            if self.countries[cid].battleground or cid in extra_battlegrounds
        ]
        total_bg = len(bg_ids)

        def controller(cid: str) -> Side | None:
            return None if cid in ignored else self.control(cid)

        opponent = side.opponent
        side_count = sum(1 for cid in country_ids if controller(cid) is side)
        opp_count = sum(1 for cid in country_ids if controller(cid) is opponent)
        side_bg = sum(1 for cid in bg_ids if controller(cid) is side)
        opp_bg = sum(1 for cid in bg_ids if controller(cid) is opponent)

        if total_bg > 0 and side_bg == total_bg and side_count > opp_count:
            return ScoringTier.CONTROL
        if (
            side_count > opp_count
            and side_bg > opp_bg
            and side_count > side_bg  # must also Control >=1 non-Battleground (10.1.1)
        ):
            return ScoringTier.DOMINATION
        if side_count > 0:
            return ScoringTier.PRESENCE
        return ScoringTier.NONE

    def region_bonus_vp(
        self,
        side: Side,
        region: Region,
        extra_battlegrounds: frozenset[str] = frozenset(),
        ignored: frozenset[str] = frozenset(),
    ) -> int:
        """Additional VP `side` scores in `region` on top of its Presence/
        Domination/Control tier (10.1.2): +1 VP per Battleground country it
        Controls there, plus +1 VP per country it Controls there that is
        adjacent to the enemy superpower. `extra_battlegrounds`/`ignored`
        mirror region_tier's scoring overrides."""
        bonus = 0
        for cid in self.countries_in(region):
            if cid in ignored:
                continue
            if self.control(cid) is not side:
                continue
            if self.countries[cid].battleground or cid in extra_battlegrounds:
                bonus += 1
            if self.is_adjacent(side.opponent.value, cid):
                bonus += 1
        return bonus

    def score_region(self, region: Region) -> int:
        """Net VP swing from scoring `region` now (positive favors US,
        negative favors USSR): each side's Presence/Domination/Control tier
        value, plus its 10.1.2 bonuses (+1 VP per Battleground Controlled,
        +1 VP per country Controlled adjacent to the enemy superpower)."""
        presence_vp, domination_vp, control_vp = RULES["scoring"][region.name]
        tier_value = {
            ScoringTier.NONE: 0,
            ScoringTier.PRESENCE: presence_vp,
            ScoringTier.DOMINATION: domination_vp,
        }

        def value_for(side: Side) -> int:
            tier = self.region_tier(side, region)
            if tier is ScoringTier.CONTROL:
                if control_vp is None:
                    raise RuntimeError(
                        f"{region} reached CONTROL tier for {side}, but has no scoring "
                        "value defined (Europe's full control is an immediate win, not "
                        "a scoring-card outcome — see Board.controls_all_of_europe)."
                    )
                base = control_vp
            else:
                base = tier_value[tier]
            return base + self.region_bonus_vp(side, region)

        return value_for(Side.US) - value_for(Side.USSR)

    # -- serialization ------------------------------------------------------

    def serialize(self) -> dict:
        return {"influence": copy.deepcopy(self.influence)}

    def load_influence(self, data: dict) -> None:
        for cid, values in data["influence"].items():
            self.influence[cid]["US"] = values["US"]
            self.influence[cid]["USSR"] = values["USSR"]
