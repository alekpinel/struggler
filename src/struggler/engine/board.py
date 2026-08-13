"""Board: country data, influence, control, adjacency, and region scoring."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from struggler.engine.rules import SCORING
from struggler.engine.types import Region, ScoringTier, Side, Subregion

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "countries.json"


@dataclass(frozen=True)
class CountryInfo:
    id: str
    name: str
    region: Region
    subregion: Subregion | None
    stability: int
    battleground: bool


class Board:
    """Owns country metadata, the adjacency graph, and influence markers.

    Adjacency is loaded exactly as declared in data/countries.json (no
    auto-symmetrization) and then validated for reciprocity, so a
    one-directional edge in the data file is a hard error, not a silent
    bug.
    """

    def __init__(self, data_path: Path | None = None) -> None:
        raw = _load_raw(data_path or DEFAULT_DATA_PATH)

        self.countries: dict[str, CountryInfo] = {}
        self._adjacency: dict[str, set[str]] = {"US": set(), "USSR": set()}

        for cid, entry in raw["countries"].items():
            self.countries[cid] = CountryInfo(
                id=cid,
                name=entry["name"],
                region=Region[entry["region"]],
                subregion=Subregion[entry["subregion"]] if entry.get("subregion") else None,
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

    def is_reachable(self, side: Side, country_id: str) -> bool:
        """Whether `side` may add influence to `country_id` at all.

        A side can place influence in a country that's adjacent to its own
        superpower, that it already has influence in, or that's adjacent to
        another country it already has influence in.
        """
        if country_id in self._adjacency[side.value]:
            return True
        if self.influence[country_id][side.value] > 0:
            return True
        return any(
            self.influence[n][side.value] > 0
            for n in self._adjacency.get(country_id, set())
            if n in self.influence
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

        Confirmed: this does NOT win the game by itself. The win happens
        when the Europe Scoring card is played while a side holds this
        condition — a card event, out of scope until M2/M3. This method
        is a pure query for that future check to use; nothing in M1 calls
        it to end the game.
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
        presence_vp, domination_vp, control_vp = SCORING[region]
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


def _load_raw(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
