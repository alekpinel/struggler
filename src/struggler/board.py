"""Board: country data, influence, control, adjacency, and region scoring."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

from struggler.types import Region, ScoringTier, Side, Subregion

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "countries.json"

# Regional scoring VP table: (presence, domination, control).
#
# PROVISIONAL. Two independent cross-checks against public sources gave
# contradictory numbers for these constants, and they have not been
# verified against the physical rulebook. Per CLAUDE.md's milestone gate,
# M1 is not "done" until this table is confirmed against an authoritative
# source. Everything else in this module (the tier logic itself) is not
# in question, only these specific numbers.
#
# Europe's control value is intentionally None: controlling every country
# in Europe ends the game immediately (see Board.controls_all_of_europe),
# so a scoring card should never see that state — score_region() raises
# rather than silently return a made-up number.
SCORING: dict[Region, tuple[int, int, int | None]] = {
    Region.EUROPE: (3, 7, None),
    Region.ASIA: (3, 7, 9),
    Region.MIDDLE_EAST: (2, 3, 5),
    Region.AFRICA: (1, 2, 4),
    Region.CENTRAL_AMERICA: (1, 3, 5),
    Region.SOUTH_AMERICA: (2, 3, 5),
}


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
        europe = self.countries_in(Region.EUROPE)
        if all(self.control(cid) is Side.US for cid in europe):
            return Side.US
        if all(self.control(cid) is Side.USSR for cid in europe):
            return Side.USSR
        return None

    # -- region scoring ---------------------------------------------------------

    def region_tier(self, side: Side, region: Region) -> ScoringTier:
        country_ids = self.countries_in(region)
        bg_ids = [cid for cid in country_ids if self.countries[cid].battleground]
        total_bg = len(bg_ids)

        opponent = side.opponent
        side_count = sum(1 for cid in country_ids if self.control(cid) is side)
        opp_count = sum(1 for cid in country_ids if self.control(cid) is opponent)
        side_bg = sum(1 for cid in bg_ids if self.control(cid) is side)
        opp_bg = sum(1 for cid in bg_ids if self.control(cid) is opponent)

        if total_bg > 0 and side_bg == total_bg and side_count > opp_count:
            return ScoringTier.CONTROL
        if side_count > opp_count and side_bg > opp_bg:
            return ScoringTier.DOMINATION
        if side_count > 0:
            return ScoringTier.PRESENCE
        return ScoringTier.NONE

    def score_region(self, region: Region) -> int:
        """Net VP swing from scoring `region` now (positive favors US,
        negative favors USSR), per the standalone Presence/Domination/
        Control tier each side independently achieves."""
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
                return control_vp
            return tier_value[tier]

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
