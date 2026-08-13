"""Loads data/rules.json: tunable facts about the physical game that belong
in data, not hardcoded as Python literals (mandate #5). `RULES` is the one
deliberate exception to "no bare module-level globals" here: a single
Mapping instance, keyed by the same snake_case names already used in
rules.json, replacing the twelve individually-typed constants this module
used to export. Every consumer does `from struggler.engine.rules import
RULES` and `RULES["name"]` at each use site.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from struggler.engine.data_loader import load_json
from struggler.engine.types import Region, Side, Subregion


class Rules(Mapping[str, Any]):
    """Parsed data/rules.json, converted once at construction into the same
    enum/tuple/frozenset-keyed shapes the old module-level constants used."""

    def __init__(self, raw: dict) -> None:
        self._data: dict[str, Any] = {}

        # Regional scoring VP table: (presence, domination, control).
        #
        # Confirmed against the physical game for Middle East, Africa, and
        # South America. Europe, Asia, and Central America are still
        # UNCONFIRMED (best guess, not yet checked against the
        # rulebook/board) — verify before trusting them beyond structural
        # testing.
        #
        # Europe's control value is intentionally None: controlling every
        # country in Europe does not win immediately — it wins when the
        # Europe Scoring card is played while that control holds (see
        # Board.controls_all_of_europe, and the M2/M3 note there). No M1
        # code path should ever hit CONTROL tier for Europe, since nothing
        # in M1 scores a region; score_region() raises rather than silently
        # return a made-up number if it ever does.
        self._data["scoring"] = {
            Region[name]: tuple(values) for name, values in raw["scoring"].items()
        }

        # VP required to win outright; the track runs to 20 in either direction.
        self._data["vp_to_win"] = raw["vp_to_win"]

        # Minimum DEFCON level required to attempt a coup in a region; regions
        # not listed have no restriction. Confirmed against the physical game.
        self._data["coup_min_defcon"] = {
            Region[name]: value for name, value in raw["coup_min_defcon"].items()
        }

        # The China Card starts face-up with the USSR.
        self._data["china_card_id"] = raw["china_card_id"]

        # UN Intervention (a Tier 4 rule-modifier): held in hand, it lets its
        # player use an *opponent's* card for Ops while cancelling that
        # card's event.
        self._data["un_intervention_id"] = raw["un_intervention_id"]

        # The "war" cards, tracked so Flower Power can score the USSR each
        # time the US plays one (for its Event or Operations).
        self._data["war_cards"] = frozenset(raw["war_cards"])

        # Additional influence each side places by choice during setup, after
        # the printed at-start influence: the USSR into Eastern Europe, the
        # US into Western Europe. VERIFY the exact counts against the
        # rulebook.
        self._data["setup_additional"] = {
            Subregion[name]: (Side(entry["side"]), entry["amount"])
            for name, entry in raw["setup_additional"].items()
        }

        # Space Race track, boxes 1..8. Per box: minimum Ops the played card
        # must be worth to attempt entry, the die roll needed (success iff
        # d6 <= roll_max), and the VP awarded to the first / second
        # superpower to reach the box.
        #
        # VERIFY: these numeric constants are best-effort from knowledge of
        # the physical Space Race track and have NOT been reconfirmed
        # line-by-line. The *mechanism* around them (attempt -> seeded
        # CHANCE roll -> advance -> award) is the part M2 proves; only the
        # numbers here are provisional. Box 4's headline-reveal-order perk
        # (6.4.4) remains unmodeled; boxes 2, 6 and 8's perks are
        # implemented in core.py.
        self._data["space_race_boxes"] = {
            int(box): values for box, values in raw["space_race_boxes"].items()
        }
        self._data["space_race_max_box"] = raw["space_race_max_box"]
        # A side that has reached this box may make two Space Race attempts
        # per turn instead of one. VERIFY exact box.
        self._data["space_race_two_attempts_from_box"] = raw[
            "space_race_two_attempts_from_box"
        ]
        # Space Race boxes whose special ability (6.4.3-6.4.4) is modeled as
        # a granted/cancelled game_effects flag rather than a direct
        # position check; see Engine._update_space_race_ability.
        self._data["space_race_ability_keys"] = {
            int(box): key for box, key in raw["space_race_ability_keys"].items()
        }

        # Hand size each player is dealt up to at the start of a turn. 8 in
        # the Early War, 9 once the Mid War begins. The China Card never
        # counts toward this. VERIFY against the printed rulebook before
        # relying on these beyond the structural full-game proof M2 is
        # about.
        self._data["hand_limit_early"] = raw["hand_limit_early"]
        self._data["hand_limit_mid_late"] = raw["hand_limit_mid_late"]

        # Action rounds per turn: 6 in the Early War (turns 1-3), 7 from the
        # Mid War on (turns 4-10). VERIFY before relying on these beyond
        # structural testing.
        self._data["action_rounds_early"] = raw["action_rounds_early"]
        self._data["action_rounds_mid_late"] = raw["action_rounds_mid_late"]

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


RULES = Rules(load_json("rules.json"))
