"""Loads data/rules.json: tunable facts about the physical game that belong
in data, not hardcoded as Python literals (mandate #5). Each constant here
keeps the exact name/shape the rest of the engine already expects, so
board.py/core.py/cards.py only change where the value comes from, not how
it's used.
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import Region, Side, Subregion

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "data" / "rules.json"


def _load(path: Path = DEFAULT_RULES_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


_raw = _load()

# Regional scoring VP table: (presence, domination, control).
#
# Confirmed against the physical game for Middle East, Africa, and South
# America. Europe, Asia, and Central America are still UNCONFIRMED (best
# guess, not yet checked against the rulebook/board) — verify before
# trusting them beyond structural testing.
#
# Europe's control value is intentionally None: controlling every country
# in Europe does not win immediately — it wins when the Europe Scoring
# card is played while that control holds (see Board.controls_all_of_europe,
# and the M2/M3 note there). No M1 code path should ever hit CONTROL tier
# for Europe, since nothing in M1 scores a region; score_region() raises
# rather than silently return a made-up number if it ever does.
SCORING: dict[Region, tuple[int, int, int | None]] = {
    Region[name]: tuple(values) for name, values in _raw["scoring"].items()
}

# VP required to win outright; the track runs to 20 in either direction.
VP_TO_WIN: int = _raw["vp_to_win"]

# Minimum DEFCON level required to attempt a coup in a region; regions not
# listed have no restriction. Confirmed against the physical game.
COUP_MIN_DEFCON: dict[Region, int] = {
    Region[name]: value for name, value in _raw["coup_min_defcon"].items()
}

# The China Card starts face-up with the USSR.
CHINA_CARD_ID: str = _raw["china_card_id"]

# UN Intervention (a Tier 4 rule-modifier): held in hand, it lets its player use
# an *opponent's* card for Ops while cancelling that card's event.
UN_INTERVENTION_ID: str = _raw["un_intervention_id"]

# The "war" cards, tracked so Flower Power can score the USSR each time the US
# plays one (for its Event or Operations).
WAR_CARDS: frozenset[str] = frozenset(_raw["war_cards"])

# Additional influence each side places by choice during setup, after the
# printed at-start influence: the USSR into Eastern Europe, the US into
# Western Europe. VERIFY the exact counts against the rulebook.
SETUP_ADDITIONAL: dict[Subregion, tuple[Side, int]] = {
    Subregion[name]: (Side(entry["side"]), entry["amount"])
    for name, entry in _raw["setup_additional"].items()
}

# Space Race track, boxes 1..8. Per box: minimum Ops the played card must be
# worth to attempt entry, the die roll needed (success iff d6 <= roll_max),
# and the VP awarded to the first / second superpower to reach the box.
#
# VERIFY: these numeric constants are best-effort from knowledge of the
# physical Space Race track and have NOT been reconfirmed line-by-line. The
# *mechanism* around them (attempt -> seeded CHANCE roll -> advance -> award)
# is the part M2 proves; only the numbers here are provisional. Box 4's
# headline-reveal-order perk (6.4.4) remains unmodeled; boxes 2, 6 and 8's
# perks are implemented in core.py.
SPACE_RACE_BOXES: dict[int, dict[str, int]] = {
    int(box): values for box, values in _raw["space_race_boxes"].items()
}
SPACE_RACE_MAX_BOX: int = _raw["space_race_max_box"]
# A side that has reached this box may make two Space Race attempts per turn
# instead of one. VERIFY exact box.
SPACE_RACE_TWO_ATTEMPTS_FROM_BOX: int = _raw["space_race_two_attempts_from_box"]
# Space Race boxes whose special ability (6.4.3-6.4.4) is modeled as a
# granted/cancelled game_effects flag rather than a direct position check;
# see Engine._update_space_race_ability.
SPACE_RACE_ABILITY_KEYS: dict[int, str] = {
    int(box): key for box, key in _raw["space_race_ability_keys"].items()
}

# Hand size each player is dealt up to at the start of a turn. 8 in the Early
# War, 9 once the Mid War begins. The China Card never counts toward this.
# VERIFY against the printed rulebook before relying on these beyond the
# structural full-game proof M2 is about.
HAND_LIMIT_EARLY: int = _raw["hand_limit_early"]
HAND_LIMIT_MID_LATE: int = _raw["hand_limit_mid_late"]

# Action rounds per turn: 6 in the Early War (turns 1-3), 7 from the Mid War
# on (turns 4-10). VERIFY before relying on these beyond structural testing.
ACTION_ROUNDS_EARLY: int = _raw["action_rounds_early"]
ACTION_ROUNDS_MID_LATE: int = _raw["action_rounds_mid_late"]
