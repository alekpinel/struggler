"""Reads `card_playbook.json`: per-card, per-side play advice for the LLM
bot's prompt.

Deliberately NOT under `struggler/data/`. That directory holds the game's
facts -- what a card mechanically does (`cards.json`), what a country is
(`countries.json`), what the rules constants are -- and the engine reads
it. This file holds *opinions about how to play those cards well*, which
only one bot consumes and which a different bot (or a re-tuned version of
this one) would legitimately disagree with. Keeping it next to the player
that uses it is the same separation `GreedyWeights` already draws between
the rules and one bot's judgement of them.

Coverage is incomplete on purpose and grows card by card -- the same
pattern `GreedyPlayer._SCORERS` uses for decision kinds. A card with no
entry simply contributes no advice line, never a placeholder.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from struggler.engine import Side

_PLAYBOOK_PATH = Path(__file__).resolve().parent / "card_playbook.json"


@lru_cache(maxsize=1)
def _playbook() -> dict:
    with _PLAYBOOK_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def scoring_card_rule() -> str:
    """The standing advice that applies to every Scoring card, kept in one
    place rather than repeated in each of the seven entries."""
    return _playbook()["_scoring_default"]


def advice_for(card_id: str, side: Side) -> str | None:
    """This seat's advice for `card_id`: the card's side-agnostic note and
    its note for `side`, joined. `None` when the playbook says nothing
    about this card for this seat -- callers must render nothing at all in
    that case, not an empty heading."""
    entry = _playbook()["cards"].get(card_id)
    if not entry:
        return None
    parts = [entry[key] for key in ("any", side.value) if entry.get(key)]
    if not parts:
        return None
    return " ".join(parts)


def known_card_ids() -> frozenset[str]:
    """Every card id the playbook has an entry for -- so a test can assert
    the file never drifts to naming a card that doesn't exist."""
    return frozenset(_playbook()["cards"])
