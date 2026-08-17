"""Card data loading and deck construction for M2.

Cards are pure data (see CLAUDE.md's card data policy and data/cards.json);
no event mechanics live here. This module owns:

- loading data/cards.json into immutable `Card` objects, and
- the deterministic deck operations the engine needs (which cards enter in
  each period, initial deal order), all as plain-list transforms so the
  engine can keep them as JSON-native state (mandate #5).

Shuffling itself is NOT done here and is NOT a CHANCE decision: the engine
performs it through its injected, seeded RNG (mandate #3), which keeps the
draw order out of `legal_actions()` and out of every player's observation
(mandate #4). These helpers only ever return cards in a fixed, canonical
order; the engine shuffles the result.
"""

from __future__ import annotations

from struggler.engine.data_loader import load_json
from struggler.engine.rules import RULES
from struggler.engine.types import Card, CardSide, Period


def load_cards() -> dict[str, Card]:
    """Load every card from the data file into immutable `Card` objects."""
    raw = load_json("cards.json")

    cards: dict[str, Card] = {}
    for cid, entry in raw["cards"].items():
        cards[cid] = Card(
            id=cid,
            number=entry["number"],
            name=entry["name"],
            ops=entry["ops"],
            side=CardSide[entry["side"]],
            period=Period[entry["period"]],
            scoring=entry["scoring"],
            remove_after_event=entry["remove_after_event"],
            optional=entry["optional"],
            in_deck=entry["in_deck"],
            event_summary=entry["event_summary"],
        )
    return cards


def cards_entering(
    cards: dict[str, Card], period: Period, include_optional: bool
) -> tuple[str, ...]:
    """Ids of the shuffle-able cards that enter the deck for `period`.

    Excludes The China Card (in_deck=False) and, unless `include_optional`,
    the Deluxe optional cards. Returned in canonical card-number order so the
    pre-shuffle sequence is deterministic; the engine shuffles it.
    """
    ids = [
        cid
        for cid, card in cards.items()
        if card.in_deck
        and card.period is period
        and (include_optional or not card.optional)
    ]
    ids.sort(key=lambda cid: cards[cid].number)
    return tuple(ids)


def hand_limit(turn: int) -> int:
    """Cards each player is dealt up to at the start of `turn`."""
    return RULES["hand_limit_early"] if turn <= 3 else RULES["hand_limit_mid_late"]


def action_rounds(turn: int) -> int:
    """Number of action rounds each player takes during `turn`."""
    return RULES["action_rounds_early"] if turn <= 3 else RULES["action_rounds_mid_late"]
