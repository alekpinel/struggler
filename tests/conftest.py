"""Shared test fixtures/helpers.

Centralized here so a mechanic that grows (e.g. the headline-pending
state or Our Man in Tehran's peek queue) only needs to be taught to one
"where do cards live" helper instead of several near-duplicate copies
drifting out of sync (see the ``_headline_pending`` incident in
test_engine_m2.py, fixed by consolidating here).
"""

from __future__ import annotations

from collections import Counter

from struggler.engine import Engine, Period
from struggler.engine.cards import cards_entering
from struggler.engine.core import HIDDEN_CARD
from struggler.engine.rules import RULES


def bare_engine(seed: int = 0) -> Engine:
    """A minimal engine with the event layer on but no turn loop running."""
    engine = Engine(seed=seed)
    engine.events_enabled = True
    return engine


def headline_setup(engine: Engine, ussr_card: str, us_card: str) -> None:
    """Put a controlled headline in front of a bare, events-on engine."""
    engine.phase = "headline"
    engine.hands = {"USSR": [ussr_card], "US": [us_card]}
    engine._advance()  # pushes the USSR headline choice


def cards_in_play(engine: Engine) -> Counter:
    """Tally every card by id across every location the engine can hold one.

    Every piece of state that can transiently own a card id must be listed
    here — this is the single source of truth `_assert_invariants` checks
    against, so a new mechanic that introduces a new such location (like
    `_headline_pending` or Our Man in Tehran's peek queue did) only needs
    one edit, not one per test file.
    """
    c: Counter = Counter()
    # HIDDEN_CARD placeholders (physical mode) are not real card ids — skip
    # them here and count `hidden_pool` instead (see below), the "no fixed
    # location yet" bucket for a physical hand's true, unknown contents.
    for cards in engine.hands.values():
        c.update(cid for cid in cards if cid != HIDDEN_CARD)
    c.update(cid for cid in engine.draw_pile if cid != HIDDEN_CARD)
    c.update(engine.discard_pile)
    c.update(engine.removed_cards)
    for cid in engine._headline.values():
        if cid is not None:
            c.update([cid])
    # A headlined card whose event is mid-resolution (its sub-decisions still
    # draining) lives here until it is filed to a pile.
    for _side, cid in engine._headline_pending:
        c.update([cid])
    # Our Man in Tehran's peeked-but-undecided cards live here mid-resolution;
    # they are deliberately excluded from observe() (mandate #4) but must
    # still be accounted for exactly once.
    c.update(engine._our_man_queue)
    c.update(engine._our_man_kept)
    c.update(engine.hidden_pool)
    return c


def expected_in_play(engine: Engine) -> set[str]:
    ids = set(cards_entering(engine.cards, Period.EARLY_WAR, engine.include_optional))
    if engine.turn >= 4:
        ids |= set(cards_entering(engine.cards, Period.MID_WAR, engine.include_optional))
    if engine.turn >= 8:
        ids |= set(cards_entering(engine.cards, Period.LATE_WAR, engine.include_optional))
    return ids


def assert_invariants(engine: Engine) -> None:
    assert 1 <= engine.defcon <= 5
    for values in engine.board.influence.values():
        assert values["US"] >= 0 and values["USSR"] >= 0
    if not engine.is_terminal:
        assert engine.pending_decision is not None
        assert len(engine.legal_actions()) > 0  # never deadlock on a live decision
    else:
        # Mandate #1: pending is None iff the game ended. An event that
        # scores the winning VP and then pushes a choice once broke this.
        assert engine.pending_decision is None

    # No card is ever in two places at once, and The China Card is tracked
    # separately (never in a hand or pile).
    in_play = cards_in_play(engine)
    assert all(count == 1 for count in in_play.values())
    assert RULES["china_card_id"] not in in_play
    assert set(in_play) == expected_in_play(engine)

    if engine.physical_mode:
        placeholder_slots = sum(cards.count(HIDDEN_CARD) for cards in engine.hands.values())
        placeholder_slots += engine.draw_pile.count(HIDDEN_CARD)
        assert placeholder_slots == len(engine.hidden_pool)
        assert HIDDEN_CARD not in engine.discard_pile
        assert HIDDEN_CARD not in engine.removed_cards
