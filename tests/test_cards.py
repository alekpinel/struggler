"""Card data loading and deck-composition tests (M2 foundation)."""

from struggler.engine import CardSide, Period
from struggler.engine.cards import (
    action_rounds,
    cards_entering,
    hand_limit,
    load_cards,
)


def test_loads_all_110_cards_with_complete_numbering():
    cards = load_cards()
    assert len(cards) == 110
    assert sorted(c.number for c in cards.values()) == list(range(1, 111))


def test_scoring_cards_have_zero_ops_and_vice_versa():
    # Ops==0 iff scoring: the two must never disagree, or ops enumeration breaks.
    for card in load_cards().values():
        assert (card.ops == 0) == card.scoring


def test_the_china_card_is_the_only_card_never_shuffled_in():
    cards = load_cards()
    not_in_deck = [cid for cid, c in cards.items() if not c.in_deck]
    assert not_in_deck == ["The_China_Card"]


def test_exactly_the_seven_regional_scoring_cards():
    cards = load_cards()
    scoring = {cid for cid, c in cards.items() if c.scoring}
    assert scoring == {
        "Asia_Scoring",
        "Europe_Scoring",
        "Middle_East_Scoring",
        "Central_America_Scoring",
        "Southeast_Asia_Scoring",
        "Africa_Scoring",
        "South_America_Scoring",
    }


def test_five_year_plan_is_a_us_card():
    # Cross-checked fact: its event forces a USSR discard, so it is US-owned.
    assert load_cards()["Five_Year_Plan"].side is CardSide.US


def test_cards_entering_excludes_china_and_optional_by_default():
    cards = load_cards()
    early = cards_entering(cards, Period.EARLY_WAR, include_optional=False)
    assert "The_China_Card" not in early
    assert all(not cards[cid].optional for cid in early)
    assert all(cards[cid].period is Period.EARLY_WAR for cid in early)


def test_cards_entering_is_deterministic_canonical_order():
    cards = load_cards()
    ids = cards_entering(cards, Period.MID_WAR, include_optional=True)
    numbers = [cards[cid].number for cid in ids]
    assert numbers == sorted(numbers)


def test_optional_cards_add_in_only_when_requested():
    cards = load_cards()
    base = cards_entering(cards, Period.EARLY_WAR, include_optional=False)
    with_opt = cards_entering(cards, Period.EARLY_WAR, include_optional=True)
    assert set(base) < set(with_opt)
    added = set(with_opt) - set(base)
    assert all(cards[cid].optional for cid in added)


def test_every_shuffleable_card_belongs_to_exactly_one_period_bucket():
    cards = load_cards()
    buckets = [
        cards_entering(cards, period, include_optional=True)
        for period in Period
    ]
    flat = [cid for bucket in buckets for cid in bucket]
    # No card appears in two periods, and every in-deck card lands somewhere.
    assert len(flat) == len(set(flat))
    assert set(flat) == {cid for cid, c in cards.items() if c.in_deck}


def test_hand_limit_and_action_rounds_by_turn():
    assert [hand_limit(t) for t in (1, 3, 4, 10)] == [8, 8, 9, 9]
    assert [action_rounds(t) for t in (1, 3, 4, 10)] == [6, 6, 7, 7]
