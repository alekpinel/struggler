"""Board mechanics: adjacency, control, region scoring."""

from struggler.board import SCORING, Board
from struggler.types import Region, ScoringTier, Side


def test_scoring_table_matches_confirmed_values():
    # Regression guard for the numbers confirmed against the physical game.
    assert SCORING[Region.MIDDLE_EAST] == (3, 5, 7)
    assert SCORING[Region.AFRICA] == (1, 4, 6)
    assert SCORING[Region.SOUTH_AMERICA] == (2, 5, 6)


def test_all_countries_reciprocal_adjacency_loaded_without_error():
    # Board() raises on load if any adjacency edge isn't reciprocated;
    # constructing it successfully is itself the assertion.
    board = Board()
    assert len(board.countries) > 0


def test_control_requires_margin_at_least_stability():
    board = Board()
    # Guatemala has stability 1: a 1-point margin is enough to control.
    board.influence["Guatemala"]["US"] = 1
    assert board.control("Guatemala") is Side.US

    board2 = Board()
    # Costa Rica has stability 3: a 2-point margin is NOT enough.
    board2.influence["Costa_Rica"]["US"] = 2
    assert board2.control("Costa_Rica") is None
    board2.influence["Costa_Rica"]["US"] = 3
    assert board2.control("Costa_Rica") is Side.US


def test_control_uses_margin_not_absolute_influence():
    board = Board()
    # Poland has stability 3. US has more raw influence than USSR (5 vs 3)
    # but the margin (2) is below stability, so nobody controls it.
    board.influence["Poland"]["US"] = 5
    board.influence["Poland"]["USSR"] = 3
    assert board.control("Poland") is None


def test_is_reachable_from_superpower_adjacency():
    board = Board()
    assert board.is_reachable(Side.USSR, "Poland")  # adjacent to USSR
    assert not board.is_reachable(Side.USSR, "Cuba")  # not adjacent, no influence chain yet


def test_is_reachable_transitively_through_own_influence():
    board = Board()
    assert not board.is_reachable(Side.US, "Guatemala")
    board.influence["Mexico"]["US"] = 1  # Mexico is adjacent to US
    assert board.is_reachable(Side.US, "Guatemala")  # Guatemala is adjacent to Mexico


def test_influence_cost_doubles_in_opponent_controlled_country():
    board = Board()
    board.influence["Guatemala"]["USSR"] = 1  # stability 1 -> USSR controls it
    assert board.control("Guatemala") is Side.USSR
    assert board.influence_cost(Side.US, "Guatemala") == 2
    assert board.influence_cost(Side.USSR, "Guatemala") == 1


def test_controls_all_of_europe():
    board = Board()
    europe = board.countries_in(Region.EUROPE)
    assert board.controls_all_of_europe() is None
    for cid in europe:
        stability = board.countries[cid].stability
        board.influence[cid]["US"] = stability
    assert board.controls_all_of_europe() is Side.US


def test_region_tier_presence_domination_control():
    board = Board()
    ca = board.countries_in(Region.CENTRAL_AMERICA)
    assert board.region_tier(Side.US, Region.CENTRAL_AMERICA) is ScoringTier.NONE

    # Give US control of exactly one non-battleground country -> presence only.
    one = next(cid for cid in ca if not board.countries[cid].battleground)
    board.influence[one]["US"] = board.countries[one].stability
    assert board.region_tier(Side.US, Region.CENTRAL_AMERICA) is ScoringTier.PRESENCE

    # Control every country including every battleground -> CONTROL tier.
    for cid in ca:
        board.influence[cid]["US"] = board.countries[cid].stability
    assert board.region_tier(Side.US, Region.CENTRAL_AMERICA) is ScoringTier.CONTROL


def test_region_tier_domination_requires_a_controlled_non_battleground():
    # Rule 10.1.1: Domination requires controlling more countries AND more
    # Battlegrounds than the opponent, *and* at least one non-Battleground
    # country. Controlling a single lone Battleground (with the opponent
    # controlling nothing) satisfies the first two conditions but not the
    # third, so it must stay at PRESENCE, not DOMINATION.
    board = Board()
    board.influence["Mexico"]["US"] = board.countries["Mexico"].stability  # Battleground
    assert board.countries["Mexico"].battleground
    assert board.region_tier(Side.US, Region.CENTRAL_AMERICA) is ScoringTier.PRESENCE

    # Add a controlled non-Battleground country -> now DOMINATION.
    board.influence["Guatemala"]["US"] = board.countries["Guatemala"].stability
    assert board.region_tier(Side.US, Region.CENTRAL_AMERICA) is ScoringTier.DOMINATION


def test_score_region_net_swing_favors_us_positive():
    board = Board()
    ca = board.countries_in(Region.CENTRAL_AMERICA)
    for cid in ca:
        board.influence[cid]["US"] = board.countries[cid].stability
    swing = board.score_region(Region.CENTRAL_AMERICA)
    assert swing > 0  # US controls the whole region, VP swing favors US


def test_score_region_rulebook_worked_example_10_1_2():
    # 10.1.2's own worked example: USSR Controls Cuba (Battleground, adjacent
    # to the US), Haiti and the Dominican Republic in Central America; the US
    # Controls only Guatemala. USSR: Domination (3) + 1 VP (Battleground Cuba)
    # + 1 VP (Cuba adjacent to the US) = 5. US: Presence only = 1. Net = -4.
    board = Board()
    board.influence["Cuba"]["USSR"] = board.countries["Cuba"].stability
    board.influence["Haiti"]["USSR"] = board.countries["Haiti"].stability
    board.influence["Dominican_Republic"]["USSR"] = board.countries["Dominican_Republic"].stability
    board.influence["Guatemala"]["US"] = board.countries["Guatemala"].stability

    assert board.region_tier(Side.USSR, Region.CENTRAL_AMERICA) is ScoringTier.DOMINATION
    assert board.region_bonus_vp(Side.USSR, Region.CENTRAL_AMERICA) == 2  # Battleground + adjacency
    assert board.region_bonus_vp(Side.US, Region.CENTRAL_AMERICA) == 0
    assert board.score_region(Region.CENTRAL_AMERICA) == 1 - 5


def test_score_region_europe_control_raises_instead_of_guessing():
    import pytest

    board = Board()
    europe = board.countries_in(Region.EUROPE)
    for cid in europe:
        board.influence[cid]["US"] = board.countries[cid].stability
    with pytest.raises(RuntimeError):
        board.score_region(Region.EUROPE)
