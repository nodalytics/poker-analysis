"""
Texas Hold'em hand evaluator.
Determines the best 5-card hand from any combination of cards.
Hand ranks: Royal Flush(10) > Straight Flush(9) > Four of a Kind(8) >
Full House(7) > Flush(6) > Straight(5) > Three of a Kind(4) >
Two Pair(3) > One Pair(2) > High Card(1)
"""
from typing import List, Tuple
from collections import Counter
from itertools import combinations
from src.core.cards import Card


class HandRank:
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10

    NAMES = {
        1: "High Card", 2: "One Pair", 3: "Two Pair",
        4: "Three of a Kind", 5: "Straight", 6: "Flush",
        7: "Full House", 8: "Four of a Kind",
        9: "Straight Flush", 10: "Royal Flush"
    }


class HandResult:
    """Result of evaluating a poker hand."""
    __slots__ = ['rank', 'values', 'cards']

    def __init__(self, rank: int, values: Tuple[int, ...], cards: List[Card]):
        self.rank = rank
        self.values = values
        self.cards = cards

    @property
    def name(self) -> str:
        return HandRank.NAMES[self.rank]

    def __lt__(self, other: 'HandResult') -> bool:
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.values < other.values

    def __eq__(self, other: 'HandResult') -> bool:
        return self.rank == other.rank and self.values == other.values

    def __le__(self, other: 'HandResult') -> bool:
        return self == other or self < other

    def __gt__(self, other: 'HandResult') -> bool:
        return other < self

    def __ge__(self, other: 'HandResult') -> bool:
        return self == other or self > other

    def __repr__(self) -> str:
        cards_str = ' '.join(str(c) for c in self.cards)
        return f"{self.name}: {cards_str}"


def evaluate_five(cards: List[Card]) -> HandResult:
    """Evaluate exactly 5 cards and return the hand rank."""
    ranks = sorted([c.rank for c in cards], reverse=True)
    suits = [c.suit for c in cards]
    rank_counts = Counter(ranks)

    is_flush = len(set(suits)) == 1

    # Check for straight
    is_straight = False
    straight_high = 0
    unique_ranks = sorted(set(ranks), reverse=True)

    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        # Ace-low straight (A-2-3-4-5)
        elif unique_ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            straight_high = 5

    if is_straight and is_flush:
        if straight_high == 14:
            return HandResult(HandRank.ROYAL_FLUSH, (14,), cards)
        return HandResult(HandRank.STRAIGHT_FLUSH, (straight_high,), cards)

    if is_flush:
        return HandResult(HandRank.FLUSH, tuple(ranks), cards)

    if is_straight:
        return HandResult(HandRank.STRAIGHT, (straight_high,), cards)

    counts = sorted(rank_counts.values(), reverse=True)
    by_freq = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    if counts == [4, 1]:
        quad_rank = by_freq[0][0]
        kicker = by_freq[1][0]
        return HandResult(HandRank.FOUR_OF_A_KIND, (quad_rank, kicker), cards)

    if counts == [3, 2]:
        trip_rank = by_freq[0][0]
        pair_rank = by_freq[1][0]
        return HandResult(HandRank.FULL_HOUSE, (trip_rank, pair_rank), cards)

    if counts == [3, 1, 1]:
        trip_rank = by_freq[0][0]
        kickers = sorted([r for r, c in by_freq if c == 1], reverse=True)
        return HandResult(HandRank.THREE_OF_A_KIND, (trip_rank,) + tuple(kickers), cards)

    if counts == [2, 2, 1]:
        pairs = sorted([r for r, c in by_freq if c == 2], reverse=True)
        kicker = [r for r, c in by_freq if c == 1][0]
        return HandResult(HandRank.TWO_PAIR, (pairs[0], pairs[1], kicker), cards)

    if counts == [2, 1, 1, 1]:
        pair_rank = by_freq[0][0]
        kickers = sorted([r for r, c in by_freq if c == 1], reverse=True)
        return HandResult(HandRank.ONE_PAIR, (pair_rank,) + tuple(kickers), cards)

    return HandResult(HandRank.HIGH_CARD, tuple(ranks), cards)


def evaluate_best(cards: List[Card]) -> HandResult:
    """Find the best 5-card hand from any number of cards (5-7)."""
    if len(cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(cards)}")
    if len(cards) == 5:
        return evaluate_five(cards)

    best = None
    for combo in combinations(cards, 5):
        result = evaluate_five(list(combo))
        if best is None or result > best:
            best = result
    return best
