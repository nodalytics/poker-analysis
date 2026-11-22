"""
Card and deck representation for Texas Hold'em poker.
Optimized for fast hand evaluation using integer encoding.
"""
from enum import IntEnum
from typing import List
import random


class Rank(IntEnum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


RANK_CHARS = {
    2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
    9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'
}
CHAR_TO_RANK = {v: k for k, v in RANK_CHARS.items()}

SUIT_CHARS = {0: 'c', 1: 'd', 2: 'h', 3: 's'}
CHAR_TO_SUIT = {v: k for k, v in SUIT_CHARS.items()}


class Card:
    """A playing card encoded as a single integer for fast operations."""
    __slots__ = ['_code']

    def __init__(self, rank: int, suit: int):
        self._code = rank * 4 + suit

    @classmethod
    def from_string(cls, s: str) -> 'Card':
        if len(s) != 2:
            raise ValueError(f"Invalid card string: {s}")
        rank = CHAR_TO_RANK.get(s[0].upper())
        suit = CHAR_TO_SUIT.get(s[1].lower())
        if rank is None or suit is None:
            raise ValueError(f"Invalid card string: {s}")
        return cls(rank, suit)

    @property
    def rank(self) -> int:
        return self._code >> 2

    @property
    def suit(self) -> int:
        return self._code & 3

    @property
    def code(self) -> int:
        return self._code

    def __str__(self) -> str:
        return RANK_CHARS[self.rank] + SUIT_CHARS[self.suit]

    def __repr__(self) -> str:
        return f"Card('{self}')"

    def __eq__(self, other) -> bool:
        if isinstance(other, Card):
            return self._code == other._code
        return NotImplemented

    def __hash__(self) -> int:
        return self._code

    def __lt__(self, other) -> bool:
        if isinstance(other, Card):
            return self.rank < other.rank
        return NotImplemented


class Deck:
    """Standard 52-card deck with efficient dealing."""

    def __init__(self):
        self._cards = [Card(r, s) for r in range(2, 15) for s in range(4)]
        self._index = 0

    def shuffle(self) -> None:
        random.shuffle(self._cards)
        self._index = 0

    def deal(self, n: int = 1) -> List[Card]:
        if self._index + n > 52:
            raise ValueError("Not enough cards in deck")
        cards = self._cards[self._index:self._index + n]
        self._index += n
        return cards

    def deal_one(self) -> Card:
        return self.deal(1)[0]

    def remove(self, cards: List[Card]) -> None:
        codes = {c.code for c in cards}
        self._cards = [c for c in self._cards if c.code not in codes]

    def remaining(self) -> int:
        return len(self._cards) - self._index

    def reset(self) -> None:
        self.__init__()


def parse_hand(s: str) -> List[Card]:
    """Parse a hand string like 'AhKd' into cards."""
    s = s.strip()
    return [Card.from_string(s[i:i+2]) for i in range(0, len(s), 2)]
