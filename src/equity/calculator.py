"""
Equity Calculator Module
Uses eval7 for fast Cython-backed equity calculations.
Supports hand vs hand, hand vs range, and multi-way equity.
"""

import eval7
from itertools import combinations
from typing import List, Optional, Tuple, Dict


def card_to_eval7(card_str: str) -> eval7.Card:
    """Convert a card string like 'Ah' to an eval7 Card."""
    return eval7.Card(card_str)


def cards_to_eval7(card_strs: List[str]) -> List[eval7.Card]:
    """Convert a list of card strings to eval7 Cards."""
    return [eval7.Card(s) for s in card_strs]


def hand_vs_hand_equity(
    hand1: List[str],
    hand2: List[str],
    board: Optional[List[str]] = None,
    num_simulations: int = 50000
) -> Dict[str, float]:
    """
    Calculate equity of hand1 vs hand2 with optional board cards.
    
    Args:
        hand1: List of 2 card strings, e.g. ['Ah', 'Kh']
        hand2: List of 2 card strings, e.g. ['Qd', 'Qs']
        board: Optional list of 0-5 board card strings
        num_simulations: Number of Monte Carlo simulations (for preflop/flop)
    
    Returns:
        Dict with 'hand1_equity', 'hand2_equity', 'tie'
    """
    if board is None:
        board = []
    
    h1 = cards_to_eval7(hand1)
    h2 = cards_to_eval7(hand2)
    b = cards_to_eval7(board)
    
    # Dead cards = hand2's cards
    dead = h2
    
    remaining_cards = 5 - len(board)
    
    if remaining_cards <= 2:
        # Use exact enumeration for turn/river
        eq1 = eval7.py_hand_vs_range_exact(
            h1, eval7.HandRange(' '.join(hand2)), b
        )
    else:
        # Use Monte Carlo for preflop/flop
        eq1 = eval7.py_hand_vs_range_monte_carlo(
            h1, eval7.HandRange(' '.join(hand2)), b, num_simulations
        )
    
    return {
        'hand1_equity': round(eq1 * 100, 2),
        'hand2_equity': round((1 - eq1) * 100, 2),
    }


def hand_vs_range_equity(
    hand: List[str],
    range_str: str,
    board: Optional[List[str]] = None,
    num_simulations: int = 50000
) -> float:
    """
    Calculate equity of a hand vs a range.
    
    Args:
        hand: List of 2 card strings, e.g. ['Ah', 'Kh']
        range_str: PokerStove-style range string, e.g. 'QQ+, AKs, AKo'
        board: Optional board cards
        num_simulations: Monte Carlo iterations
    
    Returns:
        Equity as a percentage (0-100)
    """
    if board is None:
        board = []
    
    h = cards_to_eval7(hand)
    b = cards_to_eval7(board)
    villain_range = eval7.HandRange(range_str)
    
    remaining_cards = 5 - len(board)
    
    if remaining_cards <= 2:
        eq = eval7.py_hand_vs_range_exact(h, villain_range, b)
    else:
        eq = eval7.py_hand_vs_range_monte_carlo(
            h, villain_range, b, num_simulations
        )
    
    return round(eq * 100, 2)


def evaluate_hand_strength(
    hand: List[str],
    board: List[str]
) -> Dict[str, any]:
    """
    Evaluate the absolute strength of a hand on a given board.
    
    Args:
        hand: List of 2 hole cards
        board: List of 3-5 board cards
    
    Returns:
        Dict with 'score', 'hand_type', 'best_five' info
    """
    all_cards = cards_to_eval7(hand + board)
    score = eval7.evaluate(all_cards)
    hand_type = eval7.handtype(score)
    
    return {
        'score': score,
        'hand_type': hand_type,
        'num_board_cards': len(board),
    }


def range_equity_table(
    hand: List[str],
    ranges: Dict[str, str],
    board: Optional[List[str]] = None,
    num_simulations: int = 50000
) -> Dict[str, float]:
    """
    Calculate equity of a hand against multiple named ranges.
    Useful for tournament analysis - how does your hand do vs
    different player types?
    
    Args:
        hand: Your hole cards
        ranges: Dict mapping range name to range string
                e.g. {'tight': 'QQ+, AKs', 'loose': '22+, A2s+, KTs+'}
        board: Optional board cards
        num_simulations: Monte Carlo iterations per range
    
    Returns:
        Dict mapping range name to equity percentage
    """
    results = {}
    for name, range_str in ranges.items():
        eq = hand_vs_range_equity(hand, range_str, board, num_simulations)
        results[name] = eq
    return results


def all_hands_equity(
    range_str: str,
    board: Optional[List[str]] = None
) -> List[Tuple[str, float]]:
    """
    Calculate equity for ALL possible hands against a given range.
    Useful for building preflop charts.
    
    Args:
        range_str: Villain's range string
        board: Optional board cards
    
    Returns:
        Sorted list of (hand_string, equity) tuples, best hands first
    """
    if board is None:
        board = []
    
    b = cards_to_eval7(board)
    villain_range = eval7.HandRange(range_str)
    board_set = set(board)
    
    ranks = '23456789TJQKA'
    suits = 'shdc'
    deck = [r + s for r in ranks for s in suits]
    
    # Remove board cards from deck
    available = [c for c in deck if c not in board_set]
    
    results = []
    for i, c1 in enumerate(available):
        for c2 in available[i+1:]:
            h = [eval7.Card(c1), eval7.Card(c2)]
            try:
                eq = eval7.py_hand_vs_range_monte_carlo(
                    h, villain_range, b, 5000
                )
                results.append((f'{c1}{c2}', round(eq * 100, 2)))
            except Exception:
                continue
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results
