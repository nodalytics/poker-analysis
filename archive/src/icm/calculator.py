"""
ICM (Independent Chip Model) Calculator
Converts tournament chip stacks into prize money equity.
Essential for final table and bubble decision-making.
"""

from typing import List, Dict, Optional, Tuple
from itertools import permutations


def icm_equity(
    stacks: List[int],
    payouts: List[float]
) -> List[float]:
    """
    Calculate ICM equity for each player given chip stacks and payouts.
    Uses the Malmuth-Harville method.
    
    Args:
        stacks: List of chip counts for each player
        payouts: Prize pool distribution (descending order)
    
    Returns:
        List of dollar equity for each player
    """
    n_players = len(stacks)
    n_payouts = len(payouts)
    total_chips = sum(stacks)
    
    if total_chips == 0:
        return [0.0] * n_players
    
    # Probability each player finishes in each position
    equities = [0.0] * n_players
    
    # Calculate probability of each player getting each payout position
    for player_idx in range(n_players):
        for payout_idx in range(min(n_payouts, n_players)):
            prob = _position_probability(
                stacks, total_chips, player_idx, payout_idx
            )
            equities[player_idx] += prob * payouts[payout_idx]
    
    return [round(eq, 2) for eq in equities]


def _position_probability(
    stacks: List[int],
    total_chips: int,
    player: int,
    position: int,
    eliminated: Optional[List[bool]] = None
) -> float:
    """
    Recursively calculate probability that 'player' finishes in 'position'.
    Position 0 = 1st place, position 1 = 2nd place, etc.
    
    Uses Malmuth-Harville model:
    - P(player wins) = player_chips / total_chips
    - P(player finishes 2nd) = sum over all possible winners w:
        P(w wins) * P(player wins among remaining | w eliminated)
    """
    n_players = len(stacks)
    
    if eliminated is None:
        eliminated = [False] * n_players
    
    if stacks[player] == 0 or eliminated[player]:
        return 0.0
    
    active_chips = sum(s for i, s in enumerate(stacks) if not eliminated[i])
    
    if active_chips == 0:
        return 0.0
    
    # Base case: probability of finishing 1st
    if position == 0:
        return stacks[player] / active_chips
    
    # Recursive case: sum over all possible players who finish before
    prob = 0.0
    for winner in range(n_players):
        if eliminated[winner] or stacks[winner] == 0:
            continue
        if winner == player:
            continue
        
        # Probability this winner takes the previous position
        p_winner = stacks[winner] / active_chips
        
        # Eliminate the winner and recurse
        eliminated[winner] = True
        p_player_given_winner = _position_probability(
            stacks, total_chips, player, position - 1, eliminated
        )
        eliminated[winner] = False
        
        prob += p_winner * p_player_given_winner
    
    return prob


def icm_pressure(
    stacks: List[int],
    payouts: List[float],
    player_idx: int,
    chip_delta: int = None
) -> Dict[str, float]:
    """
    Calculate the ICM pressure on a specific player.
    Shows how much a chip gain vs loss affects dollar equity.
    
    Args:
        stacks: Current chip stacks
        payouts: Prize payouts
        player_idx: Which player to analyze
        chip_delta: Amount to test (default: 10% of player's stack)
    
    Returns:
        Dict with current_equity, equity_if_win, equity_if_loss,
        chip_ev_of_win, chip_ev_of_loss, risk_premium
    """
    if chip_delta is None:
        chip_delta = max(1, stacks[player_idx] // 10)
    
    current = icm_equity(stacks, payouts)
    
    # Simulate winning chips
    win_stacks = stacks.copy()
    # Take chips from the smallest non-zero stack other than player
    others = [(s, i) for i, s in enumerate(stacks) 
              if i != player_idx and s > 0]
    if not others:
        return {
            'current_equity': current[player_idx],
            'equity_if_win': current[player_idx],
            'equity_if_loss': current[player_idx],
            'chip_ev_of_win': 0.0,
            'chip_ev_of_loss': 0.0,
            'risk_premium': 0.0
        }
    
    others.sort()
    victim = others[0][1]
    actual_delta = min(chip_delta, stacks[victim])
    
    win_stacks[player_idx] += actual_delta
    win_stacks[victim] -= actual_delta
    win_eq = icm_equity(win_stacks, payouts)
    
    # Simulate losing chips
    lose_stacks = stacks.copy()
    lose_stacks[player_idx] -= min(actual_delta, stacks[player_idx])
    lose_stacks[victim] += min(actual_delta, stacks[player_idx])
    lose_eq = icm_equity(lose_stacks, payouts)
    
    gain = win_eq[player_idx] - current[player_idx]
    loss = current[player_idx] - lose_eq[player_idx]
    risk_premium = loss - gain if loss > 0 else 0.0
    
    return {
        'current_equity': current[player_idx],
        'equity_if_win': win_eq[player_idx],
        'equity_if_loss': lose_eq[player_idx],
        'chip_ev_of_win': round(gain, 2),
        'chip_ev_of_loss': round(-loss, 2),
        'risk_premium': round(risk_premium, 2)
    }


def bubble_factor(
    stacks: List[int],
    payouts: List[float],
    player_idx: int,
    opponent_idx: int
) -> float:
    """
    Calculate the bubble factor between two players.
    Bubble factor = |ICM loss if busted| / |ICM gain if double up|
    
    A bubble factor > 1 means you need more than 50% equity to call.
    Higher = more pressure to fold.
    
    Args:
        stacks: Chip stacks
        payouts: Payouts
        player_idx: Hero
        opponent_idx: Villain
    
    Returns:
        Bubble factor (typically 1.0 to 5.0+)
    """
    current = icm_equity(stacks, payouts)
    
    hero_stack = stacks[player_idx]
    villain_stack = stacks[opponent_idx]
    at_risk = min(hero_stack, villain_stack)
    
    # Hero wins
    win_stacks = stacks.copy()
    win_stacks[player_idx] += at_risk
    win_stacks[opponent_idx] -= at_risk
    win_eq = icm_equity(win_stacks, payouts)
    
    # Hero loses
    lose_stacks = stacks.copy()
    lose_stacks[player_idx] -= at_risk
    lose_stacks[opponent_idx] += at_risk
    lose_eq = icm_equity(lose_stacks, payouts)
    
    gain = win_eq[player_idx] - current[player_idx]
    loss = current[player_idx] - lose_eq[player_idx]
    
    if gain == 0:
        return float('inf')
    
    return round(loss / gain, 2)


def icm_deal(
    stacks: List[int],
    payouts: List[float],
    player_names: Optional[List[str]] = None
) -> List[Dict[str, any]]:
    """
    Calculate a fair ICM chop deal for remaining players.
    
    Args:
        stacks: Current chip stacks
        payouts: Remaining prize pool payouts
        player_names: Optional player names
    
    Returns:
        List of dicts with player info and their ICM deal amount
    """
    equities = icm_equity(stacks, payouts)
    total_chips = sum(stacks)
    
    results = []
    for i, (stack, equity) in enumerate(zip(stacks, equities)):
        name = player_names[i] if player_names else f'Player {i+1}'
        results.append({
            'name': name,
            'chips': stack,
            'chip_share': round(stack / total_chips * 100, 1) if total_chips > 0 else 0,
            'icm_equity': equity,
            'total_remaining_prizes': sum(payouts),
        })
    
    results.sort(key=lambda x: x['icm_equity'], reverse=True)
    return results
