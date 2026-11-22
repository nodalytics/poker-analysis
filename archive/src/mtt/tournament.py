"""
MTT (Multi-Table Tournament) Analysis Module
Handles tournament structures, blind levels, stack-depth analysis,
and multi-table tournament modeling.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class BlindLevel:
    """Represents a single blind level in a tournament structure."""
    level: int
    small_blind: int
    big_blind: int
    ante: int = 0
    duration_minutes: int = 15
    
    @property
    def pot_preflop(self) -> int:
        """Starting pot with blinds and antes for a full table."""
        return self.small_blind + self.big_blind + (self.ante * 9)
    
    def effective_bb(self, stack: int) -> float:
        """Calculate effective big blinds for a given stack."""
        if self.big_blind == 0:
            return float('inf')
        return stack / self.big_blind
    
    def m_ratio(self, stack: int, num_players: int = 9) -> float:
        """
        Calculate Harrington's M-ratio.
        M = stack / (SB + BB + antes)
        """
        cost_per_orbit = self.small_blind + self.big_blind + (self.ante * num_players)
        if cost_per_orbit == 0:
            return float('inf')
        return stack / cost_per_orbit


@dataclass
class TournamentStructure:
    """Defines a complete tournament blind structure."""
    name: str
    levels: List[BlindLevel] = field(default_factory=list)
    starting_chips: int = 10000
    players: int = 100
    
    def add_level(self, small_blind: int, big_blind: int, 
                  ante: int = 0, duration: int = 15):
        """Add a blind level to the structure."""
        level_num = len(self.levels) + 1
        self.levels.append(BlindLevel(
            level=level_num,
            small_blind=small_blind,
            big_blind=big_blind,
            ante=ante,
            duration_minutes=duration
        ))
    
    @property
    def total_chips(self) -> int:
        """Total chips in play."""
        return self.starting_chips * self.players
    
    def average_stack_at_level(self, level_idx: int, 
                                players_remaining: int) -> int:
        """Average stack when players_remaining are left."""
        if players_remaining == 0:
            return 0
        return self.total_chips // players_remaining
    
    def stack_depth_at_level(self, level_idx: int,
                              players_remaining: int) -> Dict[str, float]:
        """
        Analyze stack depth at a given level.
        Returns M-ratio and BB count for average stack.
        """
        if level_idx >= len(self.levels):
            level_idx = len(self.levels) - 1
        
        level = self.levels[level_idx]
        avg_stack = self.average_stack_at_level(level_idx, players_remaining)
        
        return {
            'level': level.level,
            'blinds': f'{level.small_blind}/{level.big_blind}',
            'ante': level.ante,
            'avg_stack': avg_stack,
            'avg_bb': level.effective_bb(avg_stack),
            'avg_m': level.m_ratio(avg_stack),
        }


def categorize_stack(bb_count: float) -> Dict[str, str]:
    """
    Categorize a stack size and suggest strategy zone.
    Based on Harrington's zone system.
    
    Args:
        bb_count: Number of big blinds
    
    Returns:
        Dict with zone, color, strategy description
    """
    if bb_count >= 50:
        return {
            'zone': 'Green',
            'status': 'Deep stacked',
            'strategy': 'Full range of plays available. Can speculate with '
                       'small pairs, suited connectors. Position matters most.'
        }
    elif bb_count >= 30:
        return {
            'zone': 'Yellow',
            'status': 'Comfortable',
            'strategy': 'Standard play. Tighten slightly from early position. '
                       'Still room for 3-bet bluffs and speculative hands.'
        }
    elif bb_count >= 20:
        return {
            'zone': 'Orange',
            'status': 'Playable',
            'strategy': 'Reduce speculative plays. Focus on premium hands and '
                       'position. 3-bet or fold preferred over flatting.'
        }
    elif bb_count >= 12:
        return {
            'zone': 'Red',
            'status': 'Short',
            'strategy': 'Push/fold territory approaching. Open-shove with '
                       'wider range. Avoid calling raises without premiums.'
        }
    elif bb_count >= 6:
        return {
            'zone': 'Danger',
            'status': 'Critical',
            'strategy': 'Push/fold mode. Any playable hand is a shove. '
                       'Cannot afford to wait for premiums.'
        }
    else:
        return {
            'zone': 'Dead',
            'status': 'Desperate',
            'strategy': 'Shove any two cards at first opportunity. '
                       'Even a double-up leaves you short.'
        }


def push_fold_range(bb_count: float, position: str) -> str:
    """
    Suggest push/fold ranges based on stack size and position.
    Returns a PokerStove-style range string.
    
    Args:
        bb_count: Number of big blinds
        position: 'utg', 'mp', 'co', 'btn', 'sb'
    
    Returns:
        Range string for eval7 compatibility
    """
    ranges = {
        'utg': {
            15: '77+, ATs+, AJo+, KQs',
            12: '55+, A9s+, ATo+, KJs+, KQo',
            10: '44+, A7s+, A9o+, KTs+, KJo+, QJs',
            8: '33+, A5s+, A8o+, K9s+, KTo+, QTs+, QJo, JTs',
            6: '22+, A2s+, A5o+, K7s+, K9o+, Q9s+, QTo+, J9s+, JTo, T9s',
        },
        'mp': {
            15: '55+, A9s+, ATo+, KJs+, KQo',
            12: '44+, A7s+, A9o+, KTs+, KJo+, QJs',
            10: '33+, A5s+, A8o+, K9s+, KTo+, QTs+, QJo, JTs',
            8: '22+, A3s+, A7o+, K8s+, K9o+, Q9s+, QTo+, J9s+, JTo, T9s',
            6: '22+, A2s+, A3o+, K5s+, K8o+, Q8s+, Q9o+, J8s+, J9o+, T8s+, T9o, 98s',
        },
        'co': {
            15: '44+, A7s+, A9o+, KTs+, KJo+, QJs',
            12: '33+, A5s+, A8o+, K9s+, KTo+, QTs+, QJo, JTs',
            10: '22+, A3s+, A7o+, K8s+, K9o+, Q9s+, QTo+, J9s+, JTo, T9s',
            8: '22+, A2s+, A5o+, K6s+, K8o+, Q8s+, Q9o+, J8s+, J9o+, T8s+, T9o, 98s',
            6: '22+, A2s+, A2o+, K3s+, K7o+, Q6s+, Q8o+, J7s+, J9o+, T7s+, T8o+, 97s+, 98o, 87s',
        },
        'btn': {
            15: '33+, A5s+, A8o+, K9s+, KTo+, QTs+, QJo, JTs',
            12: '22+, A3s+, A7o+, K8s+, K9o+, Q9s+, QTo+, J9s+, JTo, T9s',
            10: '22+, A2s+, A5o+, K6s+, K8o+, Q8s+, Q9o+, J8s+, J9o+, T8s+, T9o, 98s',
            8: '22+, A2s+, A3o+, K4s+, K7o+, Q6s+, Q8o+, J7s+, J9o+, T7s+, T8o+, 97s+, 98o, 87s',
            6: '22+, A2s+, A2o+, K2s+, K5o+, Q4s+, Q7o+, J6s+, J8o+, T6s+, T8o+, 96s+, 97o+, 86s+, 87o, 76s, 75s',
        },
        'sb': {
            15: '22+, A3s+, A7o+, K8s+, K9o+, Q9s+, QTo+, J9s+, JTo, T9s',
            12: '22+, A2s+, A5o+, K6s+, K8o+, Q8s+, Q9o+, J8s+, J9o+, T8s+, T9o, 98s',
            10: '22+, A2s+, A3o+, K4s+, K7o+, Q6s+, Q8o+, J7s+, J9o+, T7s+, T8o+, 97s+, 98o, 87s',
            8: '22+, A2s+, A2o+, K2s+, K5o+, Q4s+, Q7o+, J6s+, J8o+, T6s+, T8o+, 96s+, 97o+, 86s+, 87o, 76s',
            6: '22+, A2s+, A2o+, K2s+, K3o+, Q2s+, Q5o+, J4s+, J7o+, T5s+, T7o+, 95s+, 97o+, 85s+, 86o+, 75s+, 76o, 65s, 64s',
        }
    }
    
    position = position.lower()
    if position not in ranges:
        position = 'btn'
    
    pos_ranges = ranges[position]
    
    # Find the closest BB threshold
    thresholds = sorted(pos_ranges.keys(), reverse=True)
    for threshold in thresholds:
        if bb_count >= threshold:
            return pos_ranges[threshold]
    
    # Below minimum threshold, use the widest range
    return pos_ranges[thresholds[-1]]


def create_standard_structure(name: str = "Standard MTT") -> TournamentStructure:
    """Create a standard MTT blind structure."""
    structure = TournamentStructure(name=name, starting_chips=10000, players=100)
    
    levels = [
        (25, 50, 0), (50, 100, 0), (75, 150, 0),
        (100, 200, 25), (125, 250, 25), (150, 300, 50),
        (200, 400, 50), (250, 500, 75), (300, 600, 75),
        (400, 800, 100), (500, 1000, 125), (600, 1200, 150),
        (800, 1600, 200), (1000, 2000, 250), (1250, 2500, 300),
        (1500, 3000, 400), (2000, 4000, 500), (2500, 5000, 600),
        (3000, 6000, 750), (4000, 8000, 1000), (5000, 10000, 1250),
    ]
    
    for sb, bb, ante in levels:
        structure.add_level(sb, bb, ante)
    
    return structure


def tournament_life_analysis(
    stack: int,
    structure: TournamentStructure,
    current_level: int,
    players_remaining: int
) -> List[Dict]:
    """
    Analyze how long a player can survive without playing a hand.
    Shows urgency at each future level.
    
    Args:
        stack: Current chip count
        structure: Tournament structure
        current_level: Current blind level (0-indexed)
        players_remaining: Players left
    
    Returns:
        List of dicts showing stack erosion over levels
    """
    results = []
    remaining = stack
    
    for i in range(current_level, min(current_level + 10, len(structure.levels))):
        level = structure.levels[i]
        cost_per_orbit = level.small_blind + level.big_blind + (level.ante * 9)
        orbits = remaining / cost_per_orbit if cost_per_orbit > 0 else float('inf')
        bb_count = level.effective_bb(remaining)
        zone = categorize_stack(bb_count)
        
        results.append({
            'level': level.level,
            'blinds': f'{level.small_blind}/{level.big_blind}',
            'ante': level.ante,
            'stack': remaining,
            'bb': round(bb_count, 1),
            'orbits_left': round(orbits, 1),
            'zone': zone['zone'],
            'status': zone['status'],
        })
        
        # Subtract one orbit of blinds/antes
        remaining -= cost_per_orbit
        if remaining <= 0:
            results[-1]['note'] = 'BUSTED - cannot survive this level without winning'
            break
    
    return results
