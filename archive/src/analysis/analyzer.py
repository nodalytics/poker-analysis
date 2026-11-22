"""
Tournament Analysis Module
Ties together equity, ICM, and MTT modules for complete
tournament hand and scenario analysis.
"""

from typing import List, Dict, Optional, Tuple
from src.equity.calculator import (
    hand_vs_hand_equity, hand_vs_range_equity,
    evaluate_hand_strength, range_equity_table
)
from src.icm.calculator import (
    icm_equity, icm_pressure, bubble_factor, icm_deal
)
from src.mtt.tournament import (
    BlindLevel, TournamentStructure, categorize_stack,
    push_fold_range, tournament_life_analysis
)


class TournamentAnalyzer:
    """
    Main analysis engine that combines equity, ICM, and MTT
    for comprehensive tournament decision support.
    """
    
    def __init__(self, structure: Optional[TournamentStructure] = None):
        self.structure = structure
    
    def analyze_spot(
        self,
        hero_hand: List[str],
        villain_range: str,
        hero_stack: int,
        villain_stack: int,
        all_stacks: List[int],
        payouts: List[float],
        board: Optional[List[str]] = None,
        blind_level: Optional[BlindLevel] = None,
        position: str = 'btn'
    ) -> Dict:
        """
        Complete analysis of a tournament spot.
        Combines card equity, ICM equity, and stack depth.
        
        Args:
            hero_hand: Hero's 2 hole cards
            villain_range: Villain's range string
            hero_stack: Hero's chips
            villain_stack: Villain's chips
            all_stacks: All players' chip stacks
            payouts: Tournament payout structure
            board: Board cards (empty for preflop)
            blind_level: Current blind level
            position: Hero's position
        
        Returns:
            Comprehensive analysis dict
        """
        if board is None:
            board = []
        
        # 1. Card equity
        card_equity = hand_vs_range_equity(hero_hand, villain_range, board)
        
        # 2. ICM analysis
        hero_idx = 0  # Assume hero is first in stacks list
        current_icm = icm_equity(all_stacks, payouts)
        bf = bubble_factor(all_stacks, payouts, hero_idx, 
                          all_stacks.index(villain_stack) if villain_stack in all_stacks else 1)
        pressure = icm_pressure(all_stacks, payouts, hero_idx)
        
        # 3. Required equity (based on bubble factor)
        # With bubble factor B, you need equity > B / (1 + B) to call
        required_equity = (bf / (1 + bf)) * 100
        
        # 4. Stack depth analysis
        bb_count = None
        stack_zone = None
        if blind_level:
            bb_count = blind_level.effective_bb(hero_stack)
            stack_zone = categorize_stack(bb_count)
        
        # 5. Decision
        ev_positive = card_equity > required_equity
        
        result = {
            'hand': ' '.join(hero_hand),
            'villain_range': villain_range,
            'board': ' '.join(board) if board else 'preflop',
            'card_equity': card_equity,
            'required_equity': round(required_equity, 2),
            'bubble_factor': bf,
            'icm_equity': current_icm[hero_idx],
            'icm_pressure': pressure,
            'decision': 'CALL/PUSH' if ev_positive else 'FOLD',
            'ev_positive': ev_positive,
            'equity_surplus': round(card_equity - required_equity, 2),
        }
        
        if bb_count is not None:
            result['bb_count'] = round(bb_count, 1)
            result['stack_zone'] = stack_zone
        
        return result
    
    def analyze_push_fold(
        self,
        hero_stack: int,
        all_stacks: List[int],
        payouts: List[float],
        blind_level: BlindLevel,
        position: str,
        villain_calling_range: str = '22+, A2s+, A9o+, KTs+, KQo, QJs'
    ) -> Dict:
        """
        Analyze whether hero should push or fold.
        Used for short-stack tournament decisions.
        
        Args:
            hero_stack: Hero's chips
            all_stacks: All chip stacks
            payouts: Tournament payouts
            blind_level: Current blind level
            position: Hero's position
            villain_calling_range: Expected calling range
        
        Returns:
            Push/fold analysis with recommended range
        """
        bb_count = blind_level.effective_bb(hero_stack)
        zone = categorize_stack(bb_count)
        suggested_range = push_fold_range(bb_count, position)
        m_ratio = blind_level.m_ratio(hero_stack)
        
        hero_idx = 0
        bf = bubble_factor(all_stacks, payouts, hero_idx, 1)
        required_equity = (bf / (1 + bf)) * 100
        
        return {
            'stack': hero_stack,
            'bb_count': round(bb_count, 1),
            'm_ratio': round(m_ratio, 1),
            'zone': zone,
            'position': position,
            'bubble_factor': bf,
            'required_equity': round(required_equity, 2),
            'suggested_push_range': suggested_range,
            'villain_calling_range': villain_calling_range,
            'action': 'PUSH' if bb_count <= 15 else 'OPEN RAISE or PUSH',
        }
    
    def final_table_report(
        self,
        stacks: List[int],
        payouts: List[float],
        player_names: Optional[List[str]] = None,
        blind_level: Optional[BlindLevel] = None
    ) -> Dict:
        """
        Generate a complete final table analysis report.
        
        Args:
            stacks: All player chip stacks
            payouts: Prize payouts
            player_names: Optional player names
            blind_level: Current blind level
        
        Returns:
            Comprehensive final table report
        """
        n = len(stacks)
        if player_names is None:
            player_names = [f'Player {i+1}' for i in range(n)]
        
        # ICM equities
        equities = icm_equity(stacks, payouts)
        total_chips = sum(stacks)
        
        # Player details
        players = []
        for i in range(n):
            player = {
                'name': player_names[i],
                'chips': stacks[i],
                'chip_pct': round(stacks[i] / total_chips * 100, 1),
                'icm_equity': equities[i],
                'icm_pct': round(equities[i] / sum(payouts) * 100, 1),
            }
            
            if blind_level:
                bb = blind_level.effective_bb(stacks[i])
                player['bb_count'] = round(bb, 1)
                player['zone'] = categorize_stack(bb)
                player['m_ratio'] = round(blind_level.m_ratio(stacks[i]), 1)
            
            players.append(player)
        
        # Bubble factors matrix (each pair of players)
        bf_matrix = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    bf = bubble_factor(stacks, payouts, i, j)
                    bf_matrix[f'{player_names[i]}_vs_{player_names[j]}'] = bf
        
        # ICM deal
        deal = icm_deal(stacks, payouts, player_names)
        
        players.sort(key=lambda x: x['chips'], reverse=True)
        
        return {
            'num_players': n,
            'total_chips': total_chips,
            'total_prizes': sum(payouts),
            'players': players,
            'bubble_factors': bf_matrix,
            'icm_deal': deal,
        }
    
    def scenario_compare(
        self,
        hero_hand: List[str],
        scenarios: List[Dict],
        board: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Compare multiple tournament scenarios.
        Useful for "what if" analysis.
        
        Args:
            hero_hand: Hero's cards
            scenarios: List of dicts with 'name', 'villain_range', 
                      'stacks', 'payouts'
            board: Optional board cards
        
        Returns:
            List of analysis results for each scenario
        """
        results = []
        for scenario in scenarios:
            eq = hand_vs_range_equity(
                hero_hand, 
                scenario['villain_range'],
                board
            )
            icm = icm_equity(scenario['stacks'], scenario['payouts'])
            
            results.append({
                'name': scenario['name'],
                'card_equity': eq,
                'icm_equities': icm,
                'total_prizes': sum(scenario['payouts']),
            })
        
        return results
