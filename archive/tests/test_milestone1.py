"""
Milestone 1 Comprehensive Test Suite
Tests all modules for the poker analysis system.
"""
import sys
import os
import time
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0
ERRORS = []


def test(name, func):
    global PASS, FAIL, ERRORS
    try:
        result = func()
        if result:
            PASS += 1
            print(f"  PASS: {name}")
        else:
            FAIL += 1
            ERRORS.append(f"FAIL: {name} - returned False")
            print(f"  FAIL: {name}")
    except Exception as e:
        FAIL += 1
        ERRORS.append(f"ERROR: {name} - {type(e).__name__}: {e}")
        print(f"  ERROR: {name} - {e}")


# ============================================================
# 1. CARD MODULE TESTS
# ============================================================
print("\n" + "="*60)
print("1. CARD MODULE (src/core/cards.py)")
print("="*60)

def test_card_creation():
    from src.core.cards import Card
    c = Card(14, 3)  # Ace of spades
    return c.rank == 14 and c.suit == 3

def test_card_from_string():
    from src.core.cards import Card
    c = Card.from_string("Ah")
    return c.rank == 14 and c.suit == 2 and str(c) == "Ah"

def test_card_all_suits():
    from src.core.cards import Card
    cards = [Card.from_string(f"A{s}") for s in "cdhs"]
    suits = [c.suit for c in cards]
    return suits == [0, 1, 2, 3]

def test_card_all_ranks():
    from src.core.cards import Card
    for r, ch in [("2",2),("3",3),("T",10),("J",11),("Q",12),("K",13),("A",14)]:
        c = Card.from_string(f"{r}h")
        if c.rank != ch:
            return False
    return True

def test_deck_creation():
    from src.core.cards import Deck
    d = Deck()
    return d.remaining() == 52

def test_deck_deal():
    from src.core.cards import Deck
    d = Deck()
    d.shuffle()
    cards = d.deal(5)
    return len(cards) == 5 and d.remaining() == 47

def test_deck_remove():
    from src.core.cards import Deck, Card
    d = Deck()
    to_remove = [Card.from_string("Ah"), Card.from_string("Kh")]
    d.remove(to_remove)
    return d.remaining() == 50

def test_parse_hand():
    from src.core.cards import parse_hand
    hand = parse_hand("AhKd")
    return len(hand) == 2 and str(hand[0]) == "Ah" and str(hand[1]) == "Kd"

def test_card_invalid():
    from src.core.cards import Card
    try:
        Card.from_string("Xz")
        return False
    except ValueError:
        return True

test("Card creation (Ace of spades)", test_card_creation)
test("Card from string 'Ah'", test_card_from_string)
test("All four suits", test_card_all_suits)
test("All rank characters", test_card_all_ranks)
test("Deck has 52 cards", test_deck_creation)
test("Deal 5 cards", test_deck_deal)
test("Remove cards from deck", test_deck_remove)
test("Parse hand string 'AhKd'", test_parse_hand)
test("Invalid card raises ValueError", test_card_invalid)

# ============================================================
# 2. HAND EVALUATOR TESTS
# ============================================================
print("\n" + "="*60)
print("2. HAND EVALUATOR (src/core/evaluator.py)")
print("="*60)

def test_royal_flush():
    from src.core.cards import Card
    from src.core.evaluator import evaluate_five, HandRank
    cards = [Card.from_string(s) for s in ["Ah","Kh","Qh","Jh","Th"]]
    r = evaluate_five(cards)
    return r.rank == HandRank.ROYAL_FLUSH

def test_straight_flush():
    from src.core.cards import Card
    from src.core.evaluator import evaluate_five, HandRank
    cards = [Card.from_string(s) for s in ["9h","8h","7h","6h","5h"]]
    r = evaluate_five(cards)
    return r.rank == HandRank.STRAIGHT_FLUSH

def test_four_of_a_kind():
    from src.core.cards import Card
    from src.core.evaluator import evaluate_five, HandRank
    cards = [Card.from_string(s) for s in ["Ah","Ad","Ac","As","Kh"]]
    r = evaluate_five(cards)
    return r.rank == HandRank.FOUR_OF_A_KIND

def test_full_house():
    from src.core.cards import Card
    from src.core.evaluator import evaluate_five, HandRank
    cards = [Card.from_string(s) for s in ["Ah","Ad","Ac","Kh","Kd"]]
    r = evaluate_five(cards)
    return r.rank == HandRank.FULL_HOUSE

def test_flush():
    from src.core.cards import Card
    from src.core.evaluator import evaluate_five, HandRank
    cards = [Card.from_string(s) for s in ["Ah","Kh","9h","5h","2h"]]
    r = evaluate_five(cards)
    return r.rank == HandRank.FLUSH

def test_straight():
    from src.core.cards import Card
    from src.core.evaluator import evaluate_five, HandRank
    cards = [Card.from_string(s) for s in ["9h","8d","7c","6s","5h"]]
    r = evaluate_five(cards)
    return r.rank == HandRank.STRAIGHT

def test_ace_low_straight():
    from src.core.cards import Card
    from src.core.evaluator import evaluate_five, HandRank
    cards = [Card.from_string(s) for s in ["Ah","2d","3c","4s","5h"]]
    r = evaluate_five(cards)
    return r.rank == HandRank.STRAIGHT and r.values == (5,)

def test_three_of_a_kind():
    from src.core.cards import Card
    from src.core.evaluator import evaluate_five, HandRank
    cards = [Card.from_string(s) for s in ["Ah","Ad","Ac","Kh","9d"]]
    r = evaluate_five(cards)
    return r.rank == HandRank.THREE_OF_A_KIND

def test_two_pair():
    from src.core.cards import Card
    from src.core.evaluator import evaluate_five, HandRank
    cards = [Card.from_string(s) for s in ["Ah","Ad","Kh","Kd","9c"]]
    r = evaluate_five(cards)
    return r.rank == HandRank.TWO_PAIR

def test_one_pair():
    from src.core.cards import Card
    from src.core.evaluator import evaluate_five, HandRank
    cards = [Card.from_string(s) for s in ["Ah","Ad","Kh","9d","5c"]]
    r = evaluate_five(cards)
    return r.rank == HandRank.ONE_PAIR

def test_high_card():
    from src.core.cards import Card
    from src.core.evaluator import evaluate_five, HandRank
    cards = [Card.from_string(s) for s in ["Ah","Kd","9h","5c","2s"]]
    r = evaluate_five(cards)
    return r.rank == HandRank.HIGH_CARD

def test_evaluate_best_7_cards():
    from src.core.cards import Card
    from src.core.evaluator import evaluate_best, HandRank
    # 7 cards where best 5 make a flush
    cards = [Card.from_string(s) for s in ["Ah","Kh","9h","5h","2h","3d","7c"]]
    r = evaluate_best(cards)
    return r.rank == HandRank.FLUSH

def test_hand_comparison():
    from src.core.cards import Card
    from src.core.evaluator import evaluate_five
    flush = [Card.from_string(s) for s in ["Ah","Kh","9h","5h","2h"]]
    pair = [Card.from_string(s) for s in ["Ah","Ad","Kh","9d","5c"]]
    r_flush = evaluate_five(flush)
    r_pair = evaluate_five(pair)
    return r_flush > r_pair

def test_hand_result_name():
    from src.core.cards import Card
    from src.core.evaluator import evaluate_five
    cards = [Card.from_string(s) for s in ["Ah","Kh","Qh","Jh","Th"]]
    r = evaluate_five(cards)
    return r.name == "Royal Flush"

test("Royal Flush", test_royal_flush)
test("Straight Flush", test_straight_flush)
test("Four of a Kind", test_four_of_a_kind)
test("Full House", test_full_house)
test("Flush", test_flush)
test("Straight", test_straight)
test("Ace-low straight (wheel)", test_ace_low_straight)
test("Three of a Kind", test_three_of_a_kind)
test("Two Pair", test_two_pair)
test("One Pair", test_one_pair)
test("High Card", test_high_card)
test("Best 5 from 7 cards", test_evaluate_best_7_cards)
test("Hand comparison (flush > pair)", test_hand_comparison)
test("HandResult.name property", test_hand_result_name)

# ============================================================
# 3. EQUITY CALCULATOR TESTS
# ============================================================
print("\n" + "="*60)
print("3. EQUITY CALCULATOR (src/equity/calculator.py)")
print("="*60)

def test_eval7_import():
    import eval7
    return True

def test_hand_vs_hand_preflop():
    from src.equity.calculator import hand_vs_hand_equity
    # AA vs KK - AA should win ~80%
    result = hand_vs_hand_equity(['Ah', 'Ad'], ['Kh', 'Kd'])
    eq = result['hand1_equity']
    return 75 < eq < 90

def test_hand_vs_hand_river():
    from src.equity.calculator import hand_vs_hand_equity
    # Nut flush vs top pair on completed board
    result = hand_vs_hand_equity(
        ['Ah', 'Kh'],
        ['Qs', 'Jd'],
        ['Qh', '9h', '3h', '2d', '7c']
    )
    return result['hand1_equity'] == 100.0

def test_hand_vs_range():
    from src.equity.calculator import hand_vs_range_equity
    # AA vs top 10% range
    eq = hand_vs_range_equity(['Ah', 'Ad'], 'TT+, AJs+, AQo+, KQs')
    return 60 < eq < 90

def test_evaluate_hand_strength():
    from src.equity.calculator import evaluate_hand_strength
    result = evaluate_hand_strength(['Ah', 'Kh'], ['Qh', 'Jh', 'Th'])
    return result['hand_type'] == 'Straight Flush'

def test_range_equity_table():
    from src.equity.calculator import range_equity_table
    ranges = {
        'tight': 'QQ+, AKs',
        'medium': '88+, ATs+, AJo+, KQs',
    }
    result = range_equity_table(['Ah', 'Ad'], ranges)
    return 'tight' in result and 'medium' in result

test("eval7 imports", test_eval7_import)
test("AA vs KK preflop (~80%)", test_hand_vs_hand_preflop)
test("Nut flush vs pair on river (100%)", test_hand_vs_hand_river)
test("AA vs top 10% range", test_hand_vs_range)
test("Hand strength evaluation", test_evaluate_hand_strength)
test("Range equity table", test_range_equity_table)

# ============================================================
# 4. ICM CALCULATOR TESTS
# ============================================================
print("\n" + "="*60)
print("4. ICM CALCULATOR (src/icm/calculator.py)")
print("="*60)

def test_icm_equal_stacks():
    from src.icm.calculator import icm_equity
    # 3 equal stacks, total prize 300
    equities = icm_equity([1000, 1000, 1000], [150, 100, 50])
    # Each should get 100 (equal share of 300)
    return all(abs(e - 100.0) < 0.01 for e in equities)

def test_icm_chip_leader():
    from src.icm.calculator import icm_equity
    # Big stack should have most equity
    equities = icm_equity([5000, 3000, 2000], [150, 100, 50])
    return equities[0] > equities[1] > equities[2]

def test_icm_zero_stack():
    from src.icm.calculator import icm_equity
    equities = icm_equity([5000, 3000, 0], [150, 100, 50])
    return equities[2] == 0.0

def test_icm_sum_equals_prizes():
    from src.icm.calculator import icm_equity
    payouts = [1000, 600, 400, 200]
    equities = icm_equity([5000, 4000, 3000, 2000], payouts)
    total = sum(equities)
    return abs(total - sum(payouts)) < 1.0

def test_bubble_factor():
    from src.icm.calculator import bubble_factor
    bf = bubble_factor([5000, 3000, 2000, 1000], [1000, 600, 400, 200], 3, 0)
    # Short stack vs big stack on bubble should have high BF
    return bf > 1.0

def test_bubble_factor_chip_leader():
    from src.icm.calculator import bubble_factor
    bf = bubble_factor([5000, 3000, 2000, 1000], [1000, 600, 400, 200], 0, 3)
    # Chip leader has less ICM pressure
    return bf >= 1.0

def test_icm_pressure():
    from src.icm.calculator import icm_pressure
    result = icm_pressure([5000, 3000, 2000], [150, 100, 50], 0)
    return ('current_equity' in result and 'risk_premium' in result
            and result['current_equity'] > 0)

def test_icm_deal():
    from src.icm.calculator import icm_deal
    deal = icm_deal([5000, 3000, 2000], [150, 100, 50], ['Alice', 'Bob', 'Carol'])
    return len(deal) == 3 and all('icm_equity' in d for d in deal)

test("ICM with equal stacks", test_icm_equal_stacks)
test("ICM chip leader has highest equity", test_icm_chip_leader)
test("ICM zero-stack player gets $0", test_icm_zero_stack)
test("ICM equities sum to total prizes", test_icm_sum_equals_prizes)
test("Bubble factor > 1 for short stack", test_bubble_factor)
test("Bubble factor for chip leader", test_bubble_factor_chip_leader)
test("ICM pressure calculation", test_icm_pressure)
test("ICM deal calculation", test_icm_deal)

# ============================================================
# 5. MTT TOURNAMENT MODULE TESTS
# ============================================================
print("\n" + "="*60)
print("5. MTT MODULE (src/mtt/tournament.py)")
print("="*60)

def test_blind_level():
    from src.mtt.tournament import BlindLevel
    bl = BlindLevel(level=1, small_blind=25, big_blind=50, ante=0)
    return bl.effective_bb(5000) == 100.0

def test_blind_level_m_ratio():
    from src.mtt.tournament import BlindLevel
    bl = BlindLevel(level=5, small_blind=100, big_blind=200, ante=25)
    m = bl.m_ratio(5000, 9)
    expected = 5000 / (100 + 200 + 25*9)  # 5000/525 = 9.52
    return abs(m - expected) < 0.01

def test_standard_structure():
    from src.mtt.tournament import create_standard_structure
    s = create_standard_structure()
    return len(s.levels) == 21 and s.starting_chips == 10000 and s.players == 100

def test_categorize_stack_green():
    from src.mtt.tournament import categorize_stack
    result = categorize_stack(60)
    return result['zone'] == 'Green'

def test_categorize_stack_danger():
    from src.mtt.tournament import categorize_stack
    result = categorize_stack(8)
    return result['zone'] == 'Danger'

def test_categorize_stack_dead():
    from src.mtt.tournament import categorize_stack
    result = categorize_stack(3)
    return result['zone'] == 'Dead'

def test_push_fold_range():
    from src.mtt.tournament import push_fold_range
    r = push_fold_range(10, 'btn')
    return len(r) > 0 and '22+' in r

def test_tournament_life():
    from src.mtt.tournament import tournament_life_analysis, create_standard_structure
    s = create_standard_structure()
    results = tournament_life_analysis(10000, s, 0, 100)
    return len(results) > 0 and results[0]['zone'] in ['Green','Yellow','Orange','Red','Danger','Dead']

def test_stack_depth():
    from src.mtt.tournament import create_standard_structure
    s = create_standard_structure()
    depth = s.stack_depth_at_level(0, 100)
    return 'avg_bb' in depth and 'avg_m' in depth

test("Blind level effective BB", test_blind_level)
test("M-ratio calculation", test_blind_level_m_ratio)
test("Standard 21-level structure", test_standard_structure)
test("Green zone (60bb)", test_categorize_stack_green)
test("Danger zone (8bb)", test_categorize_stack_danger)
test("Dead zone (3bb)", test_categorize_stack_dead)
test("Push/fold range (10bb BTN)", test_push_fold_range)
test("Tournament life analysis", test_tournament_life)
test("Stack depth at level", test_stack_depth)

# ============================================================
# 6. TOURNAMENT ANALYZER TESTS
# ============================================================
print("\n" + "="*60)
print("6. TOURNAMENT ANALYZER (src/analysis/analyzer.py)")
print("="*60)

def test_analyzer_creation():
    from src.analysis.analyzer import TournamentAnalyzer
    from src.mtt.tournament import create_standard_structure
    a = TournamentAnalyzer(create_standard_structure())
    return a.structure is not None

def test_analyze_spot():
    from src.analysis.analyzer import TournamentAnalyzer
    from src.mtt.tournament import BlindLevel
    a = TournamentAnalyzer()
    result = a.analyze_spot(
        hero_hand=['Ah', 'Kh'],
        villain_range='QQ+, AKs',
        hero_stack=50000,
        villain_stack=30000,
        all_stacks=[50000, 30000, 20000],
        payouts=[1000, 600, 400],
        blind_level=BlindLevel(level=5, small_blind=200, big_blind=400, ante=50)
    )
    return ('card_equity' in result and 'bubble_factor' in result
            and 'decision' in result and result['decision'] in ['CALL/PUSH', 'FOLD'])

def test_push_fold_analysis():
    from src.analysis.analyzer import TournamentAnalyzer
    from src.mtt.tournament import BlindLevel
    a = TournamentAnalyzer()
    result = a.analyze_push_fold(
        hero_stack=8000,
        all_stacks=[8000, 30000, 20000, 15000],
        payouts=[1000, 600, 400, 200],
        blind_level=BlindLevel(level=8, small_blind=300, big_blind=600, ante=75),
        position='btn'
    )
    return ('zone' in result and 'suggested_push_range' in result
            and 'bubble_factor' in result)

def test_final_table_report():
    from src.analysis.analyzer import TournamentAnalyzer
    from src.mtt.tournament import BlindLevel
    a = TournamentAnalyzer()
    result = a.final_table_report(
        stacks=[50000, 35000, 25000, 15000],
        payouts=[1000, 600, 400, 200],
        player_names=['Alice', 'Bob', 'Carol', 'Dave'],
        blind_level=BlindLevel(level=10, small_blind=500, big_blind=1000, ante=125)
    )
    return ('players' in result and 'bubble_factors' in result
            and 'icm_deal' in result and len(result['players']) == 4)

def test_scenario_compare():
    from src.analysis.analyzer import TournamentAnalyzer
    a = TournamentAnalyzer()
    scenarios = [
        {'name': 'Bubble', 'villain_range': 'TT+, AQs+',
         'stacks': [30000, 20000, 15000, 10000], 'payouts': [1000, 600, 400]},
        {'name': 'Final 3', 'villain_range': '77+, ATs+, KQs',
         'stacks': [40000, 20000, 15000], 'payouts': [1000, 600, 400]},
    ]
    results = a.scenario_compare(['Ah', 'Kh'], scenarios)
    return len(results) == 2 and all('card_equity' in r for r in results)

test("Analyzer creation", test_analyzer_creation)
test("Analyze spot (full analysis)", test_analyze_spot)
test("Push/fold analysis", test_push_fold_analysis)
test("Final table report", test_final_table_report)
test("Scenario comparison", test_scenario_compare)

# ============================================================
# 7. SOLVER WRAPPER TESTS
# ============================================================
print("\n" + "="*60)
print("7. SOLVER WRAPPER (src/solver/wrapper.py)")
print("="*60)

def test_solver_binary_exists():
    path = os.path.expanduser("~/poker-analysis/tools/TexasSolver/build/console_solver")
    return os.path.exists(path) and os.access(path, os.X_OK)

def test_solver_resources_exist():
    path = os.path.expanduser("~/poker-analysis/tools/TexasSolver/resources")
    return os.path.isdir(path)

def test_wrapper_init():
    from src.solver.wrapper import TexasSolverWrapper
    solver = TexasSolverWrapper()
    return os.path.exists(solver.solver_path)

def test_config_generation():
    from src.solver.wrapper import SolverConfig
    config = SolverConfig()
    config.set_spot(10, 95, "Qs,Jh,2h", "AA,KK,QQ", "TT,99,88")
    config.set_default_bet_sizes()
    output = config.generate_config("/tmp/test.json")
    return "set_pot 10" in output and "set_board Qs,Jh,2h" in output and "start_solve" in output

test("Solver binary exists and executable", test_solver_binary_exists)
test("Solver resources directory exists", test_solver_resources_exist)
test("Wrapper initialization", test_wrapper_init)
test("Config file generation", test_config_generation)

# ============================================================
# 8. MODULE INTEGRATION TEST
# ============================================================
print("\n" + "="*60)
print("8. INTEGRATION TESTS")
print("="*60)

def test_all_imports():
    from src.core.cards import Card, Deck, parse_hand
    from src.core.evaluator import evaluate_five, evaluate_best, HandRank, HandResult
    from src.equity.calculator import hand_vs_hand_equity, hand_vs_range_equity, evaluate_hand_strength
    from src.icm.calculator import icm_equity, icm_pressure, bubble_factor, icm_deal
    from src.mtt.tournament import BlindLevel, TournamentStructure, categorize_stack, push_fold_range
    from src.analysis.analyzer import TournamentAnalyzer
    from src.solver.wrapper import TexasSolverWrapper, SolverConfig
    return True

def test_full_pipeline():
    """End-to-end: cards -> evaluation -> equity -> ICM -> analysis"""
    from src.core.cards import Card, parse_hand
    from src.core.evaluator import evaluate_best
    from src.equity.calculator import hand_vs_range_equity
    from src.icm.calculator import icm_equity, bubble_factor
    from src.mtt.tournament import BlindLevel, categorize_stack
    from src.analysis.analyzer import TournamentAnalyzer

    # Step 1: Parse cards
    hand = parse_hand("AhKh")
    board = parse_hand("Qh9h3h")
    assert len(hand) == 2
    assert len(board) == 3

    # Step 2: Evaluate hand
    all_cards = hand + board
    result = evaluate_best(all_cards)
    assert result.name == "Flush"

    # Step 3: Get equity vs range
    eq = hand_vs_range_equity(['Ah', 'Kh'], 'JJ, TT, KQs, AJo', ['Qh', '9h', '3h'])
    assert 20 < eq < 100  # Has flush but range can improve on turn/river

    # Step 4: ICM calculation
    stacks = [50000, 30000, 20000, 10000]
    payouts = [1000, 600, 400, 200]
    equities = icm_equity(stacks, payouts)
    assert abs(sum(equities) - 2200) < 1.0

    # Step 5: Full analyzer
    analyzer = TournamentAnalyzer()
    blind = BlindLevel(level=8, small_blind=300, big_blind=600, ante=75)
    spot = analyzer.analyze_spot(
        hero_hand=['Ah', 'Kh'],
        villain_range='QQ, JJ, AQs',
        hero_stack=50000,
        villain_stack=30000,
        all_stacks=stacks,
        payouts=payouts,
        board=['Qh', '9h', '3d'],
        blind_level=blind
    )
    assert 'decision' in spot
    assert 'card_equity' in spot
    assert 'bubble_factor' in spot
    return True

test("All modules import cleanly", test_all_imports)
test("Full pipeline: cards -> eval -> equity -> ICM -> analysis", test_full_pipeline)

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "="*60)
print("MILESTONE 1 TEST RESULTS")
print("="*60)
print(f"\n  Total tests: {PASS + FAIL}")
print(f"  Passed:      {PASS}")
print(f"  Failed:      {FAIL}")
print(f"  Pass rate:   {PASS/(PASS+FAIL)*100:.1f}%")

if ERRORS:
    print(f"\n  FAILURES AND ERRORS:")
    for e in ERRORS:
        print(f"    - {e}")

if FAIL == 0:
    print("\n  VERDICT: ALL TESTS PASSED - Milestone 1 is COMPLETE")
else:
    print(f"\n  VERDICT: {FAIL} ISSUE(S) FOUND - Review required")

print("="*60)