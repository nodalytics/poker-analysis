#!/usr/bin/env python3
"""
Poker Tournament Analysis Tool — Interactive Preview
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.cards import parse_hand
from src.core.evaluator import evaluate_best
from src.equity.calculator import hand_vs_hand_equity, hand_vs_range_equity
from src.icm.calculator import icm_equity, bubble_factor
from src.mtt.tournament import BlindLevel, categorize_stack, push_fold_range
from src.analysis.analyzer import TournamentAnalyzer

analyzer = TournamentAnalyzer()

def clear():
    os.system('clear')

def get_input(prompt, default=None):
    if default:
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else default
    return input(f"  {prompt}: ").strip()

def show_menu():
    clear()
    print("="*60)
    print("   POKER TOURNAMENT ANALYSIS TOOL")
    print("="*60)
    print()
    print("  1. Hand vs Hand Equity")
    print("  2. Hand vs Range Equity")
    print("  3. ICM Calculator")
    print("  4. Tournament Spot Analysis")
    print("  5. Push/Fold Helper")
    print("  6. Final Table Report")
    print("  0. Exit")
    print()
    return input("  Choose (0-6): ").strip()

def hand_vs_hand():
    clear()
    print("="*60)
    print("   HAND vs HAND EQUITY")
    print("="*60)
    print("  Enter cards like: AhKd, QsQc, TdTh")
    print()
    h1 = get_input("Your hand (e.g. AhKd)")
    h2 = get_input("Opponent hand (e.g. QsQc)")
    board_str = get_input("Board cards (e.g. Qh9h3d) or leave blank", "")

    hand1 = [h1[i:i+2] for i in range(0, len(h1), 2)]
    hand2 = [h2[i:i+2] for i in range(0, len(h2), 2)]
    board = [board_str[i:i+2] for i in range(0, len(board_str), 2)] if board_str else None

    print("\n  Calculating...")
    result = hand_vs_hand_equity(hand1, hand2, board)

    print(f"\n  {''.join(hand1):8s}  =>  {result['hand1_equity']:.1f}%")
    print(f"  {''.join(hand2):8s}  =>  {result['hand2_equity']:.1f}%")
    if result.get('ties'):
        print(f"  Ties:       {result['ties']:.1f}%")

    if board:
        all_cards = parse_hand(h1) + parse_hand(board_str)
        if len(all_cards) >= 5:
            best = evaluate_best(all_cards)
            print(f"\n  Your best hand: {best.name}")

def hand_vs_range():
    clear()
    print("="*60)
    print("   HAND vs RANGE EQUITY")
    print("="*60)
    print("  Ranges: TT+, AJs+, AQo+, KQs, 22+, etc.")
    print()
    h = get_input("Your hand (e.g. AhKh)")
    r = get_input("Opponent range (e.g. TT+, AJs+, AQo+)")
    board_str = get_input("Board cards or leave blank", "")

    hand = [h[i:i+2] for i in range(0, len(h), 2)]
    board = [board_str[i:i+2] for i in range(0, len(board_str), 2)] if board_str else None

    print("\n  Calculating...")
    eq = hand_vs_range_equity(hand, r, board)
    print(f"\n  {''.join(hand)} vs [{r}]")
    print(f"  Your equity: {eq:.1f}%")

def icm_calc():
    clear()
    print("="*60)
    print("   ICM CALCULATOR")
    print("="*60)
    print()
    n = int(get_input("How many players", "4"))
    print()

    stacks = []
    names = []
    for i in range(n):
        name = get_input(f"Player {i+1} name", f"Player {i+1}")
        stack = int(get_input(f"  {name}'s chip stack"))
        names.append(name)
        stacks.append(stack)

    print()
    payouts = []
    for i in range(n):
        p = float(get_input(f"  {i+1}{'st' if i==0 else 'nd' if i==1 else 'rd' if i==2 else 'th'} place prize ($)"))
        payouts.append(p)

    equities = icm_equity(stacks, payouts)

    print(f"\n  {'='*50}")
    print(f"  {'Player':12s} {'Chips':>8s} {'Chip%':>6s} {'ICM $':>10s}")
    print(f"  {'-'*50}")
    total_chips = sum(stacks)
    for name, stack, eq in zip(names, stacks, equities):
        pct = stack / total_chips * 100
        print(f"  {name:12s} {stack:>8,} {pct:>5.1f}% ${eq:>9,.2f}")
    print(f"  {'-'*50}")
    print(f"  {'Total':12s} {total_chips:>8,}       ${sum(equities):>9,.2f}")

def spot_analysis():
    clear()
    print("="*60)
    print("   TOURNAMENT SPOT ANALYSIS")
    print("="*60)
    print()
    hero = get_input("Your hand (e.g. AhKs)")
    villain_range = get_input("Villain range (e.g. TT+, AQs+)")
    board_str = get_input("Board cards or leave blank", "")

    hero_stack = int(get_input("Your stack"))
    villain_stack = int(get_input("Villain stack"))

    print("\n  Other players at table (enter stacks comma-separated):")
    others = get_input("Other stacks (e.g. 15000,8000) or blank", "")
    all_stacks = [hero_stack, villain_stack]
    if others:
        all_stacks += [int(x.strip()) for x in others.split(",")]

    print("\n  Payouts (comma-separated, e.g. 1000,600,400):")
    pay_str = get_input("Payouts")
    payouts = [float(x.strip()) for x in pay_str.split(",")]

    sb = int(get_input("Small blind", "200"))
    bb = int(get_input("Big blind", "400"))
    ante = int(get_input("Ante", "50"))

    blind = BlindLevel(level=1, small_blind=sb, big_blind=bb, ante=ante)
    hand = [hero[i:i+2] for i in range(0, len(hero), 2)]
    board = [board_str[i:i+2] for i in range(0, len(board_str), 2)] if board_str else None

    print("\n  Analyzing...")
    spot = analyzer.analyze_spot(
        hero_hand=hand,
        villain_range=villain_range,
        hero_stack=hero_stack,
        villain_stack=villain_stack,
        all_stacks=all_stacks,
        payouts=payouts,
        board=board,
        blind_level=blind
    )

    zone = spot['stack_zone']
    zone_name = zone['zone'] if isinstance(zone, dict) else zone

    print(f"\n  {'='*50}")
    print(f"  ANALYSIS RESULT")
    print(f"  {'='*50}")
    print(f"  Card Equity:      {spot['card_equity']:.1f}%")
    print(f"  Bubble Factor:    {spot['bubble_factor']:.2f}x")
    print(f"  Required Equity:  {spot['required_equity']:.1f}%")
    print(f"  Stack Zone:       {zone_name}")
    print(f"  BB Count:         {spot['bb_count']:.0f}")
    print(f"  EV Positive:      {spot['ev_positive']}")
    print(f"\n  >>> DECISION: {spot['decision']} <<<")

def push_fold():
    clear()
    print("="*60)
    print("   PUSH/FOLD HELPER")
    print("="*60)
    print()
    stack = int(get_input("Your stack"))
    bb = int(get_input("Big blind"))
    bb_count = stack / bb

    zone = categorize_stack(bb_count)
    print(f"\n  Stack: {stack:,} chips = {bb_count:.1f} BB")
    print(f"  Zone: {zone['zone']} — {zone['strategy'][:60]}")

    print(f"\n  Push ranges by position:")
    for pos in ['btn', 'co', 'sb']:
        r = push_fold_range(bb_count, pos)
        print(f"  {pos.upper():3s}: {r}")

def final_table():
    clear()
    print("="*60)
    print("   FINAL TABLE REPORT")
    print("="*60)
    print()
    n = int(get_input("How many players", "4"))

    stacks = []
    names = []
    for i in range(n):
        name = get_input(f"Player {i+1} name", f"Player {i+1}")
        stack = int(get_input(f"  {name}'s stack"))
        names.append(name)
        stacks.append(stack)

    print()
    payouts = []
    for i in range(n):
        p = float(get_input(f"  {i+1}{'st' if i==0 else 'nd' if i==1 else 'rd' if i==2 else 'th'} place prize ($)"))
        payouts.append(p)

    sb = int(get_input("Small blind", "200"))
    bb = int(get_input("Big blind", "400"))
    ante = int(get_input("Ante", "50"))
    blind = BlindLevel(level=1, small_blind=sb, big_blind=bb, ante=ante)

    print("\n  Generating report...")
    ft = analyzer.final_table_report(stacks=stacks, payouts=payouts,
                                      player_names=names, blind_level=blind)

    print(f"\n  {'='*55}")
    print(f"  {'Player':12s} {'Chips':>8s} {'Zone':>8s} {'ICM $':>10s}")
    print(f"  {'-'*55}")
    for p in ft['players']:
        z = p['zone']['zone'] if isinstance(p['zone'], dict) else p['zone']
        print(f"  {p['name']:12s} {p['chips']:>8,} {z:>8s} ${p['icm_equity']:>9,.2f}")

    print(f"\n  Bubble Factor Matrix:")
    for pair, bf in ft['bubble_factors'].items():
        if bf > 1.5:
            print(f"  {pair}: {bf:.2f}x  *** HIGH PRESSURE ***")

    print(f"\n  Fair ICM Deal:")
    for d in ft['icm_deal']:
        print(f"  {d['name']:12s} => ${d['icm_equity']:>9,.2f}")

# Main loop
while True:
    choice = show_menu()
    try:
        if choice == '1': hand_vs_hand()
        elif choice == '2': hand_vs_range()
        elif choice == '3': icm_calc()
        elif choice == '4': spot_analysis()
        elif choice == '5': push_fold()
        elif choice == '6': final_table()
        elif choice == '0':
            print("\n  Goodbye!\n")
            break
        else:
            continue
    except Exception as e:
        print(f"\n  Error: {e}")

    input("\n  Press Enter to return to menu...")
