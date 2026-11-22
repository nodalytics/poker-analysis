"""Parser and replay tests on one hand per supported layout (written from each site's documented format)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from advisor import handhistory, review  # noqa: E402

review.engine.pa.SOLVER = "/nonexistent"  # tests never run the solver

STARS = """PokerStars Hand #245112233445: Hold'em No Limit ($0.01/$0.02 USD) - 2026/11/01 20:15:03 ET
Table 'Alcor III' 6-max Seat #4 is the button
Seat 1: villainA ($2.10 in chips)
Seat 2: Hero ($2.00 in chips)
Seat 4: villainB ($1.85 in chips)
Seat 5: villainC ($2.40 in chips)
Seat 6: villainD ($0.95 in chips)
villainC: posts small blind $0.01
villainD: posts big blind $0.02
*** HOLE CARDS ***
Dealt to Hero [Ah Kd]
villainA: folds
Hero: raises $0.04 to $0.06
villainB: folds
villainC: calls $0.05
villainD: folds
*** FLOP *** [Qs 7h 2c]
villainC: checks
Hero: bets $0.08
villainC: raises $0.16 to $0.24
Hero: folds
Uncalled bet ($0.16) returned to villainC
villainC collected $0.29 from pot
*** SUMMARY ***
Total pot $0.30 | Rake $0.01
Board [Qs 7h 2c]
Seat 2: Hero folded on the Flop
Seat 5: villainC (small blind) collected ($0.29)
"""

GG = """Poker Hand #HD100200300: Hold'em No Limit ($0.05/$0.10) - 2026/11/02 18:00:00
Table 'RushAndCash123' 2-max Seat #1 is the button
Seat 1: Hero ($10.00 in chips)
Seat 2: 7f3a9b ($12.50 in chips)
Hero: posts small blind $0.05
7f3a9b: posts big blind $0.10
*** HOLE CARDS ***
Dealt to Hero [Td Tc]
Dealt to 7f3a9b
Hero: raises $0.15 to $0.25
7f3a9b: calls $0.15
*** FLOP *** [9s 4h 2d]
7f3a9b: checks
Hero: bets $0.30
7f3a9b: calls $0.30
*** TURN *** [9s 4h 2d] [Jc]
7f3a9b: checks
Hero: checks
*** RIVER *** [9s 4h 2d Jc] [3h]
7f3a9b: bets $0.60
Hero: calls $0.60
*** SHOWDOWN ***
7f3a9b: shows [Jh 8h] (a pair of Jacks)
Hero: shows [Td Tc] (a pair of Tens)
7f3a9b collected $2.24 from pot
*** SUMMARY ***
Total pot $2.30 | Rake $0.06
Board [9s 4h 2d Jc 3h]
"""

ACR = """Game Hand #2662312367 - Holdem(No Limit) - $0.02/$0.05 - 2026/11/03 19:34:12 UTC
Table 'Kenai' 6-max Seat #4 is the button
Seat 1: playerA ($5.21)
Seat 3: Hero ($5.00)
Seat 4: playerD ($7.12)
Seat 5: playerE ($5.00)
Seat 6: playerF ($2.31)
playerE posts the small blind $0.02
playerF posts the big blind $0.05
*** HOLE CARDS ***
Main pot $0.07
Dealt to Hero [7c 2d]
playerA folds
Hero raises $0.15 to $0.15
playerD folds
playerE folds
playerF calls $0.10
*** FLOP *** [Qs 7h 2c]
Main pot $0.32
playerF checks
Hero bets $0.18
playerF folds
Uncalled bet ($0.18) returned to Hero
Hero does not show
*** SUMMARY ***
Total pot $0.32 | Rake $0.01 | JP Fee $0.00
Board [Qs 7h 2c]
Seat 3: Hero did not show and won $0.31
"""

WINAMAX = """Winamax Poker - CashGame - HandId: #19283746-55-1730000000 - Holdem no limit (0.01€/0.02€) - 2026/11/04 10:00:00 UTC
Table: 'Nice 05' 5-max (real money) Seat #2 is the button
Seat 1: Hero (2.00€)
Seat 2: joueur2 (2.30€)
Seat 3: joueur3 (1.50€)
joueur3 posts small blind 0.01€
Hero posts big blind 0.02€
*** HOLE CARDS ***
Dealt to Hero [9h 9d]
joueur2 raises 0.04€ to 0.06€
joueur3 folds
Hero raises 0.16€ to 0.22€
joueur2 folds
Uncalled bet (0.16€) returned to Hero
Hero collected 0.13€ from pot
*** SUMMARY ***
Total pot 0.13€ | No rake
"""

PHH = """variant = 'NT'
antes = [0, 0, 0]
blinds_or_straddles = [1, 2, 0]
min_bet = 2
starting_stacks = [200, 200, 200]
actions = ['d dh p1 ????', 'd dh p2 ????', 'd dh p3 AsAd', 'p3 cbr 6', 'p1 f', 'p2 cc', 'd db Kh7c2s', 'p2 cc', 'p3 cbr 8', 'p2 f']
players = ['sb', 'bb', 'Hero']
winnings = [0, 0, 21]
"""


def one(text):
    hs = handhistory.parse_text(text)
    assert len(hs) == 1, hs
    return hs[0]


def test_pokerstars_order_and_actions():
    h = one(STARS)
    assert h["site"] == "PokerStars"
    assert h["players"] == ["villainC", "villainD", "villainA", "Hero", "villainB"]
    assert h["blinds"] == (1, 2) and h["stacks"][3] == 200
    assert h["hero"] == "Hero" and h["hero_cards"] == "AhKd"
    assert ("act", "Hero", "raise", 6) in h["actions"] and ("act", "villainC", "raise", 24) in h["actions"]
    assert h["collected"] == {"villainC": 29} and h["returned"] == {"villainC": 16}


def test_gg_heads_up_order():
    h = one(GG)
    assert h["site"] == "GGPoker"
    assert h["players"] == ["7f3a9b", "Hero"]  # heads-up: big blind first, button (small blind) second
    assert h["shown"]["7f3a9b"] == "Jh8h" and h["board"] == ["9s", "4h", "2d", "Jc", "3h"]


def test_acr_layout():
    h = one(ACR)
    assert h["site"] == "ACR / WPN"
    assert h["players"] == ["playerE", "playerF", "playerA", "Hero", "playerD"]
    assert h["blinds"] == (2, 5)
    assert h["returned"] == {"Hero": 18}
    assert h["collected"] == {"Hero": 31}


def test_winamax_euros():
    h = one(WINAMAX)
    assert h["site"] == "Winamax" and h["blinds"] == (1, 2)
    assert h["players"] == ["joueur3", "Hero", "joueur2"]
    assert ("act", "Hero", "raise", 22) in h["actions"]


def test_phh():
    h = one(PHH)
    assert h["site"] == "PHH" and h["hero"] == "Hero" and h["hero_cards"] == "AsAd"


def test_review_nets_and_grades():
    expect = {STARS: -14, GG: -115, ACR: 16, WINAMAX: 7, PHH: 7}  # hero result in units (cents or chips)
    for text, units in expect.items():
        h = one(text)
        r = review.review_hand(h, h["hero"], use_solver=False)
        assert r is not None, h["site"]
        decisions, facts = r
        assert round(facts["net_bb"] * h["blinds"][1]) == units, (h["site"], facts["net_bb"])
        assert decisions and all(d["grade"] in ("agrees", "questionable", "mistake") for d in decisions)
        assert facts["showdown"] == (text is GG), h["site"]
        assert all(d["grade"] != "mistake" for d in decisions if d["advice"]["source"] == "Monte Carlo equity")
