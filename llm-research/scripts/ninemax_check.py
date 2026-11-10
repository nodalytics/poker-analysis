"""Check the 9-max engine (ninemax_engine.py) against PokerKit: random legal games, replayed card for card.

For every table: the same hole and board cards are dealt in PokerKit, every action of the engine is applied there
(fold, check/call, complete-bet-or-raise-to the same total), and the check asserts that
  - PokerKit's actor is the engine's actor at every step, and PokerKit accepts every action;
  - the hand ends when the engine says it ends;
  - every seat's payoff agrees within P chips (PokerKit gives a split pot's odd chips to particular players).
Random play leans towards calls and small raises so that hands reach later streets, all-ins and side pots.

With STACKS "mix", every seat starts with its own stack, 2-200bb (log-uniform), so short all-ins, incomplete raises
and side pots between unequal stacks are exercised too.

Usage (lab, GPU venv on the CPU): .venv-gpu/bin/python scripts/ninemax_check.py [TABLES] [PLAYERS] [MENU] [STACKS]
"""

import sys
import warnings

import torch
from pokerkit import Automation, NoLimitTexasHoldem

from ninemax_engine import BB, SB, STACK, Dealer, Menu, Tables

RANKS, SUITS = "23456789TJQKA", "cdhs"


def card(c):
    return RANKS[c // 4] + SUITS[c % 4]


def main():
    warnings.filterwarnings("ignore")  # PokerKit warns about dealing chosen cards; they are the engine's cards
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    P = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    menu_name = sys.argv[3] if len(sys.argv) > 3 else "medium"
    mix = len(sys.argv) > 4 and sys.argv[4] == "mix"
    dev = "cpu"
    torch.manual_seed(0)
    dealer = Dealer("blueprint/libbp5.so", "blueprint/out", P, 200, dev)
    bk, rk, cards = dealer.deal(n, cards=True)
    menu = Menu(menu_name, dev)
    stacks = (2 * (100 ** torch.rand(n, P)) * BB).round().long() if mix else torch.full((n, P), STACK, dtype=torch.long)
    T = Tables(n, P, menu, dev, stacks=stacks)
    log = [[] for _ in range(n)]
    weights = torch.tensor([1.0, 4.0] + [1.5] * menu.S + [0.4])
    while not bool(T.done.all()):
        g = (~T.done).nonzero().squeeze(1)
        i = T.actor[g]
        legal, t = T.legal(g, i)
        p = legal.float() * weights
        a = torch.multinomial(p / p.sum(1, keepdim=True), 1).squeeze(1)
        bet_before = T.bet[g].clone()
        T.step(g, a)
        for k in range(len(g)):
            log[g[k]].append((int(i[k]), int(a[k]), int(T.contrib[g[k], i[k]]), int(bet_before[k])))
    ours = T.payoffs(rk)
    stats = {"tables": n, "showdowns": int(T.showdown.sum()), "max_actions": max(map(len, log))}
    side = int(((T.allin.sum(1) >= 2) & T.showdown).sum())
    bad = 0
    for k in range(n):
        st = NoLimitTexasHoldem.create_state(
            (
                Automation.ANTE_POSTING, Automation.BET_COLLECTION, Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.CARD_BURNING, Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING,
            ),  # fmt: skip
            True, 0, (SB, BB), BB, tuple(stacks[k].tolist()), P,
        )  # fmt: skip
        cd = cards[k]
        board = [card(c) for c in cd[2 * P :]]
        dealt = 0

        def deal():
            nonlocal dealt
            while st.can_deal_hole():
                s = st.hole_dealee_index
                st.deal_hole(card(cd[2 * s]) + card(cd[2 * s + 1]))
            while st.can_deal_board():
                m = 3 if dealt == 0 else 1
                st.deal_board("".join(board[dealt : dealt + m]))
                dealt += m

        try:
            for seat, a, total, bet_before in log[k]:
                deal()
                assert st.actor_index == seat, f"actor {st.actor_index} != engine {seat}"
                if a == 0:
                    st.fold()
                elif a == 1:
                    st.check_or_call()
                else:  # PokerKit raises to an amount for this street; the engine records totals for the hand
                    before = int(stacks[k, seat]) - st.stacks[seat] - st.bets[seat]
                    st.complete_bet_or_raise_to(total - before)
            deal()
            assert not st.status, "PokerKit hand not over"
            diff = max(abs(float(st.payoffs[s]) - float(ours[k, s])) for s in range(P))
            # PokerKit gives the odd chips of a split pot to particular players; the engine splits exactly
            assert diff < P and abs(sum(st.payoffs)) < 1e-6, (
                f"payoffs differ by {diff}: pokerkit {list(st.payoffs)} engine {ours[k].tolist()}"
            )
        except Exception as e:  # noqa: BLE001
            bad += 1
            if bad <= 5:
                print(f"table {k}: {e}\n  actions {log[k]}")
    print({**stats, "multi_allin_showdowns": side, "mismatches": bad})
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
