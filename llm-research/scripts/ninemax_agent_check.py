"""Check the 9-max adapter (ninemax_agent.py): at every decision of random engine games replayed in PokerKit, the
features and the legal mask the adapter rebuilds from PokerKit's state must equal the engine's own.

Usage (lab, from llm-research): PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_agent_check.py [TABLES] [RUN]
"""

import sys
import warnings

import torch
from pokerkit import Automation, NoLimitTexasHoldem

from ninemax_agent import Seat
from ninemax_check import card
from ninemax_engine import BB, SB, STACK, Tables
from ninemax_selfplay import features


def main():
    warnings.filterwarnings("ignore")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    seat = Seat(sys.argv[2] if len(sys.argv) > 2 else "nm3")
    P, menu, dealer = 9, seat.menu, seat.dealer
    torch.manual_seed(1)
    bk, rk, cards = dealer.deal(n, cards=True)
    T = Tables(n, P, menu, "cpu")
    snaps = [[] for _ in range(n)]  # per table: (seat, action, total, engine features, legal)
    weights = torch.tensor([1.0, 4.0] + [1.5] * menu.S + [0.4])
    while not bool(T.done.all()):
        g = (~T.done).nonzero().squeeze(1)
        i = T.actor[g]
        feat, legal = features(T, g, i)
        p = legal.float() * weights
        a = torch.multinomial(p / p.sum(1, keepdim=True), 1).squeeze(1)
        T.step(g, a)
        for k in range(len(g)):
            snaps[g[k]].append((int(i[k]), int(a[k]), int(T.contrib[g[k], i[k]]), feat[k], legal[k]))
    bad = checked = 0
    for k in range(n):
        st = NoLimitTexasHoldem.create_state(
            (
                Automation.ANTE_POSTING, Automation.BET_COLLECTION, Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.CARD_BURNING, Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING,
            ),  # fmt: skip
            True, 0, (SB, BB), BB, (STACK,) * P, P,
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
            for s, a, total, f_eng, l_eng in snaps[k]:
                deal()
                view = seat.table(st, s)
                assert view is not None, "adapter declined the hand"
                Ta = view[0]
                f_ad, l_ad = features(Ta, torch.zeros(1, dtype=torch.long), torch.tensor([s]))
                assert torch.equal(l_ad[0], l_eng), f"legal {l_ad[0].tolist()} != engine {l_eng.tolist()}"
                d = (f_ad[0] - f_eng).abs().max().item()
                assert d < 1e-5, f"features differ by {d} at {(f_ad[0] - f_eng).abs().argmax().item()}"
                checked += 1
                if a == 0:
                    st.fold()
                elif a == 1:
                    st.check_or_call()
                else:
                    before = STACK - st.stacks[s] - st.bets[s]
                    st.complete_bet_or_raise_to(total - before)
        except AssertionError as e:
            bad += 1
            if bad <= 5:
                print(f"table {k}: {e}")
    print({"tables": n, "decisions_checked": checked, "tables_with_mismatch": bad})
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
