"""How big is a no-limit hold'em betting tree for N players under a given bet-size menu, and does it fit in memory?

Counts betting decision nodes per street by dynamic programming over betting states (a subtree's size depends only
on its state), so 9-player trees with billions of nodes are counted in seconds without being built. Table entries
are decision nodes x card buckets x actions; memory assumes 4-byte regrets everywhere and 4-byte average strategies
either everywhere (as our heads-up runs do) or pre-flop only (Pluribus: post-flop it keeps the current strategy and
relies on search).

Rules: blinds 0.5/1 bb, every player starts with STACK bb, no ante. Seats 0 = small blind, 1 = big blind,
2.. = UTG .. button; pre-flop UTG acts first, after the flop the small blind. A raise must be at least the last
raise; a size at or above the stack becomes all-in. Players all in or folded do not act; when at most one player
can still act and the bets are matched, the board runs out with no further decisions.

Menus (see MENUS): pre-flop raises are multiples of the current bet by raise level (open, 3-bet, 4-bet...), with
the open in big blinds; post-flop bets and raises are pot fractions, with a separate, smaller menu once three or
more players are in the pot. Every node also has fold (when facing a bet), check/call and all-in.

Usage: python blueprint/tree_size.py [PLAYERS ...]   (default 2 6 9)
"""

import sys
from functools import lru_cache

BB = 100  # chips per big blind
STACK = 100  # big blinds

# name: (pre-flop open sizes in bb, pre-flop re-raise multiples by level, max pre-flop raises,
#        post-flop heads-up bet fractions, heads-up raise fractions, multiway bet fractions, multiway raise fractions,
#        max post-flop raises per street)
MENUS = {
    "lean": ((2.5,), ((3.0,), (2.3,)), 3, (0.5, 1.0), (1.0,), (0.66,), (), 2),
    "medium": ((2.5,), ((3.0,), (2.3,)), 4, (0.33, 0.75, 1.5), (1.0,), (0.5, 1.0), (1.0,), 3),
    "wide": (
        (2.0, 2.5, 3.5),
        ((2.5, 3.5), (2.2, 3.0)),
        4,
        (0.25, 0.5, 0.75, 1.25, 2.0),
        (0.75, 1.5),
        (0.33, 0.75),
        (1.0,),
        3,
    ),  # fmt: skip
}


def count(n, menu):
    opens, reraises, max_pre, hu_bets, hu_raises, mw_bets, mw_raises, max_post = menu
    stack = STACK * BB
    pre_order = tuple(list(range(2, n)) + [0, 1]) if n > 2 else (0, 1)
    post_order = tuple(range(n)) if n > 2 else (1, 0)  # heads-up: the big blind acts first after the flop

    @lru_cache(maxsize=None)
    def node(street, c, folded, allin, pos, bet, last_raise, acted, raises):
        """Returns (decision nodes by street as a 4-tuple, table entries by street: nodes x actions)."""
        live = [i for i in range(n) if not folded >> i & 1]
        if len(live) == 1:
            return (0, 0, 0, 0), (0, 0, 0, 0)
        order = pre_order if street == 0 else post_order
        can_act = [i for i in order if not folded >> i & 1 and not allin >> i & 1]
        done = all(acted >> i & 1 and c[i] == bet for i in can_act) or (
            len(can_act) <= 1 and all(c[i] >= bet for i in can_act)
        )
        if done:
            if street == 3 or len(can_act) <= 1:
                return (0, 0, 0, 0), (0, 0, 0, 0)  # showdown, possibly after running the board out
            nxt = next(k for k, i in enumerate(post_order) if i in can_act)
            return node(street + 1, c, folded, allin, nxt, bet, BB, 0, 0)
        # the next player to act, from position pos in the order
        k = pos
        while True:
            i = order[k % n]
            if not folded >> i & 1 and not allin >> i & 1 and not (acted >> i & 1 and c[i] == bet):
                break
            k += 1
        i = order[k % n]
        nxt_pos = (k + 1) % n
        to_call = bet - c[i]
        pot = sum(c)
        targets = set()  # totals this player could raise to
        multiway = len(live) > 2
        if street == 0:
            if raises < max_pre:
                if bet <= BB:  # the open
                    targets |= {int(o * BB) for o in opens}
                else:
                    lvl = min(raises - 1, len(reraises) - 1)
                    targets |= {int(m * bet) for m in reraises[lvl]}
        elif raises < max_post:
            fr = (mw_bets if multiway else hu_bets) if bet == c[i] == max(c) and to_call == 0 else (
                mw_raises if multiway else hu_raises)  # fmt: skip
            targets |= {bet + int(f * (pot + to_call)) for f in fr}
        legal = set()
        for t in targets:
            if t - bet >= last_raise and t < stack:  # a size reaching the stack is the all-in action
                legal.add(t)
        acts = []
        if to_call > 0:
            acts.append(("fold", None))
        acts.append(("call", None))
        acts += [("raise", t) for t in sorted(legal)]
        if bet < stack:  # otherwise calling is already all in
            acts.append(("allin", None))
        cnt = [0, 0, 0, 0]
        ent = [0, 0, 0, 0]
        cnt[street] += 1
        ent[street] += len(acts)
        for kind, t in acts:
            cc, f2, a2, b2, lr2, ac2, r2 = list(c), folded, allin, bet, last_raise, acted | 1 << i, raises
            if kind == "fold":
                f2 |= 1 << i
            elif kind == "call":
                cc[i] = min(bet, stack)
                if cc[i] >= stack:
                    a2 |= 1 << i
            else:
                tot = stack if kind == "allin" else t
                if tot - bet > 0:
                    lr2 = max(last_raise, tot - bet)
                    b2 = max(bet, tot)
                    ac2 = 1 << i  # everyone else must act again
                    r2 += 1
                cc[i] = tot
                if tot >= stack:
                    a2 |= 1 << i
            sub_cnt, sub_ent = node(street, tuple(cc), f2, a2, nxt_pos, b2, lr2, ac2, r2)
            for s in range(4):
                cnt[s] += sub_cnt[s]
                ent[s] += sub_ent[s]
        return tuple(cnt), tuple(ent)

    c0 = [0] * n
    c0[0], c0[1] = BB // 2, BB
    return node(0, tuple(c0), 0, 0, 0, BB, BB, 0, 0)  # heads-up, seat 0 is the button: it acts first pre-flop


def main():
    players = [int(x) for x in sys.argv[1:]] or [2, 6, 9]
    print(f"{'players':>7} {'menu':>7} {'pre-flop':>12} {'flop':>14} {'turn':>15} {'river':>16} |"
          f" {'GB, 200 buckets':>16} {'avg pre-flop only':>18} | {'GB, 500 buckets':>16} {'avg pre-flop only':>18}")  # fmt: skip
    for n in players:
        for name, menu in MENUS.items():
            cnt, ent = count(n, menu)
            row = f"{n:>7} {name:>7} " + " ".join(f"{x:>{w},}" for x, w in zip(cnt, (12, 14, 15, 16)))
            for buckets in (200, 500):
                cells = ent[0] * 169 + sum(ent[1:]) * buckets  # (node, bucket, action) entries
                full = cells * 8 / 1e9  # regret + average, 4 bytes each
                lean = (cells * 4 + ent[0] * 169 * 4) / 1e9  # regrets everywhere, averages pre-flop only
                row += f" | {full:>16,.1f} {lean:>18,.1f}"
            print(row, flush=True)


if __name__ == "__main__":
    main()
