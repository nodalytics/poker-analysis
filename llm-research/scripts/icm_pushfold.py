"""Push/fold equilibrium under ICM for one tournament table of up to 9 players with any stacks, blinds 0.5/1 and an
ante: the planned check of the tournament network's short-stack play (ninemax_selfplay.py --tourney).

The game (the standard push/fold model, as in poker_math.solve_push_fold_multiway, but with unequal stacks and
prizes):
  - Players act in order: pos0 (first to act) ... button, small blind, big blind. Each player folded to either
    shoves all in or folds; once someone shoves, each player behind calls or folds.
  - At most one caller: the first to call; everyone after folds. All-ins are heads-up, so the 169x169 preflop
    equity table (results/preflop/equity169.json) gives every result. Card removal is ignored; a tie counts as half
    a win (the equity), which is exact for chips and close for ICM.
  - Every result is valued by ICM: the stacks after the hand -> each player's share of the prizes
    (Malmuth-Harville). With a single prize (winner takes all) this is the chip result, and the solver agrees with
    poker_math.solve_push_fold_multiway (checked in --selftest).

Solved by fictitious play: every iteration each decision (each shove position, each caller against each shover)
best-responds to the others' running averages over the 169 hand classes. The tree is evaluated exactly: folding
is worth what happens next given everyone else's strategy (later shoves, later callers, the big blind's walk).
Reported with each chart: the most any one player can gain by deviating from the averaged strategy, in big-blind
equivalents (equity x table chips) - near 0 means converged.

Usage:
  uv run python scripts/icm_pushfold.py --selftest
  uv run python scripts/icm_pushfold.py --stacks 12,8,25,10,6,15,9,30,11 --payouts 50,30,20 [--ante 0.125]
      (stacks in big blinds in action order: pos0 ... button, small blind, big blind; payouts for places 1..)
  uv run python scripts/icm_pushfold.py --charts OUT.json   (the scenario set used for the network check)
"""

import argparse
import itertools
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, "results", "preflop", "equity169.json")


def icm_np(stacks, payouts):
    """stacks [N, P] chips, payouts [P] (place 1..P; shorter lists are padded with 0): ICM equities [N, P]."""
    s = np.asarray(stacks, dtype=float).clip(min=0)
    N, P = s.shape
    pay = np.zeros(P)
    pay[: min(P, len(payouts))] = payouts[:P]
    M = 1 << P
    bits = ((np.arange(M)[:, None] >> np.arange(P)[None, :]) & 1).astype(bool)
    pop = bits.sum(1)
    msum = s @ bits.T.astype(float)
    total = s.sum(1, keepdims=True)
    prob = np.zeros((N, M))
    prob[:, 0] = 1.0
    eq = np.zeros((N, P))
    one = 1 << np.arange(P)
    for k in range(P):
        if pay[k:].sum() == 0:
            break
        ms = np.nonzero(pop == k)[0]
        rem = np.maximum(total - msum[:, ms], 1e-300)
        p = s[:, None, :] * (~bits[ms])[None] / rem[:, :, None]
        w = prob[:, ms, None] * p
        eq += w.sum(1) * pay[k]
        if k + 1 < P:
            nxt = (ms[:, None] | one[None, :]).reshape(-1)
            np.add.at(prob, (slice(None), nxt), w.reshape(N, -1))
    return eq


def _icm_brute(stacks, payouts):
    P = len(stacks)
    pay = list(payouts) + [0.0] * (P - len(payouts))
    eq = [0.0] * P
    for order in itertools.permutations(range(P)):
        pr, left = 1.0, float(sum(stacks))
        for j in order:
            pr *= stacks[j] / left if left > 0 else 0.0
            left -= stacks[j]
        for place, j in enumerate(order):
            eq[j] += pr * pay[place]
    return eq


def load_matrix(path=MATRIX):
    m = json.load(open(path))
    cls = m["classes"]
    eq = m["equity"]
    E = np.array([[0.5 if eq[a][b] is None else eq[a][b] for b in cls] for a in cls], dtype=float)
    w = np.array([6.0 if len(c) == 2 else 4.0 if c[2] == "s" else 12.0 for c in cls])
    return cls, E, w


def names_for(n):
    return [f"pos{i}" for i in range(n - 3)] + ["button", "small blind", "big blind"]


def solve(stacks, payouts, ante=0.0, iters=2000, matrix=None):
    """Push/fold equilibrium. stacks: big blinds in action order (last two are the blinds). Returns the chart."""
    cls, E, w = matrix or load_matrix()
    k = len(cls)
    S = np.asarray(stacks, dtype=float)
    n = len(S)
    post = np.full(n, float(ante))
    post[n - 2] += 0.5
    post[n - 1] += 1.0
    post = np.minimum(post, S)
    D = post.sum()
    base = S - post

    # every terminal result as a stack vector, valued once by ICM
    rows, index = [], {}

    def add(key, vec):
        index[key] = len(rows)
        rows.append(vec)

    v = base.copy()
    v[n - 1] += D
    add(("walk",), v)
    for p in range(n - 1):
        v = base.copy()
        v[p] += D
        add(("steal", p), v)
        for q in range(p + 1, n):
            e = min(S[p], S[q])
            pot = 2 * e + D - post[p] - post[q]
            for winner, loser in ((p, q), (q, p)):
                v = base.copy()
                v[p] = S[p] - e
                v[q] = S[q] - e
                v[winner] += pot
                add(("allin", p, q, winner), v)
    U = icm_np(np.array(rows), payouts)  # [results, players]
    u = lambda *key: U[index[key]]  # noqa: E731

    push = np.ones((n - 1, k))
    call = np.ones((n, n, k))
    wsum = w.sum()

    def values(push, call):
        """Expected ICM value of every player at each first-in node, and the best-response values per class."""
        fr_push = (push * w).sum(1) / wsum  # [n-1]
        node = np.zeros((n + 1, n))  # node[j] = values when everyone before j folded; node[n-1] = walk to BB
        node[n - 1] = u("walk")
        br_push = np.zeros((n - 1, k))
        gain_push = np.zeros((n - 1, k))
        br_call = np.zeros((n, n, k))
        gain_call = np.zeros((n, n, k))
        for p in range(n - 2, -1, -1):
            # after p shoves: callers q = p+1 .. n-1 in turn; tail[q] = values once everyone before q has folded
            rp = push[p] * w
            rp_tot = rp.sum()
            eq_vs_push = E @ rp / rp_tot if rp_tot > 0 else np.full(k, 0.5)  # a caller's class vs p's range
            tail = u("steal", p).copy()  # nobody calls
            tail_by_class = np.tile(u("steal", p)[p], (k,))  # p's value by p's class (does not depend on it)
            for q in range(n - 1, p, -1):
                rq = call[p, q] * w
                fq = rq.sum() / wsum
                eq_pq = E @ rq / rq.sum() if rq.sum() > 0 else np.full(k, 0.5)  # p's class vs q's calling range
                # q's decision given p's range: call vs fold (fold = tail)
                ev_call = eq_vs_push * u("allin", p, q, q)[q] + (1 - eq_vs_push) * u("allin", p, q, p)[q]
                ev_fold = tail[q]
                br_call[p, q] = ev_call > ev_fold
                d = ev_call - ev_fold  # regret of the averaged strategy: what switching to the better action gains
                gain_call[p, q] = (1 - call[p, q]) * np.maximum(d, 0) + call[p, q] * np.maximum(-d, 0)
                # p's value by class and everyone's average, with q calling at its current frequency
                pv = eq_pq * u("allin", p, q, p)[p] + (1 - eq_pq) * u("allin", p, q, q)[p]
                tail_by_class = fq * pv + (1 - fq) * tail_by_class
                # average over both ranges for all players
                if rp_tot > 0 and rq.sum() > 0:
                    pwin = float(rp @ E @ rq) / (rp_tot * rq.sum())
                else:
                    pwin = 0.5
                called = pwin * u("allin", p, q, p) + (1 - pwin) * u("allin", p, q, q)
                tail = fq * called + (1 - fq) * tail
            fold_v = node[p + 1][p]
            br_push[p] = tail_by_class > fold_v
            d = tail_by_class - fold_v
            gain_push[p] = (1 - push[p]) * np.maximum(d, 0) + push[p] * np.maximum(-d, 0)
            node[p] = fr_push[p] * tail + (1 - fr_push[p]) * node[p + 1]
        return node, br_push, gain_push, br_call, gain_call

    for t in range(1, iters + 1):
        _, bp, _, bc, _ = values(push, call)
        lr = 1.0 / (t + 1)
        push += lr * (bp - push)
        call += lr * (bc - call)

    node, _, gp, _, gc = values(push, call)
    chips = S.sum()
    # the most one player gains by best-responding at one decision (weighted over hands), in bb equivalents
    dev = max([float((gp[p] * w).sum() / wsum) for p in range(n - 1)] +
              [float((gc[p, q] * w).sum() / wsum) for p in range(n - 1) for q in range(p + 1, n)])  # fmt: skip
    pay_total = float(np.sum(payouts))
    names = names_for(n)
    share = lambda r: round(float((w * (r >= 0.5)).sum() / wsum), 3)  # noqa: E731
    rng = lambda r: [cls[i] for i in range(k) if r[i] >= 0.5]  # noqa: E731
    return {
        "stacks_bb": [float(x) for x in S],
        "payouts": [float(x) for x in payouts],
        "ante": ante,
        "icm_equity": [round(float(x), 4) for x in node[0] / pay_total],
        "chip_share": [round(float(x), 4) for x in S / chips],
        "max_gain_bb": round(dev / pay_total * chips, 4),
        "shove": {names[p]: {"share": share(push[p]), "range": rng(push[p])} for p in range(n - 1)},
        "call": {
            f"{names[q]} vs {names[p]}": {"share": share(call[p, q]), "range": rng(call[p, q])}
            for p in range(n - 1)
            for q in range(p + 1, n)
        },
    }


SCENARIOS = [  # (name, stacks in action order: pos0..button, SB, BB; payouts for places 1..; ante)
    ("chip EV, 9 x 10bb", [10] * 9, [1], 0.0),
    ("final table, 9 x 10bb", [10] * 9, [30, 20, 14, 10, 8, 6.5, 5, 4, 3.5], 0.0),
    ("bubble (1 out), 9 x 10bb", [10] * 9, [1] * 8, 0.0),
    ("bubble (1 out), top-heavy above it", [10] * 9, [30, 20, 14, 10, 8, 6.5, 5, 4], 0.0),
    ("bubble (1 out), big stack on the button", [8, 8, 8, 8, 8, 8, 30, 8, 8], [1] * 8, 0.0),
    ("bubble (1 out), short stack in the big blind", [10, 10, 10, 10, 10, 10, 10, 10, 3], [1] * 8, 0.0),
    ("final table, mixed stacks, ante", [12, 8, 25, 10, 6, 15, 9, 30, 11], [30, 20, 14, 10, 8, 6.5, 5, 4, 3.5], 0.125),
]


def selftest():
    rng = np.random.default_rng(0)
    for P in (3, 5, 7):
        st = rng.random((3, P)) * 100 + 1
        pay = sorted(rng.random(P), reverse=True)[: P - 1]
        got = icm_np(st, pay)
        for g in range(3):
            ref = _icm_brute(list(st[g]), pay)
            assert max(abs(a - b) for a, b in zip(got[g], ref)) < 1e-9
    print("ICM matches brute force (3, 5, 7 players)")
    m = load_matrix()
    import poker_math

    mm = json.load(open(MATRIX))
    for n, s in ((4, 10), (6, 8)):
        ref = poker_math.solve_push_fold_multiway(mm, n, s, 0.0, iters=1500)
        got = solve([s] * n, [1.0], 0.0, iters=1500, matrix=m)
        diffs = [abs(ref["shove"][a]["share"] - got["shove"][a]["share"]) for a in ref["shove"]]
        print(f"winner-take-all = chips, {n} x {s}bb: shove shares {[got['shove'][a]['share'] for a in got['shove']]}"
              f" vs poker_math {[ref['shove'][a]['share'] for a in ref['shove']]}")  # fmt: skip
        assert max(diffs) <= 0.03, diffs
    print("winner-take-all agrees with the chip-EV multi-way solver")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stacks")
    ap.add_argument("--payouts", default="1")
    ap.add_argument("--ante", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--charts")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.stacks:
        c = solve([float(x) for x in a.stacks.split(",")], [float(x) for x in a.payouts.split(",")], a.ante, a.iters)
        print(json.dumps({k: v for k, v in c.items() if k != "call"}, indent=1))
        return
    if a.charts:
        m = load_matrix()
        out = []
        for name, st, pay, ante in SCENARIOS:
            c = solve(st, pay, ante, a.iters, matrix=m)
            c["name"] = name
            out.append(c)
            sh = " ".join(f"{v['share']:.0%}" for v in c["shove"].values())
            bb = c["call"][f"big blind vs {names_for(len(st))[-2]}"]["share"]
            print(f"{name:45s} shove by position: {sh} | BB calls SB {bb:.0%} | max gain {c['max_gain_bb']} bb")
        json.dump({"method": "push/fold under ICM, at most one caller, fictitious play", "charts": out},
                  open(a.charts, "w"), indent=1)  # fmt: skip


if __name__ == "__main__":
    main()
