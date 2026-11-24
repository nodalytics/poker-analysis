"""Push/fold equilibrium under ICM for one tournament table of up to 9 players with any stacks, blinds 0.5/1 and an
ante: the planned check of the tournament network's short-stack play (ninemax_selfplay.py --tourney).

The game (the standard push/fold model, as in poker_math.solve_push_fold_multiway, but with unequal stacks and
prizes):
  - Players act in order: pos0 (first to act) ... button, small blind, big blind. Each player folded to either
    shoves all in or folds; once someone shoves, each player behind calls or folds.
  - At most one caller: the first to call; everyone after folds. All-ins are heads-up, so the 169x169 preflop
    equity table (results/preflop/equity169.json) gives every result. Card removal is ignored; a tie counts as half
    a win (the equity), which is exact for chips and close for ICM.
  - Or (solve2, --callers 2) up to two callers: the first caller may be followed by one overcaller, after whom
    everyone folds. Three-way all-ins are valued from the three-way table (results/preflop/equity3_169.npz, built by
    equity3_169.py with card removal among the three hands), with side pots when the stacks differ.
  - A player busted by the hand keeps a vanishing stack (bust_floor, as the engine's reward does), so it still takes
    the lowest place left; a stack of exactly 0 got no prize at all before 2026-11-24.
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
      [--callers 2] (stacks in big blinds in action order: pos0 ... button, small blind, big blind; payouts for
      places 1..)
  uv run python scripts/icm_pushfold.py --charts OUT.json   (the scenario set used for the network check)
  uv run python scripts/icm_pushfold.py --charts OUT.json --bb-ante 1   (the same scenarios with a big-blind ante,
      results/ninemax/icm_pushfold_ante.json, for runs trained with --ante 1)
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


def icm_mw_np(stacks, payouts):
    """Malmuth-Weitzman: places are given from the bottom; among the players left, the next to bust is drawn with
    probability proportional to 1 / stack (a short stack busts first; a stack of 0 has already busted). The same
    shapes as icm_np."""
    s = np.asarray(stacks, dtype=float).clip(min=0)
    N, P = s.shape
    pay = np.zeros(P)
    pay[: min(P, len(payouts))] = payouts[:P]
    # a busted player (stack ~0) takes the lowest place left: 1/stack capped at a billionth of the table's chips, so
    # it dominates without overflowing
    inv = 1.0 / np.maximum(s, 1e-9 * s.sum(1, keepdims=True))
    M = 1 << P
    bits = ((np.arange(M)[:, None] >> np.arange(P)[None, :]) & 1).astype(bool)  # who has busted
    pop = bits.sum(1)
    isum = inv @ bits.T.astype(float)
    total = inv.sum(1, keepdims=True)
    prob = np.zeros((N, M))
    prob[:, 0] = 1.0
    eq = np.zeros((N, P))
    one = 1 << np.arange(P)
    for k in range(P):  # the next bust takes place P - k
        ms = np.nonzero(pop == k)[0]
        rem = np.maximum(total - isum[:, ms], 1e-300)
        p = inv[:, None, :] * (~bits[ms])[None] / rem[:, :, None]
        w = prob[:, ms, None] * p
        eq += w.sum(1) * pay[P - 1 - k]
        if k + 1 < P:
            nxt = (ms[:, None] | one[None, :]).reshape(-1)
            np.add.at(prob, (slice(None), nxt), w.reshape(N, -1))
    return eq


EQUITY = {"mh": icm_np, "mw": icm_mw_np}  # Malmuth-Harville (the default) and Malmuth-Weitzman


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


def _mw_brute(stacks, payouts):
    """Malmuth-Weitzman by enumeration: orders of busting, each bust drawn with probability 1/stack among those left."""
    P = len(stacks)
    pay = list(payouts) + [0.0] * (P - len(payouts))
    eq = [0.0] * P
    for order in itertools.permutations(range(P)):  # order[0] busts first (place P)
        pr, left = 1.0, list(range(P))
        for j in order:
            pr *= (1 / stacks[j]) / sum(1 / stacks[i] for i in left)
            left.remove(j)
        for k, j in enumerate(order):
            eq[j] += pr * pay[P - 1 - k]
    return eq


def bust_floor(rows, S):
    """Stacks after a hand [R, n] with every busted player kept at a vanishing stack proportional to its starting stack
    S (icm.end_stacks, as the engine's reward does): a bust then takes the lowest place left, ordered by starting
    stack when several bust at once. (Until 2026-11-24 a busted player's stack was 0, which Malmuth-Harville pays
    nothing at all: the lowest prize was lost, overstating every all-in's downside by it at a final table.)"""
    rows = np.asarray(rows, dtype=float)
    return np.where(rows <= 1e-9, np.asarray(S, dtype=float)[None, :] * 1e-9, rows)


def load_matrix(path=MATRIX):
    m = json.load(open(path))
    cls = m["classes"]
    eq = m["equity"]
    E = np.array([[0.5 if eq[a][b] is None else eq[a][b] for b in cls] for a in cls], dtype=float)
    # the table's two entries for a pair are separate Monte Carlo estimates (E[a, b] + E[b, a] is 1 only to within
    # 0.019); their average keeps every all-in's two sides consistent, as the two-caller checks require
    E = (E + 1 - E.T) / 2
    w = np.array([6.0 if len(c) == 2 else 4.0 if c[2] == "s" else 12.0 for c in cls])
    return cls, E, w


def names_for(n):
    return [f"pos{i}" for i in range(n - 3)] + ["button", "small blind", "big blind"]


def solve(stacks, payouts, ante=0.0, iters=2000, matrix=None, bb_ante=0.0, model="mh"):
    """Push/fold equilibrium. stacks: big blinds in action order (last two are the blinds). ante: every player's
    ante; bb_ante: the big-blind ante (the big blind alone posts it, as in the engine's RULES["ante"]). Antes are dead
    money: they are in the pot but no part of a player's all-in that the caller has to match. Returns the chart."""
    cls, E, w = matrix or load_matrix()
    k = len(cls)
    S = np.asarray(stacks, dtype=float)
    n = len(S)
    dead = np.full(n, float(ante))
    dead[n - 1] += float(bb_ante)
    dead = np.minimum(dead, np.maximum(S - np.where(np.arange(n) == n - 1, 1.0, 0.0), 0))
    post = dead.copy()
    post[n - 2] += 0.5
    post[n - 1] += 1.0
    post = np.minimum(post, S)
    D = post.sum()
    base = S - post
    live = S - dead  # what a player can put in that the caller must match

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
            e = min(live[p], live[q])
            pot = 2 * e + D - post[p] - post[q] + dead[p] + dead[q]
            for winner, loser in ((p, q), (q, p)):
                v = base.copy()
                v[p] = live[p] - e
                v[q] = live[q] - e
                v[winner] += pot
                add(("allin", p, q, winner), v)
    U = EQUITY[model](bust_floor(np.array(rows), S), payouts)  # [results, players]
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
        "bb_ante": bb_ante,
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


EQUITY3 = os.path.join(ROOT, "results", "preflop", "equity3_169.npz")
ORDERS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]  # positions, best hand first


def load_equity3(path=EQUITY3):
    """The three-way all-in table (equity3_169.py) for every ordered triple of classes: [169, 169, 169, 6] float32,
    the probability of each strength order of ORDERS (positions 0, 1, 2 = the classes a, b, c)."""
    z = np.load(path)
    C = z["probs"].astype(np.float32)
    C /= C.sum(1, keepdims=True)
    k = len(z["classes"])
    r = np.arange(k)
    a, b, c = np.meshgrid(r, r, r, indexing="ij")
    keep = (a <= b) & (b <= c)
    idx = np.full((k, k, k), -1, np.int64)
    idx[keep] = np.arange(int(keep.sum()))  # the builder's order (equity3_169.triples)
    t = np.stack([a.ravel(), b.ravel(), c.ravel()], 1)
    del a, b, c, keep
    s = np.argsort(t, 1, kind="stable")  # sorted position j holds ordered position s[:, j]
    srt = np.take_along_axis(t, s, 1)
    row = idx[srt[:, 0], srt[:, 1], srt[:, 2]]
    lut = np.full(27, -1)
    for i, o in enumerate(ORDERS):
        lut[o[0] * 9 + o[1] * 3 + o[2]] = i
    T3 = np.empty((len(t), 6), np.float32)
    ar = np.arange(len(t))
    for j, (j0, j1, j2) in enumerate(ORDERS):
        T3[ar, lut[s[:, j0] * 9 + s[:, j1] * 3 + s[:, j2]]] = C[row, j]
    return T3.reshape(k, k, k, 6)


def solve2(stacks, payouts, ante=0.0, iters=1000, matrix=None, bb_ante=0.0, model="mh", equity3=None, overcall=True,
           init=None, arrays=False):  # fmt: skip
    """Push/fold equilibrium with up to two callers: after a shove the first player to call may be followed by one
    more caller (an overcall); everyone after the second caller folds. Heads-up all-ins use the 169x169 table,
    three-way all-ins the three-way table (load_equity3), with side pots when the stacks differ (each level of the
    contributions goes to the best hand among the players who reach it; the dead money goes to the main pot).
    overcall=False forbids every overcall and is then the one-caller game of solve(). init (push, call1, call2):
    start from these strategies (iters=0 evaluates them). arrays: also return the strategies as arrays (push [n-1,
    169], call1 [n, n, 169] indexed [shover, caller], call2 [n, n, n, 169] indexed [shover, caller, overcaller]),
    the terminal values and the node values."""
    cls, E, w = matrix or load_matrix()
    T3 = None
    if overcall:
        T3 = equity3 if equity3 is not None else load_equity3()
    k = len(cls)
    S = np.asarray(stacks, dtype=float)
    n = len(S)
    dead = np.full(n, float(ante))
    dead[n - 1] += float(bb_ante)
    dead = np.minimum(dead, np.maximum(S - np.where(np.arange(n) == n - 1, 1.0, 0.0), 0))
    post = dead.copy()
    post[n - 2] += 0.5
    post[n - 1] += 1.0
    post = np.minimum(post, S)
    D = post.sum()
    base = S - post
    live = S - dead

    def settle(players, order):
        """Stacks after an all-in among players (the shover first, then the callers in order), the hands ranked by
        order (indices into players, best first)."""
        v = base.copy()
        lp = live[players[0]]
        c = {x: min(live[x], lp) for x in players[1:]}
        c[players[0]] = min(lp, max(c[x] for x in players[1:]))  # the shove beyond the biggest call comes back
        for x in players:
            v[x] = live[x] - c[x]
        extra = D - sum(post[x] - dead[x] for x in players)  # the folded players' posts and the participants' antes
        prev, first = 0.0, True
        for L in sorted(set(c.values())):
            part = sum(min(c[x], L) - min(c[x], prev) for x in players)
            win = next(players[j] for j in order if c[players[j]] >= L)
            v[win] += part + (extra if first else 0.0)
            prev, first = L, False
        return v

    rows, index = [], {}

    def add(key, vec):
        index[key] = len(rows)
        rows.append(vec)

    v = base.copy()
    v[n - 1] += D
    add(("walk",), v)
    trip = [(p, q, r) for p in range(n - 1) for q in range(p + 1, n) for r in range(q + 1, n)]
    for p in range(n - 1):
        v = base.copy()
        v[p] += D
        add(("steal", p), v)
        for q in range(p + 1, n):
            add(("allin", p, q, p), settle((p, q), (0, 1)))
            add(("allin", p, q, q), settle((p, q), (1, 0)))
    for p, q, r in trip:
        for j, o in enumerate(ORDERS):
            add(("allin3", p, q, r, j), settle((p, q, r), o))
    R = np.array(rows)
    assert np.allclose(R.sum(1), S.sum()), "chips are conserved"
    U = EQUITY[model](bust_floor(R, S), payouts)
    u = lambda *key: U[index[key]]  # noqa: E731
    U3 = {t: U[[index[("allin3", *t, j)] for j in range(6)]] for t in trip}  # [6, n]

    if init is None:
        push, call1 = np.ones((n - 1, k)), np.ones((n, n, k))
        call2 = np.ones((n, n, n, k)) if overcall else np.zeros((n, n, n, k))
    else:
        push, call1, call2 = (np.array(x, dtype=float) for x in init)
    wsum = w.sum()
    Of = Ocl = None
    if overcall:
        Of = T3.reshape(k, -1)  # [a, (b c j)]
        Ocl = np.ascontiguousarray(T3.transpose(0, 1, 3, 2)).reshape(k * k * 6, k)  # [(a b j), c]
    safe = lambda r: r if r.sum() > 0 else w  # noqa: E731  (an unreachable node: read against all hands)

    diag = {}  # the last evaluation's values by class at every decision, and the batched three-way sums (checks)

    def values(push, call1, call2):
        node = np.zeros((n + 1, n))
        node[n - 1] = u("walk")
        gain_push = np.zeros((n - 1, k))
        gain1 = np.zeros((n, n, k))
        gain2 = np.zeros((n, n, n, k))
        br_push, br1, br2 = np.zeros((n - 1, k)), np.zeros((n, n, k)), np.zeros((n, n, n, k))
        Y = X = None
        if overcall and trip:
            RP = np.array([safe(push[p] * w) for p in range(n - 1)], dtype=np.float32)
            X = (RP @ Of).reshape(n - 1, k, k, 6) / RP.sum(1)[:, None, None, None]  # [p, b, c, j], over p's range
            RQ = np.array([safe(call1[p, q] * w) for p, q, _ in trip], dtype=np.float32)
            R2 = np.array([safe(call2[p, q, r] * w) for p, q, r in trip], dtype=np.float32)
            Z = (Ocl @ R2.T).reshape(k, k, 6, len(trip))  # [a, b, j, t]: over the overcaller's range
            Yt = np.matmul(Z.transpose(3, 0, 2, 1), RQ[:, None, :, None])[..., 0]  # [t, a, j]
            Y = {t: Yt[i] / (RQ[i].sum() * R2[i].sum()) for i, t in enumerate(trip)}
            diag["Y"] = Y
        for p in range(n - 2, -1, -1):
            rp = safe(push[p] * w)
            eq_vs_push = E @ rp / rp.sum()  # every class against p's shoving range
            tail = u("steal", p).copy()
            tail_p = np.full(k, u("steal", p)[p])
            for q in range(n - 1, p, -1):
                rq_raw = call1[p, q] * w
                fq = rq_raw.sum() / wsum
                rq = safe(rq_raw)
                eq_pq = E @ rq / rq.sum()  # p's classes against q's calling range
                U2p, U2q = u("allin", p, q, p), u("allin", p, q, q)
                pwin = float(rp @ E @ rq) / (rp.sum() * rq.sum())
                s_all = pwin * U2p + (1 - pwin) * U2q  # everyone's value once q calls, nobody overcalls
                s_a = eq_pq * U2p[p] + (1 - eq_pq) * U2q[p]  # p's, by p's class
                s_b = eq_vs_push * U2q[q] + (1 - eq_vs_push) * U2p[q]  # q's, by q's class
                if overcall:
                    M = np.tensordot(rq, X[p], (0, 0)) / rq.sum()  # [c, j]: the orders, by the overcaller's class
                    for r in range(n - 1, q, -1):
                        U3t = U3[(p, q, r)]
                        ev_call = M @ U3t[:, r]
                        diag[("call2", p, q, r)] = ev_call
                        d = ev_call - s_all[r]
                        br2[p, q, r] = d > 0
                        c2 = call2[p, q, r]
                        gain2[p, q, r] = (1 - c2) * np.maximum(d, 0) + c2 * np.maximum(-d, 0)
                        r2_raw = c2 * w
                        f2 = r2_raw.sum() / wsum
                        if f2 <= 0:
                            continue
                        Nb = np.tensordot(X[p], r2_raw, (1, 0)) / r2_raw.sum()  # [b, j]: by q's class
                        P3 = rq @ Nb / rq.sum()
                        s_all = f2 * (P3 @ U3t) + (1 - f2) * s_all
                        s_a = f2 * (Y[(p, q, r)] @ U3t[:, p]) + (1 - f2) * s_a
                        s_b = f2 * (Nb @ U3t[:, q]) + (1 - f2) * s_b
                diag[("call1", p, q)] = s_b
                d = s_b - tail[q]
                br1[p, q] = d > 0
                gain1[p, q] = (1 - call1[p, q]) * np.maximum(d, 0) + call1[p, q] * np.maximum(-d, 0)
                tail_p = fq * s_a + (1 - fq) * tail_p
                tail = fq * s_all + (1 - fq) * tail
            fr = (push[p] * w).sum() / wsum
            d = tail_p - node[p + 1][p]
            diag[("push", p)] = tail_p
            br_push[p] = d > 0
            gain_push[p] = (1 - push[p]) * np.maximum(d, 0) + push[p] * np.maximum(-d, 0)
            node[p] = fr * tail + (1 - fr) * node[p + 1]
        return node, (br_push, br1, br2), (gain_push, gain1, gain2)

    for t in range(1, iters + 1):
        _, (bp, b1, b2), _ = values(push, call1, call2)
        lr = 1.0 / (t + 1)
        push += lr * (bp - push)
        call1 += lr * (b1 - call1)
        if overcall:
            call2 += lr * (b2 - call2)

    node, _, (gp, g1, g2) = values(push, call1, call2)
    chips = S.sum()
    avg = lambda g: float((g * w).sum() / wsum)  # noqa: E731
    dev = [avg(gp[p]) for p in range(n - 1)] + [avg(g1[p, q]) for p in range(n - 1) for q in range(p + 1, n)]
    if overcall:
        dev += [avg(g2[t]) for t in trip]
    pay_total = float(np.sum(payouts))
    names = names_for(n)
    share = lambda r: round(float((w * (r >= 0.5)).sum() / wsum), 3)  # noqa: E731  (as solve: the pure range)
    rng = lambda r: [cls[i] for i in range(k) if r[i] >= 0.5]  # noqa: E731
    out = {
        "stacks_bb": [float(x) for x in S],
        "payouts": [float(x) for x in payouts],
        "ante": ante,
        "bb_ante": bb_ante,
        "callers": 2 if overcall else 1,
        "icm_equity": [round(float(x), 4) for x in node[0] / pay_total],
        "chip_share": [round(float(x), 4) for x in S / chips],
        "max_gain_bb": round(max(dev) / pay_total * chips, 4),
        "shove": {names[p]: {"share": share(push[p]), "range": rng(push[p])} for p in range(n - 1)},
        "call": {
            f"{names[q]} vs {names[p]}": {"share": share(call1[p, q]), "range": rng(call1[p, q])}
            for p in range(n - 1)
            for q in range(p + 1, n)
        },
    }
    if overcall:
        out["overcall"] = {
            f"{names[r]} vs {names[p]} and {names[q]}": {"share": share(call2[p, q, r]), "range": rng(call2[p, q, r])}
            for p, q, r in trip
        }
    if arrays:
        out["arrays"] = {"push": push, "call1": call1, "call2": call2, "node": node, "U": U, "index": index,
                         "diag": diag}  # fmt: skip
    return out


def _brute3(stacks, payouts, bb_ante, strat, matrix, T3):
    """Everyone's expected ICM value at a 3-handed table (button, small blind, big blind) for given strategies, by
    enumerating every triple of classes (the tree written out by hand): an independent check of solve2's walk.
    Returns (brute force, solve2's value of the same strategies)."""
    cls, E, w = matrix
    push, call1, call2 = strat
    ref = solve2(stacks, payouts, iters=0, matrix=matrix, bb_ante=bb_ante, equity3=T3, init=strat, arrays=True)
    U, ix = ref["arrays"]["U"], ref["arrays"]["index"]
    u = lambda *key: U[ix[key]]  # noqa: E731
    W = w / w.sum()
    prob = W[:, None, None] * W[None, :, None] * W[None, None, :]  # [a, b, c]

    def two(P, Q, eq):  # an all-in between P and Q, P's equity eq [a, b, c]
        return eq[..., None] * u("allin", P, Q, P) + (1 - eq[..., None]) * u("allin", P, Q, Q)

    shape = T3.shape[:3]
    p0 = push[0][:, None, None, None]
    c01 = call1[0, 1][None, :, None, None]
    c012 = call2[0, 1, 2][None, None, :, None]
    c02 = call1[0, 2][None, None, :, None]
    p1 = push[1][None, :, None, None]
    c12 = call1[1, 2][None, None, :, None]
    three = np.einsum("abcj,jn->abcn", T3, np.array([u("allin3", 0, 1, 2, j) for j in range(6)]))
    v01 = two(0, 1, np.broadcast_to(E[:, :, None], shape))
    v02 = two(0, 2, np.broadcast_to(E[:, None, :], shape))
    v12 = two(1, 2, np.broadcast_to(E[None, :, :], shape))
    val = p0 * (c01 * (c012 * three + (1 - c012) * v01) + (1 - c01) * (c02 * v02 + (1 - c02) * u("steal", 0)))
    val = val + (1 - p0) * (p1 * (c12 * v12 + (1 - c12) * u("steal", 1)) + (1 - p1) * u("walk"))
    # the values each decision is taken on, by the decider's class: the button shoving, the small blind calling the
    # button's shove, the big blind overcalling both
    shove = c01 * (c012 * three + (1 - c012) * v01) + (1 - c01) * (c02 * v02 + (1 - c02) * u("steal", 0))
    rp = push[0] * w / (push[0] * w).sum()
    rq = call1[0, 1] * w / (call1[0, 1] * w).sum()
    called = c012 * three + (1 - c012) * v01
    brute = {
        ("push", 0): np.einsum("abc,b,c->a", shove[..., 0], W, W),
        ("call1", 0, 1): np.einsum("abc,a,c->b", called[..., 1], rp, W),
        ("call2", 0, 1, 2): np.einsum("abc,a,b->c", three[..., 2], rp, rq),
    }
    diag = ref["arrays"]["diag"]
    gap = max(float(np.abs(v - diag[key]).max()) for key, v in brute.items())
    return (val * prob[..., None]).sum((0, 1, 2)), ref["arrays"]["node"][0], gap


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
    for P in (3, 5, 7):
        st = rng.random((3, P)) * 100 + 1
        pay = sorted(rng.random(P), reverse=True)[: P - 1]
        got = icm_mw_np(st, pay)
        for g in range(3):
            ref = _mw_brute(list(st[g]), pay)
            assert max(abs(a - b) for a, b in zip(got[g], ref)) < 1e-9
        assert np.allclose(got.sum(1), sum(pay))  # every prize is paid out
    print("Malmuth-Weitzman matches brute force (3, 5, 7 players)")
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
    if not os.path.exists(EQUITY3):
        print(f"no {EQUITY3}: the two-caller checks skipped (build it with equity3_169.py)")
        return
    selftest2(m)


FT9 = [26.0, 17.5, 12.6, 9.4, 7.2, 5.6, 4.4, 3.6, 3.0]  # a 9-handed final-table ladder (% of the pool at the table)


def selftest2(m):
    """The two-caller solver: the three-way table, side pots, an independent tree walk, the one-caller reduction and
    chip-EV sanity."""
    cls = m[0]
    T3 = load_equity3()
    i = {c: j for j, c in enumerate(cls)}
    aa = T3[i["AA"], i["KK"], i["QQ"]]
    p_aa = aa[0] + aa[1]  # orders with position 0 (AA) best
    assert np.allclose(T3.sum(3), 1, atol=1e-3) and 0.64 < p_aa < 0.70, p_aa
    j = T3[i["AKs"], i["72o"], i["QQ"]]
    jt = T3[i["QQ"], i["AKs"], i["72o"]]  # the same triple in another order: (a, b, c) -> positions (1, 2, 0)
    best_qq = jt[0] + jt[1]
    assert abs(best_qq - (j[4] + j[5])) < 1e-6, (best_qq, j)
    print(f"three-way table: AA best against KK and QQ {p_aa:.3f}; orders consistent under permutation")
    # side pots: button 10, small blind 4, big blind 7 all in, small blind best, then big blind
    r = solve2([10, 4, 7], [1], iters=0, matrix=m, equity3=T3, arrays=True)
    U, ix = r["arrays"]["U"], r["arrays"]["index"]
    got = U[ix[("allin3", 0, 1, 2, ORDERS.index((1, 2, 0)))]] * 21
    assert np.allclose(got, [3, 12, 6], atol=1e-6), got
    print("side pots: 10/4/7 all in, the 4 best then the 7: stacks 3, 12, 6")
    rng = np.random.default_rng(1)
    k = len(cls)
    for stacks, pay, ante in (([12, 5, 9], FT9[:3], 1.0), ([6, 15, 3], [1], 0.0), ([8, 8, 8], [50, 30, 0], 1.0)):
        strat = (rng.random((2, k)), rng.random((3, 3, k)), rng.random((3, 3, 3, k)))
        bf, sv, gap = _brute3(stacks, pay, ante, strat, m, T3)
        assert np.allclose(bf, sv, atol=1e-5 * sum(pay)), (bf, sv)
        assert gap < 1e-5 * sum(pay), gap
    print("solve2's tree walk matches enumerating every class triple (3 tables, random strategies): everyone's value,"
          " and the shove, call and overcall values by class")  # fmt: skip
    n = 5  # the batched three-way sums over many (shover, caller, overcaller) triples at once
    strat = (rng.random((n - 1, k)), rng.random((n, n, k)), rng.random((n, n, n, k)))
    r = solve2([9, 12, 7, 10, 8], FT9[:5], iters=0, matrix=m, equity3=T3, init=strat, arrays=True)
    w = m[2]
    for (p, q, s), Yb in r["arrays"]["diag"]["Y"].items():
        rq, r2 = strat[1][p, q] * w, strat[2][p, q, s] * w
        Yd = np.einsum("abcj,b,c->aj", T3, rq, r2) / (rq.sum() * r2.sum())
        assert np.abs(Yb - Yd).max() < 1e-5, (p, q, s)
    print(f"the batched three-way sums match a direct sum for all {len(r['arrays']['diag']['Y'])} triples of 5 players")
    for stacks, pay in (([10, 7, 12, 9, 14], FT9[:5]), ([9, 9, 9, 9, 9, 9], FT9[:5] + [0])):
        a1 = solve(stacks, pay, iters=300, matrix=m, bb_ante=1.0)
        a2 = solve2(stacks, pay, iters=300, matrix=m, bb_ante=1.0, overcall=False)
        d = max(abs(a1["shove"][x]["share"] - a2["shove"][x]["share"]) for x in a1["shove"])
        d = max(d, max(abs(a1["call"][x]["share"] - a2["call"][x]["share"]) for x in a1["call"]))
        d = max(d, max(abs(x - y) for x, y in zip(a1["icm_equity"], a2["icm_equity"])))
        assert d < 1e-3, d
    print("overcalls forbidden: solve2 reproduces the one-caller solver (5 and 6 players)")
    c = solve2([10, 10, 10, 10], [1], iters=600, matrix=m, equity3=T3, arrays=True)
    eq = c["icm_equity"]
    c2 = c["arrays"]["call2"]
    assert abs(sum(eq) - 1) < 1e-3, eq
    assert all(
        c2[p, q, r, i["AA"]] > 0.95 and c2[p, q, r, i["72o"]] < 0.05 for p, q, r in itertools.combinations(range(4), 3)
    )
    assert c["max_gain_bb"] < 0.05, c["max_gain_bb"]
    one = solve([10, 10, 10, 10], [1], iters=600, matrix=m)
    print(f"winner-take-all, 4 x 10bb: equities sum to {sum(eq):.4f}, AA always overcalls and 72o never, max gain "
          f"{c['max_gain_bb']} bb; shove shares {[v['share'] for v in c['shove'].values()]} with two callers, "
          f"{[v['share'] for v in one['shove'].values()]} with one")  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stacks")
    ap.add_argument("--payouts", default="1")
    ap.add_argument("--ante", type=float, default=0.0)
    ap.add_argument("--bb-ante", type=float, default=0.0, help="big-blind ante (bb), for every scenario with --charts")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--charts")
    ap.add_argument("--model", choices=("mh", "mw"), default="mh", help="ICM: Malmuth-Harville or Malmuth-Weitzman")
    ap.add_argument("--callers", type=int, choices=(1, 2), default=1, help="2: an overcall is allowed (solve2)")
    a = ap.parse_args()
    run = solve if a.callers == 1 else solve2
    if a.selftest:
        selftest()
        return
    if a.stacks:
        c = run([float(x) for x in a.stacks.split(",")], [float(x) for x in a.payouts.split(",")], a.ante, a.iters,
                  bb_ante=a.bb_ante, model=a.model)  # fmt: skip
        print(json.dumps({k: v for k, v in c.items() if k not in ("call", "overcall")}, indent=1))
        return
    if a.charts:
        m = load_matrix()
        out = []
        for name, st, pay, ante in SCENARIOS:
            c = run(st, pay, 0.0 if a.bb_ante else ante, a.iters, matrix=m, bb_ante=a.bb_ante, model=a.model)
            c["name"] = name + (f", big-blind ante {a.bb_ante:g}bb" if a.bb_ante else "")
            out.append(c)
            sh = " ".join(f"{v['share']:.0%}" for v in c["shove"].values())
            bb = c["call"][f"big blind vs {names_for(len(st))[-2]}"]["share"]
            print(f"{name:45s} shove by position: {sh} | BB calls SB {bb:.0%} | max gain {c['max_gain_bb']} bb")
        method = "Malmuth-Weitzman" if a.model == "mw" else "Malmuth-Harville"
        callers = "at most one caller" if a.callers == 1 else "up to two callers"
        json.dump({"method": f"push/fold under ICM ({method}), {callers}, fictitious play", "charts": out},
                  open(a.charts, "w"), indent=1)  # fmt: skip


if __name__ == "__main__":
    main()
