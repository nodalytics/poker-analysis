"""Poker maths at scale: multi-way Monte Carlo equity, a 169x169 preflop equity table, and push/fold equilibria.

Layer 1  equity_multiway(hero, villains, board, dead, sims, workers)
         Hero's equity against any number of opponents, each a PokerStove range ("TT+, AKs") or "random", on any
         board, with dead cards; run-outs are split across processes. Returns equity, tie share and a 95% interval.
Layer 2  build_matrix(path)  ->  169x169 all-in preflop equity of every hand class against every other (eval7
         Monte Carlo per pair), written once to JSON.
         solve_push_fold(matrix, stack_bb)  ->  heads-up push/fold equilibrium: small blind shoves or folds, big blind
         calls or folds, blinds 0.5/1, found by fictitious play (each side best-responds to the other's running
         average until both settle). The standard simplification: card removal between the two hands is ignored,
         and it is heads-up only; multi-way preflop equilibria are a much larger problem.

CLI:  uv run python scripts/poker_math.py matrix OUT.json [SIMS] [WORKERS]
      uv run python scripts/poker_math.py nash MATRIX.json OUT.json
"""

import itertools
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import eval7

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def classes():
    """The 169 starting-hand classes, strongest ranks first: 'AA', 'AKs', 'AKo', ..."""
    out = []
    for i in range(12, -1, -1):
        for j in range(i, -1, -1):
            a, b = RANKS[i], RANKS[j]
            out += [a + b] if i == j else [a + b + "s", a + b + "o"]
    return out


def combos(cls):
    a, b = cls[0], cls[1]
    if len(cls) == 2:
        return [(a + s1, b + s2) for s1, s2 in itertools.combinations(SUITS, 2)]
    if cls[2] == "s":
        return [(a + s, b + s) for s in SUITS]
    return [(a + s1, b + s2) for s1 in SUITS for s2 in SUITS if s1 != s2]


def n_combos(cls):
    return 6 if len(cls) == 2 else 4 if cls[2] == "s" else 12


# ----------------------------------------------------------------------------------------------- layer 1
def _range_hands(spec):
    if spec in ("random", "any", "*"):
        return None
    return [tuple(str(c) for c in h) for h, _w in eval7.HandRange(spec).hands]


def _mc_chunk(args):
    hero, villains, board, dead, sims, seed = args
    rng = random.Random(seed)
    deck = [str(c) for c in eval7.Deck().cards]
    fixed = set(hero + board + dead)
    pools = [_range_hands(v) for v in villains]
    win = tie = 0.0
    done = 0
    for _ in range(sims * 3):  # retries for range collisions are bounded
        if done >= sims:
            break
        used = set(fixed)
        hands, ok = [], True
        for pool in pools:
            if pool is None:
                rest = [c for c in deck if c not in used]
                h = tuple(rng.sample(rest, 2))
            else:
                choices = [h for h in pool if h[0] not in used and h[1] not in used]
                if not choices:
                    ok = False
                    break
                h = rng.choice(choices)
            used.update(h)
            hands.append(h)
        if not ok:
            continue
        rest = [c for c in deck if c not in used]
        run = board + rng.sample(rest, 5 - len(board))
        b7 = [eval7.Card(c) for c in run]
        me = eval7.evaluate([eval7.Card(c) for c in hero] + b7)
        best = max(eval7.evaluate([eval7.Card(c) for c in h] + b7) for h in hands)
        if me > best:
            win += 1
        elif me == best:
            n_best = 1 + sum(eval7.evaluate([eval7.Card(c) for c in h] + b7) == me for h in hands)
            tie += 1 / n_best
        done += 1
    return win, tie, done


def equity_multiway(hero, villains, board=(), dead=(), sims=200_000, workers=None):
    """hero: two cards like ['Ah', 'Kd']; villains: list of ranges ('random' or 'TT+, AQs+'); board: 0-5 cards."""
    hero, board, dead = list(hero), list(board), list(dead)
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    per = math.ceil(sims / workers)
    jobs = [(hero, list(villains), board, dead, per, 1000 + k) for k in range(workers)]
    if workers == 1:
        parts = [_mc_chunk(jobs[0])]
    else:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_mc_chunk, jobs))
    win, tie, n = (sum(p[i] for p in parts) for i in range(3))
    eq = (win + tie) / n if n else float("nan")
    half = 1.96 * math.sqrt(max(eq * (1 - eq), 1e-12) / n) if n else float("nan")
    return {
        "equity": round(eq, 4),
        "win": round(win / n, 4),
        "tie_share": round(tie / n, 4),
        "runouts": n,
        "ci95": [round(eq - half, 4), round(eq + half, 4)],
        "opponents": len(villains),
    }


# ----------------------------------------------------------------------------------------------- layer 2
def _pair_equity(args):
    ci, cj, sims = args
    # hero combo: the first combo of class i; villain: every combo of class j that does not collide
    hero = [eval7.Card(c) for c in combos(ci)[0]]
    vill = [h for h in combos(cj) if not ({h[0], h[1]} & set(combos(ci)[0]))]
    if not vill:
        return ci, cj, None
    rng = eval7.HandRange(",".join(a + b for a, b in vill))
    return ci, cj, eval7.py_hand_vs_range_monte_carlo(hero, rng, [], sims)


def build_matrix(path, sims=20000, workers=None):
    cls = classes()
    jobs = [(a, b, sims) for a in cls for b in cls]
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    eq = {}
    with ProcessPoolExecutor(workers) as ex:
        for a, b, v in ex.map(_pair_equity, jobs, chunksize=64):
            eq.setdefault(a, {})[b] = v
    json.dump({"classes": cls, "sims_per_pair": sims, "equity": eq}, open(path, "w"))
    return path


def solve_push_fold(matrix, stack_bb, iters=4000):
    """Heads-up push/fold equilibrium at an effective stack of stack_bb big blinds (blinds 0.5/1, no ante).

    Returns the shove and call sets and both players' expected value (in big blinds) at the average strategies.
    """
    cls, E = matrix["classes"], matrix["equity"]
    w = {c: n_combos(c) for c in cls}
    total = sum(w.values())
    S = stack_bb
    push = {c: 1.0 for c in cls}  # running averages, start: shove everything / call everything
    call = {c: 1.0 for c in cls}

    def eq(a, b):
        v = E[a][b]
        return 0.5 if v is None else v

    for t in range(1, iters + 1):
        # BB best response to SB's average push range
        pr = {c: push[c] * w[c] for c in cls}
        prw = sum(pr.values()) or 1e-9
        br_call = {}
        for b in cls:
            e = sum(pr[a] * (1 - eq(a, b)) for a in cls) / prw  # BB's equity when called vs the push range
            br_call[b] = 1.0 if (2 * S * e - S) > -1 else 0.0  # call (net) beats folding the big blind (-1)
        # SB best response to BB's average call range
        cw = {b: call[b] * w[b] for b in cls}
        p_call = sum(cw.values()) / total
        br_push = {}
        for a in cls:
            cws = sum(cw.values()) or 1e-9
            e = sum(cw[b] * eq(a, b) for b in cls) / cws
            ev_push = (1 - p_call) * 1.0 + p_call * (2 * S * e - S)
            br_push[a] = 1.0 if ev_push > -0.5 else 0.0  # folding the small blind loses 0.5
        lr = 1 / (t + 1)
        for c in cls:
            push[c] += lr * (br_push[c] - push[c])
            call[c] += lr * (br_call[c] - call[c])
    return {
        "stack_bb": S,
        "push": sorted([c for c in cls if push[c] >= 0.5], key=cls.index),
        "call": sorted([c for c in cls if call[c] >= 0.5], key=cls.index),
        "push_share": round(sum(w[c] for c in cls if push[c] >= 0.5) / total, 3),
        "call_share": round(sum(w[c] for c in cls if call[c] >= 0.5) / total, 3),
    }


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "matrix":
        build_matrix(
            sys.argv[2],
            int(sys.argv[3]) if len(sys.argv) > 3 else 20000,
            int(sys.argv[4]) if len(sys.argv) > 4 else None,
        )
        print("wrote", sys.argv[2])
    elif cmd == "nash":
        m = json.load(open(sys.argv[2]))
        charts = [solve_push_fold(m, s) for s in range(1, 26)]
        json.dump(
            {
                "blinds": "0.5/1",
                "ante": 0,
                "method": "heads-up push/fold, fictitious play, no card removal",
                "charts": charts,
            },
            open(sys.argv[3], "w"),
            indent=1,
        )
        for c in charts:
            print(f"{c['stack_bb']:>2}bb  shove {c['push_share']:.0%}  call {c['call_share']:.0%}")
