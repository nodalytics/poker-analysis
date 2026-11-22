"""Luck-adjusted Slumbot results: AIVAT's chance-node corrections in a simple, unbiased form.

Usage: uv run python scripts/aivat_lite.py OUTDIR DIR [DIR ...]   (slumbot_client output directories)
Our logs show both players' cards at the end of every hand, so two kinds of luck can be removed without bias:
  1. All-in adjustment: when both players are all-in before the river, the remaining cards no longer change any
     decision. The hand's result is replaced by its expected value: equity on the board so far x the pot, minus what
     we put in.
  2. Pre-flop card luck (a control variate): the result minus a x (our pre-flop equity against Slumbot's actual hand
     - 0.5). Over random deals that equity averages exactly 0.5 whatever anyone plays, so the mean is unchanged and
     only the noise of who was dealt the better hand is removed. The slope a is fitted on the other half of the hands
     (cross-fitting), per position, so it cannot bias the estimate.
  3. Board-card luck: for each flop, turn and river the hand reached, the pot at that street x (our equity after
     the card - our equity before it). Equity is a martingale over the cards, so each term averages zero whatever
     anyone plays; its weight is cross-fitted like the slope above.
Full AIVAT also corrects our own random choices and uses proper value estimates instead of pot x equity; this
keeps the corrections that need no learned model. Reports bb/100 with a 95% interval, raw and adjusted, per directory
and pooled.
"""

import json
import math
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

BIG, STACK = 100, 20000


def equity(args):
    hero, vill, board, sims, seed = args
    import eval7

    rng = random.Random(seed)
    h = [eval7.Card(c) for c in hero]
    v = [eval7.Card(c) for c in vill]
    b = [eval7.Card(c) for c in board]
    used = set(hero) | set(vill) | set(board)
    deck = [eval7.Card(r + s) for r in "23456789TJQKA" for s in "cdhs" if r + s not in used]
    need = 5 - len(board)
    if need == 0:
        a, o = eval7.evaluate(h + b), eval7.evaluate(v + b)
        return 1.0 if a > o else 0.5 if a == o else 0.0
    if need == 1:  # exact: every river card
        tot = 0.0
        for c in deck:
            a, o = eval7.evaluate(h + b + [c]), eval7.evaluate(v + b + [c])
            tot += 1.0 if a > o else 0.5 if a == o else 0.0
        return tot / len(deck)
    win = 0.0
    for _ in range(sims):
        rest = rng.sample(deck, need)
        a, o = eval7.evaluate(h + b + rest), eval7.evaluate(v + b + rest)
        win += 1.0 if a > o else 0.5 if a == o else 0.0
    return win / sims


def contributions(action, pos):
    """Chips each position put in, the street on which betting ended, and whether both ended all-in."""
    total = [0, 0]
    streets = action.split("/")
    for si, st in enumerate(streets):
        street = [BIG, BIG // 2] if si == 0 else [0, 0]
        actor, last = (1, BIG) if si == 0 else (0, 0)
        for m in re.finditer(r"b(\d+)|[kcf]", st):
            t = m.group()
            if t[0] == "b":
                last = int(m.group(1))
                street[actor] = last
            elif t == "c":
                street[actor] = last
            actor = 1 - actor
        for p in (0, 1):
            total[p] += street[p]
    allin = total[0] >= STACK and total[1] >= STACK
    live_streets = sum(1 for s in streets if s)  # streets with action
    return total, allin, live_streets - 1


def street_pots(action):
    """Pot (both players' chips) at the start of each street, from the action string."""
    pots, total = [], [0, 0]
    for si, st in enumerate(action.split("/")):
        pots.append(total[0] + total[1])
        street = [BIG, BIG // 2] if si == 0 else [0, 0]
        actor, last = (1, BIG) if si == 0 else (0, 0)
        for m in re.finditer(r"b(\d+)|[kcf]", st):
            t = m.group()
            if t[0] == "b":
                last = int(m.group(1))
                street[actor] = last
            elif t == "c":
                street[actor] = last
            actor = 1 - actor
        total = [total[0] + street[0], total[1] + street[1]]
    return pots


def street_luck(rows, pre, allin_rows):
    """Per hand: sum over flop, turn and river of pot at that street x (our equity after the card - before).
    Equity is a martingale over the cards, so each term averages zero whatever anyone plays."""
    jobs, keys = [], []
    for k, r in enumerate(rows):
        if k in allin_rows:
            continue
        board = r["board"] or []
        for n in (3, 4, 5):
            if len(board) >= n:
                jobs.append((r["hole"], r["slumbot_hole"], board[:n], 3000, 20 * k + n))
                keys.append((k, n))
    with ProcessPoolExecutor(32) as pool:
        res = dict(zip(keys, pool.map(equity, jobs, chunksize=64)))
    luck = np.zeros(len(rows))
    for k, r in enumerate(rows):
        if k in allin_rows:
            continue
        pots = street_pots(r["action"])
        prev = pre[k]
        for si, n in ((1, 3), (2, 4), (3, 5)):
            if (k, n) in res and si < len(pots):
                luck[k] += pots[si] * (res[(k, n)] - prev) / BIG
                prev = res[(k, n)]
    return luck


def cross_fit(y, x, seed=1):
    """y - b*x with b fitted on the other half (x has mean zero by construction)."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, len(y))
    out = y.copy()
    for f in (0, 1):
        tr, te = fold != f, fold == f
        b = float((x[tr] * (y[tr] - y[tr].mean())).sum() / max((x[tr] ** 2).sum(), 1e-9))
        out[te] = y[te] - b * x[te]
    return out


def adjust(rows):
    """Add allin-adjusted results; returns arrays (raw, allin_adj, pre_eq, pos)."""
    jobs = []
    for k, r in enumerate(rows):
        jobs.append((r["hole"], r["slumbot_hole"], [], 4000, k))
        total, allin, last_street = contributions(r["action"], r["pos"])
        board_at_allin = (r["board"] or [])[: [0, 3, 4, 5][last_street]] if allin else None
        if allin and len(board_at_allin) < 5:
            jobs.append((r["hole"], r["slumbot_hole"], board_at_allin, 4000, k + 10**7))
    with ProcessPoolExecutor(32) as pool:
        res = list(pool.map(equity, jobs, chunksize=64))
    # results come back in job order: one pre-flop equity per row, interleaved with all-in equities
    pre, allin_eq = [], {}
    it = iter(res)
    for k, r in enumerate(rows):
        pre.append(next(it))
        total, allin, last_street = contributions(r["action"], r["pos"])
        if allin and len((r["board"] or [])[: [0, 3, 4, 5][last_street]]) < 5:
            allin_eq[k] = (next(it), total)
    raw = np.array([r["winnings_chips"] / BIG for r in rows])
    adj = raw.copy()
    for k, (eq, total) in allin_eq.items():
        mine = total[rows[k]["pos"]]
        pot = total[0] + total[1]
        adj[k] = (eq * pot - mine) / BIG
    return raw, adj, np.array(pre), np.array([r["pos"] for r in rows]), set(allin_eq)


def control_variate(y, eq, pos, seed=0):
    """Cross-fitted y - a*(eq - 0.5), slope per position fitted on the other half."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, len(y))
    out = y.copy()
    for f in (0, 1):
        for p in (0, 1):
            tr = (fold != f) & (pos == p)
            te = (fold == f) & (pos == p)
            x = eq[tr] - 0.5
            a = float((x * (y[tr] - y[tr].mean())).sum() / max((x * x).sum(), 1e-9))
            out[te] = y[te] - a * (eq[te] - 0.5)
    return out


def ci(x):
    m = 100 * x.mean()
    se = 100 * x.std(ddof=1) / math.sqrt(len(x))
    return {"bb100": round(m, 1), "ci95": [round(m - 1.96 * se, 1), round(m + 1.96 * se, 1)], "se": round(se, 1)}


def main(outdir, *dirs):
    os.makedirs(outdir, exist_ok=True)
    rep = {}
    for d in dirs:
        rows = []
        for f in sorted(os.listdir(d)):
            if f.startswith("hands_") and f.endswith(".jsonl"):
                rows += [json.loads(x) for x in open(os.path.join(d, f))]
        rows = [r for r in rows if r.get("slumbot_hole")]
        raw, adj, eq, pos, allin_rows = adjust(rows)
        n_allin = len(allin_rows)
        cards = control_variate(adj, eq, pos)
        luck = street_luck(rows, eq, allin_rows)
        full = cross_fit(cards, luck)
        rep[os.path.basename(d.rstrip("/"))] = {
            "hands": len(rows),
            "allin_hands_adjusted": n_allin,
            "raw": ci(raw),
            "allin_adjusted": ci(adj),
            "allin_and_preflop_card_luck_adjusted": ci(cards),
            "plus_flop_turn_river_card_luck": ci(full),
            "variance_reduction_factor": round(float(raw.var() / full.var()), 2),
        }
        print(d, json.dumps(rep[os.path.basename(d.rstrip("/"))]), flush=True)
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
