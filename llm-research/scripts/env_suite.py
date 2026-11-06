"""Environment suite: which strategy profits most, and most reliably, across many kinds of poker game?

Usage: uv run python scripts/env_suite.py OUTDIR SESSIONS_PER_CELL [WORKERS] [CANDIDATE ...]
Environments: table size (heads-up, 6, 9) x stack depth (5-20bb dealt fresh each hand, or 100bb) x opponent mix:
  fish     loose, passive players (bob, carl and random styles that call a lot)
  regs     strong players (solver bots, sol, tina+, monte, tina)
  maniacs  max and random styles that raise and bluff a lot
  styles   random styles from the whole personality space (seeds never used in RL training)
  mixed    one of each
  shifters players who switch between two random styles during the session (regime changes, ~17 hands a regime)
Each candidate takes the same seat against the same opponents with the same seeds (paired), 24-hand sessions.
Default candidates: solverbot, adaptbot, rlavg, rlbot, monte, tinaplus.

Reports big blinds per 100 hands with a 95% bootstrap interval per environment and candidate, and per candidate the
mean over environments, the worst and best environment, and how many environments it wins in with 95% confidence.
"""

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poker_selfplay as ps  # noqa: E402

HANDS = 24
SIZES = [2, 6, 9]
DEPTHS = ["short", "100bb"]
MIXES = ["fish", "regs", "maniacs", "styles", "mixed", "shifters"]
CANDIDATES = ["solverbot", "adaptbot", "adaptbot_recent", "rlavg", "rlbot", "monte", "tinaplus"]


def styles_where(rng, test, n, start):
    out, seed = [], start + rng.randrange(1_000_000)
    while len(out) < n:
        name = f"style_{seed}"
        if test(ps.style_params(name)):
            out.append(name)
        seed += 1
    return out


def opponents(mix, n, rng):
    loose = lambda p: p["call"] <= 6 and p["bluff"] < 0.15  # noqa: E731
    wild = lambda p: p["raise_"] <= 6 and p["bluff"] >= 0.2  # noqa: E731
    anyp = lambda p: True  # noqa: E731
    if mix == "fish":
        pool = ["bob", "carl"] + styles_where(rng, loose, n, 5_000_000)
    elif mix == "regs":
        pool = ["solverbot#o1", "sol", "tinaplus#o", "monte", "tina", "solverbot#o2", "solverbot#o3", "solverbot#o4"]
    elif mix == "maniacs":
        pool = ["max"] + styles_where(rng, wild, n, 6_000_000)
    elif mix == "styles":
        pool = styles_where(rng, anyp, n, 7_000_000)
    elif mix == "shifters":  # players who switch style during the session (regime changes)
        base = 9_000_000 + rng.randrange(100_000) * 10
        pool = [f"shift_{base + k}" for k in range(n)]
    else:
        pool = ["carl", "solverbot#o1", "max", "nora", "tina", "bob"] + styles_where(rng, anyp, n, 8_000_000)
    pool = pool[:]
    rng.shuffle(pool)
    return pool[:n]


def one(args):
    outdir, size, depth, mix, cand, k = args
    rng = random.Random(hash((size, depth, mix, k)) & 0xFFFFFFFF)  # same opponents for every candidate
    opp = opponents(mix, size - 1, rng)
    seats = [cand] + opp
    seats = seats[k % len(seats) :] + seats[: k % len(seats)]
    random.seed(k)
    d = os.path.join(outdir, "sessions", f"{size}_{depth}_{mix}_{cand}_{k}")
    chips, _ = ps.main(d, HANDS, seats, short_stack=(5, 20) if depth == "short" else None)
    import shutil

    shutil.rmtree(d, ignore_errors=True)
    return size, depth, mix, cand, k, (chips[cand] - ps.START) / 2 / HANDS * 100


def main(outdir, sessions, workers=40, candidates=None):
    candidates = candidates or CANDIDATES
    os.makedirs(outdir, exist_ok=True)
    jobs = [
        (outdir, s, d, m, c, k)
        for s in SIZES
        for d in DEPTHS
        for m in MIXES
        for c in candidates
        for k in range(sessions)
    ]
    random.Random(0).shuffle(jobs)  # spread the slow cells across workers
    res = {}
    log = open(os.path.join(outdir, "sessions.jsonl"), "a")
    with ProcessPoolExecutor(workers) as pool:
        for n, f in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
            s, d, m, c, k, bb = f.result()
            res.setdefault((s, d, m, c), []).append(bb)
            log.write(json.dumps({"size": s, "depth": d, "mix": m, "cand": c, "k": k, "bb100": bb}) + "\n")
            if n % 500 == 0:
                log.flush()
                print(f"{n}/{len(jobs)} sessions", flush=True)
    rng = random.Random(1)

    def ci(x):
        b = sorted(sum(rng.choice(x) for _ in x) / len(x) for _ in range(1000))
        return round(b[25], 1), round(b[974], 1)

    cells = {}
    for (s, d, m, c), x in res.items():
        lo, hi = ci(x)
        cells.setdefault(f"{s}-max {d} {m}", {})[c] = {
            "bb100": round(sum(x) / len(x), 1),
            "ci95": [lo, hi],
            "n": len(x),
        }
    summary = {}
    for c in candidates:
        vals = {e: v[c] for e, v in cells.items() if c in v}
        means = [v["bb100"] for v in vals.values()]
        worst = min(vals, key=lambda e: vals[e]["bb100"])
        best = max(vals, key=lambda e: vals[e]["bb100"])
        summary[c] = {
            "mean_bb100": round(sum(means) / len(means), 1),
            "worst": [worst, vals[worst]["bb100"]],
            "best": [best, vals[best]["bb100"]],
            "envs_won_95": sum(v["ci95"][0] > 0 for v in vals.values()),
            "envs_lost_95": sum(v["ci95"][1] < 0 for v in vals.values()),
            "envs": len(vals),
        }
    json.dump(
        {"hands_per_session": HANDS, "sessions_per_cell": sessions, "summary": summary, "cells": cells},
        open(os.path.join(outdir, "report.json"), "w"),
        indent=1,
    )
    for c, v in sorted(summary.items(), key=lambda x: -x[1]["mean_bb100"]):
        print(
            f"{c:10} mean {v['mean_bb100']:+7.1f} bb/100  won {v['envs_won_95']}/{v['envs']}  lost {v['envs_lost_95']}"
            f"  worst {v['worst']}  best {v['best']}"
        )


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), *(int(x) for x in a[2:3]), a[3:] or None)
