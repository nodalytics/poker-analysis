"""The strategy population as a game: payoff matrix, Nash mix, evolution (replicator dynamics) and ecology.

Usage: uv run python scripts/meta_game.py OUTDIR SESSIONS_PER_PAIR [WORKERS] [--100bb]
Every pair of strategies plays heads-up (24-hand sessions, seats alternating; 5-20bb dealt fresh each hand, or 100bb
with --100bb). M[i][j] is i's big blinds per 100 hands against j; heads-up poker without rake is zero-sum, so
M[j][i] = -M[i][j]. On that matrix:
  game theory     the Nash mix of the meta-game (the population no single strategy can beat), by linear programming
                  ("empirical game-theoretic analysis"); and which strategies no other strategy beats heads-up
  biology         replicator dynamics from an even population: each strategy's share grows with its payoff against
                  the current population; who survives. An evolutionarily stable strategy here would have to beat
                  every other strategy outright
  ecology         an open ecosystem: weak players (fish) keep arriving at a fixed rate and leave as they lose;
                  which strategies the inflow feeds, and their long-run profit per hand
"""

import itertools
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poker_selfplay as ps  # noqa: E402

HANDS = 24
STRATEGIES = [
    "solverbot",
    "adaptbot",
    "rlavg",
    "rlbot",
    "monte",
    "sol",
    "tina",
    "tinaplus",
    "bob",
    "max",
    "carl",
    "nora",
    "style_9000001",
    "style_9000002",
]
FISH = {"bob", "max", "carl", "style_9000001", "style_9000002"}


def one(args):
    outdir, a, b, k, short = args
    seats = [a, b] if k % 2 == 0 else [b, a]
    random.seed(k)
    d = os.path.join(outdir, "sessions", f"{a}_{b}_{k}")
    chips, _ = ps.main(d, HANDS, seats, short_stack=(5, 20) if short else None)
    import shutil

    shutil.rmtree(d, ignore_errors=True)
    return a, b, (chips[a] - ps.START) / 2 / HANDS * 100


def nash(M):
    """Maximin mixed strategy of the symmetric zero-sum meta-game."""
    from scipy.optimize import linprog

    n = len(M)
    # maximise v subject to sum_i x_i M[i][j] >= v for all j, sum x = 1, x >= 0
    c = np.zeros(n + 1)
    c[-1] = -1
    A = np.hstack([-M.T, np.ones((n, 1))])
    res = linprog(c, A_ub=A, b_ub=np.zeros(n), A_eq=[[1] * n + [0]], b_eq=[1], bounds=[(0, None)] * n + [(None, None)])
    return res.x[:n], res.x[-1]


def replicator(M, x0, steps=20000, dt=0.01, inflow=None):
    x = x0.copy()
    traj = []
    for t in range(steps):
        f = M @ x
        x = x + dt * x * (f - x @ f) / 100
        if inflow is not None:
            x = x + dt * inflow
        x = np.clip(x, 0, None)
        x /= x.sum()
        if t % 2000 == 0:
            traj.append(x.round(4).tolist())
    return x, traj


def main(outdir, sessions, workers=40, short=True):
    os.makedirs(outdir, exist_ok=True)
    pairs = list(itertools.combinations(STRATEGIES, 2))
    jobs = [(outdir, a, b, k, short) for a, b in pairs for k in range(sessions)]
    random.Random(0).shuffle(jobs)
    res = {}
    log = open(os.path.join(outdir, "sessions.jsonl"), "a")
    with ProcessPoolExecutor(workers) as pool:
        for n, f in enumerate(as_completed([pool.submit(one, j) for j in jobs]), 1):
            a, b, bb = f.result()
            res.setdefault((a, b), []).append(bb)
            log.write(json.dumps({"a": a, "b": b, "bb100": bb}) + "\n")
            if n % 500 == 0:
                log.flush()
                print(f"{n}/{len(jobs)}", flush=True)
    n = len(STRATEGIES)
    M, SE = np.zeros((n, n)), np.zeros((n, n))
    for (a, b), x in res.items():
        i, j = STRATEGIES.index(a), STRATEGIES.index(b)
        m, se = float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))
        M[i, j], M[j, i] = m, -m
        SE[i, j] = SE[j, i] = se
    x_nash, v = nash(M)
    x_rep, traj = replicator(M, np.ones(n) / n)
    fish = np.array([s in FISH for s in STRATEGIES], dtype=float)
    x_eco, _ = replicator(M, np.ones(n) / n, inflow=0.02 * fish / fish.sum())
    eco_profit = M @ x_eco
    unbeaten = [s for i, s in enumerate(STRATEGIES) if all(M[i, j] + 1.96 * SE[i, j] >= 0 for j in range(n) if j != i)]
    beats_all = [s for i, s in enumerate(STRATEGIES) if all(M[i, j] - 1.96 * SE[i, j] > 0 for j in range(n) if j != i)]
    rep = {
        "short_stacks": short,
        "sessions_per_pair": sessions,
        "strategies": STRATEGIES,
        "payoff_bb100": M.round(1).tolist(),
        "stderr": SE.round(1).tolist(),
        "round_robin_mean": dict(zip(STRATEGIES, (M.sum(1) / (n - 1)).round(1).tolist())),
        "worst_matchup": {
            s: [
                STRATEGIES[int(np.argmin(np.where(np.arange(n) == i, np.inf, M[i])))],
                round(float(np.min(np.where(np.arange(n) == i, np.inf, M[i]))), 1),
            ]
            for i, s in enumerate(STRATEGIES)
        },
        "never_beaten_95": unbeaten,
        "beats_everyone_95 (ESS candidates)": beats_all,
        "nash_mix": {s: round(float(w), 3) for s, w in zip(STRATEGIES, x_nash) if w > 0.005},
        "nash_value": round(float(v), 2),
        "replicator_survivors": {s: round(float(w), 3) for s, w in zip(STRATEGIES, x_rep) if w > 0.005},
        "replicator_trajectory": traj,
        "ecosystem_shares": {s: round(float(w), 3) for s, w in zip(STRATEGIES, x_eco) if w > 0.005},
        "ecosystem_profit_bb100": {s: round(float(p), 1) for s, p in zip(STRATEGIES, eco_profit)},
    }
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(
        json.dumps(
            {k: v for k, v in rep.items() if k not in ("payoff_bb100", "stderr", "replicator_trajectory")}, indent=1
        )
    )


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--100bb"]
    main(a[0], int(a[1]), *(int(x) for x in a[2:3]), short="--100bb" not in sys.argv)
