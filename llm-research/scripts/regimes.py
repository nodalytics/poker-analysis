"""Do real players change regime? Tilt after big pots and hidden-Markov regimes in HandHQ, each against a null.

Usage: uv run python scripts/regimes.py HANDHQ_DIR OUTDIR [HANDS_PER_FOLDER] [WORKERS]
Players' hands in time order (ids are stable within a site/stake folder; results exact from a PokerKit replay,
predictive.read_folder).

1. Tilt: after a hand lost by 30+ big blinds, does a player play more hands (VPIP) or raise more (PFR) over the next
   10 hands than their own baseline? Paired within each player. Controls: the 10 hands after a 30+ big blind WIN (is
   it the loss, or any big pot?), and placebo events at random hands (should be zero).
2. Regimes: a two-state hidden Markov model of each regular's VPIP (one hidden style switching to the other with some
   probability per hand), fitted by Baum-Welch; its log-likelihood gain over a single fixed rate. Null: the same
   player's hands shuffled, which keeps their overall rate but destroys any order. Real regimes show a gain beyond
   the shuffled one; a model that finds "regimes" in shuffled data is only fitting noise.
"""

import glob
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predictive  # noqa: E402


def sequences(data):
    """player -> list of (vpip, pfr, net_bb) in time order."""
    seq = {}
    for name, hands in data.items():
        for hd in hands:
            vp, pr = set(), set()
            for r in hd["rows"]:
                if r["street"] != "preflop":
                    continue
                if r["action"] in ("cc", "cbr") and not (r["action"] == "cc" and r["to_call"] == 0):
                    vp.add(r["player"])
                if r["action"] == "cbr":
                    pr.add(r["player"])
            for p, x in zip(hd["players"], hd["net"]):
                seq.setdefault(f"{name}|{p}", []).append((p in vp, p in pr, x))
    return seq


def tilt(seq, rng, threshold=30, window=10, min_hands=300):
    out = {"loss": [], "win": [], "placebo": []}
    for p, s in seq.items():
        if len(s) < min_hands:
            continue
        v = np.array([x[0] for x in s], float)
        r = np.array([x[1] for x in s], float)
        net = np.array([x[2] for x in s])
        for kind, idx in (("loss", np.flatnonzero(net <= -threshold)), ("win", np.flatnonzero(net >= threshold))):
            idx = idx[idx < len(s) - window]
            if len(idx) == 0:
                continue
            mask = np.zeros(len(s), bool)
            for k in idx:
                mask[k + 1 : k + 1 + window] = True
            if mask.all() or not mask.any():
                continue
            out[kind].append((v[mask].mean() - v[~mask].mean(), r[mask].mean() - r[~mask].mean()))
            if kind == "loss":
                fake = rng.choice(np.arange(len(s) - window), len(idx), replace=False)
                fm = np.zeros(len(s), bool)
                for k in fake:
                    fm[k + 1 : k + 1 + window] = True
                if fm.any() and not fm.all():
                    out["placebo"].append((v[fm].mean() - v[~fm].mean(), r[fm].mean() - r[~fm].mean()))
    rep = {}
    for k, xs in out.items():
        a = np.array(xs)
        if len(a) < 3:
            continue
        rep[k] = {"players": len(a)}
        for j, lab in enumerate(("vpip", "pfr")):
            m, se = a[:, j].mean(), a[:, j].std(ddof=1) / math.sqrt(len(a))
            rep[k][f"{lab}_change"] = round(float(m), 4)
            rep[k][f"{lab}_t"] = round(float(m / se), 2)
    return rep


def hmm2(x, iters=40):
    """Two-state Bernoulli HMM by Baum-Welch; returns (log-likelihood, rates, stay probabilities)."""
    T = len(x)
    p = np.array([max(0.02, x.mean() * 0.6), min(0.98, x.mean() * 1.4 + 0.02)])
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    pi = np.array([0.5, 0.5])
    ll = -np.inf
    for _ in range(iters):
        B = np.where(x[:, None] == 1, p[None], 1 - p[None])
        alpha = np.zeros((T, 2))
        c = np.zeros(T)
        alpha[0] = pi * B[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]
        beta = np.ones((T, 2))
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        g = alpha * beta
        g /= g.sum(1, keepdims=True)
        xi = (alpha[:-1, :, None] * A[None] * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
        A = xi.sum(0) / np.maximum(g[:-1].sum(0)[:, None], 1e-12)
        p = np.clip((g * x[:, None]).sum(0) / np.maximum(g.sum(0), 1e-12), 1e-3, 1 - 1e-3)
        pi = g[0]
        ll = float(np.log(c).sum())
    return ll, p, np.diag(A)


def hmm_gain(args):
    x, shuffles, seed = args
    x = np.array(x, float)
    q = min(max(x.mean(), 1e-3), 1 - 1e-3)
    ll1 = float((x * math.log(q) + (1 - x) * math.log(1 - q)).sum())
    ll2, p, stay = hmm2(x)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(shuffles):
        y = rng.permutation(x)
        null.append((hmm2(y)[0] - ll1) / len(x))
    order = np.argsort(p)
    return {
        "hands": len(x),
        "gain": (ll2 - ll1) / len(x),
        "null_gains": null,
        "rates": p[order].round(3).tolist(),
        "stay": stay[order].round(4).tolist(),
    }


def main(src, outdir, per_folder=40000, workers=24):
    os.makedirs(outdir, exist_ok=True)
    folders = sorted(d for d in glob.glob(os.path.join(src, "*")) if os.path.isdir(d))
    with ProcessPoolExecutor(workers) as pool:
        data = dict(pool.map(predictive.read_folder, [(f, per_folder) for f in folders]))
    seq = sequences(data)
    rng = np.random.default_rng(0)
    rep = {"hands": sum(len(v) for v in data.values()), "players": len(seq), "tilt": tilt(seq, rng)}
    regs = [(s, p) for p, s in seq.items() if len(s) >= 800]
    random.Random(0).shuffle(regs)
    regs = regs[:400]
    jobs = [([x[0] for x in s], 10, k) for k, (s, _) in enumerate(regs)]
    with ProcessPoolExecutor(workers) as pool:
        fits = list(pool.map(hmm_gain, jobs))
    beat = [f["gain"] > np.quantile(f["null_gains"], 0.95) for f in fits]
    persist = [f for f, b in zip(fits, beat) if b]
    rep["hmm"] = {
        "players_fitted": len(fits),
        "mean_gain_per_hand_real": round(float(np.mean([f["gain"] for f in fits])), 5),
        "mean_gain_per_hand_shuffled": round(float(np.mean([np.mean(f["null_gains"]) for f in fits])), 5),
        "share_beating_own_shuffled_95": round(float(np.mean(beat)), 3),
        "expected_share_if_no_regimes": 0.05,
        "regime_players_median_rates": [round(float(np.median([f["rates"][k] for f in persist])), 3) for k in (0, 1)]
        if persist
        else None,
        "regime_players_median_length_hands": [
            round(float(np.median([1 / max(1 - f["stay"][k], 1e-3) for f in persist])), 1) for k in (0, 1)
        ]
        if persist
        else None,
    }
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:4]))
