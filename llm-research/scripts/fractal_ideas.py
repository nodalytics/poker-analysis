"""Fractal ideas applied to a blueprint table (docs/2026-11-09-fractal-ideas.md).

Box counting along hand strength. At every post-flop decision node, the buckets are sorted by mean hand strength
and the strategy is read as a curve over strength. The curve is coarsened to k groups of adjacent buckets (each
replaced by its reach-weighted mean), and the error E(k) is the reach-weighted total-variation distance to the
original. The slope of log E against log k is the curve's roughness: a smooth curve falls like 1/k, a noisy one
barely falls. A control shuffles the bucket order: the gap between the two measures how much of the strategy
follows hand strength at all.

Mariani–Silver (the Mandelbrot renderer's rule: fill a region whose border is one colour). A node whose strategy
is within EPS of its k-group coarsening for small k needs only k stored rows. The adaptive storage is the sum of
each node's smallest such k, compared with storing every bucket.

The reach weight of a (node, bucket) is the row sum of the unnormalised average strategy. The average strategy
accumulates reach times the strategy, so that sum is proportional to how often the bucket arrives there.

Hurst exponent (fractal time-series analysis). Detrended fluctuation analysis of per-hand winnings in the
Slumbot sessions. Every bb/100 interval in this project assumes independent hands (exponent 0.5).

Usage (lab): .venv/bin/python scripts/fractal_ideas.py blueprint/run5 blueprint/out_pa [out.json]
             .venv/bin/python scripts/fractal_ideas.py hurst out.json SESSION_DIR...
"""

import json
import os
import sys

import numpy as np

EPS = 0.02
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def strength(absdir, B):
    """Mean strength of each bucket per post-flop street (1..3), and its order."""
    out = {}
    for st, name in ((1, "flop"), (2, "turn"), (3, "river")):
        e = os.path.join(absdir, f"{name}_ehs.f32") if name != "river" else os.path.join(absdir, "river_hs.f32")
        if os.path.exists(e):  # percentile buckets: mean strength of the states in each bucket
            v = np.fromfile(e, dtype=np.float32)
            bp = os.path.join(absdir, f"{name}_buckets.u8")
            b = np.fromfile(bp, dtype=np.uint8) if os.path.exists(bp) else np.fromfile(bp[:-2] + "u16", np.uint16)
            s = np.bincount(b, weights=v, minlength=B) / np.maximum(np.bincount(b, minlength=B), 1)
        else:  # distribution-aware: each centre is a CDF over strength bins (flop/turn) or 8 OCHS win rates (river)
            c = np.fromfile(os.path.join(absdir, f"{name}_centres.f32"), dtype=np.float32).reshape(B, -1)
            s = 1 - c.mean(1) if name != "river" else c.mean(1)  # mean of a CDF on [0,1] is 1 - E[x]
        out[st] = np.argsort(s, kind="stable")
    return out


def coarsen_error(S, w, k):
    """Reach-weighted TV distance between S (B x na, rows in strength order) and its k-group piecewise mean."""
    B = len(S)
    edges = np.unique(np.linspace(0, B, k + 1).astype(int))
    cw = np.concatenate([[0.0], np.cumsum(w)])
    cws = np.vstack([np.zeros((1, S.shape[1])), np.cumsum(S * w[:, None], 0)])
    gw = cw[edges[1:]] - cw[edges[:-1]]
    means = (cws[edges[1:]] - cws[edges[:-1]]) / np.maximum(gw, 1e-30)[:, None]
    M = np.repeat(means, np.diff(edges), axis=0)
    return float((w * 0.5 * np.abs(S - M).sum(1)).sum() / max(w.sum(), 1e-30))


def dfa(x, scales):
    """Detrended fluctuation analysis (order 1): the slope of log F(n) on log n is the Hurst-like exponent alpha.
    0.5: independent hands; above 0.5: persistent (streaks), so sqrt(n) intervals are too narrow."""
    y = np.cumsum(x - x.mean())
    F = []
    for n in scales:
        m = len(y) // n
        seg = y[: m * n].reshape(m, n)
        t = np.arange(n)
        coef = np.polyfit(t, seg.T, 1)
        fit = np.outer(coef[0], t) + coef[1][:, None]
        F.append(np.sqrt(((seg - fit) ** 2).mean()))
    return float(np.polyfit(np.log(scales), np.log(F), 1)[0])


def hurst(dirs):
    """Per-hand winnings of Slumbot sessions (hands_*.jsonl, one file per worker, in play order): DFA exponent and
    the variance ratio var(sum of m hands) / (m var(one hand)), each against a shuffled copy of the same hands."""
    rng = np.random.default_rng(0)
    scales = [8, 16, 32, 64, 128, 256]
    out = {}
    for d in dirs:
        series = []
        for f in sorted(os.listdir(d)):
            if f.startswith("hands_") and f.endswith(".jsonl"):
                x = np.array([json.loads(ln)["winnings_chips"] for ln in open(os.path.join(d, f))], np.float64)
                if len(x) >= 1000:
                    series.append(x)
        if not series:
            continue
        a = np.mean([dfa(x, scales) for x in series])
        a_sh = np.mean([dfa(rng.permutation(x), scales) for x in series for _ in range(20)])
        vr = {}
        for m in (10, 50, 200):
            num = sum(((x[: len(x) // m * m].reshape(-1, m).sum(1) - m * x.mean()) ** 2).sum() for x in series)
            cnt = sum(len(x) // m for x in series)
            vr[m] = round(float(num / cnt / (m * np.mean([x.var() for x in series]))), 3)
        out[os.path.basename(d.rstrip("/"))] = {
            "hands": int(sum(map(len, series))),
            "dfa": round(a, 3),
            "dfa_shuffled": round(a_sh, 3),
            "variance_ratio": vr,
        }
        print(out[os.path.basename(d.rstrip("/"))], d, flush=True)
    return out


def main():
    if sys.argv[1] == "hurst":
        json.dump(hurst(sys.argv[3:]), open(sys.argv[2], "w"), indent=1)
        return
    run, absdir = sys.argv[1:3]
    B = int(open(os.path.join(absdir, "meta.txt")).read().split()[1])
    order = strength(absdir, B)
    rows = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(run, "tree.tsv"))][1:]
    avg = np.fromfile(os.path.join(run, "avg.f32"), dtype=np.float32)
    rng = np.random.default_rng(0)
    res = {}
    for st in (1, 2, 3):
        E = np.zeros(len(KS))
        Eshuf = np.zeros(len(KS))
        W = 0.0
        full = adaptive = 0
        for r in rows:
            if r[1] != "0" or int(r[3]) != st:
                continue
            na = len(r[7].split(","))
            raw = avg[int(r[6]) : int(r[6]) + B * na].astype(np.float64).reshape(B, na)
            w = raw.sum(1)
            if w.sum() <= 0:
                continue
            S = np.where(w[:, None] > 0, np.maximum(raw, 0) / np.maximum(w[:, None], 1e-30), 1.0 / na)
            o = order[st]
            Ss, ws = S[o], w[o]
            p = rng.permutation(B)
            node_w = w.sum()
            ek = np.array([coarsen_error(Ss, ws, k) for k in KS])
            E += node_w * ek
            Eshuf += node_w * np.array([coarsen_error(S[p], w[p], k) for k in KS])
            W += node_w
            full += B * na
            kmin = next((k for k, e in zip(KS, ek) if e <= EPS), B)
            adaptive += min(kmin, B) * na + (kmin + 1 if kmin < B else 0)  # rows plus group edges
        E /= W
        Eshuf /= W
        fit = slice(2, 7)  # k = 4 .. 64
        slope = np.polyfit(np.log(KS[fit]), np.log(np.maximum(E[fit], 1e-9)), 1)[0]
        res[{1: "flop", 2: "turn", 3: "river"}[st]] = {
            "E_by_k": dict(zip(map(str, KS), np.round(E, 4).tolist())),
            "E_shuffled_by_k": dict(zip(map(str, KS), np.round(Eshuf, 4).tolist())),
            "slope_k4_64": round(float(slope), 3),
            "adaptive_storage_share": round(adaptive / full, 4),
        }
        print(st, res[{1: "flop", 2: "turn", 3: "river"}[st]], flush=True)
    if len(sys.argv) > 3:
        json.dump(res, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
