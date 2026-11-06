"""Value network (AlphaPoker's value net, DeepStack style): predict every hand's value in a turn subgame.

Usage: uv run python scripts/value_net.py VALUE_LIBRARY_DIR OUTDIR [WORKERS]
Reads value_library.py's solves. One example per (solve, player, hand in range): the hand's value as a share of the
pot. Features: the board (ranks, suit and pairing structure), the hand (ranks, made-hand class, draws), its equity
against the opponent's range over all river cards, the share of each range kept, the average equity of each range
against the other (range advantage), the hand's equity percentile within its own range, stack-to-pot, position.

Held out by board (25% of solves). Scored on mean absolute error in pots, R^2, and the error of each range's average
value (what a depth-limited search needs at its leaves). Baselines: the mean, and raw equity (value = equity - 1/2,
i.e. every hand realises exactly its equity). Models: ridge, gradient boosting, extra trees, MLP.
"""

import glob
import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RANKS = "23456789TJQKA"
CATS = ["High Card", "Pair", "Two Pair", "Trips", "Straight", "Flush", "Full House", "Quads", "Straight Flush"]


def cls(h):
    r1, s1, r2, s2 = h[0], h[1], h[2], h[3]
    hi, lo = sorted((r1, r2), key=RANKS.index, reverse=True)
    return hi + lo if hi == lo else hi + lo + ("s" if s1 == s2 else "o")


def equities(board, hands, opp, w_opp):
    """Equity of each hand against the weighted opponent range, averaged over every river card (blockers between the
    two hands ignored; the river card and the board are removed)."""
    import eval7

    b = [eval7.Card(c) for c in board]
    used = set(board)
    rivers = [r + s for r in RANKS for s in "cdhs" if r + s not in used]
    hc = [(eval7.Card(h[:2]), eval7.Card(h[2:])) for h in hands]
    oc = [(eval7.Card(h[:2]), eval7.Card(h[2:])) for h in opp]
    eq, n = np.zeros(len(hands)), 0
    for rv in rivers:
        rc = eval7.Card(rv)
        mine = np.array([eval7.evaluate([a, c, *b, rc]) if rv not in h else -1 for (a, c), h in zip(hc, hands)])
        theirs = np.array([eval7.evaluate([a, c, *b, rc]) if rv not in h else -1 for (a, c), h in zip(oc, opp)])
        w = np.where(theirs >= 0, w_opp, 0.0)
        tot = w.sum()
        if tot <= 0:
            continue
        order = np.argsort(theirs)
        s_sorted, w_cum = theirs[order], np.concatenate([[0], np.cumsum(w[order])])
        lo = np.searchsorted(s_sorted, mine, "left")
        hi = np.searchsorted(s_sorted, mine, "right")
        e = (w_cum[lo] + 0.5 * (w_cum[hi] - w_cum[lo])) / tot
        eq += np.where(mine >= 0, e, 0.5)
        n += 1
    return eq / max(n, 1)


def features_of(path):
    import eval7

    r = json.load(gzip.open(path, "rt"))
    board, pot = r["board"], r["pot"]
    br = sorted((RANKS.index(c[0]) for c in board), reverse=True)
    bs = [c[1] for c in board]
    board_f = [
        *(x / 12 for x in br),
        max(bs.count(s) for s in "cdhs") / 4,
        float(len(set(br)) < 4),
        (br[0] - br[3]) / 12,
    ]
    sides = {}
    for p, vals, rng in (("ip", r["values_ip"], r["range_ip"]), ("oop", r["values_oop"], r["range_oop"])):
        hands = sorted(vals)
        sides[p] = (hands, np.array([vals[h] for h in hands]) / pot, np.array([rng.get(cls(h), 0.0) for h in hands]))
    X, y, w, group_id, side = [], [], [], [], []
    eqs = {}
    for p, q in (("ip", "oop"), ("oop", "ip")):
        eqs[p] = equities(board, sides[p][0], sides[q][0], sides[q][2])
    for p, q in (("ip", "oop"), ("oop", "ip")):
        hands, vals, wts = sides[p]
        e = eqs[p]
        wsum = wts.sum() or 1.0
        range_eq = float((e * wts).sum() / wsum)
        opp_eq = float((eqs[q] * sides[q][2]).sum() / (sides[q][2].sum() or 1.0))
        order = np.argsort(e)
        pct = np.empty(len(e))
        pct[order] = np.cumsum(wts[order]) / wsum
        for k, h in enumerate(hands):
            cards = [eval7.Card(h[:2]), eval7.Card(h[2:])] + [eval7.Card(c) for c in board]
            cat = eval7.handtype(eval7.evaluate(cards))
            hr = sorted((RANKS.index(h[0]), RANKS.index(h[2])), reverse=True)
            suits = bs + [h[1], h[3]]
            allr = sorted(set(br) | set(hr))
            run = best = 1
            for a, b in zip(allr, allr[1:]):
                run = run + 1 if b == a + 1 else 1
                best = max(best, run)
            X.append(
                [
                    *board_f,
                    hr[0] / 12,
                    hr[1] / 12,
                    float(h[1] == h[3]),
                    float(hr[0] == hr[1]),
                    *(float(cat == c) for c in CATS),
                    float(max(suits.count(s) for s in "cdhs") == 4),
                    float(best >= 4),
                    e[k],
                    pct[k],
                    range_eq,
                    opp_eq,
                    wsum / 1326,
                    sides[q][2].sum() / 1326,
                    r["stack"] / pot,
                    float(p == "ip"),
                ]
            )
            y.append(vals[k])
            w.append(wts[k])
            group_id.append(r["id"])
            side.append(f"{r['id']}|{p}")
    return np.array(X), np.array(y), np.array(w), np.array(group_id), np.array(side)


def main(libdir, outdir, workers=32):
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    os.makedirs(outdir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(libdir, "solves", "*.json.gz")))
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(features_of, files, chunksize=8))
    X = np.vstack([p[0] for p in parts])
    y, w = np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])
    G, S = np.concatenate([p[3] for p in parts]), np.concatenate([p[4] for p in parts])
    rng = np.random.default_rng(0)
    solves = np.unique(G)
    test_solves = set(rng.choice(solves, len(solves) // 4, replace=False).tolist())
    te = np.array([g in test_solves for g in G])
    tr_idx = np.flatnonzero(~te)
    if len(tr_idx) > 800_000:
        tr_idx = rng.choice(tr_idx, 800_000, replace=False)
    rep = {
        "solves": len(solves),
        "examples": int(len(y)),
        "train_examples": int(len(tr_idx)),
        "test_examples": int(te.sum()),
        "units": "share of the pot (values are chips relative to each player having put in half the pot)",
    }

    def score(pred):
        err = np.abs(pred - y[te])
        # range-average value: weighted mean over each (solve, player) range, predicted vs solved
        sides = S[te]
        uniq, inv = np.unique(sides, return_inverse=True)
        ww = w[te]
        tot = np.bincount(inv, ww)
        rv_true = np.bincount(inv, ww * y[te]) / np.maximum(tot, 1e-9)
        rv_pred = np.bincount(inv, ww * pred) / np.maximum(tot, 1e-9)
        return {
            "mae_pots": round(float(np.average(err, weights=ww)), 4),
            "r2": round(
                float(
                    1
                    - np.average((pred - y[te]) ** 2, weights=ww)
                    / np.average((y[te] - np.average(y[te], weights=ww)) ** 2, weights=ww)
                ),
                4,
            ),
            "range_value_mae_pots": round(float(np.abs(rv_pred - rv_true).mean()), 4),
        }

    rep["baseline mean"] = score(np.full(te.sum(), np.average(y[tr_idx], weights=w[tr_idx])))
    eq_col = 7 + 4 + len(CATS) + 2  # board 7, hand 4, made-hand classes 9, draws 2: then equity
    rep["baseline raw equity (value = equity - 1/2)"] = score(X[te][:, eq_col] - 0.5)
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(1.0)),
        "gradient boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=0),
        "extra trees": ExtraTreesRegressor(200, min_samples_leaf=10, n_jobs=workers, random_state=0),
        "MLP": make_pipeline(
            StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, random_state=0)
        ),
    }
    for name, m in models.items():
        t0 = time.time()
        if name == "MLP" or name == "ridge":
            m.fit(X[tr_idx], y[tr_idx])
        else:
            m.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx] + 1e-3)
        fit = time.time() - t0
        t0 = time.time()
        pred = m.predict(X[te])
        rep[name] = {
            **score(pred),
            "fit_seconds": round(fit, 1),
            "predict_ms_per_1000_hands": round((time.time() - t0) / te.sum() * 1e6, 2),
        }
        print(name, rep[name], flush=True)
        if name == "MLP":
            from joblib import dump

            dump(m, os.path.join(outdir, "value_mlp.joblib"))
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:3]))
