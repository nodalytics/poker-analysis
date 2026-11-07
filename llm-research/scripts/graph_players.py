"""Graph methods on real player networks (HandHQ): can the network tell who is good, and who plays together oddly?

Usage: uv run python scripts/graph_players.py HANDHQ_DIR OUTDIR [HANDS_PER_FOLDER] [WORKERS]
Per site/stake folder (ids are stable within one), hands in time order, split at the median time. From the first
half: the co-play graph (players linked by hands played together) and the money-flow graph (each hand's losers'
chips allocated to its winners). Target: each regular's big blinds per 100 hands in the second half (300+ hands in
each half). Scored by Spearman rank correlation with a player bootstrap.

Predictors (first half only):
  own result        bb/100, shrunk toward 0 (persistence of results; it was -0.02 in predictive.py)
  PageRank          on the money-flow graph (money flows to winners; flows from winners count more)
  network rating    opponent-adjusted (Massey / least squares on the co-play graph): r solves (L + lambda I) r = net,
                    so beating strong opponents counts more than beating weak ones
  style only        ridge on each player's style (VPIP, PFR, aggression, hands), leave-one-folder-out
  graph network     the same ridge on style propagated over the co-play graph (simplified graph convolution, SGC:
                    [X, AX, A^2X], A the normalised co-play adjacency): does who you play with add signal?
Collusion screen: pairs seated together 300+ hands; how often both enter the same pot versus their independent rates
(per pair, a z score); colluders soft-play, entering pots together less than expected. Null: the same pair with one
player's hands permuted among their shared hands (keeps both rates, removes same-hand coupling).
"""

import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import behaviour  # noqa: E402

MIN_HANDS = 300


def half_stats(hands):
    n, net = defaultdict(int), defaultdict(float)
    vp, pr, ar, ac = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    co, flow = defaultdict(float), defaultdict(float)
    for hd in hands:
        v, p, r, c = behaviour.vpip_pfr_af(hd)
        ps = hd["players"]
        for j, x in enumerate(ps):
            n[x] += 1
            net[x] += hd["net"][j]
            vp[x] += x in v
            pr[x] += x in p
            ar[x] += r[x]
            ac[x] += c[x]
        for a in range(len(ps)):
            for b in range(a + 1, len(ps)):
                co[(ps[a], ps[b])] += 1
        win = [(ps[j], x) for j, x in enumerate(hd["net"]) if x > 0]
        tot = sum(x for _, x in win)
        if tot > 0:
            for j, x in enumerate(hd["net"]):
                if x < 0:
                    for w, y in win:
                        flow[(ps[j], w)] += -x * y / tot
    style = {x: [vp[x] / n[x], pr[x] / n[x], ar[x] / max(1, ar[x] + ac[x]), math.log(n[x])] for x in n}
    return n, net, style, co, flow


def pagerank(nodes, flow, d=0.85, iters=60):
    idx = {x: k for k, x in enumerate(nodes)}
    out = defaultdict(float)
    for (a, _), w in flow.items():
        out[a] += w
    pr = np.ones(len(nodes)) / len(nodes)
    edges = [(idx[a], idx[b], w / out[a]) for (a, b), w in flow.items() if a in idx and b in idx]
    src = np.array([e[0] for e in edges], int)
    dst = np.array([e[1] for e in edges], int)
    wt = np.array([e[2] for e in edges])
    for _ in range(iters):
        new = np.full(len(nodes), (1 - d) / len(nodes))
        np.add.at(new, dst, d * pr[src] * wt)
        new += d * pr[[k for k, x in enumerate(nodes) if out[x] == 0]].sum() / len(nodes)
        pr = new / new.sum()
    return dict(zip(nodes, pr))


def massey(nodes, n, net, co, lam=50.0):
    """Opponent-adjusted rating per hand: (L + lam I) r = net / hands-weighted; L the co-play Laplacian."""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve

    idx = {x: k for k, x in enumerate(nodes)}
    rows, cols, vals = [], [], []
    for (a, b), w in co.items():
        if a in idx and b in idx:
            rows += [idx[a], idx[b]]
            cols += [idx[b], idx[a]]
            vals += [w, w]
    A = coo_matrix((vals, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()
    L = diags(np.asarray(A.sum(1)).ravel()) - A
    b = np.array([net[x] for x in nodes])
    r = spsolve((L + lam * diags(np.ones(len(nodes)))).tocsc(), b)
    return dict(zip(nodes, r)), A


def sgc(A, X, hops=2):
    deg = np.asarray(A.sum(1)).ravel() + 1.0
    dinv = 1 / np.sqrt(deg)
    from scipy.sparse import diags, identity

    Ah = diags(dinv) @ (A + identity(A.shape[0])) @ diags(dinv)
    feats, cur = [X], X
    for _ in range(hops):
        cur = Ah @ cur
        feats.append(cur)
    return np.hstack(feats)


def spearman(a, b):
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def boot(a, b, rng, k=1000):
    a, b = np.array(a), np.array(b)
    vals = []
    for _ in range(k):
        s = rng.integers(0, len(a), len(a))
        vals.append(spearman(a[s], b[s]))
    vals.sort()
    return [round(vals[25], 3), round(vals[974], 3)]


def collusion(hands, rng, min_shared=300):
    shared = defaultdict(list)
    for hd in hands:
        v, _, _, _ = behaviour.vpip_pfr_af(hd)
        ps = hd["players"]
        for a in range(len(ps)):
            for b in range(a + 1, len(ps)):
                key = tuple(sorted((ps[a], ps[b])))
                shared[key].append((key[0] in v, key[1] in v))
    zs, zn = [], []
    for key, xs in shared.items():
        if len(xs) < min_shared:
            continue
        x = np.array(xs, float)
        pa, pb = x[:, 0].mean(), x[:, 1].mean()
        exp = pa * pb
        if exp <= 0 or exp >= 1:
            continue
        se = math.sqrt(exp * (1 - exp) / len(x))
        zs.append(((x[:, 0] * x[:, 1]).mean() - exp) / se)
        perm = rng.permutation(len(x))
        zn.append(((x[:, 0] * x[perm, 1]).mean() - exp) / se)
    return zs, zn


def main(src, outdir, per_folder=100000, workers=27):
    os.makedirs(outdir, exist_ok=True)
    data = behaviour.load(src, per_folder, workers)
    rng = np.random.default_rng(0)
    rows = []  # per player: folder, target, predictors..., style features, graph features
    zs_all, zn_all = [], []
    for name, hands in data.items():
        if len(hands) < 1000:
            continue
        cut = len(hands) // 2
        n1, net1, style1, co1, flow1 = half_stats(hands[:cut])
        n2, net2, _, _, _ = half_stats(hands[cut:])
        nodes = sorted(n1)
        pr = pagerank(nodes, flow1)
        rating, A = massey(nodes, n1, net1, co1)
        X = np.array([style1[x] for x in nodes])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        G = sgc(A, (X - mu) / sd)
        for k, x in enumerate(nodes):
            if n1[x] >= MIN_HANDS and n2.get(x, 0) >= MIN_HANDS:
                rows.append(
                    {
                        "folder": name,
                        "target": 100 * net2[x] / n2[x],
                        "own": 100 * net1[x] / (n1[x] + 500),
                        "pagerank": pr[x] * len(nodes),
                        "rating": rating[x],
                        "style": ((X[k] - mu) / sd).tolist(),
                        "graph": G[k].tolist(),
                    }
                )
        zs, zn = collusion(hands, rng)
        zs_all += zs
        zn_all += zn
    # leave-one-folder-out ridge for the learned predictors
    from sklearn.linear_model import Ridge

    folders = sorted({r["folder"] for r in rows})
    for feat in ("style", "graph"):
        for f in folders:
            tr = [r for r in rows if r["folder"] != f]
            te = [r for r in rows if r["folder"] == f]
            if not te:
                continue
            y = np.clip([r["target"] for r in tr], -200, 200)
            m = Ridge(alpha=10.0).fit(np.array([r[feat] for r in tr]), y)
            for r, p in zip(te, m.predict(np.array([r[feat] for r in te]))):
                r[f"pred_{feat}"] = float(p)
    y = [r["target"] for r in rows]
    rep = {"hands": sum(len(v) for v in data.values()), "regulars_scored": len(rows), "folders": len(folders)}
    preds = {
        "own result (persistence)": "own",
        "PageRank on money flow": "pagerank",
        "network rating (opponent-adjusted)": "rating",
        "style only (ridge)": "pred_style",
        "style + graph convolution (SGC)": "pred_graph",
    }
    rep["spearman_with_second_half_bb100"] = {
        lab: {"rho": round(spearman([r[k] for r in rows], y), 3), "ci95": boot([r[k] for r in rows], y, rng)}
        for lab, k in preds.items()
    }
    top = {}
    for lab, k in preds.items():
        order = sorted(rows, key=lambda r: -r[k])
        dec = max(1, len(order) // 10)
        top[lab] = {
            "top_10pct_second_half_bb100": round(float(np.mean([r["target"] for r in order[:dec]])), 1),
            "bottom_10pct_second_half_bb100": round(float(np.mean([r["target"] for r in order[-dec:]])), 1),
        }
    rep["deciles"] = top
    zs_all, zn_all = np.array(zs_all), np.array(zn_all)
    if len(zs_all):
        rep["collusion_screen"] = {
            "pairs": len(zs_all),
            "real_z_quantiles_1_50_99": [round(float(np.percentile(zs_all, q)), 2) for q in (1, 50, 99)],
            "null_z_quantiles_1_50_99": [round(float(np.percentile(zn_all, q)), 2) for q in (1, 50, 99)],
            "pairs_below_null_1st_percentile": int((zs_all < np.percentile(zn_all, 1)).sum()),
            "expected_if_like_null": round(0.01 * len(zs_all), 1),
            "pairs_below_minus_6": int((zs_all < -6).sum()),
            # same-hand coupling is naturally negative (one player in the pot crowds the other out), so outliers
            # are judged against the real population: 5 robust sd below its median
            "robust_outliers_5sd_below_population": int(
                (zs_all < np.median(zs_all) - 5 * 1.4826 * np.median(np.abs(zs_all - np.median(zs_all)))).sum()
            ),
        }
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    random.seed(0)
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:4]))
