"""Opponent modelling without the showdown bias: the generic model, tilted to each player's observed action rates.

Usage: HANDHQ_CACHE=... uv run python scripts/opp_adjust.py OUTDIR [WORKERS]
opp_models.py showed that models trained on showdown hands read worse than the generic one: players who show down
never fold, so those models cannot learn which hands would. This keeps the generic model (trained on Pluribus's
hands, where every hand's cards are known, folds included) and tilts it per player:
    P(action | holding, spot) proportional to P_generic(action | holding, spot) x exp(theta[bucket, action])
with one theta per spot bucket (street x facing a bet or not) and action, chosen so that the model's average rates in
that bucket, over all holdings, match how often the player actually folds / calls / raises there. Those rates come
from every hand the player played, cards or not, so they carry no showdown bias.
Variants: per player (rates from their own hands) and per style cluster (pooled rates of the cluster).
Scored as in opp_models.py on the test players' showdown hands: rank of the true holding (0 top, 0.5 nothing) and
the top-10% rate, against the untilted generic model on the same hands.
"""

import json
import os
import pickle
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opp_models as om  # noqa: E402
import ranges  # noqa: E402

ACT = {"f": 0, "cc": 1, "cbr": 2}


def bucket(r):
    return f"{r['street']}|{'facing' if r['to_call'] > 0 else 'open'}"


def generic_mean(args):
    """Average generic action probabilities over all holdings for a sample of rows (one bucket)."""
    rows = args
    m = ranges.model()
    acc = np.zeros(3)
    for r in rows:
        board = om.board_list(r["board"])
        st = ranges.strengths(board)
        cand = [h for h in ranges.ALL if not ({h[:2], h[2:]} & set(board))]
        S = np.array([st.get(h, (0.0, 0.0, 0.0)) for h in cand])
        ctx = om.features((0, 0, 0), r)[3:]
        X = np.hstack([S, np.tile(ctx, (len(cand), 1))])
        P = m.predict_proba(X)
        full = np.zeros((len(cand), 3))
        for k, c in enumerate(m.classes_):
            full[:, int(c)] = P[:, k]
        acc += full.mean(0)
    return acc / max(len(rows), 1)


def thetas(observed, generic):
    """theta per bucket: log(observed rate / generic average rate), with smoothing."""
    out = {}
    for b, f in observed.items():
        g = generic.get(b)
        if g is None:
            continue
        f = (np.array(f) + 1.0) / (np.sum(f) + 3.0)
        out[b] = np.log(f / np.clip(g, 1e-4, 1))
    return out


def read_rank(args):
    hand, th = args
    m = ranges.model()
    cand = [h for h in ranges.ALL if not ({h[:2], h[2:]} & set(hand["board"]))]
    lp_g, lp_t = np.zeros(len(cand)), np.zeros(len(cand))
    for _, y, r in hand["items"]:
        st = ranges.strengths(om.board_list(r["board"]))
        S = np.array([st.get(h, (0.0, 0.0, 0.0)) for h in cand])
        ctx = om.features((0, 0, 0), r)[3:]
        X = np.hstack([S, np.tile(ctx, (len(cand), 1))])
        P = m.predict_proba(X)
        full = np.full((len(cand), 3), 1e-6)
        for k, c in enumerate(m.classes_):
            full[:, int(c)] = P[:, k]
        lp_g += np.log(np.clip(full[:, y], 1e-4, 1))
        t = th.get(bucket(r))
        if t is not None:
            tilted = full * np.exp(t)[None, :]
            tilted /= tilted.sum(1, keepdims=True)
            lp_t += np.log(np.clip(tilted[:, y], 1e-4, 1))
        else:
            lp_t += np.log(np.clip(full[:, y], 1e-4, 1))
    if hand["hole"] not in cand:
        return None
    k = cand.index(hand["hole"])
    res = {}
    for name, lp in (("generic", lp_g), ("tilted", lp_t)):
        post = np.exp(lp - lp.max())
        rank = float((post > post[k]).sum() + 0.5 * ((post == post[k]).sum() - 1)) / len(cand)
        res[name] = rank
    return res


def summary(res, key):
    x = [r[key] for r in res]
    return {
        "hands": len(x),
        "mean_rank": round(float(np.mean(x)), 4),
        "top10_rate": round(float(np.mean([v < 0.1 for v in x])), 4),
    }


def main(outdir, workers=32):
    os.makedirs(outdir, exist_ok=True)
    # our own cache, written by behaviour.py on the lab from the HandHQ files: trusted input
    data = pickle.load(open(os.environ["HANDHQ_CACHE"], "rb"))
    observed = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))  # player -> bucket -> counts
    sample_rows = defaultdict(list)  # bucket -> rows (for the generic averages)
    rng = random.Random(0)
    hands_of = defaultdict(int)
    for name, hands in data.items():
        for hd in hands:
            for p in set(hd["players"]):
                hands_of[f"{name}|{p}"] += 1
            for r in hd["rows"]:
                p = f"{name}|{r['player']}"
                b = bucket(r)
                observed[p][b][ACT[r["action"]]] += 1
                if rng.random() < 0.002:
                    sample_rows[b].append(r)
    with ProcessPoolExecutor(workers) as pool:
        bs = sorted(sample_rows)
        gen = dict(zip(bs, pool.map(generic_mean, [sample_rows[b][:200] for b in bs])))
        items = [h for part in pool.map(om.hand_items, list(data.items())) for h in part]
    regs = [p for p, n in hands_of.items() if n >= 300]
    rng.shuffle(regs)
    test = set(regs[: len(regs) // 2])
    test_hands = [h for h in items if h["player"] in test]
    # style clusters from the same style features as opp_models (VPIP-like rates from observed counts)
    feats = {}
    for p in regs:
        o = observed[p]
        pre = o.get("preflop|facing", np.zeros(3))
        post = sum(
            (o.get(f"{s}|facing", np.zeros(3)) + o.get(f"{s}|open", np.zeros(3)) for s in ("flop", "turn", "river")),
            np.zeros(3),
        )
        feats[p] = [pre[1:].sum() / max(pre.sum(), 1), pre[2] / max(pre.sum(), 1), post[2] / max(post[1:].sum(), 1)]
    from sklearn.cluster import KMeans

    names = sorted(feats)
    km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(np.array([feats[p] for p in names]))
    cl = dict(zip(names, km.labels_))
    pooled = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))
    for p in regs:
        if p not in test:
            for b, c in observed[p].items():
                pooled[cl[p]][b] += c
    jobs_p, jobs_c = [], []
    for h in test_hands:
        jobs_p.append((h, thetas(observed[h["player"]], gen)))
        jobs_c.append((h, thetas(pooled[cl[h["player"]]], gen)))
    with ProcessPoolExecutor(workers) as pool:
        rp = [r for r in pool.map(read_rank, jobs_p, chunksize=8) if r]
        rc = [r for r in pool.map(read_rank, jobs_c, chunksize=8) if r]
    rep = {
        "test_players": len(test),
        "generic": summary(rp, "generic"),
        "tilted_per_player": summary(rp, "tilted"),
        "tilted_per_style_cluster": summary(rc, "tilted"),
        "note": "rank 0 = the true holding is the model's top guess; 0.5 = no information. Same showdown hands for all.",
    }
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(int(x) for x in a[1:2]))
