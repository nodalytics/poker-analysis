"""Opponent modelling at scale on real players: does a model of a player's style, or of the player alone, read
their hands better than the generic one?

Usage: HANDHQ_CACHE=... uv run python scripts/opp_models.py OUTDIR [WORKERS]   (reads the behaviour.py cache)
For every hand where a player's cards were shown, each of their actions becomes a training row: the strength of the
hand they really held on the board at that moment, the price, raises so far, street and seat (the features of
range_inference.py / ranges.py), and what they did. Regulars (300+ hands) are grouped by style (VPIP, PFR,
post-flop aggression) with k-means, and split by player into train and test halves.
Models compared on the test players' showdown hands (where their true hand is known):
  generic   the Pluribus-trained model ranges.py uses
  style     one model per style cluster, trained on the train players of that cluster; a test player gets the
            cluster their style (from all their hands, not just showdowns) puts them in
  personal  a model of the test player alone, trained on the first 70% of their own showdown hands, scored on the
            last 30% (the Slumbot approach, per opponent)
Scored like range_inference.py: where the true holding ranks in the posterior after all the player's actions in the
hand (0 top, 0.5 no information), and how often it is in the top 10%. Showdown hands are a biased sample
(called-down hands), the same for every model, so the comparison is fair even if the absolute numbers are not.
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
import behaviour  # noqa: E402
import ranges  # noqa: E402

STREETS = ["preflop", "flop", "turn", "river"]
ACT = {"f": 0, "cc": 1, "cbr": 2}


def seat_feature(seat, n):
    return seat / max(n - 1, 1)


def features(strength, r):
    pot = max(r["pot"], 1)
    return [
        *strength,
        *(float(r["street"] == x) for x in STREETS),
        r["to_call"] / pot,
        float(r["to_call"] > 0),
        r["raises_before"],
        seat_feature(r["seat"], r["players"]),
        6,
    ]


def board_list(s):
    return [s[k : k + 2] for k in range(0, len(s), 2)]


def hand_items(args):
    """Showdown players' decision rows with the strength of their true holding, plus what reading needs."""
    name, hands = args
    out = []
    for hd in hands:
        if not hd["shown"]:
            continue
        by_seat = defaultdict(list)
        for r in hd["rows"]:
            if r["seat"] in hd["shown"]:
                by_seat[r["seat"]].append(r)
        for seat, rs in by_seat.items():
            hole = ranges.norm(hd["shown"][seat])
            items = []
            for r in rs:
                st = ranges.strengths(board_list(r["board"])).get(hole)
                if st is None:
                    break
                items.append((features(st, r), ACT[r["action"]], r))
            else:
                if items:
                    final_board = board_list(max((r["board"] for r in rs), key=len))
                    out.append(
                        {"player": f"{name}|{hd['players'][seat]}", "hole": hole, "items": items, "board": final_board}
                    )
    return out


def read_rank(args):
    """Posterior rank of the true holding under a model (path or fitted object passed by path)."""
    hand, model_path = args
    m = ranges.model(model_path)
    cand = [h for h in ranges.ALL if not ({h[:2], h[2:]} & set(hand["board"]))]
    logp = np.zeros(len(cand))
    for _, y, r in hand["items"]:
        st = ranges.strengths(board_list(r["board"]))
        S = np.array([st.get(h, (0.0, 0.0, 0.0)) for h in cand])
        ctx = features((0, 0, 0), r)[3:]
        X = np.hstack([S, np.tile(ctx, (len(cand), 1))])
        cls = list(m.classes_)
        p = m.predict_proba(X)[:, cls.index(y)] if y in cls else np.full(len(cand), 1e-4)
        logp += np.log(np.clip(p, 1e-4, 1))
    if hand["hole"] not in cand:
        return None
    k = cand.index(hand["hole"])
    post = np.exp(logp - logp.max())
    rank = float((post > post[k]).sum() + 0.5 * ((post == post[k]).sum() - 1)) / len(cand)
    return {"rank": rank, "top10": rank < 0.1, "actions": len(hand["items"])}


def fit(rows, path):
    from joblib import dump
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.array([x for x, _ in rows])
    y = np.array([a for _, a in rows])
    if len(set(y)) < 2 or len(y) < 300:  # showdown players never fold, so two classes is normal
        return None
    m = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X, y)
    dump(m, path)
    return path


def score(hands, path, pool):
    res = [r for r in pool.map(read_rank, [(h, path) for h in hands], chunksize=8) if r]
    if not res:
        return None
    return {
        "hands": len(res),
        "mean_rank": round(float(np.mean([r["rank"] for r in res])), 4),
        "top10_rate": round(float(np.mean([r["top10"] for r in res])), 4),
    }


def main(outdir, workers=32):
    os.makedirs(outdir, exist_ok=True)
    # our own cache, written by behaviour.py on the lab from the HandHQ files: trusted input
    data = pickle.load(open(os.environ["HANDHQ_CACHE"], "rb"))
    # styles from all hands
    st = defaultdict(lambda: [0, 0, 0, 0, 0])  # hands, vpip, pfr, post raises, post calls
    for name, hands in data.items():
        for hd in hands:
            vp, pr, prr, pc = behaviour.vpip_pfr_af(hd)
            for p in hd["players"]:
                s = st[f"{name}|{p}"]
                s[0] += 1
                s[1] += p in vp
                s[2] += p in pr
                s[3] += prr[p]
                s[4] += pc[p]
    regs = {p: s for p, s in st.items() if s[0] >= 300}
    style = {p: [s[1] / s[0], s[2] / s[0], s[3] / max(1, s[3] + s[4])] for p, s in regs.items()}
    from sklearn.cluster import KMeans

    names = sorted(style)
    Xs = np.array([style[p] for p in names])
    km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(Xs)
    cluster = dict(zip(names, km.labels_))
    with ProcessPoolExecutor(workers) as pool:
        items = [h for part in pool.map(hand_items, list(data.items())) for h in part]
    items = [h for h in items if h["player"] in cluster]
    players = sorted({h["player"] for h in items})
    random.Random(0).shuffle(players)
    test_players = set(players[: len(players) // 2])
    train = [h for h in items if h["player"] not in test_players]
    test = [h for h in items if h["player"] in test_players]
    paths = {}
    for c in range(6):
        rows = [(x, y) for h in train if cluster[h["player"]] == c for x, y, _ in h["items"]]
        paths[c] = fit(rows, os.path.join(outdir, f"style_{c}.joblib"))
    generic = ranges.MODEL_PATH
    rep = {
        "hands_with_showdown_rows": len(items),
        "regulars": len(regs),
        "train_players": len(players) - len(test_players),
        "test_players": len(test_players),
        "clusters": [
            {
                "vpip": round(float(c[0]), 3),
                "pfr": round(float(c[1]), 3),
                "aggression": round(float(c[2]), 3),
                "players": int((km.labels_ == k).sum()),
            }
            for k, c in enumerate(km.cluster_centers_)
        ],
    }
    with ProcessPoolExecutor(workers) as pool:
        rep["generic_on_test"] = score(test, generic, pool)
        by_c = defaultdict(list)
        for h in test:
            by_c[cluster[h["player"]]].append(h)
        style_res, gen_res = [], []
        for c, hs in by_c.items():
            if paths.get(c):
                style_res += [r for r in pool.map(read_rank, [(h, paths[c]) for h in hs], chunksize=8) if r]
                gen_res += [r for r in pool.map(read_rank, [(h, generic) for h in hs], chunksize=8) if r]
        rep["style_vs_generic_same_hands"] = {
            "hands": len(style_res),
            "style": {
                "mean_rank": round(float(np.mean([r["rank"] for r in style_res])), 4),
                "top10_rate": round(float(np.mean([r["top10"] for r in style_res])), 4),
            },
            "generic": {
                "mean_rank": round(float(np.mean([r["rank"] for r in gen_res])), 4),
                "top10_rate": round(float(np.mean([r["top10"] for r in gen_res])), 4),
            },
        }
        # personal models for test players with enough showdowns
        per_p = defaultdict(list)
        for h in test:
            per_p[h["player"]].append(h)
        pers, gen2, sty2 = [], [], []
        for n, (p, hs) in enumerate(per_p.items()):
            if len(hs) < 60:
                continue
            cut = int(0.7 * len(hs))
            # one file per player: ranges.model caches models by path
            path = fit(
                [(x, y) for h in hs[:cut] for x, y, _ in h["items"]], os.path.join(outdir, f"personal_{n}.joblib")
            )
            if not path:
                continue
            ev = hs[cut:]
            pers += [r for r in pool.map(read_rank, [(h, path) for h in ev], chunksize=4) if r]
            gen2 += [r for r in pool.map(read_rank, [(h, generic) for h in ev], chunksize=4) if r]
            if paths.get(cluster[p]):
                sty2 += [r for r in pool.map(read_rank, [(h, paths[cluster[p]]) for h in ev], chunksize=4) if r]
        if pers:
            rep["personal_vs_style_vs_generic_same_hands"] = {
                k: {
                    "hands": len(v),
                    "mean_rank": round(float(np.mean([r["rank"] for r in v])), 4),
                    "top10_rate": round(float(np.mean([r["top10"] for r in v])), 4),
                }
                for k, v in (("personal", pers), ("style", sty2), ("generic", gen2))
            }
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(int(x) for x in a[1:2]))
