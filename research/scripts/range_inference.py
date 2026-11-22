"""Reading hidden cards from betting: Bayesian filtering over every possible holding (the particle filter of physics
and robotics, here exact: one particle per two-card combination).

Usage: uv run python scripts/range_inference.py PLURIBUS_PHH_DIR OUTDIR [WORKERS]
Pluribus's 10,000 hands show every player's cards, so the answer is known. An action model, P(fold / check-call /
bet-raise | strength of the hand, street, price, raises so far, position), is trained on 70% of the hands. On the
other 30%, for every player in every hand, each combination that does not clash with the board or the observer's own
cards (the observer is Pluribus) starts equally likely and is reweighted after each of that player's actions by how
likely the model says that action is with that combination. Scored at the end of each hand on where the true
combination ranks (0 = the most likely, 0.5 = what knowing nothing gives), the information gained in bits, and how
often the true hand is in the top 10%.

Strength of a combination: before the flop, its equity against a random hand (169-class table); after, its rank
among all combinations on that board (share of combinations it beats), plus flush- and straight-draw flags.
"""

import itertools
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_hh  # noqa: E402
import hand_histories as hh  # noqa: E402
import preflop_deep  # noqa: E402

RANKS = "23456789TJQKA"
DECK = [r + s for r in RANKS for s in "cdhs"]
ALL = [a + b for a, b in itertools.combinations(DECK, 2)]
ACT = {"f": 0, "cc": 1, "cbr": 2}
STREETS = ["preflop", "flop", "turn", "river"]


def cls(h):
    hi, lo = sorted((h[0], h[2]), key=RANKS.index, reverse=True)
    return hi + lo if hi == lo else hi + lo + ("s" if h[1] == h[3] else "o")


def strengths(board):
    """Strength features for every combination on this board: (strength, flush draw, straight draw)."""
    import eval7

    vr = preflop_deep._table()["vs_random"]
    out = {}
    if not board:
        for h in ALL:
            out[h] = (vr[cls(h)], 0.0, 0.0)
        return out
    b = [eval7.Card(c) for c in board]
    ok = [h for h in ALL if h[:2] not in board and h[2:] not in board]
    vals = np.array([eval7.evaluate([eval7.Card(h[:2]), eval7.Card(h[2:])] + b) for h in ok])
    order = np.argsort(vals)
    pct = np.empty(len(ok))
    pct[order] = np.arange(len(ok)) / max(len(ok) - 1, 1)
    bs, br = [c[1] for c in board], [RANKS.index(c[0]) for c in board]
    for k, h in enumerate(ok):
        suits = bs + [h[1], h[3]]
        fd = float(len(board) < 5 and max(suits.count(s) for s in "cdhs") == 4)
        allr = sorted(set(br) | {RANKS.index(h[0]), RANKS.index(h[2])})
        run = best = 1
        for x, y in zip(allr, allr[1:]):
            run = run + 1 if y == x + 1 else 1
            best = max(best, run)
        out[h] = (float(pct[k]), fd, float(len(board) < 5 and best >= 4))
    return out


def context(s):
    raises = 0
    for e in reversed(s["history"]):
        if e.startswith("["):
            break
        raises += "raises" in e
    pot = max(s["pot"], 1)
    return [
        *(float(s["street"] == x) for x in STREETS),
        s["to_call"] / pot,
        float(s["to_call"] > 0),
        raises,
        s["seat"] / max(s["players"] - 1, 1),
        s["players"],
    ]


def norm(h):
    a, b = h[:2], h[2:]
    return a + b if DECK.index(a) < DECK.index(b) else b + a


def hand_rows(path_h):
    """Decisions of one hand with features for the true holding, plus what inference needs."""
    _, h = path_h
    if "actions" not in h:
        return None
    spots = list(backtest_hh.spots(h))
    cache = {}
    rows = []
    for s in spots:
        if not s["hole"] or "?" in s["hole"]:
            continue
        key = tuple(s["board"])
        if key not in cache:
            cache[key] = strengths(s["board"])
        st = cache[key].get(norm(s["hole"]))
        if st is None:
            continue
        rows.append(
            {
                "player": s["player"],
                "hole": norm(s["hole"]),
                "board": s["board"],
                "ctx": context(s),
                "strength": st,
                "y": ACT[s["action"]],
            }
        )
    return {"rows": rows, "pluribus_hole": next((r["hole"] for r in rows if r["player"] == "Pluribus"), None)}


def infer(args):
    hand, model_path = args
    from joblib import load

    model = load(model_path)
    rows, me = hand["rows"], hand["pluribus_hole"]
    blocked = set([me[:2], me[2:]]) if me else set()
    out = []
    by_player = {}
    for r in rows:
        if r["player"] != "Pluribus":
            by_player.setdefault(r["player"], []).append(r)
    final_board = max((r["board"] for r in rows), key=len, default=[])
    for player, rs in by_player.items():
        cand = [h for h in ALL if not ({h[:2], h[2:]} & (blocked | set(final_board)))]
        logp = np.zeros(len(cand))
        cache = {}
        for r in rs:
            key = tuple(r["board"])
            if key not in cache:
                cache[key] = strengths(r["board"])
            S = np.array([cache[key][h] for h in cand])
            X = np.hstack([S, np.tile(r["ctx"], (len(cand), 1))])
            P = model.predict_proba(X)[:, r["y"]]
            logp += np.log(np.clip(P, 1e-6, 1))
        true = rs[0]["hole"]
        if true not in cand:
            continue
        k = cand.index(true)
        post = np.exp(logp - logp.max())
        post /= post.sum()
        rank = float((post > post[k]).sum() + 0.5 * ((post == post[k]).sum() - 1)) / len(cand)
        out.append(
            {
                "actions": len(rs),
                "rank": rank,
                "bits": math.log2(post[k] * len(cand)),
                "top10": rank < 0.10,
                "entropy_bits": float(-(post * np.log2(np.clip(post, 1e-300, 1))).sum()),
                "max_entropy_bits": math.log2(len(cand)),
            }
        )
    return out


def main(phh_dir, outdir, workers=32):
    from joblib import dump
    from sklearn.ensemble import HistGradientBoostingClassifier

    os.makedirs(outdir, exist_ok=True)
    hands = list(hh.hands(phh_dir))
    random.Random(0).shuffle(hands)
    with ProcessPoolExecutor(workers) as pool:
        parsed = [p for p in pool.map(hand_rows, hands, chunksize=50) if p and p["rows"]]
    cut = int(0.7 * len(parsed))
    train, test = parsed[:cut], parsed[cut:]
    X = np.array([[*r["strength"], *r["ctx"]] for hd in train for r in hd["rows"]])
    y = np.array([r["y"] for hd in train for r in hd["rows"]])
    model = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X, y)
    mpath = os.path.join(outdir, "action_model.joblib")
    dump(model, mpath)
    Xt = np.array([[*r["strength"], *r["ctx"]] for hd in test for r in hd["rows"]])
    yt = np.array([r["y"] for hd in test for r in hd["rows"]])
    acc = float((model.predict(Xt) == yt).mean())
    with ProcessPoolExecutor(workers) as pool:
        res = [x for part in pool.map(infer, [(hd, mpath) for hd in test], chunksize=20) for x in part]
    rep = {
        "hands_train": len(train),
        "hands_test": len(test),
        "action_model_accuracy": round(acc, 4),
        "players_scored": len(res),
    }
    for label, sel in (
        ("all", lambda r: True),
        ("1 action", lambda r: r["actions"] == 1),
        ("2-3 actions", lambda r: 2 <= r["actions"] <= 3),
        ("4+ actions", lambda r: r["actions"] >= 4),
    ):
        rs = [r for r in res if sel(r)]
        if rs:
            rep[label] = {
                "n": len(rs),
                "mean_rank": round(float(np.mean([r["rank"] for r in rs])), 4),
                "mean_bits_gained": round(float(np.mean([r["bits"] for r in rs])), 3),
                "top10_rate": round(float(np.mean([r["top10"] for r in rs])), 4),
                "mean_entropy_left_bits": round(float(np.mean([r["entropy_bits"] for r in rs])), 2),
            }
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:3]))
