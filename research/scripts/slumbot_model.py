"""A hand-reading model of Slumbot itself, from our logged hands (its cards are revealed at the end of every hand).

Usage: uv run python scripts/slumbot_model.py OUTDIR 'slumbot_*'   (globs of slumbot_client output directories)
Every Slumbot action is replayed with the state it was taken in (slumbot_client.replay) and its true holding; the
same features as range_inference.py / ranges.py (holding strength on the board, street, price, raises, seat) predict
fold / check-call / bet-raise. Trained on 80% of hands, then both this model and the Pluribus-trained one read
Slumbot's hand on the other 20%: where its true holding ranks in the posterior after all its actions (0 = top,
0.5 = no information) and how often it is in the top 10%. Saves OUTDIR/action_model.joblib for ranges.py
(RANGE_MODEL_OPPONENT).
"""

import glob
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ranges  # noqa: E402
import slumbot_client as S  # noqa: E402


def features(e, hole):
    st = ranges.strengths(e["board"]).get(ranges.norm(hole))
    if st is None:
        return None
    pot = max(e["pot"], 1)
    ctx = [*(float(e["street"] == x) for x in ranges.STREETS), e["to_call"] / pot, float(e["to_call"] > 0)]
    return [*st, *ctx, e["raises"], e["seat_feature"], 6]


def hand_rows(r):
    """Slumbot's decisions in one logged hand: (features, action) plus what reading needs."""
    try:
        resp = {"client_pos": r["pos"], "hole_cards": r["hole"], "board": r["board"] or [], "action": r["action"]}
        _, _, _, _, log = S.replay(resp, random.Random(0))
    except Exception:  # an action string our replay cannot follow is skipped
        return None
    slum = 1 - r["pos"]
    hole = "".join(r["slumbot_hole"])
    rows = [(features(e, hole), e["action"]) for e in log if e["player"] == slum]
    rows = [(x, y) for x, y in rows if x is not None]
    return {"rows": rows, "log": log, "slum": slum, "hole": ranges.norm(hole), "ours": r["hole"], "board": r["board"]}


def read_rank(args):
    hd, model_path = args
    blocked = set(hd["ours"]) | set(hd["board"] or [])
    w = ranges.weights(hd["log"], hd["slum"], blocked, model_path)
    if hd["hole"] not in w:
        return None
    v = np.array(list(w.values()))
    t = w[hd["hole"]]
    rank = float((v > t).sum() + 0.5 * ((v == t).sum() - 1)) / len(v)
    return {"rank": rank, "top10": rank < 0.10, "actions": sum(1 for e in hd["log"] if e["player"] == hd["slum"])}


def main(outdir, *patterns):
    from joblib import dump
    from sklearn.ensemble import HistGradientBoostingClassifier

    os.makedirs(outdir, exist_ok=True)
    logged = []
    for p in patterns:
        for f in glob.glob(os.path.join(p, "hands_*.jsonl")):
            logged += [json.loads(x) for x in open(f)]
    logged = [r for r in logged if r.get("slumbot_hole")]
    random.Random(0).shuffle(logged)
    with ProcessPoolExecutor(32) as pool:
        hands = [h for h in pool.map(hand_rows, logged, chunksize=50) if h and h["rows"]]
    cut = int(0.8 * len(hands))
    train, test = hands[:cut], hands[cut:]
    X = np.array([x for h in train for x, _ in h["rows"]])
    y = np.array([a for h in train for _, a in h["rows"]])
    model = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X, y)
    path = os.path.join(outdir, "action_model.joblib")
    dump(model, path)
    Xt = np.array([x for h in test for x, _ in h["rows"]])
    yt = np.array([a for h in test for _, a in h["rows"]])
    generic = ranges.MODEL_PATH
    from joblib import load

    rep = {
        "hands_logged": len(logged),
        "hands_train": len(train),
        "hands_test": len(test),
        "slumbot_decisions_train": len(y),
        "action_accuracy": {
            "slumbot_model": round(float((model.predict(Xt) == yt).mean()), 4),
            "pluribus_model": round(float((load(generic).predict(Xt) == yt).mean()), 4),
        },
        "hand_reading_on_test_hands": {},
    }
    with ProcessPoolExecutor(32) as pool:
        for name, mp in (("pluribus_model", generic), ("slumbot_model", path)):
            res = [x for x in pool.map(read_rank, [(h, mp) for h in test], chunksize=20) if x]
            for lab, sel in (("all", lambda r: True), ("3+ Slumbot actions", lambda r: r["actions"] >= 3)):
                rs = [r for r in res if sel(r)]
                rep["hand_reading_on_test_hands"].setdefault(lab, {})[name] = {
                    "n": len(rs),
                    "mean_rank": round(float(np.mean([r["rank"] for r in rs])), 4),
                    "top10_rate": round(float(np.mean([r["top10"] for r in rs])), 4),
                }
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], *sys.argv[2:])
