"""Predict TexasSolver's flop strategy in milliseconds: a network trained on the solver library.

Usage: uv run python scripts/ml_flop.py LIBRARY_DIR OUT_DIR
Reads LIBRARY_DIR/solves/*.json.gz (from solver_library.py) and trains on three decision points per solve: the first
player's opening action, the second player's response after a check, and their response facing the bet. One example
per hand in the range at each point; the target is the solver's action probabilities, mapped to four slots
(passive: check or call; bet or raise; all-in; fold).

Evaluation holds out whole flops (grouped split), so the model is scored on boards it never saw, against the
baseline of predicting each decision point's average strategy. Reports mean absolute error per action probability
and how often the most likely action matches the solver's. Compares a neural network with tree models (gradient
boosting, random forest, extra trees) and two averaging ensembles. Writes OUT_DIR/<model>.joblib and report.json.
"""

import glob
import gzip
import json
import os
import sys

import eval7
import numpy as np

RANKS = "23456789TJQKA"
SLOTS = ["passive", "aggressive", "allin", "fold"]
NODES = ["oop_open", "ip_after_check", "ip_facing_bet"]
SPOT_IDS = ["spr9.5_wide", "spr4.5_wide", "spr9.5_tight_vs_wide", "spr4.5_tight_vs_wide"]


def slot(action):
    a = action.split()[0]
    if a in ("CHECK", "CALL"):
        return 0
    if a == "FOLD":
        return 3
    amt = float(action.split()[1]) if len(action.split()) > 1 else 0
    return 2 if amt >= 40 else 1  # stacks are 45 or 95 behind a pot of 10: the big size is the all-in


def features(board, hand, node, spot):
    b = [eval7.Card(c) for c in board]
    h = [eval7.Card(c) for c in (hand[:2], hand[2:])]
    br = sorted((RANKS.index(c[0]) for c in board), reverse=True)
    hr = sorted((RANKS.index(hand[0]), RANKS.index(hand[2])), reverse=True)
    suits_b = [c[1] for c in board]
    nsuit = len(set(suits_b))
    all_suits = suits_b + [hand[1], hand[3]]
    flush_cards = max(all_suits.count(s) for s in "cdhs")
    ranks_all = sorted({RANKS.index(c[0]) for c in board} | set(hr))
    run = best_run = 1
    for x, y in zip(ranks_all, ranks_all[1:]):
        run = run + 1 if y == x + 1 else 1
        best_run = max(best_run, run)
    cat = eval7.handtype(eval7.evaluate(h + b))
    cats = ["High Card", "Pair", "Two Pair", "Trips", "Straight", "Flush", "Full House", "Quads", "Straight Flush"]
    pair_rank = next((i for i, r in enumerate(br) if r in hr), -1)  # 0 top pair, 1 middle, 2 bottom
    f = [
        *[r / 12 for r in br],
        *[r / 12 for r in hr],
        float(hand[1] == hand[3]),
        float(hr[0] == hr[1]),
        float(nsuit == 1),
        float(nsuit == 2),
        float(nsuit == 3),
        float(len(set(br)) < 3),
        (br[0] - br[2]) / 12,
        *[float(cat == c) for c in cats],
        float(flush_cards == 4),
        float(best_run >= 4),
        sum(r > br[0] for r in hr) / 2,
        float(pair_rank == 0),
        float(pair_rank == 1),
        float(pair_rank == 2),
        float(hr[0] == hr[1] and hr[0] > br[0]),
        *[float(node == n) for n in NODES],
        *[float(spot == s) for s in SPOT_IDS],
    ]
    return f


def examples(tree, board, spot):
    nodes = [("oop_open", tree)]
    for name, child in (tree.get("childrens") or {}).items():
        if child.get("node_type") != "action_node":
            continue
        nodes.append(("ip_after_check" if name.startswith("CHECK") else "ip_facing_bet", child))
    for node_name, node in nodes:
        acts = node.get("actions", [])
        strat = (node.get("strategy") or {}).get("strategy") or {}
        for hand, probs in strat.items():
            y = np.zeros(4)
            for a, p in zip(acts, probs):
                y[slot(a)] += p
            yield features(board, hand, node_name, spot), y, node_name


def load(libdir):
    X, Y, G, N = [], [], [], []
    for path in sorted(glob.glob(os.path.join(libdir, "solves", "*.json.gz"))):
        sid = os.path.basename(path)[: -len(".json.gz")]
        flop_s, spot = sid.split("_", 1)
        board = [flop_s[i : i + 2] for i in (0, 2, 4)]
        tree = json.load(gzip.open(path, "rt"))
        for x, y, node in examples(tree, board, spot):
            X.append(x)
            Y.append(y)
            G.append(flop_s)
            N.append(node)
    return np.array(X), np.array(Y), np.array(G), np.array(N)


def normalise(P):
    P = np.clip(P, 0, None)
    return P / np.maximum(P.sum(axis=1, keepdims=True), 1e-9)


def score(P, Y, N):
    out = {
        "mae": round(float(np.abs(P - Y).mean()), 4),
        "top_action_agreement": round(float((P.argmax(1) == Y.argmax(1)).mean()), 4),
        "by_node": {},
    }
    for node in NODES:
        m = N == node
        if m.any():
            out["by_node"][node] = {
                "mae": round(float(np.abs(P[m] - Y[m]).mean()), 4),
                "top_agree": round(float((P[m].argmax(1) == Y[m].argmax(1)).mean()), 4),
            }
    return out


def models():
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neural_network import MLPRegressor

    return {
        "mlp": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=0),
        "gradient_boosting": MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=0)
        ),
        "random_forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=8, random_state=0),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=8, random_state=0),
    }


def main(libdir, outdir):
    import time

    from joblib import dump
    from sklearn.model_selection import GroupShuffleSplit

    os.makedirs(outdir, exist_ok=True)
    X, Y, G, N = load(libdir)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, Y, G))
    base = np.zeros_like(Y[te])
    for node in NODES:  # baseline: that decision point's average strategy in training
        m_tr, m_te = N[tr] == node, N[te] == node
        if m_tr.any():
            base[m_te] = Y[tr][m_tr].mean(axis=0)
    rep = {
        "solves": len(glob.glob(os.path.join(libdir, "solves", "*.json.gz"))),
        "flops": len(set(G)),
        "examples": int(len(X)),
        "test_flops": int(len(set(G[te]))),
        "test_examples": int(len(te)),
        "baseline": score(base, Y[te], N[te]),
        "models": {},
    }
    preds = {}
    for name, model in models().items():
        t0 = time.time()
        model.fit(X[tr], Y[tr])
        fit_s = time.time() - t0
        P = normalise(model.predict(X[te]))
        t0 = time.time()
        model.predict(X[te][:1])
        rep["models"][name] = {
            **score(P, Y[te], N[te]),
            "fit_seconds": round(fit_s, 1),
            "predict_ms_one_hand": round((time.time() - t0) * 1000, 2),
        }
        preds[name] = P
        dump(model, os.path.join(outdir, f"{name}.joblib"))
    trees = ["gradient_boosting", "random_forest", "extra_trees"]
    rep["models"]["ensemble_trees"] = score(normalise(sum(preds[k] for k in trees) / len(trees)), Y[te], N[te])
    rep["models"]["ensemble_all"] = score(normalise(sum(preds.values()) / len(preds)), Y[te], N[te])
    best = min(rep["models"], key=lambda k: rep["models"][k]["mae"])
    rep["best_by_mae"] = best
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(
        f"{rep['solves']} solves, {rep['flops']} flops, {rep['test_flops']} held out ({rep['test_examples']} examples)"
    )
    print(
        f"  {'baseline':18} MAE {rep['baseline']['mae']:.4f}  top-action {rep['baseline']['top_action_agreement']:.1%}"
    )
    for k, v in rep["models"].items():
        extra = f"  fit {v['fit_seconds']}s  predict {v['predict_ms_one_hand']} ms" if "fit_seconds" in v else ""
        print(f"  {k:18} MAE {v['mae']:.4f}  top-action {v['top_action_agreement']:.1%}{extra}")
    print("best:", best)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
