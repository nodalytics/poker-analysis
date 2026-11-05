"""Predict TexasSolver's flop strategy in milliseconds: a network trained on the solver library.

Usage: uv run python scripts/ml_flop.py LIBRARY_DIR OUT_DIR
Reads LIBRARY_DIR/solves/*.json.gz (from solver_library.py) and trains on three decision points per solve: the first
player's opening action, the second player's response after a check, and their response facing the bet. One example
per hand in the range at each point; the target is the solver's action probabilities, mapped to four slots
(passive: check or call; bet or raise; all-in; fold).

Evaluation holds out whole flops (grouped split), so the model is scored on boards it never saw, against the
baseline of predicting each decision point's average strategy. Reports mean absolute error per action probability
and how often the most likely action matches the solver's. Writes OUT_DIR/model.joblib and OUT_DIR/report.json.
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


def main(libdir, outdir):
    from joblib import dump
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.neural_network import MLPRegressor

    os.makedirs(outdir, exist_ok=True)
    X, Y, G, N = load(libdir)
    flops = len(set(G))
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    tr, te = next(split.split(X, Y, G))
    model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=0)
    model.fit(X[tr], Y[tr])
    P = np.clip(model.predict(X[te]), 0, None)
    P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-9)
    base = np.zeros_like(Y[te])
    for node in NODES:  # baseline: that decision point's average strategy in training
        m_tr, m_te = N[tr] == node, N[te] == node
        if m_tr.any():
            base[m_te] = Y[tr][m_tr].mean(axis=0)
    rep = {
        "solves": int(len(set(zip(G, N)))) and len(glob.glob(os.path.join(libdir, "solves", "*.json.gz"))),
        "flops": flops,
        "examples": int(len(X)),
        "test_flops": int(len(set(G[te]))),
        "test_examples": int(len(te)),
        "model_mae": round(float(np.abs(P - Y[te]).mean()), 4),
        "baseline_mae": round(float(np.abs(base - Y[te]).mean()), 4),
        "model_top_action_agreement": round(float((P.argmax(1) == Y[te].argmax(1)).mean()), 4),
        "baseline_top_action_agreement": round(float((base.argmax(1) == Y[te].argmax(1)).mean()), 4),
        "by_node": {},
    }
    for node in NODES:
        m = N[te] == node
        if m.any():
            rep["by_node"][node] = {
                "examples": int(m.sum()),
                "model_mae": round(float(np.abs(P[m] - Y[te][m]).mean()), 4),
                "baseline_mae": round(float(np.abs(base[m] - Y[te][m]).mean()), 4),
                "model_top_agree": round(float((P[m].argmax(1) == Y[te][m].argmax(1)).mean()), 4),
            }
    import time

    t0 = time.time()
    model.predict(X[te][:1])
    rep["predict_ms_one_hand"] = round((time.time() - t0) * 1000, 3)
    dump(model, os.path.join(outdir, "model.joblib"))
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
