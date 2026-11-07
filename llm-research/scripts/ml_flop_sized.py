"""A flop model that knows bet sizes, trained on solver_library_sized.py's multi-size solves.

Usage: uv run python scripts/ml_flop_sized.py LIBRARY_DIR OUT_DIR
Decision points (one example per hand in range at each):
  oop_open          first to act on the flop
  ip_after_check    in position after a check
  ip_facing_bet     in position facing a bet of 33/66/100/150% pot or all-in (one point per size)
  oop_facing_bet    out of position after checking, facing a bet of each size (the old model had no such point)
Inputs: ml_flop.features (hand and board), the decision point, the bet faced as a fraction of the pot (0 when none)
and the stack-to-pot ratio. Outputs: probabilities of check/call, bet 33/66/100/150%, raise, all-in, fold.
Scored on flops held out entirely (grouped split): mean absolute error and top-action agreement, per decision point
and per bet size faced. Writes OUT_DIR/mlp.joblib, meta.json and report.json.
"""

import glob
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_flop  # noqa: E402

SLOTS = ["passive", "bet33", "bet66", "bet100", "bet150", "raise", "allin", "fold"]
SIZES = [0.33, 0.66, 1.0, 1.5]
NODES = ["oop_open", "ip_after_check", "ip_facing_bet", "oop_facing_bet"]
POT = 10.0
STACKS = {"spr9.5_wide": 95.0, "spr4.5_wide": 45.0}


def slot(action, pot, stack):
    p = action.split()
    if p[0] in ("CHECK", "CALL"):
        return 0
    if p[0] == "FOLD":
        return 7
    amt = float(p[1]) if len(p) > 1 else 0.0
    if amt >= 0.98 * stack:
        return 6
    if p[0] == "RAISE":
        return 5
    frac = amt / pot
    return 1 + min(range(4), key=lambda k: abs(SIZES[k] - frac))


def feats(board, hand, node, spot, bet_frac):
    base = ml_flop.features(board, hand, "oop_open", spot)
    return base + [float(node == n) for n in NODES] + [bet_frac, STACKS.get(spot, 95.0) / POT / 10]


def points(tree, stack):
    """(node name, node, pot at the node, bet faced as a fraction of the pot before it)."""
    out = [("oop_open", tree, POT, 0.0)]
    for name, child in (tree.get("childrens") or {}).items():
        if child.get("node_type") != "action_node":
            continue
        if name.startswith("CHECK"):
            out.append(("ip_after_check", child, POT, 0.0))
            for n2, c2 in (child.get("childrens") or {}).items():
                if c2.get("node_type") == "action_node" and n2.startswith("BET"):
                    amt = float(n2.split()[1])
                    out.append(("oop_facing_bet", c2, POT + amt, amt / POT))
        elif name.startswith("BET"):
            amt = float(name.split()[1])
            out.append(("ip_facing_bet", child, POT + amt, amt / POT))
    return out


def load(libdir):
    X, Y, G, N, B = [], [], [], [], []
    for path in sorted(glob.glob(os.path.join(libdir, "solves", "*.json.gz"))):
        sid = os.path.basename(path)[: -len(".json.gz")]
        flop_s, spot = sid.split("_", 1)
        board = [flop_s[i : i + 2] for i in (0, 2, 4)]
        tree = json.load(gzip.open(path, "rt"))
        stack = STACKS.get(spot, 95.0)
        for name, node, pot, bet in points(tree, stack):
            acts = node.get("actions", [])
            strat = (node.get("strategy") or {}).get("strategy") or {}
            for hand, probs in strat.items():
                y = np.zeros(len(SLOTS))
                for a, p in zip(acts, probs):
                    y[slot(a, pot, stack)] += p
                X.append(feats(board, hand, name, spot, bet))
                Y.append(y)
                G.append(flop_s)
                N.append(name)
                B.append(round(bet, 2))
    return np.array(X), np.array(Y), np.array(G), np.array(N), np.array(B)


def score(P, Y, mask):
    return {
        "examples": int(mask.sum()),
        "mae": round(float(np.abs(P[mask] - Y[mask]).mean()), 4),
        "top_agree": round(float((P[mask].argmax(1) == Y[mask].argmax(1)).mean()), 4),
    }


def main(libdir, outdir):
    from joblib import dump
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.neural_network import MLPRegressor

    os.makedirs(outdir, exist_ok=True)
    X, Y, G, N, B = load(libdir)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, Y, G))
    model = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=200, early_stopping=True, random_state=0)
    model.fit(X[tr], Y[tr])
    P = ml_flop.normalise(model.predict(X[te]))
    base = np.zeros_like(Y[te])  # baseline: each decision point's average strategy (per bet size) on training flops
    for n in NODES:
        for b in np.unique(B):
            m_tr = (N[tr] == n) & (B[tr] == b)
            m_te = (N[te] == n) & (B[te] == b)
            if m_tr.any() and m_te.any():
                base[m_te] = Y[tr][m_tr].mean(0)
    rep = {
        "solves": len(set(G)),
        "examples": len(Y),
        "test_flops": len(set(G[te])),
        "model": score(P, Y[te], np.ones(len(te), bool)),
        "baseline": score(base, Y[te], np.ones(len(te), bool)),
        "by_node": {n: score(P, Y[te], N[te] == n) for n in NODES if (N[te] == n).any()},
        "by_bet_faced": {
            str(b): score(P, Y[te], (B[te] == b) & np.isin(N[te], ["ip_facing_bet", "oop_facing_bet"]))
            for b in np.unique(B[te])
            if b > 0
        },
    }
    dump(model, os.path.join(outdir, "mlp.joblib"))
    json.dump({"slots": SLOTS, "sizes": SIZES, "nodes": NODES}, open(os.path.join(outdir, "meta.json"), "w"))
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
