"""Predictive analysis on real hand histories: next-action prediction (incl. an echo state network), PCA of
playing styles, and whether style predicts profit out of sample.

Usage: uv run python scripts/predictive.py HANDHQ_DIR OUTDIR [HANDS_PER_FOLDER] [WORKERS]
Reads every site/stake folder of the HandHQ PHH data (anonymised 2009 hands, ids stable within a folder) with
PokerKit, which replays each hand to the final stacks, so every player's net result is exact.

Splits are by time within each folder, so nothing from the future is used:
  A (first 40% of hands)  player profiles (VPIP, PFR, 3-bet, aggression, fold to a bet) for task 1
  B (next 40%)            training decisions for task 1
  C (last 20%)            test decisions for task 1
1. Next action (fold / check-call / bet-raise) at each decision. Features: street, table size, seat, pot and price
   in big blinds, raises so far, counts of earlier actions this hand, and the player's profile from A. Models: the
   majority action per street, logistic regression, gradient boosting, random forest, an MLP, and an echo state
   network (a fixed random recurrent reservoir that reads the hand's earlier actions one by one; only its linear
   read-out is trained) with and without the static features.
2. PCA of styles: players with 500+ hands, five standardised style rates; explained variance and loadings.
3. Style -> profit: style from each player's first half of hands, big blinds won per 100 hands in the second half;
   cross-validated R^2 of a linear and a gradient-boosted model, and win rate by predicted quintile.
"""

import collections
import glob
import json
import math
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hand_histories as hh  # noqa: E402

warnings.filterwarnings("ignore")
STREETS = ["preflop", "flop", "turn", "river"]
ACTS = {"f": 0, "cc": 1, "cbr": 2}
STYLE = ["vpip", "pfr", "three_bet", "aggression", "fold_to_bet"]


def read_folder(args):
    """One site/stake folder: per hand, the decision rows in order and each player's net result in big blinds."""
    folder, limit = args
    from pokerkit import HandHistory

    files = sorted(glob.glob(os.path.join(folder, "**", "*.phhs"), recursive=True))
    out, n = [], 0
    for f in files:
        for part in re.split(r"(?m)^\[\d+\]\s*$", open(f, errors="replace").read()):
            if not part.strip() or n >= limit:
                continue
            try:
                h = hh.parse_phh(part)
                if "actions" not in h or "players" not in h:
                    continue
                hist = HandHistory.loads(part)
                st = None
                for st in hist:
                    pass
                bb = abs(float(max(h.get("blinds_or_straddles", [1]) or [1]))) or 1.0
                net = [(float(b) - float(a)) / bb for a, b in zip(hist.starting_stacks, st.stacks)]
                rows = list(hh.replay(h))
                if not all(math.isfinite(x) for x in net):
                    continue
            except Exception:  # a hand PokerKit cannot replay is skipped, not guessed
                continue
            n += 1
            out.append({"players": h["players"], "net": net, "bb": bb, "rows": rows})
    return os.path.basename(folder.rstrip("/")), out


def profile(hands):
    """Style rates per player id from a list of hands (same definitions as hand_histories.py)."""
    c = collections.defaultdict(collections.Counter)
    for hd in hands:
        vp, pr, three = set(), set(), set()
        for r in hd["rows"]:
            who, pre = r["player"], r["street"] == "preflop"
            if pre and r["action"] in ("cc", "cbr") and not (r["action"] == "cc" and r["to_call"] == 0):
                vp.add(who)
            if pre and r["action"] == "cbr":
                pr.add(who)
                if r["raises_before"] >= 1:
                    three.add(who)
            if pre and r["raises_before"] == 1:
                c[who]["three_chances"] += 1
            if not pre and r["action"] == "cbr":
                c[who]["post_raises"] += 1
            if not pre and r["action"] == "cc" and r["to_call"] > 0:
                c[who]["post_calls"] += 1
            if not pre and r["to_call"] > 0:
                c[who]["faced"] += 1
                c[who]["folded"] += r["action"] == "f"
        for who, x in zip(hd["players"], hd["net"]):
            c[who]["hands"] += 1
            c[who]["vpip"] += who in vp
            c[who]["pfr"] += who in pr
            c[who]["three"] += who in three
            c[who]["net"] += x
    out = {}
    for who, k in c.items():
        h = k["hands"]
        out[who] = {
            "hands": h,
            "vpip": k["vpip"] / h,
            "pfr": k["pfr"] / h,
            "three_bet": k["three"] / k["three_chances"] if k["three_chances"] else 0.0,
            "aggression": k["post_raises"] / (k["post_raises"] + k["post_calls"])
            if k["post_raises"] + k["post_calls"]
            else 0.5,
            "fold_to_bet": k["folded"] / k["faced"] if k["faced"] else 0.5,
            "bb_per_100": 100 * k["net"] / h,
        }
    return out


class ESN:
    """Leaky echo state network: fixed random input and recurrent weights, spectral radius below 1."""

    def __init__(self, n_in, n=200, rho=0.9, leak=0.3, density=0.1, seed=0):
        r = np.random.default_rng(seed)
        W = r.normal(size=(n, n)) * (r.random((n, n)) < density)
        self.W = W * (rho / max(abs(np.linalg.eigvals(W))))
        self.Win = r.uniform(-0.5, 0.5, size=(n, n_in))
        self.leak, self.n = leak, n

    def step(self, x, u):
        return (1 - self.leak) * x + self.leak * np.tanh(self.Win @ u + self.W @ x)


def esn_input(r, who):
    u = np.zeros(10)
    u[ACTS[r["action"]]] = 1
    u[3 + STREETS.index(r["street"])] = 1
    u[7] = min(r.get("size_x_pot", 0) or 0, 3) / 3
    u[8] = 1.0 if r["player"] == who else 0.0
    u[9] = min(max(r["to_call"], 0) / max(r["pot"], 1e-9), 2) / 2
    return u


def decisions(hands, prof, esn, mean_prof, cap, rng):
    """Feature rows for every decision (sampled to `cap`), with the ESN state over the hand's earlier actions."""
    X, E, y, street = [], [], [], []
    idx = [(k, j) for k, hd in enumerate(hands) for j in range(len(hd["rows"]))]
    keep = set(map(tuple, rng.permutation(np.array(idx, dtype=np.int64))[:cap].tolist())) if len(idx) > cap else None
    for k, hd in enumerate(hands):
        bb, rows = hd["bb"], hd["rows"]
        x_state = np.zeros(esn.n)
        counts = np.zeros(3)
        for j, r in enumerate(rows):
            if keep is None or (k, j) in keep:
                p = prof.get(r["player"], mean_prof)
                X.append(
                    [
                        *(float(r["street"] == s) for s in STREETS),
                        r["players"],
                        r["seat"] / max(r["players"], 1),
                        math.log1p(max(r["pot"], 0) / bb),
                        max(r["to_call"], 0) / max(r["pot"], 1e-9),
                        math.log1p(max(r["to_call"], 0) / bb),
                        r["raises_before"],
                        *counts,
                        *(p[s] for s in STYLE),
                        math.log1p(p["hands"]),
                    ]
                )
                E.append(x_state.copy())
                y.append(ACTS[r["action"]])
                street.append(r["street"])
            x_state = esn.step(x_state, esn_input(r, r["player"]))
            counts[ACTS[r["action"]]] += 1
    return np.array(X), np.array(E), np.array(y), np.array(street)


def main(src, outdir, per_folder=60000, workers=24):
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    os.makedirs(outdir, exist_ok=True)
    folders = sorted(d for d in glob.glob(os.path.join(src, "*")) if os.path.isdir(d))
    with ProcessPoolExecutor(workers) as pool:
        data = dict(pool.map(read_folder, [(f, per_folder) for f in folders]))
    rep = {"folders": len(data), "hands": sum(len(v) for v in data.values())}
    print(rep, flush=True)
    rng = np.random.default_rng(0)
    esn = ESN(10)

    # 1. next-action prediction with time splits per folder
    A, B, C = [], [], []
    for name, hands in data.items():
        for hd in hands:  # ids are only stable within a folder
            hd["players"] = [f"{name}|{p}" for p in hd["players"]]
            for r in hd["rows"]:
                r["player"] = f"{name}|{r['player']}"
        n = len(hands)
        A += hands[: int(0.4 * n)]
        B += hands[int(0.4 * n) : int(0.8 * n)]
        C += hands[int(0.8 * n) :]
    prof = profile(A)
    mean_prof = {s: float(np.mean([p[s] for p in prof.values()])) for s in STYLE} | {"hands": 0}
    Xb, Eb, yb, sb = decisions(B, prof, esn, mean_prof, 300000, rng)
    Xc, Ec, yc, sc = decisions(C, prof, esn, mean_prof, 100000, rng)
    res = {
        "train_decisions": len(yb),
        "test_decisions": len(yc),
        "class_share_test": np.bincount(yc, minlength=3).tolist(),
    }
    maj = {s: np.bincount(yb[sb == s], minlength=3) for s in STREETS}
    Pm = np.array([(maj[s] + 1) / (maj[s] + 1).sum() for s in sc])
    res["majority per street"] = {
        "accuracy": round(accuracy_score(yc, Pm.argmax(1)), 4),
        "log_loss": round(log_loss(yc, Pm), 4),
    }
    st_c = np.array([[float(s == t) for t in STREETS] for s in sc])
    st_b = np.array([[float(s == t) for t in STREETS] for s in sb])
    models = {
        "logistic regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Xb, Xc),
        "gradient boosting": (HistGradientBoostingClassifier(max_iter=300, random_state=0), Xb, Xc),
        "random forest": (RandomForestClassifier(300, min_samples_leaf=20, n_jobs=workers, random_state=0), Xb, Xc),
        "MLP (deep learning)": (
            make_pipeline(StandardScaler(), MLPClassifier((128, 64), early_stopping=True, random_state=0)),
            Xb,
            Xc,
        ),
        "ESN only (reservoir + street)": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            np.hstack([Eb, st_b]),
            np.hstack([Ec, st_c]),
        ),
        "ESN + static features": (
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            np.hstack([Eb, Xb]),
            np.hstack([Ec, Xc]),
        ),
    }
    for name, (m, xtr, xte) in models.items():
        m.fit(xtr, yb)
        P = m.predict_proba(xte)
        res[name] = {"accuracy": round(accuracy_score(yc, P.argmax(1)), 4), "log_loss": round(log_loss(yc, P), 4)}
        print(name, res[name], flush=True)
    rep["1_next_action"] = res
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)

    # 2. PCA of styles (all hands) and 3. style -> profit (first half vs second half of each player's hands)
    every = [hd for hands in data.values() for hd in hands]
    allp = {k: v for k, v in profile(every).items() if v["hands"] >= 500}
    names = sorted(allp)
    S = np.array([[allp[k][s] for s in STYLE] for k in names])
    Z = StandardScaler().fit_transform(S)
    pca = PCA().fit(Z)
    pcs = pca.transform(Z)[:, :2]
    rep["2_pca"] = {
        "players": len(names),
        "explained_variance": [round(float(v), 3) for v in pca.explained_variance_ratio_],
        "loadings": {
            f"PC{k + 1}": dict(zip(STYLE, [round(float(v), 3) for v in pca.components_[k]])) for k in range(3)
        },
        "bb_per_100_corr_with_PC": {
            f"PC{k + 1}": round(
                float(np.corrcoef(pca.transform(Z)[:, k], [allp[n]["bb_per_100"] for n in names])[0, 1]), 3
            )
            for k in range(3)
        },
    }
    pick = rng.choice(len(names), min(4000, len(names)), replace=False)
    json.dump(
        [
            {
                "pc1": round(float(pcs[k, 0]), 3),
                "pc2": round(float(pcs[k, 1]), 3),
                "hands": allp[names[k]]["hands"],
                "bb_per_100": round(allp[names[k]]["bb_per_100"], 1),
                **{s: round(allp[names[k]][s], 3) for s in STYLE},
            }
            for k in pick
        ],
        open(os.path.join(outdir, "pca_points.json"), "w"),
    )
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    per_player = collections.defaultdict(list)
    for hd in every:
        for p in hd["players"]:
            per_player[p].append(hd)
    first, second = [], []
    for p, hs in per_player.items():
        if len(hs) >= 1000:
            first.append(profile(hs[: len(hs) // 2]).get(p))
            second.append(profile(hs[len(hs) // 2 :]).get(p))
    Xs = np.array([[f[s] for s in STYLE] for f in first])
    ys = np.array([s["bb_per_100"] for s in second])
    yf = np.array([f["bb_per_100"] for f in first])
    kf = KFold(5, shuffle=True, random_state=0)
    out3 = {
        "players_1000_plus_hands": len(ys),
        "persistence_corr_first_vs_second_half": round(float(np.corrcoef(yf, ys)[0, 1]), 3),
    }
    for name, m in (
        ("linear (ridge)", make_pipeline(StandardScaler(), Ridge(1.0))),
        ("gradient boosting", HistGradientBoostingRegressor(max_iter=200, random_state=0)),
    ):
        pred = cross_val_predict(m, Xs, ys, cv=kf)
        r2 = 1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
        q = np.quantile(pred, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(pred, q)
        out3[name] = {
            "cv_r2": round(float(r2), 4),
            "corr": round(float(np.corrcoef(pred, ys)[0, 1]), 3),
            "second_half_bb_per_100_by_predicted_quintile": [round(float(ys[bins == b].mean()), 1) for b in range(5)],
        }
        print(name, out3[name], flush=True)
    rep["3_style_to_profit"] = out3
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:4]))
