"""Opponent embeddings from betting sequences, as speech systems embed speakers: does a learned summary of a player's
hands name its type in fewer hands than the six HUD statistics Kestrel's profiles use?

Data: kestrel.db files from test-platform sessions with known bots (the name gives the type). Each hand a player is
dealt into becomes 26 numbers: its actions counted by street and kind (4 x 6), whether it showed cards, and the
table size. A sample is a window of N consecutive hands of one player. Three classifiers of the type:

  stats     the six HUD rates over the window (VPIP, raise, 3-bet, fold to c-bet, aggression, showdown), multinomial
            logistic regression: what Kestrel's profiles read
  xvector   a per-hand network, then mean and standard-deviation pooling over the window, then a classifier (the
            x-vector design of speaker recognition)
  gru       a recurrent network over the hands in order

Trained on some sessions, tested on others (no hands shared). Accuracy at N = 10, 20, 50.

Usage (lab, GPU venv): PYTHONPATH=scripts .venv-gpu/bin/python scripts/opponent_embed.py DB... --test-dbs DB...
"""

import argparse
import json
import sqlite3
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = ["fold", "check", "call", "bet", "raise", "allin"]


def hands_of(db):
    """Per player (named by the bot type): the list of per-hand vectors and per-hand HUD events, in order."""
    out = defaultdict(list)
    for (data,) in sqlite3.connect(db).execute("SELECT data FROM hands ORDER BY session_id, seq"):
        h = json.loads(data)
        t = h.get("table")
        if not t:
            continue
        acts, shown, me = t["actions"], t.get("shown", {}), t["me"]
        # HUD events per player in this hand (as OpponentBook counts them, simplified)
        raises, aggressor, three_chance, three, vpip, pfr = 0, None, set(), set(), set(), set()
        for p, st, k in acts:
            if st != 0:
                continue
            if raises == 1 and k != "check":
                three_chance.add(p)
            if k in ("call", "bet", "raise", "allin"):
                vpip.add(p)
            if k in ("bet", "raise", "allin"):
                if raises == 1:
                    three.add(p)
                raises, aggressor = raises + 1, p
                pfr.add(p)
        cbet, faced, folded_cbet = False, set(), set()
        for p, st, k in acts:
            if st != 1:
                continue
            if not cbet and k in ("bet", "raise", "allin"):
                if p != aggressor:
                    break
                cbet = True
                continue
            if cbet and p != aggressor and p not in faced:
                faced.add(p)
                if k == "fold":
                    folded_cbet.add(p)
        for p in t["players"]:
            if p == me:
                continue
            v = np.zeros(26, np.float32)
            agg = pas = 0
            saw_flop = False
            for q, st, k in acts:
                if q == p and k in KINDS:
                    v[st * 6 + KINDS.index(k)] += 1
                    if st >= 1:
                        saw_flop = True
                        agg += k in ("bet", "raise", "allin")
                        pas += k == "call"
            v[24] = p in shown
            v[25] = len(t["players"]) / 9
            ev = [1, p in vpip, p in pfr, p in three, p in three_chance, p in folded_cbet, p in faced, agg, agg + pas,
                  p in shown, saw_flop]  # fmt: skip
            out[p].append((v, np.array(ev, np.float32)))
    return out


def label(name):
    return name.rstrip("0123456789")


def windows(dbs, N, classes):
    X, E, y = [], [], []
    for db in dbs:
        for p, seq in hands_of(db).items():
            c = label(p)
            if c not in classes:
                continue
            for k in range(0, len(seq) - N + 1, N):
                w = seq[k : k + N]
                X.append(np.stack([a for a, _ in w]))
                E.append(np.stack([b for _, b in w]).sum(0))
                y.append(classes.index(c))
    return np.stack(X), np.stack(E), np.array(y)


def hud_rates(E, prior=8.0):
    """The six rates from summed events [hands, vpip, pfr, 3bet, 3bet chances, cbet folds, cbet faced, aggressive,
    aggressive + passive, shown, saw flop], each shrunk towards 0.3 with `prior` chances (as the profiles shrink)."""

    def r(x, n):
        return (x + prior * 0.3) / (n + prior)

    return np.stack([r(E[:, 1], E[:, 0]), r(E[:, 2], E[:, 0]), r(E[:, 3], E[:, 4]), r(E[:, 5], E[:, 6]),
                     r(E[:, 7], E[:, 8]), r(E[:, 9], E[:, 10])], 1)  # fmt: skip


class XVector(nn.Module):
    def __init__(self, d, k, h=64):
        super().__init__()
        self.frame = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, k))

    def forward(self, x):  # [B, N, d]
        f = self.frame(x)
        return self.head(torch.cat([f.mean(1), f.std(1, unbiased=False)], 1))


class GRUNet(nn.Module):
    def __init__(self, d, k, h=64):
        super().__init__()
        self.gru = nn.GRU(d, h, batch_first=True)
        self.head = nn.Linear(h, k)

    def forward(self, x):
        _, hN = self.gru(x)
        return self.head(hN[-1])


class Logistic(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.lin = nn.Linear(d, k)

    def forward(self, x):
        return self.lin(x)


def fit(model, Xtr, ytr, Xte, yte, epochs=60, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr))
        for b in range(0, len(Xtr), 256):
            i = perm[b : b + 256]
            loss = F.cross_entropy(model(Xtr[i]), ytr[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(torch.tensor(Xte)).argmax(1).numpy()
    acc = float((pred == yte).mean())
    se = float(np.sqrt(acc * (1 - acc) / len(yte)))
    return acc, se, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dbs", nargs="+")
    ap.add_argument("--test-dbs", nargs="+", required=True)
    ap.add_argument("--classes", default="tag,station,limper,odd,minraiser,tilter")
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_num_threads(8)
    classes = a.classes.split(",")
    res = {"classes": classes, "rows": []}
    for N in (10, 20, 50):
        Xtr, Etr, ytr = windows(a.dbs, N, classes)
        Xte, Ete, yte = windows(a.test_dbs, N, classes)
        mu, sd = Xtr.reshape(-1, Xtr.shape[-1]).mean(0), Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
        Xtr_n, Xte_n = (Xtr - mu) / sd, (Xte - mu) / sd
        Htr, Hte = hud_rates(Etr), hud_rates(Ete)
        hm, hs = Htr.mean(0), Htr.std(0) + 1e-6
        row = {"N": N, "train": len(ytr), "test": len(yte)}
        for name, model, xtr, xte in (
            ("stats", Logistic(6, len(classes)), (Htr - hm) / hs, (Hte - hm) / hs),
            ("xvector", XVector(26, len(classes)), Xtr_n, Xte_n),
            ("gru", GRUNet(26, len(classes)), Xtr_n, Xte_n),
        ):
            acc, se, pred = fit(model, xtr.astype(np.float32), ytr, xte.astype(np.float32), yte)
            row[name] = {"accuracy": round(acc, 4), "se": round(se, 4),
                         "per_class": {c: round(float((pred[yte == k] == k).mean()), 3) for k, c in enumerate(classes)
                                       if (yte == k).any()}}  # fmt: skip
        res["rows"].append(row)
        print(f"N {N:3d} hands ({row['train']} train / {row['test']} test windows): "
              + "  ".join(f"{m} {row[m]['accuracy']:.3f}+-{1.96 * row[m]['se']:.3f}" for m in ("stats", "xvector", "gru")),
              flush=True)  # fmt: skip
        for m in ("stats", "xvector", "gru"):
            print(f"    {m:8s} " + " ".join(f"{c} {v:.2f}" for c, v in row[m]["per_class"].items()))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
