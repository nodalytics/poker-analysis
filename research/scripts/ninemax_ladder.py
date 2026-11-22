"""The version ladder: every tournament network against every other, in one game, so their strength can be compared
over time (is each champion better than the last, and by how much).

Every ordered pair (A, B) plays a duplicate match: A in each seat in turn against 8 copies of B, on the same deals
(ninemax_selfplay.duplicate). The game is one run's settings (--game, default nm25's: big-blind ante, half the final
tables short-handed, 10-25bb), so older versions are judged on today's game. A pair's two directions are combined,
d(A, B) = (A vs 8 B - B vs 8 A) / 2, and the ratings r are fitted by weighted least squares, r(A) - r(B) ~ d(A, B),
with r(--anchor) = 0: the rating is bb/100 (in ICM equity) above or below the anchor, and its interval comes from the
fit. Results are kept per pair in --out, so adding a version only plays its new pairs.

Usage (lab, GPU venv, from research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_ladder.py blueprint/nm12 blueprint/nm13 ... \\
      [--game blueprint/nm25] [--anchor nm17] [--hands 20000] [--out results/ninemax/ladder.json]
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from ninemax_engine import Dealer, Menu
from ninemax_eval import load
from ninemax_selfplay import duplicate, set_mode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def name_of(run):
    return run.rstrip("/").split("/")[-1].removesuffix(".pt")


def ratings(names, pairs, anchor):
    """Weighted least squares on the combined pair results: ratings and their 95% half-widths."""
    idx = {n: k for k, n in enumerate(names)}
    rows, y, w = [], [], []
    for key, p in pairs.items():
        a, b = key.split(" vs ")
        rev = pairs.get(f"{b} vs {a}")
        if a not in idx or b not in idx or rev is None or a > b:
            continue
        d = (p["bb100"] - rev["bb100"]) / 2
        se = np.hypot(p["se"], rev["se"]) / 2
        row = np.zeros(len(names))
        row[idx[a]], row[idx[b]] = 1, -1
        rows.append(row), y.append(d), w.append(1 / max(se, 1e-6) ** 2)
    if not rows:
        return {}
    X, y, W = np.array(rows), np.array(y), np.diag(w)
    played = np.abs(X).sum(0) > 0  # only versions with a pair played both ways get a rating
    keep = [k for k, n in enumerate(names) if n != anchor and played[k]]  # the anchor is 0
    Xk = X[:, keep]
    cov = np.linalg.pinv(Xk.T @ W @ Xk)
    r = cov @ Xk.T @ W @ y
    out = {anchor: {"rating": 0.0, "ci95": 0.0}} if anchor in idx and played[idx[anchor]] else {}
    for j, k in enumerate(keep):
        out[names[k]] = {"rating": round(float(r[j]), 2), "ci95": round(float(1.96 * np.sqrt(max(cov[j, j], 0))), 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--game", default=os.path.join(ROOT, "blueprint", "nm25"), help="the run whose game they all play")
    ap.add_argument("--anchor", default="nm17")
    ap.add_argument("--hands", type=int, default=20000)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "ninemax", "ladder.json"))
    ap.add_argument("--lib", default=os.path.join(ROOT, "blueprint", "libbp7.so"))
    ap.add_argument("--abs", default=os.path.join(ROOT, "blueprint", "out"))
    a = ap.parse_args()
    dev = "cuda"
    state = json.load(open(a.out)) if os.path.exists(a.out) else {"pairs": {}}
    dealer = Dealer(a.lib, a.abs, 9, 0, dev, seed=2026)
    nets = {}
    for run in a.runs:
        net, hands = load(run, dealer.B, dev)  # older runs do not record their bucket count: the abstraction's
        nets[name_of(run)] = (net, hands)
    cfg = json.load(open(os.path.join(a.game, "config.json")))
    set_mode(cfg.get("stacks", "100"), cfg.get("tourney", False), cfg.get("depths"), not cfg.get("no_open_limp", False),
             cfg.get("ante", 0.0), cfg.get("short", 0.0), cfg.get("seats"), cfg.get("bounty", 0.0))  # fmt: skip
    menu = Menu(cfg.get("menu", "medium"), dev)
    names = list(nets)
    for x in names:
        for y in names:
            key = f"{x} vs {y}"
            if x == y or key in state["pairs"]:
                continue
            t0 = time.time()
            m, lo, hi = duplicate(dealer, menu, nets[x][0], nets[y][0], a.hands, dev)
            state["pairs"][key] = {"bb100": round(float(m), 3), "se": round(float(hi - lo) / 3.92, 3), "hands": a.hands}
            print(f"{key:16s} {m:+6.2f} bb/100 [{lo:+.2f}, {hi:+.2f}]  {time.time() - t0:.0f} s", flush=True)
            state["game"], state["anchor"] = os.path.basename(a.game.rstrip("/")), a.anchor
            state["hands_trained"] = {n: int(nets[n][1]) for n in names} | state.get("hands_trained", {})
            state["ratings"] = ratings(sorted(set(state.get("ratings", {})) | set(names)), state["pairs"], a.anchor)
            state["updated"] = time.strftime("%Y-%m-%d %H:%M")
            json.dump(state, open(a.out, "w"), indent=1)
    state["ratings"] = ratings(sorted({k.split(" vs ")[0] for k in state["pairs"]}), state["pairs"], a.anchor)
    json.dump(state, open(a.out, "w"), indent=1)
    print("\nversion   rating (bb/100 vs " + a.anchor + ")")
    for n, r in sorted(state["ratings"].items(), key=lambda kv: int("".join(c for c in kv[0] if c.isdigit()) or 0)):
        print(f"{n:8s} {r['rating']:+7.2f} ± {r['ci95']:.2f}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
