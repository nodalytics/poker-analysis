"""The tournament network's short-stack play against the ICM push/fold equilibrium (icm_pushfold.py).

For every chart table without an ante (the engine has none) and every first-in seat, the network is dealt random
hands with the chart's stacks and payouts, everyone before the seat folds, and its policy is read:
  shove    the probability of all-in (the chart's only way in),
  in       the probability of entering at all (all-in, call/limp or a smaller raise),
  agree    the share of hands on which it does what the chart does, in or out (by hand class, from the cards).
Then the big blind facing the small blind's shove (everyone else folded): its call probability against the chart's.

Usage (lab, GPU venv, from research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_pushfold_check.py blueprint/nm8 [blueprint/nm7 ...]
      [--charts results/ninemax/icm_pushfold.json] [--hands 20000] [--out results/ninemax/pushfold_check.json]
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from ninemax_engine import BB, Dealer, Menu, Tables, strength
from ninemax_eval import load
from ninemax_selfplay import features, masked
from icm import icm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKS = "23456789TJQKA"


def class_name(c1, c2):
    """Hand class ('AKs', 'T9o', '77') of two cards encoded rank * 4 + suit."""
    r1, r2 = c1 // 4, c2 // 4
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo:
        return RANKS[hi] * 2
    return RANKS[hi] + RANKS[lo] + ("s" if c1 % 4 == c2 % 4 else "o")


def seat_of(j, n):
    """Chart action-order index (pos0 .. button, small blind, big blind) -> engine seat (0 SB, 1 BB, 2 UTG ..)."""
    return j + 2 if j < n - 2 else (0 if j == n - 2 else 1)


def policy(net, T, g, bk, dealer):
    i = T.actor[g]
    s = strength(dealer, bk[g], T.street[g]).gather(1, i[:, None]).squeeze(1)
    feat, legal = features(T, g, i)
    with torch.no_grad():
        lg, _ = net(feat, bk[g, i], T.street[g], s)
    return F.softmax(masked(lg, legal), 1), i


def check(net, chart, dealer, menu, G, dev):
    n = len(chart["stacks_bb"])
    st = torch.zeros(n, dtype=torch.long)
    for j, b in enumerate(chart["stacks_bb"]):
        st[seat_of(j, n)] = round(b * BB)
    pay = torch.zeros(n)
    pay[: len(chart["payouts"])] = torch.tensor(chart["payouts"])
    pay = pay / pay.sum()
    names = list(chart["shove"])
    S = menu.S
    out = {}
    for j, name in enumerate(names + ["bb_vs_sb"]):
        bk, _, cd = dealer.deal(G, cards=True)
        T = Tables(G, n, menu, dev, stacks=st[None].expand(G, n).to(dev).clone())
        T.pay = pay[None].expand(G, n).to(dev).clone()
        T.eq0 = icm(T.stack.float(), T.pay).float()
        g = torch.arange(G, device=dev)
        target = seat_of(n - 2 if name == "bb_vs_sb" else j, n)
        while int(T.actor[0]) != target:  # everyone before the seat folds (all tables move together)
            T.step(g, torch.zeros(G, dtype=torch.long, device=dev))
        if name == "bb_vs_sb":  # the small blind shoves; read the big blind
            T.step(g, torch.full((G,), S + 2, dtype=torch.long, device=dev))
            assert int(T.actor[0]) == 1
            rng = set(chart["call"][f"big blind vs {names[-1]}"]["range"])
            ref = chart["call"][f"big blind vs {names[-1]}"]["share"]
        else:
            rng = set(chart["shove"][name]["range"])
            ref = chart["shove"][name]["share"]
        probs, i = policy(net, T, g, bk, dealer)
        p_in = 1 - probs[:, 0]
        p_shove = probs[:, S + 2] if name != "bb_vs_sb" else p_in
        seat = int(i[0])
        hands = [class_name(int(a), int(b)) for a, b in zip(cd[:, 2 * seat], cd[:, 2 * seat + 1])]
        inr = torch.tensor([h in rng for h in hands], device=dev, dtype=torch.float32)
        agree = (inr * p_in + (1 - inr) * (1 - p_in)).mean()
        out[name] = {
            "chart": ref,
            "in": round(float(p_in.mean()), 3),
            "shove": round(float(p_shove.mean()), 3),
            "agree": round(float(agree), 3),
            "in_when_chart_in": round(float((p_in * inr).sum() / inr.sum().clamp(min=1)), 3),
            "in_when_chart_out": round(float((p_in * (1 - inr)).sum() / (1 - inr).sum().clamp(min=1)), 3),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--charts", default=os.path.join(ROOT, "results", "ninemax", "icm_pushfold.json"))
    ap.add_argument("--hands", type=int, default=20000)
    ap.add_argument("--out")
    ap.add_argument("--lib", default=os.path.join(ROOT, "blueprint", "libbp5.so"))
    ap.add_argument("--abs", default=os.path.join(ROOT, "blueprint", "out"))
    ap.add_argument("--seed", type=int, default=77)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B = int(open(os.path.join(a.abs, "meta.txt")).read().split()[1])
    charts = [c for c in json.load(open(a.charts))["charts"] if not c["ante"]]
    res = {}
    for run in a.runs:
        net, hands = load(run, B, dev)
        d = os.path.dirname(run) if run.endswith(".pt") else run  # a run, or a kept snapshot RUN/avg_<h>h.pt
        cfg = json.load(open(os.path.join(d, "config.json")))
        menu = Menu(cfg.get("menu", "medium"), dev)
        dealer = Dealer(a.lib, a.abs, 9, B, dev, seed=a.seed)
        tag = run.rstrip("/").split("blueprint/")[-1].removesuffix(".pt")
        print(f"== {tag} ({hands / 1e6:.0f} M hands)", flush=True)
        for c in charts:
            r = check(net, c, dealer, menu, a.hands, dev)
            res[f"{tag} | {c['name']}"] = r
            cells = " ".join(f"{v['in']:.0%}/{v['chart']:.0%}" for k, v in r.items() if k != "bb_vs_sb")
            agree = sum(v["agree"] for v in r.values()) / len(r)
            bb = r["bb_vs_sb"]
            print(f"{c['name']:45s} in/chart by seat: {cells} | BB calls SB {bb['in']:.0%}/{bb['chart']:.0%}"
                  f" | agree {agree:.0%}", flush=True)  # fmt: skip
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
