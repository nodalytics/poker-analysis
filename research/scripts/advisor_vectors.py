"""Reference data for Kestrel, the C++ advisor (../kestrel/, at the repo root): network weights in a flat file, and test vectors it must
reproduce.

Writes, for one exported 9-max network (models/ninemax/<name>/):
  net.bin   every tensor of the averaged network in state_dict order, float16 little-endian, back to back (the
            exported weights are half precision already, so nothing is lost);
  net.json  the tensor names and shapes (in that order), the run's game settings (menu, ante, open-limp, tournament
            features), the bucket count and the 169 pre-flop class strengths the features use (from the trainer's
            dealer, seed 1, as in training).
and ../kestrel/testdata/vectors_<name>.json:
  hands     whole hands at random tournament tables (stacks, payouts, empty seats as in training), each decision with
            the actor, the legal mask, the raise-to targets, the feature vector, the network's logits and value, and
            the action taken (the network's choice, or 1 in 5 a uniformly random legal action, to reach odd states);
            the hand's buckets and showdown ranks, and the payoffs and starting ICM equities;
  icm       ICM equities for random stacks and payouts (2-9 players).

Usage (lab, from research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/advisor_vectors.py models/ninemax/nm17/avg_fp16.pt [--hands 120]
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from icm import icm
from ninemax_engine import BB, MENUS, RULES, Dealer, Menu, Tables, strength
from ninemax_eval import load
from ninemax_selfplay import FEAT, context, features, masked, split_pay, start_equity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = 9


def r6(x):
    return [round(float(v), 6) for v in x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="models/ninemax/<name>/avg_fp16.pt (or any run checkpoint)")
    ap.add_argument("--hands", type=int, default=120)
    ap.add_argument("--icm", type=int, default=200)
    ap.add_argument("--lib", default=os.path.join(ROOT, "blueprint", "libbp5.so"))
    ap.add_argument("--abs", default=os.path.join(ROOT, "blueprint", "out"))
    ap.add_argument("--out", help="where to write net.bin and net.json (default: beside the model)")
    ap.add_argument("--vectors-out", help="the test-vector file (default: ../kestrel/testdata/vectors_<name>.json)")
    a = ap.parse_args()
    dev = "cpu"
    torch.manual_seed(5)
    random.seed(5)
    d = os.path.dirname(a.model)
    name = os.path.basename(d)
    cfg = json.load(open(os.path.join(d, "config.json")))
    B = int(open(os.path.join(a.abs, "meta.txt")).read().split()[1])
    net, _ = load(a.model, B, dev)  # sets RULES and FEAT from the config
    dealer = Dealer(a.lib, a.abs, P, B, dev)  # seed 1: the trainer's pre-flop strengths
    menu = Menu(cfg.get("menu", "medium"), dev)

    # ---- weights ----
    # training-only outputs (the belief head) are not part of the policy Kestrel reads
    sd = {k: t for k, t in net.state_dict().items() if not k.startswith("belief.")}
    wd = a.out or d
    os.makedirs(wd, exist_ok=True)
    with open(os.path.join(wd, "net.bin"), "wb") as f:
        for t in sd.values():
            f.write(t.detach().half().contiguous().numpy().astype("<f2").tobytes())
    meta = {
        "tensors": [[k, list(t.shape)] for k, t in sd.items()],
        "seats": cfg.get("seats") or cfg.get("players", 9),  # the table size it plays (6 for 6-max), for its description
        "nfeat": cfg["nfeat"], "actions": cfg["actions"], "buckets": B, "players": P,
        "menu": cfg.get("menu", "medium"), "menu_def": MENUS[cfg.get("menu", "medium")],
        "ante_bb": RULES["ante"], "open_limp": RULES["open_limp"],
        "features": {"stacks": FEAT["stacks"] is not None, "tourney": FEAT["tourney"], "bounty": FEAT["bounty"]},
        "pre_strength": r6(dealer.pre_strength.tolist()),
        "source": a.model,
    }  # fmt: skip
    json.dump(meta, open(os.path.join(wd, "net.json"), "w"), indent=1)

    # ---- hands ----
    hands = []
    for h in range(a.hands):
        stacks, pay = context(1, P, dev)
        bk, rk, cd = dealer.deal(1, cards=True)
        T = Tables(1, P, menu, dev, stacks=stacks)
        if pay is not None:
            T.pay, T.eq0 = pay.float(), start_equity(T.stack, pay)
        reg, bty = split_pay(pay)
        g = torch.zeros(1, dtype=torch.long)
        steps = []
        while not bool(T.done.all()):
            i = T.actor[g]
            s = strength(dealer, bk, T.street).gather(1, i[:, None]).squeeze(1)
            feat, legal = features(T, g, i)
            _, t = T.legal(g, i)
            with torch.no_grad():
                lg, v = net(feat, bk[g, i], T.street, s)
            p = F.softmax(masked(lg, legal), 1)
            if random.random() < 0.2:
                act = int(random.choice(legal[0].nonzero().squeeze(1).tolist()))
            else:
                act = int(torch.multinomial(p, 1))
            steps.append({
                "actor": int(i), "street": int(T.street[0]), "legal": [int(x) for x in legal[0]],
                "targets": [int(x) for x in t[0]], "strength": round(float(s), 6), "feat": r6(feat[0]),
                "logits": r6(lg[0]), "value": round(float(v), 6), "action": act,
            })  # fmt: skip
            T.step(g, torch.tensor([act]))
        u = T.payoffs(rk)
        hands.append({
            "stacks": [int(x) for x in T.stack[0]], "pay": r6(reg[0]) if pay is not None else None,
            "bounty": r6(bty[0]) if bty is not None else None,
            "eq0": r6(T.eq0[0]) if pay is not None else None,
            "buckets": [[int(x) for x in bk[0, j]] for j in range(P)], "ranks": [int(x) for x in rk[0]],
            "cards": [int(x) for x in cd[0]], "steps": steps, "payoffs": r6(u[0]),
            "showdown": bool(T.showdown[0]),
        })  # fmt: skip

    # ---- ICM ----
    icm_cases = []
    for _ in range(a.icm):
        n = random.randint(2, P)
        st = torch.tensor([[random.choice([0, random.uniform(0.5, 60)]) if j >= 2 else random.uniform(0.5, 60)
                            for j in range(n)]], dtype=torch.float64)  # fmt: skip
        pay = torch.tensor([sorted(np.random.dirichlet(np.ones(n)).tolist(), reverse=True)], dtype=torch.float64)
        icm_cases.append({"stacks": r6(st[0]), "pay": [float(x) for x in pay[0]], "eq": r6(icm(st, pay)[0])})

    out = a.vectors_out or os.path.join(ROOT, "..", "kestrel", "testdata", f"vectors_{name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"model": name, "bb": BB, "hands": hands, "icm": icm_cases}, open(out, "w"))
    n_dec = sum(len(x["steps"]) for x in hands)
    print(
        f"{name}: {len(sd)} tensors -> {wd}/net.bin; {len(hands)} hands, {n_dec} decisions, {len(icm_cases)} ICM -> {out}"
    )


if __name__ == "__main__":
    main()
