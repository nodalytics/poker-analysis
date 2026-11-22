"""Do different versions of the 9-max network decide the same spot the same way? Where they do not, which spots?

Spots are drawn once, from tables of the reference network (nm25 by default) playing itself in its own game, and every
version is asked for its action probabilities on exactly those spots (the same features, cards and buckets). Two
measures per spot, over every pair of versions:
  disagreement  the mean total variation distance between their action distributions (0 identical, 1 disjoint)
  split         whether their most likely actions differ (one folds where another raises)
and, against the reference, each version's own distance. They are grouped by street, position, stack depth, whether
the player faces a bet, and how many are seated, and the spots where the versions disagree most are listed with the
hand and each version's choice.

A spot where the versions disagree a lot cannot be played right by all of them: a leak in some, or a decision the
training never settled. Those are the places to look first, and to test in the next run.

Versions are the runs given, or every blueprint/nm* run that reads the reference's inputs (the same number of inputs,
the same tournament, bounty and texture settings, and the same card abstraction). Others are listed as skipped.

Usage (lab, from research):
  CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_consistency.py \
      --reference blueprint/nm25 --out results/ninemax/consistency.json            [--runs blueprint/nm17,...]
The pipeline's "consistency" check runs it for one run against its reference (--runs RUN): a report, never a gate.
"""

import argparse
import glob
import json
import math
import os
import re

import torch
import torch.nn.functional as F

from ninemax_eval import load, setup
from ninemax_pushfold_check import class_name
from ninemax_selfplay import FEAT, board_of, context, masked, play

POS = ["SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO", "BTN"]  # engine seats 0..8
STREETS = ["pre-flop", "flop", "turn", "river"]
CARD = "23456789TJQKA"


def card(c):
    return CARD[c // 4] + "cdhs"[c % 4]


def compatible(cfg, ref):
    same = all(cfg.get(k) == ref.get(k) for k in ("nfeat", "actions", "abs", "menu"))
    same = same and all(bool(cfg.get(k)) == bool(ref.get(k)) for k in ("tourney", "texture"))  # missing = False
    return (
        same and float(cfg.get("bounty") or 0) == float(ref.get("bounty") or 0) and cfg.get("seats") == ref.get("seats")
    )


def versions(ref_run, given):
    ref = json.load(open(os.path.join(ref_run, "config.json")))
    if given:
        return given.split(","), []
    keep, skip = [], []
    for d in sorted(glob.glob("blueprint/nm*"), key=lambda d: int(re.sub(r"\D", "", d) or 0)):
        c, ck = os.path.join(d, "config.json"), os.path.join(d, "checkpoint.pt")
        if not (os.path.exists(c) and os.path.exists(ck)):
            continue
        (keep if compatible(json.load(open(c)), ref) else skip).append(d)
    return keep, skip


def spots(dealer, menu, net, n, dev):
    """Decisions of net playing itself on n deals: features, legal moves, buckets, street, strength, and context."""
    P = dealer.P
    bk, rk, cd = dealer.deal(n, cards=True)
    ctx = context(n, P, dev)
    _, rec = play(dealer, menu, [net] * P, bk, rk, record=True, dev=dev, ctx=ctx, board=board_of(cd, P, dev))
    g, i, f, legal, mybk, street, s, _a, _who = (torch.cat([r[k] for r in rec]) for k in range(9))
    stacks = ctx[0]
    seated = (stacks > 0).sum(1)[g] if stacks is not None else torch.full_like(g, P)
    at = P + 4 + 10 + 5 * (P - 1)  # the stack block (features(): log(stack / BB) / log(200))
    depth = torch.exp(f[:, at] * math.log(200))  # my stack in big blinds at the start of the hand
    facing = f[:, P + 5] > 0  # the price of calling is not zero
    cdt = torch.as_tensor(cd, device=dev)
    hole = torch.stack([cdt[g, 2 * i], cdt[g, 2 * i + 1]], 1)
    board = cdt[g, 2 * P : 2 * P + 5]
    return {"g": g, "seat": i, "feat": f, "legal": legal, "bk": mybk, "street": street, "s": s, "depth": depth,
            "facing": facing, "seated": seated, "hole": hole, "board": board}  # fmt: skip


def probs(net, S, chunk=65536):
    out = []
    with torch.no_grad():
        for k in range(0, len(S["feat"]), chunk):
            sl = slice(k, k + chunk)
            lg, _ = net(S["feat"][sl], S["bk"][sl], S["street"][sl], S["s"][sl])
            out.append(F.softmax(masked(lg, S["legal"][sl]), 1))
    return torch.cat(out)


def label(menu, slot):
    if slot == 0:
        return "fold"
    if slot == 1:
        return "check/call"
    if slot == menu.A - 1:
        return "all-in"
    return f"bet/raise {slot - 1}"


def groups(S):
    depth = S["depth"]
    d = torch.where(depth < 12, 0, torch.where(depth < 18, 1, torch.where(depth < 25, 2, 3)))
    seated = S["seated"]
    sz = torch.where(seated <= 2, 0, torch.where(seated <= 5, 1, torch.where(seated <= 8, 2, 3)))
    return {
        "street": (S["street"], STREETS),
        "position": (S["seat"], POS),
        "stack": (d, ["under 12bb", "12-18bb", "18-25bb", "25bb+"]),
        "facing a bet": (S["facing"].long(), ["no", "yes"]),
        "seated": (sz, ["2", "3-5", "6-8", "9"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="blueprint/nm25")
    ap.add_argument("--runs", default="", help="comma-separated runs (default: every compatible blueprint/nm*)")
    ap.add_argument("--deals", type=int, default=4000, help="deals of the reference playing itself")
    ap.add_argument("--top", type=int, default=25, help="spots listed where the versions disagree most")
    ap.add_argument("--out", required=True)
    ap.add_argument("--players", type=int, default=9)
    ap.add_argument("--menu", default="mtt")
    ap.add_argument("--abs", default="blueprint/out")
    ap.add_argument("--lib", default="blueprint/libbp7.so")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dealer, menu, B = setup(a, dev)
    runs, skipped = versions(a.reference, a.runs)
    if a.reference not in runs:
        runs = [a.reference] + runs
    nets = {}
    for r in runs:
        nets[os.path.basename(r)], _ = load(r, B, dev)
    ref_name = os.path.basename(a.reference)
    ref_net, _ = load(a.reference, B, dev)  # last: the spots are drawn in the reference's game
    assert not FEAT["texture"], "texture networks need their own spots"
    S = spots(dealer, menu, ref_net, a.deals, dev)
    n = len(S["feat"])
    names = list(nets)
    Pr = torch.stack([probs(nets[k], S) for k in names])  # [V, n, A]
    V = len(names)
    tv = torch.zeros(n, device=dev)
    split = torch.zeros(n, device=dev)
    top = Pr.argmax(2)  # [V, n]
    pairs = 0
    for x in range(V):
        for y in range(x + 1, V):
            tv += 0.5 * (Pr[x] - Pr[y]).abs().sum(1)
            split += (top[x] != top[y]).float()
            pairs += 1
    tv, split = tv / max(pairs, 1), split / max(pairs, 1)
    r0 = names.index(ref_name)
    vs_ref = {k: round(float((0.5 * (Pr[v] - Pr[r0]).abs().sum(1)).mean()), 4) for v, k in enumerate(names)}
    split_ref = {k: round(float((top[v] != top[r0]).float().mean()), 4) for v, k in enumerate(names)}
    res = {
        "reference": a.reference, "versions": names, "skipped": skipped, "spots": n, "deals": a.deals,
        "overall": {"disagreement": round(float(tv.mean()), 4), "split": round(float(split.mean()), 4)},
        "vs_reference": {"distance": vs_ref, "different_top_action": split_ref},
        "groups": {},
    }  # fmt: skip
    for gname, (key, labels) in groups(S).items():
        rows = []
        for v, lab in enumerate(labels):
            m = key == v
            if int(m.sum()) < 30:
                continue
            rows.append({"group": lab, "spots": int(m.sum()), "disagreement": round(float(tv[m].mean()), 4),
                         "split": round(float(split[m].mean()), 4),
                         "vs_reference": {k: round(float((0.5 * (Pr[j][m] - Pr[r0][m]).abs().sum(1)).mean()), 4)
                                          for j, k in enumerate(names) if k != ref_name}})  # fmt: skip
        res["groups"][gname] = sorted(rows, key=lambda r: -r["disagreement"])
    worst = torch.argsort(tv, descending=True)[: a.top]
    out = []
    for k in worst.tolist():
        st = int(S["street"][k])
        shown = [0, 3, 4, 5][st]
        choice = {}
        for v, nm in enumerate(names):
            p = Pr[v, k]
            choice[nm] = {label(menu, j): round(float(p[j]), 3) for j in range(menu.A) if float(p[j]) >= 0.05}
        out.append({
            "street": STREETS[st], "position": POS[int(S["seat"][k])], "stack_bb": round(float(S["depth"][k]), 1),
            "facing_a_bet": bool(S["facing"][k]), "seated": int(S["seated"][k]),
            "hand": class_name(int(S["hole"][k, 0]), int(S["hole"][k, 1])),
            "cards": " ".join(card(int(c)) for c in S["hole"][k].tolist()),
            "board": " ".join(card(int(c)) for c in S["board"][k, :shown].tolist()),
            "disagreement": round(float(tv[k]), 3), "choices": choice,
        })  # fmt: skip
    res["most_disputed"] = out
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{V} versions, {n:,} spots from {a.deals:,} deals; skipped {len(skipped)} with other inputs or cards")
    print(
        f"overall: disagreement {res['overall']['disagreement']:.3f}, different top action {res['overall']['split']:.1%}"
    )
    print("distance from " + ref_name + ": " + ", ".join(f"{k} {v:.3f}" for k, v in vs_ref.items() if k != ref_name))
    for gname, rows in res["groups"].items():
        print(f"by {gname}: " + ", ".join(f"{r['group']} {r['disagreement']:.3f}" for r in rows))
    for s in out[:8]:
        ch = "; ".join(f"{k} {max(v, key=v.get)}" for k, v in s["choices"].items())
        print(
            f"  {s['disagreement']:.2f}  {s['street']} {s['position']} {s['hand']} {s['board']} {s['stack_bb']}bb: {ch}"
        )


if __name__ == "__main__":
    main()
