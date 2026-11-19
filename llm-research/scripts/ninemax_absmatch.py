"""Head-to-head between networks trained on different card abstractions (bucket counts per street, or how the
buckets are cut): the same deals for both, each network reading them through its own buckets.

For every pair: a 1-vs-8 duplicate match as ninemax_selfplay.duplicate (the hero in each of the nine seats in turn,
eight copies of the other in the rest, the same deals and stacks for every seating), in the runs' own game (their
config.json; they must share it). Each run's abstraction (config "abs") is loaded through its own copy of the bucket
library, because one loaded library holds one abstraction.

Usage (lab, GPU venv, from llm-research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_absmatch.py blueprint/nmA blueprint/nmB ... \\
      [--hands 20000] [--out results/ninemax/absmatch.json]
"""

import argparse
import json
import math
import os
import shutil
import tempfile

import torch

from ninemax_engine import BB, Dealer, Menu
from ninemax_eval import load
from ninemax_selfplay import board_of, context, play

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--hands", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out")
    ap.add_argument(
        "--lib", default="blueprint/libbp7.so", help="the bucket library (with bp_bucket_cards), for every run"
    )
    a = ap.parse_args()
    dev = "cuda"
    P = 9
    tmp = tempfile.mkdtemp(prefix="absmatch-")
    runs = []
    for k, r in enumerate(a.runs):
        cfg = json.load(open(os.path.join(r if not r.endswith(".pt") else os.path.dirname(r), "config.json")))
        absdir = os.path.join(ROOT, cfg.get("abs", "blueprint/out"))
        lib = os.path.join(tmp, f"libbp_{k}.so")
        shutil.copy(os.path.join(ROOT, a.lib), lib)
        dealer = Dealer(lib, absdir, P, 0, dev, seed=a.seed)
        bk0, _, _ = dealer.raw(1 << 16)
        pre_s = dealer.pre_strength[torch.tensor(bk0[:, :, 0].ravel(), device=dev)]
        dealer.pre_q = (torch.quantile(pre_s, 0.88).item(), torch.quantile(pre_s, 0.65).item())
        net, hands = load(r, dealer.B, dev)  # also sets the game from the run's config
        runs.append({"name": os.path.basename(r.rstrip("/")).removesuffix(".pt"), "net": net, "dealer": dealer,
                     "cfg": cfg, "buckets": dealer.Bs, "hands": hands})  # fmt: skip
        print(f"{r}: buckets {dealer.Bs} ({absdir}), {hands / 1e9:.2f} B hands", flush=True)
    menu = Menu(runs[0]["cfg"].get("menu", "medium"), dev)
    base = runs[0]["dealer"]
    res = {}
    for h in runs:
        for v in runs:
            if h is v:
                continue
            bk, rk, cd = base.deal(a.hands, cards=True)
            views = {}
            for x in (h, v):
                xb, xr = x["dealer"].buckets_for(cd)
                assert torch.equal(xr, rk), "the abstractions disagree on showdown ranks"
                views[id(x["net"])] = (xb, x["dealer"])
            ctx = context(a.hands, P, dev)
            per = torch.zeros(a.hands, device=dev)
            for seat in range(P):
                pols = [v["net"]] * P
                pols[seat] = h["net"]
                u, _ = play(base, menu, pols, bk, rk, dev=dev, ctx=ctx, views=views, board=board_of(cd, P, dev))
                per += u[:, seat] / BB
            x = (per / P).cpu().numpy()
            m, se = x.mean() * 100, x.std() / math.sqrt(len(x)) * 100
            res[f"{h['name']} vs {v['name']}"] = [
                round(float(m), 4),
                round(float(m - 1.96 * se), 4),
                round(float(m + 1.96 * se), 4),
            ]  # fmt: skip  (4 places: tests use them)
            print(f"{h['name']:>8} {str(h['buckets']):>18} vs 8 x {v['name']:<8} {m:+7.1f} bb/100 "
                  f"[{m - 1.96 * se:+.1f}, {m + 1.96 * se:+.1f}]", flush=True)  # fmt: skip
    shutil.rmtree(tmp, ignore_errors=True)
    if a.out:
        json.dump({"runs": {r["name"]: {"buckets": r["buckets"], "hands": r["hands"]} for r in runs},
                   "hands": a.hands, "results": res}, open(a.out, "w"), indent=1)  # fmt: skip


if __name__ == "__main__":
    main()
