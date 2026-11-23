"""Does a 9-max network lose to pre-flop aggression, and at which stage? (2026-11-23: a tester's report that nm25
folds to almost all pre-flop aggression.)

One scripted aggressor seat against eight copies of each defender, by duplicate (every seat, the same deals):
  jam-any    all-in pre-flop with any two cards, every time it acts
  jam-40     all-in with the top 40% of hands, else plays like the defender
  raise-any  the biggest raise (3x) with any two cards pre-flop, which also re-raises every open (a 3-bet with any
             two cards); after the flop it plays like the defender
Scored in prize equity (what a tournament pays, ICM) and in chips (what a player at the table sees), at two stages:
  icm    the defender's own tables (final table, bubble, in the money; 10-25bb)
  early  winner-take-all early-stage tables, where prize equity is the chip share (no ICM pressure)
An aggressor winning prize equity at a stage is a real leak there; winning chips but losing equity is the defender
folding correctly under ICM. Also prints each defender's pre-flop fold rate facing an all-in.

Usage (lab, from research):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=scripts .venv-gpu/bin/python \
      scripts/ninemax_aggressor_check.py --defenders nm25,nm43,nm44 --out results/aggressors/2026-11-23.json
"""

import argparse
import json
import os

import torch

from ninemax_eval import load, setup
from ninemax_hybrid import Bluffer, fold_to_allin
from ninemax_selfplay import FEAT, duplicate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defenders", default="nm25,nm43,nm44", help="run names under blueprint/; the first sets the game")
    ap.add_argument("--hands", type=int, default=20000, help="deals per duplicate match (x 9 seatings)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--players", type=int, default=9)
    ap.add_argument("--menu", default="mtt")
    ap.add_argument("--abs", default="blueprint/out")
    ap.add_argument("--lib", default="blueprint/libbp7.so")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--depths", default=None, help='average stacks "LO,HI" in bb instead of the defender\'s own')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    dealer, menu, B = setup(args, dev)
    names = args.defenders.split(",")
    nets = {n: load(f"blueprint/{n}", B, dev)[0] for n in reversed(names)}  # the first loaded last: its game is played
    assert FEAT["tourney"] and not FEAT["bounty"], "a tournament game without bounties"
    if args.depths:  # deeper (or shallower) tables than the defenders trained on
        FEAT["depths"] = [tuple(float(x) for x in args.depths.split(","))]
    res_depths = FEAT["depths"]
    bk0, _, _ = dealer.raw(1 << 16)
    pre_s = dealer.pre_strength[torch.tensor(bk0[:, :, 0].ravel(), device=dev)]
    top40 = torch.quantile(pre_s, 0.60).item()
    allin, big = menu.A - 1, 2 + menu.S - 1
    res = {
        "hands": args.hands,
        "defenders": names,
        "seed": args.seed,
        "depths": res_depths,
        "results": {},
        "fold_to_allin": {},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for stage, early in (("icm", 0.0), ("early", 1.0)):
        FEAT["early"] = early
        for dn in names:
            d = nets[dn]
            aggressors = {
                "jam-any": Bluffer(d, allin),
                "jam-40": Bluffer(d, allin, top40),
                "raise-any": Bluffer(d, big),
            }
            for an, ag in aggressors.items():
                for unit in ("equity", "chips"):
                    m, lo, hi = duplicate(dealer, menu, ag, d, args.hands, dev, chips=unit == "chips")
                    res["results"][f"{stage} {unit} {an} vs 8 x {dn}"] = [round(float(x), 2) for x in (m, lo, hi)]
                    print(f"{stage:>5} {unit:>6} {an:>9} vs 8 x {dn:<5} {m:+8.2f} bb/100 [{lo:+.2f}, {hi:+.2f}]",
                          flush=True)  # fmt: skip
                    json.dump(res, open(args.out, "w"), indent=1)
            res["fold_to_allin"][f"{stage} {dn}"] = fold_to_allin(
                dealer, menu, d, aggressors["jam-any"], args.hands // 4, dev
            )
            print(f"{stage:>5} fold to an all-in, {dn}: {res['fold_to_allin'][f'{stage} {dn}']}", flush=True)
            json.dump(res, open(args.out, "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
