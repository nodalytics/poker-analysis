"""Does nm25 fold too much to pre-flop aggression, and does nm30's pre-flop (with nm25 after the flop) fix it?

A player's report (2026-11-21): bluffing nm25 with 3x raises and all-ins pre-flop works, it folds almost every time;
nm30 (the PKO network) is much harder to play against. The idea: nm30 pre-flop, nm25 from the flop on.

Everything is played in nm25's game (9-max tournament, ICM, 10-25bb, no bounties), by duplicate (ninemax_selfplay):

  bluffers   one scripted seat against eight copies of a defender (nm25, or a hybrid):
               jam-any    all-in pre-flop with any two cards, every time it acts
               jam-40     all-in with the top 40% of hands, else plays nm25
               raise-any  the biggest raise size (3x) with any two cards pre-flop, nm25 after the flop
             what a bluffer wins against each defender is how exploitable that defender's pre-flop is, in prize
             equity (ICM, what the tournament pays) and in chips (what a player at the table sees)
  head-up    hybrid vs 8 x nm25 and nm25 vs 8 x hybrid: does the hybrid give up anything against good players

nm30 reads nine bounty inputs nm25 does not have (its own bounty and the others', after the payout ladder). The
hybrid feeds them two ways: zero (no bounties, the honest game, but never seen in training) and pko (the payouts
halved and an even bounty for everyone, the game nm30 knows, which makes it call as if knockouts paid).

Also prints the defenders' pre-flop fold rate facing an all-in, with a jam-any bluffer at the table and without.

Usage (lab, from llm-research):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 PYTHONPATH=scripts .venv-gpu/bin/python \
      scripts/ninemax_hybrid.py --out results/hybrid/nm30pre-nm25post.json
"""

import argparse
import json
import os

import torch
import torch.nn as nn

from ninemax_engine import BB
from ninemax_eval import load, setup
from ninemax_selfplay import FEAT, board_of, context, duplicate, play


def bounty_at(P):
    """Where nm30's bounty inputs go in nm25's features: after the seat, street, the ten scalars, the five per-seat
    blocks, the stack depths (1 + P-1 + 1) and the payout ladder (P)."""
    return P + 4 + 10 + 5 * (P - 1) + (1 + (P - 1) + 1) + P


class Hybrid(nn.Module):
    """pre's policy before the flop (fed bounty inputs as `fill` says), post's after."""

    def __init__(self, pre, post, P, fill):
        super().__init__()
        self.pre, self.post, self.P, self.fill = pre, post, P, fill
        self.at = bounty_at(P)

    def forward(self, feat, bk, street, st):
        lg, v = self.post(feat, bk, street, st)
        m = street == 0
        if not bool(m.any()):
            return lg, v
        f = feat[m]
        P, at = self.P, self.at
        reg = f[:, at - P : at]
        if self.fill == "zero":
            b = torch.zeros(len(f), P, device=f.device)
        else:  # pko: half the money in bounties, an even share each (bty * P per seat, as features() scales it)
            reg = reg * 0.5
            b = torch.full((len(f), P), 0.5, device=f.device)
        f30 = torch.cat([f[:, : at - P], reg, b, f[:, at:]], 1)
        lg2, _ = self.pre(f30, bk[m], street[m], st[m])
        lg = lg.clone()
        lg[m] = lg2
        return lg, v


class Bluffer(nn.Module):
    """base's policy, except pre-flop it pushes one action (slot) with every hand at least as strong as thr."""

    def __init__(self, base, slot, thr=0.0):
        super().__init__()
        self.base, self.slot, self.thr = base, slot, thr

    def forward(self, feat, bk, street, st):
        lg, v = self.base(feat, bk, street, st)
        m = (street == 0) & (st >= self.thr)
        lg = lg.clone()
        lg[m, self.slot] += 50.0  # where the slot is illegal, masked() removes it and base's choice stands
        return lg, v


def fold_to_allin(dealer, menu, defender, bluffer, n, dev):
    """The defenders' pre-flop fold rate facing an all-in (calling is the only alternative), with bluffer in one seat
    (rotated) and, for reference, at a table of defenders only."""
    P = dealer.P
    out = {}
    for name, hero in (("with jam-any at the table", bluffer), ("defenders only", defender)):
        folds = faced = 0
        for seat in range(P):
            bk, rk, cd = dealer.deal(n, cards=FEAT["texture"])
            pols = [defender] * P
            pols[seat] = hero
            _, rec = play(dealer, menu, pols, bk, rk, record=True, dev=dev, learner=defender, ctx=context(n, P, dev),
                          board=board_of(cd, P, dev))  # fmt: skip
            for g, i, f, legal, mybk, street, s, a, who in rec:
                # facing a bet that only calling or folding answers: no raise size and no all-in of one's own left
                facing = (street == 0) & legal[:, 1] & ~legal[:, 2:].any(1) & legal[:, 0]
                faced += int(facing.sum())
                folds += int((facing & (a == 0)).sum())
        out[name] = round(folds / max(faced, 1), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default="blueprint/nm30")
    ap.add_argument("--post", default="blueprint/nm25")
    ap.add_argument("--hands", type=int, default=20000, help="deals per duplicate match (x 9 seatings)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--players", type=int, default=9)
    ap.add_argument("--menu", default="mtt")
    ap.add_argument("--abs", default="blueprint/out")
    ap.add_argument("--lib", default="blueprint/libbp5.so")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--early", type=float, default=0.0, help="share of early-stage (winner-take-all) tables")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    dealer, menu, B = setup(args, dev)
    pre, _ = load(args.pre, B, dev)
    post, _ = load(args.post, B, dev)  # loaded last: its game (no bounties) is the one played
    assert not FEAT["bounty"] and FEAT["tourney"], "the post-flop network must be nm25's game"
    FEAT["early"] = args.early
    P = dealer.P
    bk0, _, _ = dealer.raw(1 << 16)
    pre_s = dealer.pre_strength[torch.tensor(bk0[:, :, 0].ravel(), device=dev)]
    top40 = torch.quantile(pre_s, 0.60).item()
    A, S = menu.A, menu.S
    allin, big = A - 1, 2 + S - 1
    bluffers = {
        "jam-any": Bluffer(post, allin),
        "jam-40": Bluffer(post, allin, top40),
        "raise-any": Bluffer(post, big),
    }
    defenders = {"nm25": post, "hybrid-zero": Hybrid(pre, post, P, "zero"), "hybrid-pko": Hybrid(pre, post, P, "pko")}
    res = {"hands": args.hands, "pre": args.pre, "post": args.post, "head_up": {}, "fold_to_allin": {}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def save():
        json.dump(res, open(args.out, "w"), indent=1)

    for unit in ("equity", "chips"):  # prize equity (ICM, what a tournament pays) and the chips a player sees
        res["bluffers" if unit == "equity" else "bluffers_chips"] = {}
        for bn, b in bluffers.items():
            for dn, d in defenders.items():
                m, lo, hi = duplicate(dealer, menu, b, d, args.hands, dev, chips=unit == "chips")
                res["bluffers" if unit == "equity" else "bluffers_chips"][f"{bn} vs 8 x {dn}"] = [
                    round(float(x), 1) for x in (m, lo, hi)
                ]
                print(f"{unit:>6} {bn:>9} vs 8 x {dn:<12} {m:+8.1f} bb/100 [{lo:+.1f}, {hi:+.1f}]", flush=True)
                save()
    for dn in ("hybrid-zero", "hybrid-pko"):
        d = defenders[dn]
        for h, v, key in ((d, post, f"{dn} vs 8 x nm25"), (post, d, f"nm25 vs 8 x {dn}")):
            m, lo, hi = duplicate(dealer, menu, h, v, args.hands, dev)
            res["head_up"][key] = [round(float(x), 2) for x in (m, lo, hi)]
            print(f"{key:<28} {m:+8.2f} bb/100 [{lo:+.2f}, {hi:+.2f}]", flush=True)
            save()
    for dn, d in defenders.items():
        res["fold_to_allin"][dn] = fold_to_allin(dealer, menu, d, Bluffer(d, allin), args.hands // 4, dev)
        print(f"fold to an all-in pre-flop, {dn}: {res['fold_to_allin'][dn]}", flush=True)
        save()
    print("done", flush=True)


if __name__ == "__main__":
    _ = BB
    main()
