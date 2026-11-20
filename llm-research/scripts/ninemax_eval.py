"""Stronger yardsticks for the 9-max networks (ninemax_selfplay.py), once the rule-based opponents stop separating runs.

cross RUN [RUN ...]
    Every trained network against every other: 1-vs-8 duplicate matches (the hero in each of the nine seats in turn,
    eight copies of the villain in the others, the same deals), plus each against the rule-based tight-aggressive
    player for reference. Prints a hero x villain table in bb/100 and writes it to --out as JSON.

exploit RUN
    How exploitable a network is at a full ring: a learner starts as a copy of RUN's network and trains for
    --minutes against eight frozen copies of it (no KL pull, no magnet: a pure best response by policy gradient),
    learning only from its own seat. Every --eval-minutes the learner plays RUN 1-vs-8 by duplicate; its winnings
    are a lower bound on RUN's exploitability, the 9-max counterpart of LBR. It starts at 0 by symmetry; a network
    that is hard to exploit keeps the learner near 0.

Networks are the exported moving averages ("avg" in RUN/checkpoint.pt), as in the trainer's own evaluations.

Usage (lab, GPU venv, from llm-research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_eval.py cross blueprint/nm2 blueprint/nm3 --out X.json
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_eval.py exploit blueprint/nm3 --minutes 120 --out DIR
"""

import argparse
import copy
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ninemax_engine import BB, Dealer, Menu
from ninemax_selfplay import FEAT, NineNet, board_of, duplicate, masked, play, seats_used, set_mode


def setup(args, dev):
    B = int(open(os.path.join(args.abs, "meta.txt")).read().split()[1])
    dealer = Dealer(args.lib, args.abs, args.players, B, dev, seed=args.seed)
    bk0, _, _ = dealer.raw(1 << 16)  # the rule-based player's pre-flop thresholds, as in the trainer
    pre_s = dealer.pre_strength[torch.tensor(bk0[:, :, 0].ravel(), device=dev)]
    dealer.pre_q = (torch.quantile(pre_s, 0.88).item(), torch.quantile(pre_s, 0.65).item())
    return dealer, Menu(args.menu, dev), B


def load(run, B, dev, part="avg"):
    """RUN is a run directory (its checkpoint.pt) or a kept snapshot RUN/avg_<hours>h.pt (--keep-hours)."""
    ck = run if run.endswith(".pt") else os.path.join(run, "checkpoint.pt")
    run = os.path.dirname(ck)
    cfg = json.load(open(os.path.join(run, "config.json")))
    set_mode(
        cfg.get("stacks", "100"),
        cfg.get("tourney", False),
        cfg.get("depths"),
        not cfg.get("no_open_limp", False),
        cfg.get("ante", 0.0),
        cfg.get("short", 0.0),
        cfg.get("seats"),
        cfg.get("bounty", 0.0),
        cfg.get("texture", False),
    )  # the runs compared must share a mode
    net = NineNet(
        cfg["nfeat"],
        cfg["actions"],
        cfg.get("buckets", B),
        arch=cfg.get("arch", "mlp"),
        belief=cfg.get("belief_weight", 0) > 0,
    ).to(dev)
    st = torch.load(ck, map_location=dev, weights_only=True)
    net.load_state_dict(st[part])
    net.texture = bool(cfg.get("texture", False))  # play() strips the texture inputs for a network trained without
    return net.requires_grad_(False).eval(), st["hands"]


def cross(args, dev):
    dealer, menu, B = setup(args, dev)
    nets = {}
    for r in args.runs:
        nets[r.rstrip("/").split("blueprint/")[-1].removesuffix(".pt")], hands = load(r, B, dev)  # nm9, nm9/avg_4h
        print(f"{r}: {hands / 1e6:.0f} M hands", flush=True)
    names = list(nets)
    res = {}
    for h in names:
        for v in names + ["tag"]:
            if v == h:
                continue
            m, lo, hi = duplicate(dealer, menu, nets[h], nets.get(v, v), args.hands, dev)
            res[f"{h} vs {v}"] = [round(float(x), 1) for x in (m, lo, hi)]
            print(f"{h:>6} vs 8 x {v:<6} {m:+8.1f} bb/100 [{lo:+.1f}, {hi:+.1f}]", flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


def exploit(args, dev):
    dealer, menu, B = setup(args, dev)
    target, hands = load(args.runs[0], B, dev)
    hero = copy.deepcopy(target).requires_grad_(True).train()
    opt = torch.optim.Adam(hero.parameters(), lr=args.lr)
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "exploit.log"), "a", buffering=1)
    log.write(f"target {args.runs[0]} ({hands / 1e6:.0f} M hands), lr {args.lr}, batch {args.batch}\n")
    P = dealer.P
    t0 = last = time.time()
    played, adv_scale, steps = 0, 1.0, 0
    m, lo, hi = duplicate(dealer, menu, hero, target, args.hands, dev)
    log.write(f"0.0 min 0 hands  learner vs 8 x target: {m:+.1f} bb/100 [{lo:+.1f}, {hi:+.1f}]\n")
    while time.time() - t0 < args.minutes * 60:
        bk, rk, cd = dealer.deal(args.batch, cards=FEAT["texture"])
        board = board_of(cd, P, dev)
        recs, Gs = [], []
        n = args.batch // args.chunks
        # the seats a player can sit in: with fewer seats (6-max cash, --seats) the early ones are always empty, and a
        # hero there has no decisions
        seats = seats_used(P)
        for c in range(args.chunks):  # the hero in one seat per chunk, the seat drawn at random
            seat = seats[int(torch.randint(len(seats), ()).item())]
            pols = [target] * P
            pols[seat] = hero
            sl = slice(c * n, (c + 1) * n)
            bsl = None if board is None else board[sl]
            u, rec = play(dealer, menu, pols, bk[sl], rk[sl], record=True, dev=dev, learner=hero, board=bsl)
            recs += rec
            Gs += [u[r[0], r[1]] / BB for r in rec]
        g, i, f, legal, mybk, street, s, a = (torch.cat([r[k] for r in recs]) for k in range(8))
        G = torch.cat(Gs)
        if len(G) == 0:  # no decision of the hero's this step (short tables): nothing to learn from, and the mean of an
            continue  # empty batch is NaN, which would poison adv_scale and then every later step
        lg, v = hero(f, mybk, street, s)
        logp = F.log_softmax(masked(lg, legal), 1)
        ent = -(logp.exp() * logp).masked_fill(~legal, 0).sum(1)
        adv = G - v.detach()
        adv_scale = 0.99 * adv_scale + 0.01 * adv.abs().mean().item()
        adv = adv / max(adv_scale, 1e-3)
        lp_a = logp.gather(1, a[:, None]).squeeze(1)
        loss = (-adv * lp_a + 0.5 * (v - G) ** 2 / 25 - args.entropy * ent).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(hero.parameters(), 1.0)
        opt.step()
        played += n * args.chunks
        steps += 1
        now = time.time()
        if now - last >= args.eval_minutes * 60 or now - t0 >= args.minutes * 60:
            hero.eval()
            m, lo, hi = duplicate(dealer, menu, hero, target, args.hands, dev)
            hero.train()
            line = (f"{(now - t0) / 60:.1f} min {played} hands {steps} steps entropy {ent.mean().item():.3f}  "
                    f"learner vs 8 x target: {m:+.1f} bb/100 [{lo:+.1f}, {hi:+.1f}]")  # fmt: skip
            log.write(line + "\n")
            print(line, flush=True)
            last = time.time()
    # the exploiter itself, for leak maps (ninemax_leaks.py): where it wins is where the target leaks
    torch.save(
        {"avg": hero.state_dict(), "target": args.runs[0], "target_hands": hands},
        os.path.join(args.out, "exploiter.pt"),
    )
    log.write("stopped\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("cross", "exploit"))
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--players", type=int, default=9)
    ap.add_argument("--menu", default="medium")
    ap.add_argument("--abs", default="blueprint/out")
    ap.add_argument("--lib", default="blueprint/libbp5.so")
    ap.add_argument("--seed", type=int, default=7, help="deal seed (different from the trainers')")
    ap.add_argument("--hands", type=int, default=20000, help="deals per duplicate match (x 9 seatings)")
    ap.add_argument("--minutes", type=float, default=120, help="exploit: training time")
    ap.add_argument("--eval-minutes", type=float, default=15)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--chunks", type=int, default=3, help="exploit: hero seats drawn per batch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.003)
    args = ap.parse_args()
    if args.mode == "exploit" and (len(args.runs) != 1 or not args.out):
        ap.error("exploit takes one RUN and --out DIR")
    torch.manual_seed(args.seed)
    (cross if args.mode == "cross" else exploit)(args, "cuda")


if __name__ == "__main__":
    main()
