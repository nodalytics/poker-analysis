"""Where a network leaks: its 30-minute exploiter (ninemax_eval.py exploit, which saves exploiter.pt) plays in every
seat against 8 copies of it on fresh deals, and the exploiter's winnings are broken down by table cell (kind of table
x average stack x players seated, ninemax_selfplay.table_cell) and by the exploiter's position. One number says that a
network leaks; this says where. Cells are Holm-corrected (many are tested at once).

The map also gives prioritised-sampling weights for training (ninemax_selfplay.py --priority): a cell where the
exploiter wins clearly (z > 0) is drawn up to 3x as often, w = min(3, 1 + 0.5 z); cells with under 200 deals stay at 1.

Usage (lab, GPU venv): PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_leaks.py blueprint/nm25 \\
    results/ninemax/exploit30_nm25 [--hands 40000] [--out results/ninemax/leaks_nm25.json]
"""

import argparse
import json
import math
import os

import torch

from icm import scenarios
from multitest import holm
from ninemax_engine import BB, Menu, run_dealer
from ninemax_eval import load
from ninemax_selfplay import FEAT, NineNet, board_of, play, table_cell

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINDS = ["final table", "bubble", "in the money"]
DEPTHS = ["<12bb", "12-18bb", "18-25bb", "25bb+"]
PLAYERS = ["2", "3-5", "6-8", "9"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("exploit_dir")
    ap.add_argument("--hands", type=int, default=40000)
    ap.add_argument("--out")
    a = ap.parse_args()
    dev = "cuda"
    dealer = run_dealer(a.run, os.path.join(ROOT, "blueprint", "libbp7.so"), None, 9, 0, dev, 2027, "/tmp/leaks")
    target, _ = load(a.run, dealer.B, dev)
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    menu = Menu(cfg.get("menu", "medium"), dev)
    expl = NineNet(cfg["nfeat"], cfg["actions"], cfg.get("buckets", dealer.B), arch=cfg.get("arch", "mlp")).to(dev)
    st_ = torch.load(os.path.join(a.exploit_dir, "exploiter.pt"), map_location=dev, weights_only=True)
    expl.load_state_dict({k: v for k, v in st_["avg"].items() if not k.startswith("belief.")})
    expl.eval()
    P = dealer.P
    if not FEAT["tourney"]:
        raise SystemExit("leak maps are by tournament table cell; this run is not a tournament run")
    res_cells, res_pos = {}, {}
    n_done = 0
    chunk = 10000
    sums = {}  # cell -> [deals, sum, sum of squares] of the exploiter's mean result per deal (bb)
    pos = {}  # position group -> same, per seated exploiter seat
    groups = {0: "small blind", 1: "big blind", 2: "early", 3: "early", 4: "early", 5: "middle", 6: "middle",
              7: "late", 8: "button"}  # fmt: skip
    with torch.no_grad():
        while n_done < a.hands:
            n = min(chunk, a.hands - n_done)
            bk, rk, cd = dealer.deal(n, cards=True)
            board = board_of(cd, P, dev)
            stacks, pay, kind = scenarios(n, P, dev, avg_bb=FEAT["depths"] or (10, 60), short=FEAT["short"])
            cells = table_cell(stacks, kind)
            per = torch.zeros(n, device=dev)
            seated = (stacks > 0).float()
            for seat in range(P):
                pols = [target] * P
                pols[seat] = expl
                u, _ = play(dealer, menu, pols, bk, rk, dev=dev, ctx=(stacks, pay), board=board)
                r = u[:, seat] / BB * seated[:, seat]
                per += r
                g = groups[seat]
                m = seated[:, seat] > 0
                acc = pos.setdefault(g, [0, 0.0, 0.0])
                acc[0] += int(m.sum())
                acc[1] += float(r[m].sum())
                acc[2] += float((r[m] ** 2).sum())
            per = per / seated.sum(1).clamp(min=1)
            for c in cells.unique().tolist():
                m = cells == c
                acc = sums.setdefault(int(c), [0, 0.0, 0.0])
                acc[0] += int(m.sum())
                acc[1] += float(per[m].sum())
                acc[2] += float((per[m] ** 2).sum())
            n_done += n
            print(f"{n_done} deals", flush=True)

    def stat(acc):
        n, s, q = acc
        m = s / max(n, 1)
        sd = math.sqrt(max(q / max(n, 1) - m * m, 0.0))
        se = sd / math.sqrt(max(n, 1))
        return m * 100, se * 100

    keys = sorted(sums)
    zs, ps = [], []
    for c in keys:
        m, se = stat(sums[c])
        z = m / se if se > 0 else 0.0
        zs.append(z)
        ps.append(math.erfc(abs(z) / math.sqrt(2)))
    hs = holm(ps)
    weights = {}
    for c, z, p, h in zip(keys, zs, ps, hs):
        m, se = stat(sums[c])
        kind_, depth, players = c // 16, (c // 4) % 4, c % 4
        res_cells[str(c)] = {"kind": KINDS[kind_], "depth": DEPTHS[depth], "players": PLAYERS[players],
                             "deals": sums[c][0], "bb100": round(m, 3), "ci95": [round(m - 1.96 * se, 3),
                             round(m + 1.96 * se, 3)], "z": round(z, 2), "holm": round(h, 4)}  # fmt: skip
        weights[str(c)] = round(min(3.0, 1 + 0.5 * max(z, 0.0)), 3) if sums[c][0] >= 200 else 1.0
    for g, acc in pos.items():
        m, se = stat(acc)
        res_pos[g] = {
            "seatings": acc[0],
            "bb100": round(m, 3),
            "ci95": [round(m - 1.96 * se, 3), round(m + 1.96 * se, 3)],
        }
    total = [sum(v[0] for v in sums.values()), sum(v[1] for v in sums.values()), sum(v[2] for v in sums.values())]
    m, se = stat(total)
    out = {"run": a.run, "exploiter": a.exploit_dir, "hands": a.hands, "overall": {"bb100": round(m, 3),
           "ci95": [round(m - 1.96 * se, 3), round(m + 1.96 * se, 3)]}, "cells": res_cells, "positions": res_pos,
           "weights": weights}  # fmt: skip
    print(f"\nexploiter overall {m:+.2f} bb/100 [{m - 1.96 * se:+.2f}, {m + 1.96 * se:+.2f}]")
    print("cells where it wins most (Holm over cells):")
    for c, r in sorted(res_cells.items(), key=lambda kv: -kv[1]["z"])[:10]:
        print(f"  {r['kind']:13s} {r['depth']:8s} {r['players']:4s} players: {r['bb100']:+7.2f} bb/100 "
              f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] on {r['deals']} deals, z {r['z']:+.2f}, Holm p {r['holm']:.4f}, "
              f"weight {weights[c]}")  # fmt: skip
    print("by the exploiter's position:")
    for g, r in res_pos.items():
        print(f"  {g:12s} {r['bb100']:+7.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
