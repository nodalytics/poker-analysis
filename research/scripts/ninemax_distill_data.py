"""Distillation data: positions from a network's own self-play, each labelled by search with every legal move's value
Q (big blinds of ICM equity) and the network's own policy, so a successor can be trained towards the improved policy
pi ~ pi_net exp(Q / tau) (expert iteration, as AlphaZero and ReBeL train towards their search).

Every seat is the network (on-policy positions); in each batch one seat is the hero, searched wherever Kestrel
searches (before the flop, and after it in pots of up to three players), and play continues with the network's own
move so the positions stay the network's. Rows: features, buckets, street, strength, legal, Q, pi_net.

--ensemble RUN,RUN,...: no search; every decision of every seat is labelled with the average policy of those networks
(Q = 0, so the trainer's target is that average): distilling an ensemble into one network.

Usage (lab, GPU venv): PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_distill_data.py blueprint/nm25 OUTDIR \\
    [--decisions 500000] [--samples 2048] [--batch 256] [--seed 1] [--shard 50000]
"""

import argparse
import os
import time

import numpy as np
import torch

from ninemax_engine import Menu, Tables, run_dealer, strength
from ninemax_eval import load
from ninemax_search import NBOARD, Search, net_probs, with_pay
from ninemax_selfplay import context, features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    ap.add_argument("--decisions", type=int, default=500_000)
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=256, help="tables per batch")
    ap.add_argument("--chunk", type=int, default=4, help="decisions searched at once (GPU memory)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shard", type=int, default=50_000)
    ap.add_argument("--max-live", type=int, default=3)
    ap.add_argument("--ensemble", help="comma-separated runs: label with their average policy instead of search")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)
    os.makedirs(a.out + "/.lib", exist_ok=True)  # the run's own copy of the bucket library goes here
    dealer = run_dealer(a.run, os.path.join(ROOT, "blueprint", "libbp7.so"), None, 9, 0, dev, a.seed, a.out + "/.lib")
    net, _ = load(a.run, dealer.B, dev)
    cfg_menu = __import__("json").load(open(os.path.join(a.run, "config.json"))).get("menu", "medium")
    menu = Menu(cfg_menu, dev)
    search = Search(net, dealer, menu, a.samples, "greedy", 1.0, a.chunk)
    search.trace = []
    members = [load(r, dealer.B, dev)[0] for r in a.ensemble.split(",")] if a.ensemble else []
    if members:  # load() sets the game from each run's config: the positions are this run's game, so it goes last
        net, _ = load(a.run, dealer.B, dev)
    P = dealer.P
    rows, shard, total, t0 = [], 0, 0, time.time()

    def flush():
        nonlocal rows, shard
        if not rows:
            return
        cat = {k: torch.cat([r[k] for r in rows]).cpu() for k in rows[0]}
        torch.save(cat, os.path.join(a.out, f"shard_{a.seed}_{shard:03d}.pt"))
        shard, rows = shard + 1, []

    while total < a.decisions:
        n = a.batch
        bk, rk, cd = dealer.deal(n, cards=True)
        stacks, pay = context(n, P, dev)
        T = with_pay(Tables(n, P, menu, dev, stacks=stacks), pay)
        h = int(torch.randint(P, ()).item())  # this batch's hero seat
        hist = [[] for _ in range(n)]
        with torch.no_grad():
            while not bool(T.done.all()):
                g = (~T.done).nonzero().squeeze(1)
                i = T.actor[g]
                probs, _ = net_probs(net, dealer, T, g, bk)
                if members:  # every decision, labelled with the ensemble's average policy
                    feat, legal = features(T, g, i)
                    st = strength(dealer, bk[g], T.street[g]).gather(1, i[:, None]).squeeze(1)
                    avg_p = sum(net_probs(m, dealer, T, g, bk)[0] for m in members) / len(members)
                    rows.append({"feat": feat.half(), "bk": bk[g, i].short(), "street": T.street[g].byte(),
                                 "strength": st.half(), "legal": legal, "q": torch.zeros_like(avg_p),
                                 "pnet": avg_p.half()})  # fmt: skip
                    total += len(g)
                    if sum(len(r["q"]) for r in rows) >= a.shard:
                        flush()
                    a_ = torch.multinomial(probs, 1).squeeze(1)
                    for t, s, x in zip(g.tolist(), i.tolist(), a_.tolist()):
                        hist[t].append((s, x))
                    T.step(g, a_)
                    continue
                small = ((~T.folded[g]).sum(1) <= a.max_live) | (T.street[g] == 0)
                el = (i == h) & small
                if el.any():
                    ge, ie = g[el], i[el]
                    feat, legal = features(T, ge, ie)
                    st = strength(dealer, bk[ge], T.street[ge]).gather(1, ie[:, None]).squeeze(1)
                    gj = ge.tolist()
                    fx = np.full((len(gj), 2 * P + 5), -1, np.int32)
                    for k, t in enumerate(gj):
                        fx[k, 2 * h : 2 * h + 2] = cd[t, 2 * h : 2 * h + 2]
                        nb = NBOARD[int(T.street[t])]
                        fx[k, 2 * P : 2 * P + nb] = cd[t, 2 * P : 2 * P + nb]
                    sel = lambda x: None if x is None else x[ge]  # noqa: E731
                    search.trace.clear()
                    search.decide(sel(stacks), sel(pay), [hist[t] for t in gj], h, fx)
                    Q = torch.cat([q for q, _ in search.trace])
                    pn = torch.cat([p for _, p in search.trace])
                    rows.append({"feat": feat.half(), "bk": bk[ge, ie].short(), "street": T.street[ge].byte(),
                                 "strength": st.half(), "legal": legal, "q": Q.float(), "pnet": pn.half()})  # fmt: skip
                    total += len(gj)
                    if sum(len(r["q"]) for r in rows) >= a.shard:
                        flush()
                a_ = torch.multinomial(probs, 1).squeeze(1)  # the network's own move: on-policy positions
                for t, s, x in zip(g.tolist(), i.tolist(), a_.tolist()):
                    hist[t].append((s, x))
                T.step(g, a_)
        rate = total / (time.time() - t0)
        print(
            f"{total} decisions, {rate:.1f}/s, {(a.decisions - total) / max(rate, 1e-9) / 3600:.1f} h left", flush=True
        )
    flush()


if __name__ == "__main__":
    main()
