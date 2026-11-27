"""Search with beliefs, measured: on the same spots, how close each search's move values come to a much larger search,
and how many guesses they effectively rest on.

1. Spots. The network plays itself (ninemax_search.match, the hero in each seat) and every spot search would take (after
   the flop, at most three players in) is recorded: stacks, payouts, the hand so far, the hero's seat and the known
   cards. The match is played by the network alone, so the spots do not depend on any search.
2. Arms. Each search values every legal move of every spot (Q, in big blinds): v0 uniform guesses (today's search),
   v1 uniform x over then resampled, v2 belief guesses x over then resampled (and over 1), all with the same number of
   rollouts N.
3. Reference. A uniform search with --ref-over x --ref-samples guesses, many times larger.

Per arm: the root mean square error of Q against the reference over the legal moves (bb), how often its best move is
the reference's, and the median of the effective guesses (of N).

Usage (lab, GPU, from research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_belief_eval.py blueprint/nm25 --belief DIR --out FILE
"""

import argparse
import json
import os
import time

import torch

from ninemax_belief import load_belief
from ninemax_engine import Dealer, Menu
from ninemax_eval import load
from ninemax_search import Search, match

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Recorder(Search):
    """Records every spot search is asked about and answers with the network's own policy (an ESS floor above N)."""

    def __init__(self, net, dealer, menu):
        super().__init__(net, dealer, menu, samples=4, min_ess=2.0)
        self.spots = []

    def decide(self, stacks, pay, hists, hero, fixed):
        for j in range(len(hists)):
            self.spots.append((None if stacks is None else stacks[j : j + 1].clone(),
                               None if pay is None else pay[j : j + 1].clone(), [list(hists[j])], hero,
                               fixed[j : j + 1].copy()))  # fmt: skip
        return super().decide(stacks, pay, hists, hero, fixed)


def values(search, spots):
    """Q [n, A] (-1e9 at illegal moves) of every spot, and the median effective guesses."""
    search.trace, search.ess = [], []
    for st, pay, h, hero, fx in spots:
        search.decide(st, pay, h, hero, fx)
    return torch.cat([q for q, _ in search.trace]), float(torch.tensor(search.ess).median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--belief", required=True)
    ap.add_argument("--hands", type=int, default=150, help="deals of the spot-collecting match (x 9 seats)")
    ap.add_argument("--spots", type=int, default=300)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--over", type=int, default=8)
    ap.add_argument("--ref-samples", type=int, default=1024)
    ap.add_argument("--ref-over", type=int, default=16)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lib", default=os.path.join(ROOT, "blueprint", "libbp6.so"))
    ap.add_argument("--abs", default=os.path.join(ROOT, "blueprint", "out"))
    a = ap.parse_args()
    dev = "cuda"
    B = int(open(os.path.join(a.abs, "meta.txt")).read().split()[1])
    net, _ = load(a.run, B, dev)
    d = os.path.dirname(a.run) if a.run.endswith(".pt") else a.run
    menu = Menu(json.load(open(os.path.join(d, "config.json"))).get("menu", "medium"), dev)
    dealer = Dealer(a.lib, a.abs, 9, B, dev, seed=a.seed)
    belief = load_belief(a.belief, dev)
    t0 = time.time()
    rec = Recorder(net, dealer, menu)
    match(net, rec, dealer, menu, a.hands, dev)
    spots = rec.spots[: a.spots]
    mk = lambda prop, over, n=a.samples: Search(net, dealer, menu, n, chunk=8, proposal=prop, over=over,  # noqa: E731
                                                belief=belief if prop == "belief" else None)  # fmt: skip
    q_ref, ess_ref = values(mk("uniform", a.ref_over, a.ref_samples), spots)
    legal = q_ref > -1e8
    arms = {"v0 uniform": mk("uniform", 1), f"v1 uniform x{a.over}": mk("uniform", a.over),
            "v2 belief x1": mk("belief", 1), f"v2 belief x{a.over}": mk("belief", a.over)}  # fmt: skip
    res = {"run": a.run, "spots": len(spots), "samples": a.samples, "reference": f"uniform x{a.ref_over}, "
           f"{a.ref_samples} rollouts", "reference_ess_median": round(ess_ref, 1), "arms": {}}  # fmt: skip
    for name, s in arms.items():
        q, ess = values(s, spots)
        err = ((q - q_ref)[legal] ** 2).mean().sqrt()
        best = (q.argmax(1) == q_ref.argmax(1)).float().mean()
        res["arms"][name] = {"q_rmse_bb": round(float(err), 4), "best_move_agrees": round(float(best), 4),
                             "ess_median_of_N": round(ess, 1)}  # fmt: skip
        print(f"{name:<18} Q error {float(err):.4f} bb, best move as reference {float(best):.3f}, "
              f"effective guesses {ess:.1f} of {a.samples}", flush=True)  # fmt: skip
    res["seconds"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
