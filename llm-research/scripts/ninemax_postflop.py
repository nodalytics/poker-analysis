"""Poker-sense checks after the flop for the 9-max tournament networks: the post-flop counterpart of ninemax_sense.py.

Every spot is the same heads-up pot: nine players at the same stack, everyone folds to the button, which opens 2x,
the small blind folds and the big blind calls; then a scripted line on the later streets. Many random deals; the
network's policy is read at the target decision and grouped by the hand's strength on that street, the network's
own bucket as a percentile (0 weakest, 1 strongest: equal-population cuts of EHS2 on the flop and turn, HS on the
river), so "strong" means the same thing on every board.

  c-bets            flop, the big blind checks: the button bets 40-85% overall, its strongest 15% at least half
  strong vs c-bet   flop, the big blind facing a 1/3-pot c-bet: its strongest 15% fold at most 5%
  defends c-bet     the same spot: it continues (calls or raises) 50-85% of the time (minimum defence: 75%)
  folds air         the same spot: its weakest 30% fold at least 30% (no call-everything)
  river value       checked to the button on the river: its strongest 10% bet at least 60%
  never folds nuts  river, the big blind facing a 3/4-pot bet: its strongest 2% fold at most 2%
  commits short     8bb stacks, flop, the big blind facing an all-in: its strongest 30% call at least 85%

Weak hands win pots too (added 2026-11-16): a strategy that bets only its strong hands is read and folded against,
so a sound one bluffs and defends with some weak ones.
  c-bet bluffs      flop, checked to the button: its weakest 30% bet 15-85% (not only value)
  river bluffs      river, checked to the button: 15-45% of its bets come from its weakest 30% (a 3/4-pot bet is
                    balanced near 30% bluffs; the weakest hands cannot win at showdown, so betting them is right)
  river defence     river, the big blind facing a 3/4-pot bet: continues 30-75% (minimum defence: 57%), and its
                    middle hands (strength 0.4-0.7, bluff-catchers) call at least 20%

Chip payouts (winner takes all) and the run's own ante, as ninemax_sense.py. The bands are rough references, not
exact equilibria: a WARN is a spot to look at.

Usage (lab, GPU venv, from llm-research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_postflop.py blueprint/nm17 [...] [--hands 100000] \\
      [--out results/ninemax/postflop_nm17.json]
"""

import argparse
import json
import os
import shutil
import tempfile

import torch

from icm import icm
from ninemax_engine import BB, Menu, Tables, run_dealer, strength
from ninemax_eval import load
from ninemax_pushfold_check import policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BTN, SB_, BBS = 8, 0, 1  # seats: 0 small blind, 1 big blind, 8 the button


def spot(net, dealer, menu, dev, G, stack_bb, line):
    """Deal G hands at stack_bb, play the preflop (folds to the button, 2x open, small blind folds, big blind
    calls) and then `line`, a list of (seat, slot) on the later streets; return the policy [G, A] at the next
    decision and that player's strength on the street."""
    bk, _, _ = dealer.deal(G, cards=True)
    T = Tables(G, 9, menu, dev, stacks=torch.full((G, 9), int(stack_bb * BB), device=dev))
    T.pay = torch.zeros(G, 9, device=dev)
    T.pay[:, 0] = 1.0
    T.eq0 = icm(T.stack.float(), T.pay).float()
    g = torch.arange(G, device=dev)
    full = lambda a: torch.full((G,), a, dtype=torch.long, device=dev)  # noqa: E731
    for seat, slot in [(BTN, 2), (SB_, 0), (BBS, 1), *line]:
        while int(T.actor[0]) != seat:  # before the button: everyone folds
            T.step(g, full(0))
        T.step(g, full(slot))
    pr, i = policy(net, T, g, bk, dealer)
    s = strength(dealer, bk, T.street).gather(1, i[:, None]).squeeze(1)
    return pr, s


def check(run, dealer, dev, G):
    net, hands = load(run, dealer.B, dev)
    d = os.path.dirname(run) if run.endswith(".pt") else run
    menu = Menu(json.load(open(os.path.join(d, "config.json"))).get("menu", "medium"), dev)
    A = menu.A
    bet = lambda pr: pr[:, 2:].sum(1)  # noqa: E731  (every bet, raise and all-in)
    res = []

    def add(name, value, band, detail):
        res.append({"check": name, "value": float(value), "band": band, "ok": band[0] <= value <= band[1],
                    "detail": detail})  # fmt: skip

    def share(x, m):
        return float(x[m].mean()) if bool(m.any()) else float("nan")

    # c-bets: flop, the big blind checks, the button to act
    pr, s = spot(net, dealer, menu, dev, G, 20, [(BBS, 1)])
    b = bet(pr)
    add("c-bets when checked to (flop, 20bb)", float(b.mean()), (0.40, 0.85),
        f"strongest 15% bet {share(b, s >= 0.85):.0%}, weakest 30% bet {share(b, s <= 0.30):.0%}")  # fmt: skip
    add("strongest hands bet the flop when checked to", share(b, s >= 0.85), (0.50, 1.0), "strength >= 0.85")
    add("c-bets some of its weakest hands (bluffs)", share(b, s <= 0.30), (0.15, 0.85), "strength <= 0.30: bet rate")
    # the big blind facing a 1/3-pot c-bet
    pr, s = spot(net, dealer, menu, dev, G, 20, [(BBS, 1), (BTN, 2)])
    fold = pr[:, 0]
    add("strong hands do not fold to a small c-bet", share(fold, s >= 0.85), (0.0, 0.05), "strength >= 0.85: fold rate")
    add("defends a 1/3-pot c-bet (calls or raises)", 1 - float(fold.mean()), (0.50, 0.85),
        f"raises {float(bet(pr).mean()):.0%}, calls {float(pr[:, 1].mean()):.0%}")  # fmt: skip
    add("weakest hands fold to a small c-bet", share(fold, s <= 0.30), (0.30, 1.0), "strength <= 0.30: fold rate")
    # river value: checked through to the button on the river
    line_river = [(BBS, 1), (BTN, 1), (BBS, 1), (BTN, 1), (BBS, 1)]
    pr, s = spot(net, dealer, menu, dev, G, 20, line_river)
    b = bet(pr)
    add("value-bets the river with its strongest hands", share(b, s >= 0.90), (0.60, 1.0),
        f"strength >= 0.90; all hands bet {float(b.mean()):.0%}, weakest 30% bet {share(b, s <= 0.30):.0%}")  # fmt: skip
    weak = (s <= 0.30).float()
    add("river bets include bluffs (share of its bets from its weakest 30%)", float((b * weak).sum() / b.sum()),
        (0.15, 0.45), f"weakest 30% bet {share(b, s <= 0.30):.0%}; a 3/4-pot bet is balanced near 30% bluffs")  # fmt: skip
    # the big blind facing a 3/4-pot river bet
    pr, s = spot(net, dealer, menu, dev, G, 20, [(BBS, 1), (BTN, 1), (BBS, 1), (BTN, 1), (BBS, 1), (BTN, 3)])
    fold = pr[:, 0]
    add("never folds the nuts on the river", share(fold, s >= 0.98), (0.0, 0.02),
        f"strength >= 0.98: fold rate; all hands fold {float(fold.mean()):.0%}, weakest 30% "
        f"{share(fold, s <= 0.30):.0%}")  # fmt: skip
    add("defends a 3/4-pot river bet (calls or raises)", 1 - float(fold.mean()), (0.30, 0.75),
        f"raises {float(bet(pr).mean()):.0%}, calls {float(pr[:, 1].mean()):.0%}")  # fmt: skip
    mid = (s >= 0.40) & (s <= 0.70)
    add("calls a river bet with some bluff-catchers", share(pr[:, 1], mid), (0.20, 1.0), "strength 0.4-0.7: call rate")
    # short stacks: 8bb, the big blind facing an all-in on the flop
    pr, s = spot(net, dealer, menu, dev, G, 8, [(BBS, 1), (BTN, A - 1)])
    call = pr[:, 1]
    add("commits a short stack with a strong hand (8bb, flop all-in)", share(call, s >= 0.70), (0.85, 1.0),
        f"strength >= 0.70: call rate; all hands call {float(call.mean()):.0%}")  # fmt: skip
    return hands, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--hands", type=int, default=100000)
    ap.add_argument("--out")
    ap.add_argument("--lib", default=os.path.join(ROOT, "blueprint", "libbp5.so"))
    ap.add_argument("--abs", default=None, help="the card abstraction (default: the run's own, else blueprint/out)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    tmp = tempfile.mkdtemp(prefix="postflop-")
    for run in a.runs:
        dealer = run_dealer(run, a.lib, a.abs, 9, 0, dev, 41, tmp)
        hands, res = check(run, dealer, dev, a.hands)
        tag = run.rstrip("/").split("blueprint/")[-1].removesuffix(".pt")
        out[tag] = {"hands_trained": hands, "checks": res}
        n_ok = sum(r["ok"] for r in res)
        print(f"== {tag} ({hands / 1e6:.0f} M hands): {n_ok}/{len(res)} within the reference bands", flush=True)
        for r in res:
            print(f"  {'ok  ' if r['ok'] else 'WARN'} {r['check']:58s} {r['value']:.0%}  "
                  f"(band {r['band'][0]:.0%}-{r['band'][1]:.0%})  {r['detail']}", flush=True)  # fmt: skip
    shutil.rmtree(tmp, ignore_errors=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
