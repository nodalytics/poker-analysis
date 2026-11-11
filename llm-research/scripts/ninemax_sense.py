"""Poker-sense checks for the 9-max tournament networks: the principles strong tournament players follow,
measured on the network's own policy, each against a reference band (docs/2026-11-11-mtt-strategy.md).

Each check sets up one spot (stacks, payouts, everyone before the seat folds, optionally a forced action), deals
many random hands and reads the network's action probabilities by hand class:

  premiums     first in with AA, KK, QQ, AK: how often it raises or shoves (should be about always)
  trash        first in at UTG with 72o, 83o, 94o, T2o, J3o: how often it folds (should be about always)
  open width   first in, share of hands it plays, by position, at 20bb (tight early, wide late)
  shove short  first in at 8bb: of the hands it plays, the share that are all in (open-shove territory)
  bb defence   big blind facing the button's 2x open at 20bb, the small blind folded: share it defends
               (calls or raises); the price is good, so a wide defence

Payouts are winner-take-all, so the checks are in chips (ICM is checked by ninemax_pushfold_check.py). The bands
are approximate references from common tournament charts, not exact equilibria: a WARN is a spot to look at, not
a proof of a leak.

Usage (lab, GPU venv, from llm-research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_sense.py blueprint/nm10 [blueprint/nm8 ...] \\
      [--hands 100000] [--out results/ninemax/sense_nm10.json]
"""

import argparse
import json
import os

import torch

from icm import icm
from ninemax_engine import BB, Dealer, Menu, Tables
from ninemax_eval import load
from ninemax_pushfold_check import class_name, policy, seat_of

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["UTG", "UTG+1", "MP", "LJ", "HJ", "CO", "BTN", "SB", "BB"]  # action order before the flop
PREMIUMS = ["AA", "KK", "QQ", "AKs", "AKo"]
TRASH = ["72o", "83o", "94o", "T2o", "J3o"]
# reference bands (share of all hands played first in at 20bb, 9-handed), approximate
OPEN_BAND = {"UTG": (0.10, 0.20), "MP": (0.13, 0.24), "HJ": (0.18, 0.30), "CO": (0.24, 0.38), "BTN": (0.35, 0.58),
             "SB": (0.35, 0.70)}  # fmt: skip


def spot(net, dealer, menu, dev, G, stack_bb, j, forced=()):
    """Hands dealt at stack_bb for everyone, players before action index j fold, then the forced actions (slots)
    are taken in turn; returns the policy [G, A], the class name of each hand of the player to act, and the menu."""
    n = 9
    bk, _, cd = dealer.deal(G, cards=True)
    T = Tables(G, n, menu, dev, stacks=torch.full((G, n), int(stack_bb * BB), device=dev))
    T.pay = torch.zeros(G, n, device=dev)
    T.pay[:, 0] = 1.0  # winner takes all: chip value
    T.eq0 = icm(T.stack.float(), T.pay).float()
    g = torch.arange(G, device=dev)
    target = seat_of(j, n)
    while int(T.actor[0]) != target:
        T.step(g, torch.zeros(G, dtype=torch.long, device=dev))
    for a, folds_after in forced:  # an action by the player to act, then everyone up to the next target folds
        T.step(g, torch.full((G,), a, dtype=torch.long, device=dev))
        for _ in range(folds_after):
            T.step(g, torch.zeros(G, dtype=torch.long, device=dev))
    with torch.no_grad():
        pr, i = policy(net, T, g, bk, dealer)
    seat = int(i[0])
    names = [class_name(int(a), int(b)) for a, b in zip(cd[:, 2 * seat], cd[:, 2 * seat + 1])]
    return pr, names


def by_class(pr, names, hands):
    out = {}
    for h in hands:
        m = torch.tensor([x == h for x in names], device=pr.device)
        out[h] = pr[m].mean(0) if bool(m.any()) else None
    return out


def check(run, dealer, dev, G):
    net, hands_trained = load(run, dealer.B, dev)
    d = os.path.dirname(run) if run.endswith(".pt") else run
    menu = Menu(json.load(open(os.path.join(d, "config.json"))).get("menu", "medium"), dev)
    S = menu.S
    res = []

    def add(name, value, band, detail=""):
        ok = band[0] <= value <= band[1]
        res.append({"check": name, "value": round(value, 3), "band": band, "ok": ok, "detail": detail})

    for j in (0, 6):  # premiums at UTG and on the button, 20bb
        pr, names = spot(net, dealer, menu, dev, G, 20, j)
        bc = by_class(pr, names, PREMIUMS)
        raise_rate = [float(p[2:].sum()) for p in bc.values() if p is not None]
        limp = [float(p[1]) for p in bc.values() if p is not None]
        add(f"premiums raise first in ({NAMES[j]}, 20bb)", min(raise_rate), (0.95, 1.0),
            " ".join(f"{h} {float(p[2:].sum()):.0%}" for h, p in bc.items() if p is not None) +
            f" | limp max {max(limp):.0%}")  # fmt: skip
    pr, names = spot(net, dealer, menu, dev, G, 20, 0)
    bc = by_class(pr, names, TRASH)
    folds = [float(p[0]) for p in bc.values() if p is not None]
    add("trash folds first in (UTG, 20bb)", min(folds), (0.97, 1.0),
        " ".join(f"{h} {float(p[0]):.0%}" for h, p in bc.items() if p is not None))  # fmt: skip
    for pos, j in (("UTG", 0), ("MP", 2), ("HJ", 4), ("CO", 5), ("BTN", 6), ("SB", 7)):
        pr, names = spot(net, dealer, menu, dev, G, 20, j)
        play = float(1 - pr[:, 0].mean())
        sizes = pr[:, 2 : 2 + S].mean(0).tolist()
        add(f"opens first in ({pos}, 20bb)", play, OPEN_BAND[pos],
            f"limp {float(pr[:, 1].mean()):.1%}, opens " + "/".join(f"{x:.1%}" for x in sizes) +
            f", shove {float(pr[:, S + 2].mean()):.1%}")  # fmt: skip
    for pos, j in (("UTG", 0), ("CO", 5)):
        pr, names = spot(net, dealer, menu, dev, G, 8, j)
        play = 1 - pr[:, 0]
        shove_share = float(pr[:, S + 2].sum() / play.sum().clamp(min=1e-9))
        add(f"plays all in at 8bb ({pos}), share of hands played", shove_share, (0.75, 1.0),
            f"plays {float(play.mean()):.0%} of hands")  # fmt: skip
    # big blind vs the button's smallest open (slot 2) at 20bb, small blind folds
    pr, names = spot(net, dealer, menu, dev, G, 20, 6, forced=((2, 1),))
    add("big blind defends vs button min-open (20bb)", float(1 - pr[:, 0].mean()), (0.50, 0.80),
        f"calls {float(pr[:, 1].mean()):.0%}, raises {float(pr[:, 2:].sum(1).mean()):.0%}")  # fmt: skip
    return hands_trained, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--hands", type=int, default=100000)
    ap.add_argument("--out")
    ap.add_argument("--lib", default=os.path.join(ROOT, "blueprint", "libbp5.so"))
    ap.add_argument("--abs", default=os.path.join(ROOT, "blueprint", "out"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B = int(open(os.path.join(a.abs, "meta.txt")).read().split()[1])
    dealer = Dealer(a.lib, a.abs, 9, B, dev, seed=31)
    out = {}
    for run in a.runs:
        hands, res = check(run, dealer, dev, a.hands)
        tag = run.rstrip("/").split("blueprint/")[-1].removesuffix(".pt")
        out[tag] = {"hands_trained": hands, "checks": res}
        n_ok = sum(r["ok"] for r in res)
        print(f"== {tag} ({hands / 1e6:.0f} M hands): {n_ok}/{len(res)} within the reference bands", flush=True)
        for r in res:
            print(f"  {'ok  ' if r['ok'] else 'WARN'} {r['check']:52s} {r['value']:.0%}  "
                  f"(band {r['band'][0]:.0%}-{r['band'][1]:.0%})  {r['detail']}", flush=True)  # fmt: skip
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
