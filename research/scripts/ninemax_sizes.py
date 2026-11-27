"""Which bet sizes does a 9-max network use, where, and how much do its sizes give away about its cards?

A network on its own menu plays itself (tables drawn as in training, the run's game); every decision is recorded with
its context: the street, whether the pot is unopened, the re-raise level before the flop, and heads-up or multiway
after it. Two reports per run, for comparing menus (bet sizing, plan A: `docs/2026-11-23-review-and-plan.md` section 6):

  usage  per context, the share of decisions that fold, check or call, bet each size of the menu, or go all in: the
         sizes a wider menu adds show up here, and where they are used (the new strategies).
  tells  per context, among the decisions that bet or raise, how much of the variance of the player's hand strength
         (the bucket strength the networks read, 0..1) the chosen size explains: eta squared, the between-size variance
         over the total. 0: the size says nothing about the hand; 1: the size names it. A balanced strategy still has
         some (a big bet is stronger on average); the comparison across networks is what matters, at the same contexts.
         Weighted by decisions, overall and by street.

Usage (lab, from research):
  CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_sizes.py blueprint/nm48 \
      --baseline blueprint/nm25 --deals 20000 --out results/pipeline/nm48/sizes.json
"""

import argparse
import json
import os

import torch

from ninemax_engine import Menu
from ninemax_eval import load, setup
from ninemax_selfplay import FEAT, board_of, context, play

STREETS = ["pre-flop", "flop", "turn", "river"]


def labels(menu):
    """Per context, the name of each raise slot: {context: [label or None per slot]}."""
    out = {}
    nm = lambda v, unit: None if v != v else f"{v:g}{unit}"  # noqa: E731 (nan: unused)
    out["pre-flop open"] = [nm(float(x) / 100, "bb") for x in menu.opens.cpu()]  # chips: the big blind is 100
    for lvl in range(len(menu.rer)):
        out[f"pre-flop re-raise {lvl + 1}"] = [nm(float(x), "x") for x in menu.rer[lvl].cpu()]
    for street in STREETS[1:]:
        for key, v in (("heads-up bet", menu.hb), ("heads-up raise", menu.hr), ("multiway bet", menu.mb),
                       ("multiway raise", menu.mr)):  # fmt: skip
            out[f"{street} {key}"] = [nm(float(x), " pot") for x in v.cpu()]
    return out


def contexts(feat, street, P, menu):
    """The context name of each recorded decision, from its features (seat one-hot, street one-hot, pot, odds, mine,
    left, left/pot, raises/4, pre-raises/4, live/P: the order features() writes them)."""
    raises = (feat[:, P + 9] * 4).round().long()
    live = (feat[:, P + 11] * P).round().long()
    names = []
    for k in range(len(street)):
        st, r = int(street[k]), int(raises[k])
        if st == 0:
            names.append("pre-flop open" if r == 0 else f"pre-flop re-raise {min(r, len(menu.rer))}")
        else:
            way = "heads-up" if int(live[k]) <= 2 else "multiway"
            names.append(f"{STREETS[st]} {way} {'bet' if r == 0 else 'raise'}")
    return names


def report(net, menu, dealer, deals, dev, batch=4096):
    P = dealer.P
    lab = labels(menu)
    usage, rows = {}, {}
    done = 0
    while done < deals:
        n = min(batch, deals - done)
        bk, rk, cd = dealer.deal(n, cards=FEAT["texture"])
        _, rec = play(dealer, menu, [net] * P, bk, rk, record=True, dev=dev, ctx=context(n, P, dev),
                      board=board_of(cd, P, dev) if cd is not None else None)  # fmt: skip
        for g, i, feat, legal, mybk, street, s, a, pol in rec:
            names = contexts(feat, street, P, menu)
            for k, c in enumerate(names):
                act = int(a[k])
                if act == 0:
                    what = "fold"
                elif act == 1:
                    what = "check/call"
                elif act == menu.A - 1:
                    what = "all-in"
                else:
                    what = (lab.get(c) or [None] * menu.S)[act - 2] or f"slot {act - 2}"
                u = usage.setdefault(c, {})
                u[what] = u.get(what, 0) + 1
                if act >= 2:  # a bet, a raise or an all-in: what the size says about the hand
                    rows.setdefault(c, []).append((what, float(s[k])))
        done += n
    out = {"usage": {}, "tells": {}}
    for c, u in sorted(usage.items()):
        tot = sum(u.values())
        out["usage"][c] = {
            "decisions": tot,
            **{k: round(v / tot, 4) for k, v in sorted(u.items(), key=lambda x: -x[1])},
        }
    by_street, w_all, e_all = {}, 0, 0.0
    for c, r in sorted(rows.items()):
        if len(r) < 50:
            continue
        x = torch.tensor([v for _, v in r], dtype=torch.float64)
        groups = {}
        for what, v in r:
            groups.setdefault(what, []).append(v)
        mean, tot = float(x.mean()), float(((x - x.mean()) ** 2).sum())
        between = sum(len(v) * (sum(v) / len(v) - mean) ** 2 for v in groups.values())
        eta = between / tot if tot > 0 else 0.0
        out["tells"][c] = {"bets": len(r), "sizes": len(groups), "eta2": round(eta, 4)}
        st = c.split()[0]
        b = by_street.setdefault(st, [0, 0.0])
        b[0] += len(r)
        b[1] += eta * len(r)
        w_all += len(r)
        e_all += eta * len(r)
    out["tells_by_street"] = {k: round(v[1] / v[0], 4) for k, v in by_street.items()}
    out["tells_overall"] = round(e_all / max(w_all, 1), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--baseline", default="", help="the same report for this run too (its own menu)")
    ap.add_argument("--deals", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--players", type=int, default=9)
    ap.add_argument("--menu", default="mtt", help="unused: each network plays its own menu")
    ap.add_argument("--abs", default="blueprint/out")
    ap.add_argument("--lib", default="blueprint/libbp7.so")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    dev = "cuda"
    dealer, _, B = setup(a, dev)
    res = {"deals": a.deals}
    for who, run in (("baseline", a.baseline), ("run", a.run)):  # the run last: load() leaves its game set
        if not run:
            continue
        net, _ = load(run, B, dev)
        menu = Menu(net.menu_name, dev)
        dealer.seed = a.seed  # the same deals for both
        torch.manual_seed(a.seed)
        r = report(net, menu, dealer, a.deals, dev)
        res[who] = {"run": run, "menu": net.menu_name, **r}
        print(f"{who} {run} ({net.menu_name}): tells overall {r['tells_overall']:.4f}, by street {r['tells_by_street']}",
              flush=True)  # fmt: skip
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
