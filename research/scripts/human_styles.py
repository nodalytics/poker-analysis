"""Opponent styles shaped like real players: player types from human hands, each played by a fitted style of nm25.

Three stages, each writing a file the next one reads:

  players DIRS... --out results/human_styles/players.json
      Every player's statistics from published hand histories (the PHH dataset's HandHQ hands, 2009, anonymised):
      VPIP, PFR, 3-bet, post-flop aggression (bets and raises per call) and fold to a bet, from a sample of each
      directory, at tables of 6 or more, for players with at least --min-hands hands. Standard library only.
  types --players results/human_styles/players.json --out results/human_styles/types.json [--k 8]
      Players grouped into k types by k-means on the five statistics (standardised; weighted by hands), each type
      with its centre, its share of the players and of the hands, and a name from its centre (tight or loose,
      passive or aggressive, calling station when it rarely folds to a bet).
  fit --types results/human_styles/types.json --out results/human_styles/styles.json  (lab, GPU)
      For each type, the style (pipeline_opponents.Style: nm25 with its tendencies pushed) whose own statistics,
      measured in nm25's game with it in one seat and nm25 in the others, come closest to the type's: random search
      over the style's seven settings, then local refinement around the best.

The pipeline's styles check plays the fitted styles beside the generated ones ("human": true). Caveat: the humans
played cash games at about 100bb, the networks play 10-25bb tournaments; the statistics are matched as they are,
so a type is "plays 35% of hands and folds to a bet 40% of the time", whatever the stacks.

Usage (lab, from research):
  python3 scripts/human_styles.py players ~/datasets/phh-dataset/data/handhq/*NLH* --out results/human_styles/players.json
  python3 scripts/human_styles.py types --players results/human_styles/players.json --out results/human_styles/types.json
  CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=scripts .venv-gpu/bin/python scripts/human_styles.py fit \
      --types results/human_styles/types.json --out results/human_styles/styles.json
"""

import argparse
import collections
import json
import math
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STATS = ("vpip", "pfr", "three_bet", "aggression", "fold_to_bet")


# ---------------------------------------------------------------------------------------------------- 1: players


def tally(args):
    """One directory's per-player counts (a worker; standard library only)."""
    d, limit, min_players = args
    from hand_histories import hands, replay

    st = collections.defaultdict(collections.Counter)
    tag = os.path.basename(d.rstrip("/"))
    for _f, h in hands(d, limit):
        if "actions" not in h or "starting_stacks" not in h or len(h["starting_stacks"]) < min_players:
            continue
        names = [f"{tag}|{x}" for x in (h.get("players") or [f"p{i + 1}" for i in range(len(h["starting_stacks"]))])]
        h = dict(h, players=names)
        vp, pr, three = set(), set(), set()
        for r in replay(h):
            c, who = st[r["player"]], r["player"]
            pre = r["street"] == "preflop"
            if pre and r["action"] in ("cc", "cbr") and not (r["action"] == "cc" and r["to_call"] == 0):
                vp.add(who)
            if pre and r["action"] == "cbr":
                pr.add(who)
                if r["raises_before"] == 1:
                    three.add(who)
            if pre and r["raises_before"] == 1 and r["to_call"] > 0:
                c["three_bet_chances"] += 1
            if not pre and r["action"] == "cbr":
                c["post_raises"] += 1
            if not pre and r["action"] == "cc" and r["to_call"] > 0:
                c["post_calls"] += 1
            if not pre and r["to_call"] > 0:
                c["faced_bet"] += 1
                c["folded_to_bet"] += r["action"] == "f"
        for who in names:
            st[who]["hands"] += 1
            st[who]["vpip"] += who in vp
            st[who]["pfr"] += who in pr
            st[who]["three_bets"] += who in three
    return dict(st)


def rates(c):
    r = lambda a, b: a / b if b else None  # noqa: E731
    return {
        "hands": c["hands"],
        "vpip": r(c["vpip"], c["hands"]),
        "pfr": r(c["pfr"], c["hands"]),
        "three_bet": r(c["three_bets"], c["three_bet_chances"]),
        "aggression": r(c["post_raises"], c["post_calls"]),
        "fold_to_bet": r(c["folded_to_bet"], c["faced_bet"]),
    }


def cmd_players(a):
    jobs = [(d, a.per_dir, a.min_players) for d in a.dirs]
    with Pool(min(len(jobs), a.workers)) as pool:
        parts = pool.map(tally, jobs)
    players = {}
    for st in parts:
        for who, c in st.items():
            if c["hands"] >= a.min_hands and c["faced_bet"] >= 30 and c["post_calls"] >= 10:
                players[who] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rates(c).items()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"source": [os.path.basename(d.rstrip("/")) for d in a.dirs], "per_dir": a.per_dir,
               "min_hands": a.min_hands, "players": players}, open(a.out, "w"), indent=1)  # fmt: skip
    print(f"{len(players):,} players with {a.min_hands}+ hands from {len(a.dirs)} directories -> {a.out}")


# ---------------------------------------------------------------------------------------------------- 2: types


def vec(p):
    """The five statistics on comparable scales (aggression on a log scale: it is a ratio)."""
    return [p["vpip"], p["pfr"], p["three_bet"] or 0.0, math.log(max(p["aggression"], 0.05)), p["fold_to_bet"]]


def name_of(c):
    loose, aggr = c["vpip"] >= 0.30, c["pfr"] / max(c["vpip"], 1e-6) >= 0.55
    if c["fold_to_bet"] < 0.35 and c["vpip"] >= 0.30:
        base = "loose calling station"
    elif c["vpip"] < 0.18:
        base = "nit" if not aggr else "tight-aggressive (nitty)"
    else:
        base = ("loose" if loose else "tight") + "-" + ("aggressive" if aggr else "passive")
    if c["aggression"] >= 3.0:
        base += ", hyper-aggressive after the flop"
    return base


def cmd_types(a):
    import numpy as np

    P = json.load(open(a.players))["players"]
    rows = [p for p in P.values() if all(p[k] is not None for k in ("vpip", "pfr", "aggression", "fold_to_bet"))]
    X = np.array([vec(p) for p in rows])
    w = np.array([p["hands"] for p in rows], float)
    mu, sd = np.average(X, 0, w), np.sqrt(np.average((X - np.average(X, 0, w)) ** 2, 0, w))
    Z = (X - mu) / sd
    rng = np.random.default_rng(a.seed)
    best = None
    for _ in range(a.restarts):  # k-means++ starts, weighted Lloyd iterations; keep the lowest inertia
        cen = [Z[rng.choice(len(Z), p=w / w.sum())]]
        for _k in range(1, a.k):
            d2 = np.min([((Z - c) ** 2).sum(1) for c in cen], 0)
            cen.append(Z[rng.choice(len(Z), p=(w * d2) / (w * d2).sum())])
        cen = np.array(cen)
        for _it in range(100):
            lab = np.argmin(((Z[:, None, :] - cen[None]) ** 2).sum(2), 1)
            new = np.array(
                [np.average(Z[lab == j], 0, w[lab == j]) if (lab == j).any() else cen[j] for j in range(a.k)]
            )
            if np.allclose(new, cen):
                break
            cen = new
        inertia = float((w * ((Z - cen[lab]) ** 2).sum(1)).sum())
        if best is None or inertia < best[0]:
            best = (inertia, cen, lab)
    _, cen, lab = best
    types = []
    for j in range(a.k):
        m = lab == j
        c = cen[j] * sd + mu
        centre = {"vpip": round(float(c[0]), 3), "pfr": round(float(c[1]), 3), "three_bet": round(float(c[2]), 3),
                  "aggression": round(float(math.exp(c[3])), 2), "fold_to_bet": round(float(c[4]), 3)}  # fmt: skip
        types.append({"centre": centre, "players": int(m.sum()), "share_players": round(float(m.mean()), 3),
                      "share_hands": round(float(w[m].sum() / w.sum()), 3), "name": name_of(centre)})  # fmt: skip
    types.sort(key=lambda t: -t["share_hands"])
    for i, t in enumerate(types):
        t["id"] = f"human{i + 1}"
    spread = {k: round(float(s), 4) for k, s in zip(STATS, sd)}
    json.dump({"players": len(rows), "k": a.k, "spread": spread, "types": types}, open(a.out, "w"), indent=1)
    print(f"{len(rows):,} players into {a.k} types -> {a.out}")
    for t in types:
        c = t["centre"]
        print(f"  {t['id']:8} {t['share_hands']:5.1%} of hands  VPIP {c['vpip']:.0%} PFR {c['pfr']:.0%} "
              f"3-bet {c['three_bet']:.0%} aggr {c['aggression']:.1f} fold-to-bet {c['fold_to_bet']:.0%}  {t['name']}")  # fmt: skip


# ---------------------------------------------------------------------------------------------------- 3: fit


def measure(dealer, menu, base, style, n, dev, seats=(2, 5, 8, 1)):
    """A style's own statistics in base's game: it in one seat (each of seats in turn), base in the others."""
    import torch

    from ninemax_selfplay import board_of, context, play

    P = dealer.P
    c = collections.Counter()
    for seat in seats:
        bk, rk, cd = dealer.deal(n, cards=False)
        ctx = context(n, P, dev)
        pols = [base] * P
        pols[seat] = style
        _, rec = play(
            dealer, menu, pols, bk, rk, record=True, dev=dev, learner=style, ctx=ctx, board=board_of(cd, P, dev)
        )
        if not rec:
            continue
        g, _i, f, _legal, _bk, street, _s, act, _w = (torch.cat([r[k] for r in rec]) for k in range(9))
        to_call = f[:, P + 5] > 0
        raises = (f[:, P + 9] * 4).round()
        pre = street == 0
        aggr = act >= 2
        call = (act == 1) & to_call
        seated = ctx[0][:, seat] > 0 if ctx[0] is not None else torch.ones(n, dtype=torch.bool, device=dev)
        c["hands"] += int(seated.sum())
        vp = torch.zeros(n, dtype=torch.bool, device=dev)
        vp[g[pre & (aggr | call)]] = True
        pr = torch.zeros(n, dtype=torch.bool, device=dev)
        pr[g[pre & aggr]] = True
        c["vpip"] += int(vp.sum())
        c["pfr"] += int(pr.sum())
        ch = pre & (raises == 1) & to_call
        c["three_bet_chances"] += int(ch.sum())
        c["three_bets"] += int((ch & aggr).sum())
        c["post_raises"] += int((~pre & aggr).sum())
        c["post_calls"] += int((~pre & call).sum())
        c["faced_bet"] += int((~pre & to_call).sum())
        c["folded_to_bet"] += int((~pre & to_call & (act == 0)).sum())
    return rates(c)


def loss(got, want, spread):
    out = 0.0
    for k in STATS:
        if got[k] is None or want[k] is None:
            out += 4.0
            continue
        g, w = (math.log(max(got[k], 0.05)), math.log(max(want[k], 0.05))) if k == "aggression" else (got[k], want[k])
        out += ((g - w) / spread[k]) ** 2
    return out


def cmd_fit(a):
    import torch

    from ninemax_eval import load, setup
    from pipeline_opponents import KEYS, Style

    T = json.load(open(a.types))
    spread = dict(T["spread"])
    spread["aggression"] = max(spread["aggression"], 0.1)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = "cuda"
    dealer, menu, B = setup(a, dev)
    base, _ = load(a.baseline, B, dev)
    lo = (-2, -2, -2, -2.5, -1.5, 0, 0.8)
    hi = (2, 2, 2, 4.5, 2.5, 1, 1.5)  # post_fold to 4.5: humans fold to 60-70% of bets, nm25 to 39%
    clip = lambda p: tuple(round(min(max(x, l_), h_), 3) for x, l_, h_ in zip(p, lo, hi))  # noqa: E731
    own = measure(dealer, menu, base, Style(base, (0, 0, 0, 0, 0, 0, 1)), a.deals, dev)
    out = {"baseline": a.baseline, "baseline_stats": own, "deals": a.deals, "styles": []}
    print("nm25's own statistics: " + ", ".join(f"{k} {own[k]:.3f}" for k in STATS if own[k] is not None), flush=True)
    for t in T["types"]:
        want = t["centre"]
        tried = []
        cands = [(0, 0, 0, 0, 0, 0, 1)] + [
            clip(tuple(rng.uniform(x, y) for x, y in zip(lo, hi))) for _ in range(a.random)
        ]
        for rnd in range(a.rounds + 1):
            for p in cands:
                got = measure(dealer, menu, base, Style(base, p), a.deals, dev)
                tried.append((loss(got, want, spread), p, got))
            tried.sort(key=lambda x: x[0])
            if rnd == a.rounds:
                break
            step = 0.6 * (0.5**rnd)
            cands = [clip(tuple(x + rng.gauss(0, step) for x in tried[j % 6][1])) for j in range(a.local)]
        best_loss, p, got = tried[0]
        final = measure(dealer, menu, base, Style(base, p), a.deals * 3, dev)  # a fresh, larger measurement
        rec = {"id": t["id"], "name": t["name"], "share_hands": t["share_hands"], "target": want,
               "params": dict(zip(KEYS, p)), "achieved": {k: (round(v, 3) if isinstance(v, float) else v)
                                                          for k, v in final.items()},
               "loss": round(loss(final, want, spread), 3)}  # fmt: skip
        out["styles"].append(rec)
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"{t['id']} ({t['name']}): loss {rec['loss']:.2f}  " + "  ".join(
            f"{k} {want[k]:.2f}->{final[k]:.2f}" for k in STATS if final[k] is not None), flush=True)  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("players")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--per-dir", type=int, default=150000, help="hands read from each directory")
    p.add_argument("--min-hands", type=int, default=500)
    p.add_argument("--min-players", type=int, default=6, help="only tables with this many players or more")
    p.add_argument("--workers", type=int, default=24)
    t = sub.add_parser("types")
    t.add_argument("--players", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--k", type=int, default=8)
    t.add_argument("--restarts", type=int, default=20)
    t.add_argument("--seed", type=int, default=7)
    f = sub.add_parser("fit")
    f.add_argument("--types", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--baseline", default="blueprint/nm25")
    f.add_argument("--deals", type=int, default=1500, help="deals per seat per measurement")
    f.add_argument("--random", type=int, default=48, help="random candidates per type")
    f.add_argument("--rounds", type=int, default=3, help="rounds of local refinement")
    f.add_argument("--local", type=int, default=24, help="candidates per refinement round")
    f.add_argument("--players", type=int, default=9)
    f.add_argument("--menu", default="mtt")
    f.add_argument("--abs", default="blueprint/out")
    f.add_argument("--lib", default="blueprint/libbp7.so")
    f.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    {"players": cmd_players, "types": cmd_types, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
