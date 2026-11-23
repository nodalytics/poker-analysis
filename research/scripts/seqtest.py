"""Sequential duplicate matches, as clinical trials run them: look at the result after each stage and stop as soon as
the answer is clear, with the overall false-positive rate still alpha.

Group-sequential design (O'Brien & Fleming 1979): K looks at equal steps of n_max / K deals; at look k the match stops
for a difference when |z_k| >= C / sqrt(t_k), t_k = k / K the share of deals played. C is set so that, with no real
difference, the chance of ever crossing is alpha (computed here by simulating the z path, exactly the Brownian motion
the statistic follows). Early looks need a very large z, so stopping early costs almost no power. The match also
stops for equivalence when the 95% interval lies inside (-margin, +margin): no difference that matters.

Usage (lab, GPU venv, from research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/seqtest.py simulate          # error rates and deals saved
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/seqtest.py match blueprint/nm20 blueprint/nm17 [--max 40000]
"""

import argparse
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def obf_constant(K, alpha=0.05, sims=400_000, seed=1):
    """C for K equally spaced looks, two-sided: Pr(any |z_k| >= C / sqrt(t_k) | no difference) = alpha."""
    rng = np.random.default_rng(seed)
    w = np.cumsum(rng.standard_normal((sims, K)), 1)  # Brownian motion at t = 1..K (in steps)
    t = np.arange(1, K + 1) / K
    z = w / np.sqrt(np.arange(1, K + 1))  # the z statistic at each look
    stat = np.max(np.abs(z) * np.sqrt(t), 1)  # crossing C/sqrt(t) <=> |z| sqrt(t) >= C
    return float(np.quantile(stat, 1 - alpha))


def boundaries(K, alpha=0.05):
    C = obf_constant(K, alpha)
    return [C / math.sqrt((k + 1) / K) for k in range(K)], C


def decide(x, k, K, bounds, margin):
    """After look k (0-based) on per-deal results x (bb a deal): 'difference', 'equivalent' or None (go on)."""
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    z = m / se if se > 0 else 0.0
    if abs(z) >= bounds[k]:
        return "difference", m, se, z
    if margin and abs(m) + 1.96 * se < margin / 100:  # margin in bb/100
        return "equivalent", m, se, z
    return ("no difference at the end" if k == K - 1 else None), m, se, z


def simulate(K=8, alpha=0.05, margin=0.5, sd=1.0, n_max=40000, reps=20000, seed=7):
    """Error rates and deals used, on synthetic matches whose per-deal results are normal with the spread of our
    duplicate matches (sd bb a deal; --sd sets it): for true differences of 0 to 3 bb/100."""
    bounds, C = boundaries(K, alpha)
    rng = np.random.default_rng(seed)
    step = n_max // K
    rows = []
    for eff in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0):
        stops, used, fixed_sig = [], [], 0
        for _ in range(reps // 10):
            x = rng.normal(eff / 100, sd, n_max)
            fixed_sig += abs(x.mean()) / (x.std(ddof=1) / math.sqrt(n_max)) >= 1.96
            for k in range(K):
                v, m, se, z = decide(x[: (k + 1) * step], k, K, bounds, margin)
                if v:
                    stops.append(v), used.append((k + 1) * step)
                    break
        n = len(stops)
        rows.append({"true_bb100": eff, "difference": sum(s == "difference" for s in stops) / n,
                     "equivalent": sum(s == "equivalent" for s in stops) / n,
                     "fixed_design_significant": fixed_sig / n, "deals_used": float(np.mean(used)),
                     "saved": 1 - float(np.mean(used)) / n_max})  # fmt: skip
    return {"K": K, "alpha": alpha, "C": round(C, 4), "margin_bb100": margin, "n_max": n_max, "sd": sd, "rows": rows}


def run_lib(run):
    """The bucket library a run was trained with (its config.json), libbp7 (nm25's) where it says nothing."""
    try:
        cfg = json.load(open(os.path.join(run, "config.json")))
    except (OSError, ValueError):
        cfg = {}
    return os.path.join(ROOT, cfg.get("lib") or os.path.join("blueprint", "libbp7.so"))


def match(a_run, b_run, n_max, K, alpha, margin, dev="cuda", early_shove_bb=0.0, early=None):
    """A sequential duplicate match: a vs 8 copies of b, stage by stage on fresh deals."""
    import torch

    from ninemax_engine import Menu, run_dealer
    from ninemax_eval import load
    from ninemax_selfplay import BB, FEAT, board_of, context, play

    bounds, C = boundaries(K, alpha)
    dealer = run_dealer(a_run, run_lib(a_run), None, 9, 0, dev, 2026, "/tmp/seqtest")  # a's library and abstraction
    b, _ = load(b_run, dealer.B, dev)
    a, _ = load(a_run, dealer.B, dev)  # last: load() sets the game from the config, and the match is in a's game
    if early is not None:  # the share of early-stage tables, instead of a's own (ninemax_selfplay --early)
        FEAT["early"] = early
    if early_shove_bb:  # a plays by the early-shove rule (a separate object, so b does not; play() reads it)
        a.early_shove_bb = early_shove_bb
    cfg = json.load(open(os.path.join(a_run, "config.json")))
    menu = Menu(cfg.get("menu", "medium"), dev)
    step, xs, looks = n_max // K, [], []
    with torch.no_grad():
        for k in range(K):
            bk, rk, cd = dealer.deal(step, cards=FEAT["texture"])
            board = board_of(cd, dealer.P, dev)
            ctx = context(step, dealer.P, dev)
            per = torch.zeros(step, device=dev)
            for seat in range(dealer.P):
                pols = [b] * dealer.P
                pols[seat] = a
                u, _ = play(dealer, menu, pols, bk, rk, dev=dev, ctx=ctx, board=board)
                per += u[:, seat] / BB
            xs.append((per / dealer.P).cpu().numpy())
            x = np.concatenate(xs)
            v, m, se, z = decide(x, k, K, bounds, margin)
            m, se, z = float(m), float(se), float(z)  # numpy float32 does not serialise to JSON
            looks.append({"deals": len(x), "bb100": round(m * 100, 3), "ci95": [round((m - 1.96 * se) * 100, 3),
                          round((m + 1.96 * se) * 100, 3)], "z": round(z, 3), "boundary": round(bounds[k], 3)})  # fmt: skip
            print(f"look {k + 1}: {len(x)} deals  {m * 100:+.2f} bb/100  z {z:+.2f} (stop beyond {bounds[k]:.2f})",
                  flush=True)  # fmt: skip
            if v:
                return {
                    "a": a_run,
                    "b": b_run,
                    "early": FEAT["early"],
                    "verdict": v,
                    "deals": len(x),
                    "n_max": n_max,
                    "looks": looks,
                    "C": C,
                }
    # no boundary crossed by n_max: the reading stands as inconclusive (it used to return None, which main() crashed on)
    return {"a": a_run, "b": b_run, "early": FEAT["early"], "verdict": "inconclusive", "deals": len(x), "n_max": n_max,
            "looks": looks, "C": C}  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("simulate", "match"))
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--max", type=int, default=40000)
    ap.add_argument("--looks", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.5, help="equivalence margin, bb/100 (0: never stop for it)")
    ap.add_argument("--sd", type=float, default=1.0, help="simulate: per-deal spread, bb (duplicate matches ~0.8-1.2)")
    ap.add_argument(
        "--early-shove-bb",
        type=float,
        default=0.0,
        help="match: the first run plays by the early-shove rule from this many big blinds (0: off)",
    )
    ap.add_argument("--early", type=float, default=None,
                    help="match: share of early-stage (winner-take-all) tables, instead of the first run's own")  # fmt: skip
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "simulate":
        res = simulate(a.looks, a.alpha, a.margin, a.sd, a.max)
        print(f"K {res['K']} looks, C {res['C']}, margin {res['margin_bb100']} bb/100, up to {res['n_max']} deals")
        for r in res["rows"]:
            print(f"  true {r['true_bb100']:+.2f} bb/100: stops for a difference {r['difference']:.3f}, "
                  f"for equivalence {r['equivalent']:.3f} (fixed design significant {r['fixed_design_significant']:.3f}); "
                  f"{r['deals_used']:.0f} deals on average, {r['saved']:.0%} saved")  # fmt: skip
    else:
        res = match(
            a.runs[0], a.runs[1], a.max, a.looks, a.alpha, a.margin, early_shove_bb=a.early_shove_bb, early=a.early
        )
        print(json.dumps({k: v for k, v in res.items() if k != "looks"}))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
