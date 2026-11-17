"""The look-elsewhere effect for our results: when a family of results is tested together, some look significant by
chance. Holm's step-down correction keeps the chance of even one false claim in the family at alpha;
Benjamini-Hochberg keeps the expected share of false claims among those made at alpha (it is less strict).

Each result is an estimate with a 95% interval (bb/100); its two-sided p-value comes from the normal approximation,
se = (hi - lo) / 3.92.

Usage: uv run python scripts/multitest.py [--out results/multitest.json]   (the families below; prints a table)
"""

import argparse
import json
import math


def p_value(m, lo, hi):
    se = (hi - lo) / 3.92
    z = m / se if se > 0 else 0.0
    return math.erfc(abs(z) / math.sqrt(2))


def holm(ps, alpha=0.05):
    """Holm-adjusted p-values (monotone), in the input order."""
    order = sorted(range(len(ps)), key=lambda k: ps[k])
    adj, run = [0.0] * len(ps), 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[k]))
        adj[k] = run
    return adj


def bh(ps):
    """Benjamini-Hochberg adjusted p-values (q-values), in the input order."""
    n = len(ps)
    order = sorted(range(n), key=lambda k: ps[k])
    adj, run = [0.0] * n, 1.0
    for r in range(n - 1, -1, -1):
        k = order[r]
        run = min(run, ps[k] * n / (r + 1))
        adj[k] = run
    return adj


# Families of results already reported, as (label, estimate, lo, hi). Each family is what was looked at together.
FAMILIES = {
    "search on nm25, Python, paired (2026-11-16)": [
        ("v1 vs network", 2.18, -0.32, 4.68),
        ("v1 vs raise-biased", 6.39, 3.92, 8.86),
        ("v1 vs call-biased", 3.89, 1.30, 6.49),
        ("v1 vs fold-biased", 2.26, -0.07, 4.60),
        ("robust vs network", 1.99, 0.02, 3.96),
        ("robust vs raise-biased", 3.69, 1.54, 5.84),
        ("robust vs call-biased", 3.13, 0.98, 5.28),
        ("robust vs fold-biased", 0.82, -1.07, 2.71),
    ],
    "search in Kestrel, C++, paired (2026-11-17)": [
        ("1,024 vs raise-biased", 6.16, 3.35, 8.98),
        ("1,024 vs network", 4.12, 1.37, 6.87),
        ("1,024 vs call-biased", 4.18, 1.29, 7.07),
        ("1,024 vs fold-biased", 3.73, 0.98, 6.48),
        ("2,048 vs network", 4.20, 1.59, 6.81),
        ("256 vs raise-biased", 0.50, -2.91, 3.91),
        ("256 vs network", -5.21, -8.65, -1.78),
        ("256 vs call-biased", -6.54, -10.21, -2.88),
        ("256 vs fold-biased", -5.16, -8.52, -1.80),
        ("256 robust vs network", -2.27, -5.03, 0.50),
    ],
    "version ladder against nm17 (rating, +-ci95)": [
        (name, r, r - c, r + c)
        for name, r, c in (
            ("nm15", -0.36, 0.16),
            ("nm16", -0.26, 0.16),
            ("nm18", -0.43, 0.16),
            ("nm19", -0.36, 0.16),
            ("nm20", 1.04, 0.15),
            ("nm21", -0.56, 0.16),
            ("nm25", 0.96, 0.15),
            ("nm26", 1.03, 0.15),
            ("nm27", 1.03, 0.15),
        )  # fmt: skip
    ],
    "local best response, 9-max (2026-11-15)": [
        ("nm25", 0.26, -0.43, 0.95),
        ("nm20", 0.09, -0.40, 0.58),
        ("nm17", -0.55, -1.39, 0.29),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    out = {}
    for fam, rows in FAMILIES.items():
        ps = [p_value(m, lo, hi) for _, m, lo, hi in rows]
        hs, qs = holm(ps), bh(ps)
        print(f"\n{fam} ({len(rows)} results)")
        print(f"  {'result':28s} {'bb/100':>8s} {'p':>8s} {'Holm p':>8s} {'BH q':>8s}  verdict")
        out[fam] = []
        for (name, m, lo, hi), p, h, q in zip(rows, ps, hs, qs):
            alone, fam_ok = p < 0.05, h < 0.05
            verdict = "holds" if fam_ok else ("lost to look-elsewhere" if alone else "not significant")
            print(f"  {name:28s} {m:+8.2f} {p:8.4f} {h:8.4f} {q:8.4f}  {verdict}")
            out[fam].append({"result": name, "bb100": m, "p": round(p, 5), "holm": round(h, 5), "bh": round(q, 5),
                             "verdict": verdict})  # fmt: skip
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
