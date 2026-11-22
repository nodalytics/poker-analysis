"""Bayesian ratings for the version ladder, as sports ratings do it (Thurstone-Mosteller, the Gaussian cousin of
Bradley-Terry; TrueSkill is its online approximation): every version's rating with an uncertainty, the chance each
beats each other, and the chance each is the best.

Model: a pair's combined result d_AB = (A vs 8 B - B vs 8 A) / 2 ~ N(r_A - r_B, se_AB^2), ratings r ~ N(0, s0^2) a
priori, the anchor's rating 0. The posterior is Gaussian: precision X'WX + I/s0^2 (anchor removed), mean from the same
normal equations as the least-squares ladder plus the prior. A version seen in few pairs is pulled towards 0 instead
of getting a confident extreme. Goodness of fit: chi^2 of the residuals against the pairs' own errors; a model that
fits has chi^2 / dof near 1, more means results that ratings alone cannot explain (non-transitivity: A beats B,
B beats C, C beats A).

Usage: uv run python scripts/ratings_bayes.py [results/ninemax/ladder.json] [--anchor nm17] [--prior 3]
"""

import argparse
import json
import math

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ladder", nargs="?", default="results/ninemax/ladder.json")
    ap.add_argument("--anchor", default="nm17")
    ap.add_argument("--prior", type=float, default=3.0, help="prior sd of a rating, bb/100")
    ap.add_argument("--out")
    a = ap.parse_args()
    pairs = json.load(open(a.ladder))["pairs"]
    names = sorted({n for k in pairs for n in k.split(" vs ")}, key=lambda n: int(n[2:]))
    idx = {n: k for k, n in enumerate(names)}
    rows, y, w = [], [], []
    for key, p in pairs.items():
        x, z = key.split(" vs ")
        rev = pairs.get(f"{z} vs {x}")
        if rev is None or x > z:
            continue
        r = np.zeros(len(names))
        r[idx[x]], r[idx[z]] = 1, -1
        rows.append(r), y.append((p["bb100"] - rev["bb100"]) / 2), w.append(1 / (np.hypot(p["se"], rev["se"]) / 2) ** 2)
    X, y, W = np.array(rows), np.array(y), np.array(w)
    keep = [k for k, n in enumerate(names) if n != a.anchor]
    Xk = X[:, keep]
    # least squares (the ladder's own fit) and the Bayesian posterior
    ls = np.linalg.solve(Xk.T @ (W[:, None] * Xk), Xk.T @ (W * y))
    prec = Xk.T @ (W[:, None] * Xk) + np.eye(len(keep)) / a.prior**2
    cov = np.linalg.inv(prec)
    mu = cov @ (Xk.T @ (W * y))
    full_mu, full_ls = np.zeros(len(names)), np.zeros(len(names))
    full_mu[keep], full_ls[keep] = mu, ls
    full_cov = np.zeros((len(names), len(names)))
    full_cov[np.ix_(keep, keep)] = cov
    resid = y - X @ full_ls
    chi2, dof = float((W * resid**2).sum()), len(y) - len(keep)
    # chance of being the best, from the posterior
    rng = np.random.default_rng(1)
    draws = rng.multivariate_normal(full_mu, full_cov + 1e-12 * np.eye(len(names)), 200_000)
    best = np.bincount(draws.argmax(1), minlength=len(names)) / len(draws)
    print(f"{len(y)} pairs, {len(names)} versions, anchor {a.anchor}; fit chi2/dof {chi2:.1f}/{dof} = {chi2 / dof:.2f}")
    print(f"  {'version':8s} {'least sq':>9s} {'Bayes':>8s} {'+-95%':>7s} {'P(best)':>8s}")
    out = {"pairs": len(y), "chi2": chi2, "dof": dof, "versions": {}}
    for k, n in enumerate(names):
        sd = math.sqrt(full_cov[k, k])
        print(f"  {n:8s} {full_ls[k]:+9.2f} {full_mu[k]:+8.2f} {1.96 * sd:7.2f} {best[k]:8.3f}")
        out["versions"][n] = {"least_squares": round(float(full_ls[k]), 3), "bayes": round(float(full_mu[k]), 3),
                              "ci95": round(1.96 * sd, 3), "p_best": round(float(best[k]), 4)}  # fmt: skip
    top = [n for n in sorted(names, key=lambda n: -full_mu[idx[n]])][:4]
    print("\n  P(row beats column), the leaders:")
    print("  " + " " * 8 + "".join(f"{n:>8s}" for n in top))
    out["p_beats"] = {}
    for x in top:
        cells = []
        for z in top:
            d = full_mu[idx[x]] - full_mu[idx[z]]
            v = full_cov[idx[x], idx[x]] + full_cov[idx[z], idx[z]] - 2 * full_cov[idx[x], idx[z]]
            p = 0.5 if x == z else 0.5 * math.erfc(-d / math.sqrt(2 * max(v, 1e-12)))
            cells.append(p)
            out["p_beats"][f"{x} > {z}"] = round(p, 3)
        print(f"  {x:8s}" + "".join(f"{p:8.3f}" for p in cells))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
