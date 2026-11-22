"""Paired comparison of search runs on the same deals (ninemax_search.py --per-deal, same --seed and --hands): for each
kind of opponent, the gain of each search over the plain network, deal by deal, with a 95% interval. Pairing removes
the deals' own luck, which the unpaired intervals of each run carry.

Usage: python scripts/search_paired.py results/ninemax/search3_nm25   (reads <prefix>_<opp>_<kind>.npy)
"""

import glob
import os
import sys

import numpy as np


def main():
    prefix = sys.argv[1]
    opps = sorted(
        {os.path.basename(f)[len(os.path.basename(prefix)) + 1 :].split("_")[0] for f in glob.glob(prefix + "_*.npy")}
    )
    print(f"{'opponents':10s} {'search':8s} {'gain over plain, bb/100':>28s}   deals")
    for opp in opps:
        plain = np.load(f"{prefix}_{opp}_plain.npy")
        for kind in ("v1", "robust"):
            f = f"{prefix}_{opp}_{kind}.npy"
            if not os.path.exists(f):
                continue
            d = np.load(f) - plain
            m, se = d.mean(), d.std() / np.sqrt(len(d))
            print(f"{opp:10s} {kind:8s} {m:+10.2f} [{m - 1.96 * se:+.2f}, {m + 1.96 * se:+.2f}]   {len(d)}")


if __name__ == "__main__":
    main()
