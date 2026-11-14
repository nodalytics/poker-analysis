"""Is the trainer's dealer fair? A statistical check of libbp's bp_deal_n (ninemax_engine.Dealer) against exact odds.

The dealer shuffles with a partial Fisher-Yates per deal, each deal on its own splitmix64 stream (seed * FNV prime +
deal index). Fast and reproducible, which training needs; this checks it is also unbiased:

  slots        every card equally likely in every hole and board position (chi-square, 23 slots x 52 cards)
  distinct     no card twice in a deal
  classes      the 169 starting-hand classes at their exact frequencies (pairs 6, suited 4, offsuit 12 of 1,326)
  categories   seat 0's seven-card hand category, from the dealer's own ranks, against the exact counts in C(52,7)
  pairs        the two hole cards independent of each other (the 52 x 51 table of ordered pairs)
  neighbours   consecutive deals, and the same deal index in successive calls, share a card as often as chance says
  seeds        the same seed deals the same cards, a different seed different cards

Each test reports its p-value; a fair dealer fails one at the 0.1% level about once in a thousand runs.

Usage (lab, from llm-research):
  OMP_NUM_THREADS=8 PYTHONPATH=scripts .venv-gpu/bin/python scripts/dealer_check.py [--deals 4000000] \\
      [--out results/ninemax/dealer_check.json]
"""

import argparse
import json
import math
import os

import numpy as np
from scipy.stats import binomtest, chi2

from ninemax_engine import Dealer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = 9
SLOTS = 2 * P + 5
# seven-card hand categories: exact counts out of C(52, 7) = 133,784,560, and PokerHandEvaluator's rank ranges
CATS = [
    ("straight flush", 41584, 1, 10),
    ("four of a kind", 224848, 11, 166),
    ("full house", 3473184, 167, 322),
    ("flush", 4047644, 323, 1599),
    ("straight", 6180020, 1600, 1609),
    ("three of a kind", 6461620, 1610, 2467),
    ("two pair", 31433400, 2468, 3325),
    ("pair", 58627800, 3326, 6185),
    ("high card", 23294460, 6186, 7462),
]
TOTAL7 = math.comb(52, 7)


def chisq(obs, exp):
    obs, exp = np.asarray(obs, float), np.asarray(exp, float)
    stat = float(((obs - exp) ** 2 / exp).sum())
    df = obs.size - 1
    return {"chi2": round(stat, 2), "df": df, "p": float(chi2.sf(stat, df))}


def hand_class(c1, c2):
    r1, r2, s1, s2 = c1 // 4, c2 // 4, c1 % 4, c2 % 4
    hi, lo = np.maximum(r1, r2), np.minimum(r1, r2)
    return np.where(s1 == s2, hi * 13 + lo, lo * 13 + hi)


def class_probs():
    p = np.zeros(169)
    for hi in range(13):
        for lo in range(13):
            if hi == lo:
                p[hi * 13 + lo] = 6 / 1326
            elif hi > lo:
                p[hi * 13 + lo] = 4 / 1326
                p[lo * 13 + hi] = 12 / 1326
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=4_000_000)
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--out")
    ap.add_argument("--lib", default=os.path.join(ROOT, "blueprint", "libbp5.so"))
    ap.add_argument("--abs", default=os.path.join(ROOT, "blueprint", "out"))
    a = ap.parse_args()
    B = int(open(os.path.join(a.abs, "meta.txt")).read().split()[1])
    d = Dealer(a.lib, a.abs, P, B, "cpu", seed=12345)

    slot = np.zeros((SLOTS, 52), np.int64)
    pairs = np.zeros((52, 52), np.int64)
    cls = np.zeros(169, np.int64)
    cat = np.zeros(len(CATS), np.int64)
    dup = 0
    nb_same = nb_n = call_same = call_n = 0
    prev_last, prev_first_col = None, None
    n = 0
    while n < a.deals:
        _, rk, cd = d.raw(a.chunk, cards=True)
        n += a.chunk
        for s in range(SLOTS):
            slot[s] += np.bincount(cd[:, s], minlength=52)
        srt = np.sort(cd, 1)
        dup += int((srt[:, 1:] == srt[:, :-1]).any(1).sum())
        np.add.at(pairs, (cd[:, 0], cd[:, 1]), 1)
        for p in range(P):
            cls += np.bincount(hand_class(cd[:, 2 * p], cd[:, 2 * p + 1]), minlength=169)
        r0 = rk[:, 0]
        for k, (_, _, lo, hi) in enumerate(CATS):
            cat[k] += int(((r0 >= lo) & (r0 <= hi)).sum())
        # consecutive deals share their first card with probability 1/52
        nb_same += int((cd[1:, 0] == cd[:-1, 0]).sum())
        nb_n += len(cd) - 1
        if prev_last is not None:
            nb_same += int(cd[0, 0] == prev_last)
            nb_n += 1
            # the same deal index in the next call (seed + 1): the first 1,000 deals' first cards
            call_same += int((cd[:1000, 0] == prev_first_col).sum())
            call_n += 1000
        prev_last, prev_first_col = cd[-1, 0], cd[:1000, 0].copy()

    res = {"deals": n, "players": P, "tests": {}}
    T = res["tests"]
    per_slot = [chisq(slot[s], np.full(52, n / 52)) for s in range(SLOTS)]
    T["slots"] = {"worst_p": min(x["p"] for x in per_slot), "combined": chisq(slot.ravel(), np.full(SLOTS * 52, n / 52)),
                  "per_slot_p": [round(x["p"], 4) for x in per_slot]}  # fmt: skip
    T["slots"]["combined"]["df"] = SLOTS * 51
    T["slots"]["combined"]["p"] = float(chi2.sf(T["slots"]["combined"]["chi2"], SLOTS * 51))
    T["distinct"] = {"deals_with_a_repeated_card": dup}
    T["classes"] = chisq(cls, class_probs() * n * P)
    exp_cat = np.array([c[1] for c in CATS]) / TOTAL7 * n
    T["categories"] = chisq(cat, exp_cat)
    T["categories"]["observed_vs_exact"] = {c[0]: [int(o), round(float(e), 1)] for c, o, e in zip(CATS, cat, exp_cat)}
    off = ~np.eye(52, dtype=bool)
    T["pairs"] = chisq(pairs[off], np.full(52 * 51, n / (52 * 51)))
    T["neighbours"] = {
        "consecutive_deals": {"same_first_card": nb_same, "n": nb_n, "p": binomtest(nb_same, nb_n, 1 / 52).pvalue},
        "successive_calls": {"same_first_card": call_same, "n": call_n,
                             "p": binomtest(call_same, call_n, 1 / 52).pvalue},
    }  # fmt: skip
    s0 = d.seed
    d.seed = 777
    _, _, x1 = d.raw(1000, cards=True)
    d.seed = 777
    _, _, x2 = d.raw(1000, cards=True)
    _, _, x3 = d.raw(1000, cards=True)
    d.seed = s0
    T["seeds"] = {"same_seed_same_cards": bool((x1 == x2).all()),
                  "next_seed_identical_deals": int((x1 == x3).all(1).sum())}  # fmt: skip

    ps = [T["slots"]["combined"]["p"], T["classes"]["p"], T["categories"]["p"], T["pairs"]["p"],
          T["neighbours"]["consecutive_deals"]["p"], T["neighbours"]["successive_calls"]["p"]]  # fmt: skip
    res["fair"] = bool(min(ps) > 1e-3 and dup == 0 and T["seeds"]["same_seed_same_cards"]
                       and T["seeds"]["next_seed_identical_deals"] == 0)  # fmt: skip
    print(f"{n:,} deals, {P} players")
    print(
        f"  slots       combined p = {T['slots']['combined']['p']:.3f}, worst of {SLOTS} slots p = {T['slots']['worst_p']:.3f}"
    )
    print(f"  distinct    deals with a repeated card: {dup}")
    print(f"  classes     169 starting-hand classes p = {T['classes']['p']:.3f}")
    print(f"  categories  seven-card categories p = {T['categories']['p']:.3f}")
    for k, (o, e) in T["categories"]["observed_vs_exact"].items():
        print(f"              {k:16s} {o / n * 100:8.4f}%  exact {e / n * 100:8.4f}%")
    print(f"  pairs       hole cards independent p = {T['pairs']['p']:.3f}")
    print(f"  neighbours  consecutive deals p = {T['neighbours']['consecutive_deals']['p']:.3f}, "
          f"successive calls p = {T['neighbours']['successive_calls']['p']:.3f}")  # fmt: skip
    print(f"  seeds       same seed same cards: {T['seeds']['same_seed_same_cards']}, "
          f"next seed identical deals: {T['seeds']['next_seed_identical_deals']}")  # fmt: skip
    print("FAIR" if res["fair"] else "NOT FAIR: see the tests above")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
