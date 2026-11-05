"""Confidence intervals and decision-by-decision comparison for a backtest_hh.py run.

Usage: uv run python scripts/backtest_analyse.py BACKTEST_DIR [FLOW_A FLOW_B]
Prints each flow's mean chip regret against Pluribus with a bootstrap 95% interval, then, for two flows (default: the
first model's #plain and #advised), every spot where they chose differently: how many changes, which way, the net
chips, and the spots that cost the most.
"""

import collections
import json
import os
import random
import sys


def ev(kind, r):
    if r["eq_true"] is None:
        return None
    return 0.0 if kind == "fold" else r["eq_true"] * (r["pot"] + r["to_call"]) - r["to_call"]


def short(flow):
    return flow.split("/")[-1].replace("-GGUF:Q4_K_M", "")


def main(bdir, a=None, b=None):
    rows = [json.loads(x) for x in open(os.path.join(bdir, "spots.jsonl"))]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["flow"]].append(r)
    rng = random.Random(1)
    print("Mean chip regret against Pluribus per spot, bootstrap 95% interval:")
    for flow, rs in by.items():
        g = [ev(r["choice"], r) - ev(r["ref"], r) for r in rs if r["eq_true"] is not None]
        m = sum(g) / len(g)
        boot = sorted(sum(rng.choice(g) for _ in g) / len(g) for _ in range(2000))
        print(f"  {short(flow):38} {m:7.1f}  [{boot[50]:7.1f}, {boot[1949]:7.1f}]  n={len(g)}")
    models = [f[: -len("#plain")] for f in by if f.endswith("#plain")]
    a = a or models[0] + "#plain"
    b = b or models[0] + "#advised"
    key = lambda r: (r["street"], r["pot"], r["to_call"], r["eq_true"], r["ref"])  # noqa: E731
    pa_, pb = {key(r): r for r in by[a]}, {key(r): r for r in by[b]}
    diffs = []
    for k, r in pa_.items():
        if k in pb and r["choice"] != pb[k]["choice"] and r["eq_true"] is not None:
            diffs.append((ev(pb[k]["choice"], r) - ev(r["choice"], r), r, pb[k]["choice"]))
    diffs.sort(key=lambda x: x[0])
    total = sum(d for d, *_ in diffs)
    print(f"\n{short(b)} vs {short(a)}: {len(diffs)} decisions changed, net {total:+.0f} chips over all spots")
    print("  changes:", dict(collections.Counter(f"{r['choice']} -> {nb}" for _, r, nb in diffs)))
    print(f"  the 5 costliest changes account for {sum(d for d, *_ in diffs[:5]):+.0f} chips:")
    for d, r, nb in diffs[:5]:
        print(
            f"    {d:+8.0f}  {r['street']:7} pot {r['pot']:6} to call {r['to_call']:6} true equity {r['eq_true']:.2f}: "
            f"{r['choice']} -> {nb}  (Pluribus {r['ref']})"
        )


if __name__ == "__main__":
    main(*sys.argv[1:4])
