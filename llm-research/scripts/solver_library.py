"""Post-flop solver library: batch TexasSolver flop solves, the training data for real-time flop prediction.

Usage (on the lab): uv run python scripts/solver_library.py OUTDIR [N_FLOPS] [PARALLEL] [THREADS_EACH]

Flops: the 1,755 strategically distinct flops (suit isomorphism), sampled across textures so every kind is covered
(rainbow / two-tone / monotone, paired, connected, high and low). Spots: 2 stack depths (pot 10 with 95 or 45
behind: stack-to-pot 9.5 and 4.5) x 2 range pairs (wide vs wide; a tighter opener vs a wide defender). Each solve:
one bet size (66% pot) plus all-in on every street, accuracy 2%, at most 80 iterations; about 2.4 minutes on 16
threads. Resumable: finished solves are skipped. Writes OUTDIR/solves/<id>.json.gz (the full flop-street tree) and
OUTDIR/index.jsonl (one line per solve: flop, texture, spot, seconds, exploitability).
"""

import gzip
import itertools
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import poker_agents as pa

RANKS = "23456789TJQKA"
WIDE = pa.WIDE
TIGHT = "AA,KK,QQ,JJ,TT,99,88,77,66,AK,AQ,AJ,AT,A9s,A5s,A4s,KQ,KJ,KTs,QJ,QTs,JTs,T9s,98s"
SPOTS = [
    {"id": "spr9.5_wide", "pot": 10, "stack": 95, "oop": WIDE, "ip": WIDE},
    {"id": "spr4.5_wide", "pot": 10, "stack": 45, "oop": WIDE, "ip": WIDE},
    {"id": "spr9.5_tight_vs_wide", "pot": 10, "stack": 95, "oop": WIDE, "ip": TIGHT},
    {"id": "spr4.5_tight_vs_wide", "pot": 10, "stack": 45, "oop": WIDE, "ip": TIGHT},
]


def distinct_flops():
    """One representative per suit-isomorphism class: 1,755 flops. A flop's class is the smallest card tuple over
    all 24 relabellings of the suits (sorting by rank alone double-counts paired flops)."""
    seen, out = set(), []
    deck = [r + s for r in RANKS for s in "cdhs"]
    perms = list(itertools.permutations("cdhs"))
    for f in itertools.combinations(deck, 3):
        key = min(
            tuple(sorted((RANKS.index(c[0]), "cdhs".index(dict(zip("cdhs", p))[c[1]])) for c in f)) for p in perms
        )
        if key not in seen:
            seen.add(key)
            out.append(list(f))
    return out


def texture(flop):
    rs = sorted((RANKS.index(c[0]) for c in flop), reverse=True)
    suits = len({c[1] for c in flop})
    return {
        "suits": {1: "monotone", 2: "two-tone", 3: "rainbow"}[suits],
        "paired": len(set(rs)) < 3,
        "high": RANKS[rs[0]],
        "connected": (rs[0] - rs[2]) <= 4 and len(set(rs)) == 3,
    }


def sample_flops(n, seed=11):
    flops = distinct_flops()
    groups = {}
    for f in flops:
        t = texture(f)
        groups.setdefault((t["suits"], t["paired"], t["connected"], t["high"] in "AKQJ"), []).append(f)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    out, keys = [], sorted(groups)
    while len(out) < n and any(groups.values()):  # round-robin over textures
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def solve(flop, spot, outdir, threads):
    sid = f"{''.join(flop)}_{spot['id']}"
    path = os.path.join(outdir, "solves", sid + ".json.gz")
    if os.path.exists(path):
        return None
    lines = [
        f"set_pot {spot['pot']}",
        f"set_effective_stack {spot['stack']}",
        f"set_board {','.join(flop)}",
        f"set_range_oop {spot['oop']}",
        f"set_range_ip {spot['ip']}",
    ]
    for p in ("oop", "ip"):
        for st in ("flop", "turn", "river"):
            lines += [f"set_bet_sizes {p},{st},bet,66", f"set_bet_sizes {p},{st},allin"]
    with tempfile.TemporaryDirectory() as tmp:
        lines += [
            "set_allin_threshold 0.67",
            "build_tree",
            f"set_thread_num {threads}",
            "set_accuracy 2.0",
            "set_max_iteration 80",
            "set_print_interval 20",
            "set_use_isomorphism 1",
            "start_solve",
            "set_dump_rounds 1",
            f"dump_result {tmp}/out.json",
        ]
        open(f"{tmp}/in.txt", "w").write("\n".join(lines) + "\n")
        t0 = time.time()
        res = subprocess.run(
            ["nice", "-n", "12", pa.SOLVER, "--input_file", f"{tmp}/in.txt", "--resource_dir", pa.RESOURCES],
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=tmp,
        )
        secs = time.time() - t0
        expl = re.findall(r"Total exploitability ([\d.]+)", res.stdout)
        tree = json.load(open(f"{tmp}/out.json"))
    with gzip.open(path, "wt") as f:
        json.dump(tree, f)
    row = {
        "id": sid,
        "flop": flop,
        "texture": texture(flop),
        "spot": spot["id"],
        "seconds": round(secs, 1),
        "exploitability_pct": float(expl[-1]) if expl else None,
    }
    with open(os.path.join(outdir, "index.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def main(outdir, n_flops=100, parallel=3, threads=16):
    os.makedirs(os.path.join(outdir, "solves"), exist_ok=True)
    jobs = [(f, s) for f in sample_flops(n_flops) for s in SPOTS]
    print(
        f"{len(jobs)} solves ({n_flops} flops x {len(SPOTS)} spots), {parallel} at a time on {threads} threads",
        flush=True,
    )
    with ThreadPoolExecutor(parallel) as ex:
        for row in ex.map(lambda j: solve(j[0], j[1], outdir, threads), jobs):
            if row:
                print(f"{row['id']}: {row['seconds']}s, exploitability {row['exploitability_pct']}%", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 else 100, int(a[3]) if len(a) > 3 else 3, int(a[4]) if len(a) > 4 else 16)
