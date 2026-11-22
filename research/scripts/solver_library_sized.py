"""Flop solver library with several bet sizes: training data for a bet-size-aware flop model.

Usage (on the lab): uv run python scripts/solver_library_sized.py OUTDIR [N_FLOPS] [PARALLEL] [THREADS_EACH]
Same flops (texture-balanced sample of the 1,755 distinct flops) and solver settings as solver_library.py, but the
flop street offers bets of 33%, 66%, 100% and 150% of the pot and all-in, so every facing-bet decision exists at
several sizes (turn and river keep 66% and all-in, which keeps solves affordable). LBR showed the 66%-only model
answers a pot-size bet or a shove as if it were 66% (2026-11-07). Spots: stack-to-pot 9.5 and 4.5, wide vs wide.
Resumable. Writes OUTDIR/solves/<id>.json.gz and OUTDIR/index.jsonl.
"""

import gzip
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poker_agents as pa  # noqa: E402
import solver_library as sl  # noqa: E402

FLOP_SIZES = [33, 66, 100, 150]
SPOTS = [s for s in sl.SPOTS if s["id"] in ("spr9.5_wide", "spr4.5_wide")]


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
        lines += [
            f"set_bet_sizes {p},flop,bet,{','.join(map(str, FLOP_SIZES))}",
            f"set_bet_sizes {p},flop,raise,100",
            f"set_bet_sizes {p},flop,allin",
        ]
        for st in ("turn", "river"):
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
            timeout=3600,
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
        "texture": sl.texture(flop),
        "spot": spot["id"],
        "seconds": round(secs, 1),
        "exploitability_pct": float(expl[-1]) if expl else None,
    }
    with open(os.path.join(outdir, "index.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def main(outdir, n_flops=80, parallel=2, threads=16):
    os.makedirs(os.path.join(outdir, "solves"), exist_ok=True)
    jobs = [(f, s) for f in sl.sample_flops(n_flops) for s in SPOTS]
    print(f"{len(jobs)} solves, {parallel} at a time on {threads} threads", flush=True)
    with ThreadPoolExecutor(parallel) as pool:
        for k, r in enumerate(pool.map(lambda j: solve(j[0], j[1], outdir, threads), jobs), 1):
            if r:
                print(f"{k}/{len(jobs)} {r['id']} {r['seconds']}s expl {r['exploitability_pct']}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(int(x) for x in a[1:4]))
