"""Value library: turn subgames with random ranges, solved, with every hand's value: training data for a value net.

Usage: uv run python scripts/value_library.py OUTDIR N_SOLVES [WORKERS]
Needs the solver build that exports root values (`~/solver-values`, TexasSolver patched so dump_result also writes
`root_evs`: each hand's value at the root from the final best-response pass; validated on the textbook river game,
where it gives AA +100, the bluffs -50 and the bluff-catcher +26.8, as theory says). VALUE_SOLVER overrides the path.

Each solve: a random turn board, pot 10, an effective stack of 10, 25 or 50 (stack-to-pot 1, 2.5, 5), bets of 66%
of the pot and all-in on turn and river, and a random range for each player in the style of DeepStack's training
data: hand classes ordered by strength with noise, a random share kept (8-60%), sometimes polarised (strong hands
plus some of the weakest). Values are chips relative to the start of the turn, each player having put in half of
the pot (so +5 means winning the opponent's half). One JSON line per solve, gzipped, in OUTDIR/solves/.
"""

import gzip
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preflop_deep  # noqa: E402  (hand classes ranked by equity against a random hand)

SOLVER = os.environ.get("VALUE_SOLVER", os.path.expanduser("~/solver-values/build/console_solver"))
RESOURCES = os.path.expanduser("~/poker-analysis/tools/TexasSolver/resources")
RANKS, SUITS = "23456789TJQKA", "cdhs"
STACKS = [10, 25, 50]


def random_range(rng):
    order = preflop_deep._table()["by_strength"]
    noisy = sorted(order, key=lambda c: order.index(c) + rng.gauss(0, 12))
    share = rng.uniform(0.08, 0.6)
    n = max(3, int(share * len(noisy)))
    keep = {c: 1.0 for c in noisy[:n]}
    for c in noisy[n : n + 10]:
        keep[c] = round(rng.uniform(0.1, 0.9), 2)  # a soft edge, as real ranges have
    if rng.random() < 0.3:  # polarised: add some of the weakest hands as bluffs
        for c in noisy[-rng.randint(5, 25) :]:
            keep[c] = round(rng.uniform(0.2, 1.0), 2)
    return keep


def range_string(r):
    # TexasSolver range syntax: AA,AKs:0.5,... (pairs and suited/offsuit classes)
    return ",".join(c if w >= 0.999 else f"{c}:{w}" for c, w in r.items())


def solve(args):
    k, seed = args
    rng = random.Random(seed)
    deck = [r + s for r in RANKS for s in SUITS]
    board = rng.sample(deck, 4)
    stack = rng.choice(STACKS)
    r_oop, r_ip = random_range(rng), random_range(rng)
    tmp = tempfile.mkdtemp(prefix="vl-")
    try:
        lines = [
            "set_pot 10",
            f"set_effective_stack {stack}",
            "set_board " + ",".join(board),
            "set_range_oop " + range_string(r_oop),
            "set_range_ip " + range_string(r_ip),
        ]
        for p in ("oop", "ip"):
            for st in ("turn", "river"):
                lines += [f"set_bet_sizes {p},{st},bet,66", f"set_bet_sizes {p},{st},allin"]
        lines += [
            "set_allin_threshold 0.67",
            "build_tree",
            "set_thread_num 1",
            "set_accuracy 1.0",
            "set_max_iteration 300",
            "set_print_interval 25",
            "set_use_isomorphism 1",
            "start_solve",
            "set_dump_rounds 1",
            f"dump_result {tmp}/o.json",
        ]
        open(f"{tmp}/in.txt", "w").write("\n".join(lines) + "\n")
        t0 = time.time()
        log = subprocess.run(
            [SOLVER, "--input_file", "in.txt", "--resource_dir", RESOURCES],
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=600,
        ).stdout
        expl = [float(x.split()[2]) for x in log.splitlines() if x.startswith("Total exploitability")]
        out = json.load(open(f"{tmp}/o.json"))
        rec = {
            "id": k,
            "board": board,
            "pot": 10,
            "stack": stack,
            "range_oop": r_oop,
            "range_ip": r_ip,
            "values_ip": out["root_evs"].get("0", {}),
            "values_oop": out["root_evs"].get("1", {}),
            "root_actions": out.get("actions"),
            "exploitability_pct": expl[-1] if expl else None,
            "seconds": round(time.time() - t0, 1),
        }
        return rec
    except (subprocess.SubprocessError, OSError, ValueError, KeyError) as e:
        return {"id": k, "error": repr(e)[:200]}
    finally:
        subprocess.run(["rm", "-rf", tmp])


def main(outdir, n, workers=40):
    os.makedirs(os.path.join(outdir, "solves"), exist_ok=True)
    done = {int(f.split(".")[0]) for f in os.listdir(os.path.join(outdir, "solves"))}
    jobs = [(k, 1000 + k) for k in range(n) if k not in done]
    ok = bad = 0
    with ProcessPoolExecutor(workers) as pool, open(os.path.join(outdir, "index.jsonl"), "a") as idx:
        for f in as_completed([pool.submit(solve, j) for j in jobs]):
            r = f.result()
            if "error" in r:
                bad += 1
                idx.write(json.dumps(r) + "\n")
                continue
            ok += 1
            with gzip.open(os.path.join(outdir, "solves", f"{r['id']}.json.gz"), "wt") as g:
                json.dump(r, g)
            idx.write(json.dumps({k: r[k] for k in ("id", "board", "stack", "exploitability_pct", "seconds")}) + "\n")
            idx.flush()
            if ok % 100 == 0:
                print(f"{ok} solved, {bad} failed", flush=True)
    print(f"done: {ok} solved, {bad} failed", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), *(int(x) for x in a[2:3]))
