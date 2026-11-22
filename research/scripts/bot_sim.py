"""Bot-only poker simulation at scale: the solver stack alone against the other bots, with no language model.

Usage: uv run python scripts/bot_sim.py OUTDIR SESSIONS [--short] [--workers N] [SEAT ...]
Default seats: solverbot (charts + flop model + TexasSolver, Monte Carlo elsewhere), sol, monte, tina, tinaplus,
bob, max, carl, nora. Each session is 24 hands of poker_selfplay at 1/2 blinds; seats rotate every session.
--short deals every hand 5-20 big blinds (chips then track net winnings); otherwise 100 big blinds and busted
players sit out the rest of the session. Sessions run in parallel worker processes.

Writes OUTDIR/sessions.jsonl (chips per entrant per session) and OUTDIR/summary.json: per entrant, big blinds won
per 100 hands with a bootstrap 95% interval over sessions, and the solver call counts.
"""

import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poker_selfplay as ps  # noqa: E402

HANDS = int(os.environ.get("SIM_HANDS", 24))  # hands per session
# SIM_STACKS="200,200": deal every hand fresh stacks of 200 big blinds (a cash game at that depth, e.g. the blueprint)
STACKS = tuple(int(x) for x in os.environ["SIM_STACKS"].split(",")) if os.environ.get("SIM_STACKS") else None
SEATS = ["solverbot", "sol", "monte", "tina", "tinaplus", "bob", "max", "carl", "nora"]


def one(args):
    outdir, k, seats, short = args
    seats = seats[k % len(seats) :] + seats[: k % len(seats)]
    chips, _ = ps.main(
        os.path.join(outdir, f"session{k:04d}"), HANDS, seats, short_stack=STACKS or ((5, 20) if short else None)
    )
    return k, chips, dict(ps.pa.stats)


def main(outdir, sessions, short, workers, seats):
    os.makedirs(outdir, exist_ok=True)
    rows, calls = [], {}
    with ProcessPoolExecutor(workers) as pool, open(os.path.join(outdir, "sessions.jsonl"), "a") as log:
        futs = [pool.submit(one, (outdir, k, seats, short)) for k in range(1, sessions + 1)]
        for f in as_completed(futs):
            k, chips, st = f.result()
            rows.append(chips)
            log.write(json.dumps({"session": k, "chips": chips}) + "\n")
            log.flush()
            for key, v in st.items():
                calls[key] = calls.get(key, 0) + v
    rng, out = random.Random(1), {}
    for e in seats:
        # net big blinds per 100 hands: chips minus the 200 start, over 24 hands, at a 2-chip big blind
        x = [(r[e] - ps.START) / 2 / HANDS * 100 for r in rows if e in r]
        m = sum(x) / len(x)
        boot = sorted(sum(rng.choice(x) for _ in x) / len(x) for _ in range(2000))
        out[e] = {"sessions": len(x), "bb_per_100": round(m, 1), "ci95": [round(boot[50], 1), round(boot[1949], 1)]}
    json.dump(
        {"short": short, "hands_per_session": HANDS, "entrants": out, "solver_calls": calls},
        open(os.path.join(outdir, "summary.json"), "w"),
        indent=1,
    )
    for e, s in sorted(out.items(), key=lambda x: -x[1]["bb_per_100"]):
        print(f"{e:10} {s['bb_per_100']:+8.1f} bb/100  [{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}]  n={s['sessions']}")


if __name__ == "__main__":
    a = sys.argv[1:]
    short = "--short" in a
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 6
    rest = [x for i, x in enumerate(a) if x != "--short" and x != "--workers" and (i == 0 or a[i - 1] != "--workers")]
    main(rest[0], int(rest[1]), short, workers, rest[2:] or SEATS)
