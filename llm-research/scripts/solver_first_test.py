"""Does handing the solver's work to a model make it play better? A paired poker test.

Usage: uv run python scripts/solver_first_test.py OUTDIR SESSIONS MODEL [MODEL ...]

Each session seats every model twice, `model#plain` (no help) and `model#solverfirst` (before every decision it is
told TexasSolver's strategy for its exact hand when the spot is heads-up on the turn or river, otherwise Monte Carlo
equity against the live opponents and the pot odds), plus fixed opponents: Bob, Monte and one rotating personality.
Both seats of a model face the same cards and opponents, so the session-by-session chip difference is a paired
measurement; a one-sided sign test says whether solver-first beats plain.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena  # noqa: E402  (binom_tail, short names)
import poker_selfplay as ps  # noqa: E402

HANDS = 24
SHORT = None  # (5, 20) with --short: every hand dealt 5-20 big blinds


def main(outdir, sessions, models):
    os.makedirs(outdir, exist_ok=True)
    others = [b for b in ps.PERSONAS if b != ps.BOB]
    diffs = {m: [] for m in models}
    log = open(os.path.join(outdir, "test.log"), "a", buffering=1)
    for k in range(1, sessions + 1):
        seats = [f"{m}#{t}" for m in models for t in ("plain", "solverfirst")]
        seats += [ps.BOB, "monte", others[(k - 1) % len(others)]]
        seats = seats[k % len(seats) :] + seats[: k % len(seats)]  # rotate positions every session
        chips, _ = ps.main(os.path.join(outdir, f"session{k:02d}"), HANDS, seats, short_stack=SHORT)
        for m in models:
            diffs[m].append(chips[f"{m}#solverfirst"] - chips[f"{m}#plain"])
        log.write(
            f"session {k}: "
            + ", ".join(f"{arena.cs.short(e)} {chips[e]}" for e in sorted(chips, key=lambda e: -chips[e]))
            + "\n"
        )
    rows = []
    for m in models:
        d = [x for x in diffs[m] if x != 0]
        ahead = sum(x > 0 for x in d)
        rows.append(
            {
                "model": arena.cs.short(m),
                "sessions": len(diffs[m]),
                "solverfirst_ahead": ahead,
                "decided_sessions": len(d),
                "mean_chip_diff": round(sum(diffs[m]) / len(diffs[m]), 1),
                "p_solverfirst_better": round(arena.binom_tail(ahead, len(d)), 3) if d else None,
                "diffs": diffs[m],
            }
        )
    json.dump({"rows": rows, "solver_stats": ps.pa.stats}, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    for r in rows:
        log.write(
            f"{r['model']}: solver-first ahead in {r['solverfirst_ahead']} of {r['decided_sessions']} sessions, "
            f"mean {r['mean_chip_diff']:+} chips, p {r['p_solverfirst_better']}\n"
        )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--short"]
    if "--short" in sys.argv:
        SHORT = (5, 20)
    main(args[0], int(args[1]), args[2:])
