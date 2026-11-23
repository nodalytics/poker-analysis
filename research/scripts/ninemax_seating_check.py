"""Check the training seating (B1 of docs/2026-11-23-review-and-plan.md): at short tables and 6-max, the exploiters
and the population sit only where a player sits, the net keeps at least one seat, and PFSP scores only the net's seats
that play. Also shows how often the old seating (randperm over all P seats) put opponents in empty seats.

  python3 scripts/ninemax_seating_check.py          (CPU, a few seconds; exits 1 on a failure)
"""

import sys

import torch

from ninemax_selfplay import context, seated_mask, set_mode, split_seats

P, G = 9, 20000


def old_population(G, P):
    """The seating before the fix: per chunk, the net keeps k random seats of all P and the opponent takes the rest."""
    k = int(torch.randint(1, P, ()).item())
    theirs = torch.randperm(P)[k:]
    out = torch.zeros(G, P, dtype=torch.bool)
    out[:, theirs] = True
    return out


def main():
    torch.manual_seed(0)
    fails = []
    for name, mode in [("tournament, short 0.5", dict(tourney=True, short=0.5)),
                       ("6-max cash, short 0.5", dict(seats=6, short=0.5)),
                       ("100bb cash, full", dict())]:  # fmt: skip
        set_mode(**mode)
        st, _ = context(G, P, "cpu")
        seated = seated_mask(st, G, P, "cpu")
        n = seated.sum(1)
        theirs = split_seats(seated)
        on_empty = int((theirs & ~seated).sum())
        k = theirs.sum(1)
        net_keeps = (seated & ~theirs).sum(1)
        old = torch.cat([old_population(200, P) for _ in range(G // 200)])
        old_empty = float((old & ~seated).sum()) / float(old.sum())
        print(f"{name}: seated per table {n.float().mean():.2f} (min {int(n.min())}); opponent seats on empty seats: "
              f"new {on_empty}, old {old_empty:.1%} of the opponent's seats")  # fmt: skip
        if on_empty:
            fails.append(f"{name}: {on_empty} opponent seats on empty seats")
        if bool((k < 1).any()) or bool((net_keeps < 1).any()):
            fails.append(f"{name}: a table where the net or the opponent has no seat")
        # every split 1 .. n-1 occurs (the old seating's k was uniform over 1 .. P-1)
        for m in sorted(set(n.tolist())):
            got = set(k[n == m].tolist())
            if got != set(range(1, m)):
                fails.append(f"{name}: tables with {m} players split {sorted(got)}, not 1..{m - 1}")
        # one seat per exploiter table, always a seated one
        pick = torch.multinomial(seated.float(), 1).squeeze(1)
        if not bool(seated[torch.arange(G), pick].all()):
            fails.append(f"{name}: an exploiter seat on an empty seat")
    print("all seating checks pass" if not fails else "FAIL:\n  " + "\n  ".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
