"""Power of the learner-minus-twin test: how big an edge would the ladder detect, after how many rounds?

Usage: uv run python scripts/power_sim.py

Simulates the chess ladder's own rules (8 entrants, neighbours paired by rating, Elo K=24, draws as half points)
with one planted learner whose chance of scoring against anyone is 0.5 + edge, everyone else at 0.5. The test reads
learner minus twin Elo after R rounds and calls it real when the gap clears the 95th percentile of the same gap with
no edge. Prints the detection rate per edge and round count, i.e. where the test goes blind.
"""

import random

K, N_ENTRANTS, TRIALS = 24, 8, 2000


def expected(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))


def gap_after(rounds, edge, rng):
    elo = [1000.0] * N_ENTRANTS  # 0 = planted learner, 1 = its twin
    for _ in range(rounds):
        order = sorted(range(N_ENTRANTS), key=lambda i: (-elo[i], rng.random()))
        for a, b in zip(order[0::2], order[1::2]):
            p = 0.5 + (edge if a == 0 else -edge if b == 0 else 0.0)
            r = rng.random()
            sa = 1.0 if r < p * 0.8 else 0.5 if r < p * 0.8 + 0.2 else 0.0  # 20% draws, as in the ladder's chess
            ea = expected(elo[a], elo[b])
            elo[a] += K * (sa - ea)
            elo[b] += K * ((1 - sa) - (1 - ea))
    return elo[0] - elo[1]


def paired_detects(games, edge, rng):
    """Learner plays its own twin once per round; one-sided test on the head-to-head score (draws count half)."""

    def score(e):
        t = 0.0
        for _ in range(games):
            r = rng.random()
            p = 0.5 + e
            t += 1.0 if r < p * 0.8 else 0.5 if r < p * 0.8 + 0.2 else 0.0
        return t

    null = sorted(score(0.0) for _ in range(TRIALS))
    cut = null[int(0.95 * TRIALS)]
    return sum(score(edge) > cut for _ in range(TRIALS // 2)) / (TRIALS // 2)


def main():
    rng = random.Random(1)
    print("edge = extra chance of scoring per game for the learner; cells = share of trials the test detects it")
    rounds_list = (8, 20, 58)
    print("edge   " + "  ".join(f"{r:>3} rounds" for r in rounds_list))
    nulls = {r: sorted(gap_after(r, 0.0, rng) for _ in range(TRIALS)) for r in rounds_list}
    cut = {r: nulls[r][int(0.95 * TRIALS)] for r in rounds_list}
    for edge in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        rates = []
        for r in rounds_list:
            hits = sum(gap_after(r, edge, rng) > cut[r] for _ in range(TRIALS // 2))
            rates.append(hits / (TRIALS // 2))
        print(f"{edge:4.2f}   " + "  ".join(f"{x:9.0%}" for x in rates))
    print("\nDirect learner-vs-twin game every round (paired test):")
    print("edge   " + "  ".join(f"{r:>3} rounds" for r in rounds_list))
    for edge in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        print(f"{edge:4.2f}   " + "  ".join(f"{paired_detects(r, edge, rng):9.0%}" for r in rounds_list))
    print("95th-percentile null gap (Elo): " + ", ".join(f"{r} rounds {cut[r]:.0f}" for r in rounds_list))


if __name__ == "__main__":
    main()
