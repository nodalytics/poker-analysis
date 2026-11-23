"""Check the knockout (PKO) bounty awards of the trainer (B2 of docs/2026-11-23-review-and-plan.md): each busted
player's bounty goes, half in cash, to the winner(s) of the pot that took the player's last chips, split equally
between tied winners. Before the fix the whole hand's biggest winner (u.argmax) collected every bounty, which is wrong
with ties and with side pots, for example: B wins the main pot and busts A while C wins a bigger side pot from D; A's
bounty belongs to B, not C.

Plays thousands of random tables to the end with the real engine (unequal stacks so all-ins make side pots, ranks
drawn from three values so hands tie, several busts per table, with and without the big-blind ante), settles each one
again here by brute force, written separately and simply, and compares ninemax_selfplay.knockouts with it.

  python3 scripts/ninemax_pko_check.py          (CPU, under a minute; exits 1 on a failure)
"""

import sys
from fractions import Fraction

import torch

from ninemax_engine import BB, RULES, Menu, Tables
from ninemax_selfplay import knockouts

P, G = 9, 3000


def settle(stack, contrib, folded, dead, rank, bounty):
    """One finished table by hand: each player's net chips, who busts, and the bounty cash each player collects.

    The pot is cut at every distinct contribution: the slice (lo, hi] holds min(c, hi) - min(c, lo) chips from each
    player, and the best live hand among the players who put in at least hi wins it (tied hands split it). A slice
    with no live contender goes to the best live hand at the table. The big blind's dead ante goes to the best live
    hand. A player busts when nothing is left; its last chip lies in the slice that ends at its contribution, so that
    slice's winners share its bounty."""
    n = len(stack)
    live = [stack[j] > 0 and not folded[j] for j in range(n)]

    def best(among):
        top = min(rank[j] for j in among)
        return [j for j in among if rank[j] == top]

    win = [Fraction(0)] * n
    slice_winners = {}
    lo = 0
    for hi in sorted(set(c for c in contrib if c > 0)):
        chips = sum(min(c, hi) - min(c, lo) for c in contrib)
        contenders = [j for j in range(n) if live[j] and contrib[j] >= hi]
        ws = best(contenders or [j for j in range(n) if live[j]])
        for j in ws:
            win[j] += Fraction(chips, len(ws))
        slice_winners[hi] = ws
        lo = hi
    if dead:
        ws = best([j for j in range(n) if live[j]])
        for j in ws:
            win[j] += Fraction(dead, len(ws))
    net = [win[j] - contrib[j] - (dead if j == 1 else 0) for j in range(n)]
    busted = [stack[j] > 0 and stack[j] + net[j] <= Fraction(1, 2) for j in range(n)]
    cash = [Fraction(0)] * n
    split = 0
    for i in range(n):
        if busted[i]:
            ws = slice_winners[contrib[i]]
            split += len(ws) > 1
            for j in ws:
                cash[j] += bounty[i] / 2 / len(ws)
    return net, busted, cash, split


def random_tables(ante, gen):
    RULES["ante"] = ante
    menu = Menu("mtt", "cpu")
    stacks = (2 * 50 ** torch.rand(G, P, generator=gen) * BB).round().long()
    seated_n = torch.randint(2, P + 1, (G, 1), generator=gen)  # some short tables
    seat = torch.arange(P)[None, :]
    stacks = torch.where((seat < 2) | (seat >= P - (seated_n - 2)), stacks, torch.zeros_like(stacks))
    T = Tables(G, P, menu, "cpu", stacks=stacks)
    weights = torch.tensor([1.0, 4.0] + [1.0] * menu.S + [2.0])  # all-in-heavy: many busts and side pots
    while not bool(T.done.all()):
        g = (~T.done).nonzero().squeeze(1)
        legal, _ = T.legal(g, T.actor[g])
        p = legal.float() * weights
        T.step(g, torch.multinomial(p / p.sum(1, keepdim=True), 1, generator=gen).squeeze(1))
    ranks = torch.randint(0, 3, (G, P), generator=gen, dtype=torch.int32)  # three hand values: many ties
    return T, ranks


def named_case():
    """B (seat 3) wins the main pot and busts A (seat 2); C (seat 0) wins a bigger side pot and busts D (seat 1)."""
    RULES["ante"] = 0.0
    T = Tables(1, 4, Menu("mtt", "cpu"), "cpu", stacks=torch.tensor([[12000, 10000, 1000, 1500]]))
    T.contrib[:] = torch.tensor([[10000, 10000, 1000, 1500]])
    T.folded[:] = False
    T.done[:] = True
    ranks = torch.tensor([[1, 2, 3, 0]], dtype=torch.int32)
    return T, ranks


def compare(name, T, ranks, gen):
    bty = torch.rand(T.G, T.P, generator=gen, dtype=torch.float64).float() * (T.stack > 0)
    u = T.payoffs(ranks)
    busted, k = knockouts(T, u, ranks, bty)
    bad_net = bad_bust = bad_cash = 0
    nbust = tied = side = 0
    worst = 0.0
    for t in range(T.G):
        net, bust, cash, split = settle(T.stack[t].tolist(), T.contrib[t].tolist(), T.folded[t].tolist(), int(T.dead[t]),
                                 ranks[t].tolist(), [Fraction(float(x)) for x in bty[t]])  # fmt: skip
        bad_net += any(abs(float(net[j]) - float(u[t, j])) > 0.01 for j in range(T.P))  # payoffs are float32
        bad_bust += bust != busted[t].tolist()
        err = max(abs(float(cash[j]) - float(k[t, j])) for j in range(T.P))
        worst = max(worst, err)
        bad_cash += err > 1e-5
        nbust += sum(bust)
        side += len(set(c for c in T.contrib[t].tolist() if c > 0)) > 1 and sum(bust) > 0
        tied += split
    print(f"{name}: {T.G} tables, {nbust} busts, {side} tables with a bust and side pots, {tied} bounties split "
          f"between tied winners; settlement mismatches "
          f"{bad_net}, bust mismatches {bad_bust}, bounty mismatches {bad_cash} (worst {worst:.4f})")  # fmt: skip
    fails = []
    if bad_net:
        fails.append(f"{name}: Tables.payoffs disagrees with the brute force on {bad_net} tables")
    if bad_bust:
        fails.append(f"{name}: the busted players disagree on {bad_bust} tables")
    if bad_cash:
        fails.append(f"{name}: the bounty awards disagree on {bad_cash} of {T.G} tables")
    return fails


def main():
    gen = torch.Generator().manual_seed(0)
    fails = []
    T, ranks = named_case()
    fails += compare("named case (B busts A in the main pot, C busts D in a bigger side pot)", T, ranks, gen)
    for ante in (0.0, 1.0):
        T, ranks = random_tables(ante, gen)
        fails += compare(f"random tables, ante {ante:g}bb", T, ranks, gen)
    RULES["ante"] = 0.0
    print("all PKO checks pass" if not fails else "FAIL:\n  " + "\n  ".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
