"""ICM (the Independent Chip Model) and tournament situations for the 9-max tournament network (ninemax_selfplay.py
--tourney).

ICM turns chip stacks into shares of the prize money. Malmuth-Harville: a player finishes first with probability
stack / total chips; given who has finished above, the next place goes the same way among those left. A player's
equity is the sum over places of P(that place) x its prize. Exact for up to about 12 players: the probability of
every set of players taking the top places is carried place by place (2^P sets), batched over many tables at once.

Tournament situations (scenarios): one 9-handed table with every seat's stack in big blinds and the prizes for the
table's finishing places 1..P (in units of the prize money at stake at this table, summing to 1):
  final table  places 1-9 all paid, top-heavy (prize ~ place^-alpha);
  bubble       the next 1-3 players to bust get nothing, the rest a top-heavy ladder (as at a final table; the
               chips above the min-cash carry the deep run). Until 2026-11-10 the ladder above was nearly flat,
               which is a satellite (nm7 trained on that);
  in the money all places paid, a gentle ladder: deep in the money, between pay jumps.
Short-handed final tables (short > 0): that share of final tables seats only 2..P-1 players (uniform), the early
seats empty (stack 0, see ninemax_engine.py) and only places 1..n paid. ICM handles the empty seats as they are: a
player with no chips takes no place while anyone has chips.
A hand's reward is the change in the player's ICM equity, in big-blind equivalents: equity change x the table's chips
/ BB. With prizes proportional to chips this is the chip result, so cash play and tournament play share a scale.

Usage (a check against brute force, CPU, seconds): python scripts/icm.py
"""

import itertools

import torch

BB = 100


def icm(stacks, payouts):
    """stacks [G, P] (any non-negative units), payouts [G, P] (prize for place 1..P): ICM equities [G, P]."""
    G, P = stacks.shape
    dev = stacks.device
    s = stacks.double().clamp(min=0)
    M = 1 << P
    code = torch.arange(M, device=dev)
    bits = ((code[:, None] >> torch.arange(P, device=dev)[None, :]) & 1).bool()  # [M, P]: who has placed
    pop = bits.sum(1)
    msum = s @ bits.double().T  # [G, M] chips of the players already placed
    total = s.sum(1, keepdim=True)
    prob = torch.zeros(G, M, dtype=torch.float64, device=dev)
    prob[:, 0] = 1.0
    eq = torch.zeros(G, P, dtype=torch.float64, device=dev)
    pay = payouts.double()
    one = 1 << torch.arange(P, device=dev)
    for k in range(P):  # place k+1 goes to one of the players not yet placed, in proportion to their chips
        ms = (pop == k).nonzero().squeeze(1)
        rem = (total - msum[:, ms]).clamp(min=1e-300)  # [G, Mk]
        free = ~bits[ms]  # [Mk, P]
        p = s[:, None, :] * free[None] / rem[:, :, None]  # [G, Mk, P]
        w = prob[:, ms, None] * p
        eq += w.sum(1) * pay[:, k : k + 1]
        if k + 1 < P:
            nxt = (ms[:, None] | one[None, :]).reshape(1, -1).expand(G, -1)
            prob.scatter_add_(1, nxt, w.reshape(G, -1))  # placed players' entries carry w = 0
    return eq.to(stacks.dtype if stacks.is_floating_point() else torch.float32)


def end_stacks(start, payoff):
    """Stacks after a hand, with busted players kept below everyone else and ordered by their starting stack (the
    usual rule when several bust in one hand): a busted player keeps a vanishing stack proportional to the start."""
    after = start.double() + payoff.double()
    bust = after <= 0.5
    return torch.where(bust, start.double() * 1e-9, after)


def scenarios(G, P, dev, kinds=(0.4, 0.4, 0.2), avg_bb=(10, 60), spread=0.7, gen=None, short=0.0):
    """Random tournament tables: stacks [G, P] in chips (>= 2bb, rounded to the chip), payouts [G, P] summing to 1,
    and the kind of each table (0 final table, 1 bubble, 2 in the money, 3 early: kinds may have a fourth share). avg_bb: the range (LO, HI) of a table's
    average stack in big blinds, log-uniform, or a list of ranges, one picked per table with equal odds (nm10:
    [(5, 15), (20, 50)], push/fold depths and realistic mid-stack depths). short: the share of final tables with
    2..P-1 players (empty seats have stack 0 and the places below the last player pay nothing)."""
    r = lambda *shape: torch.rand(*shape, device=dev, generator=gen)  # noqa: E731
    kind = torch.multinomial(torch.tensor(kinds, device=dev), G, replacement=True, generator=gen)
    ranges = torch.tensor(avg_bb if isinstance(avg_bb[0], (list, tuple)) else [avg_bb], device=dev, dtype=torch.float)
    pick = torch.randint(len(ranges), (G,), device=dev, generator=gen)
    lo, hi = ranges[pick, 0:1], ranges[pick, 1:2]
    avg = lo * (hi / lo) ** r(G, 1)  # log-uniform average stack within the table's range
    z = torch.randn(G, P, device=dev, generator=gen)
    stacks = (avg * torch.exp(spread * z - spread**2 / 2) * BB).clamp(min=2 * BB).round().long()
    place = torch.arange(1, P + 1, device=dev, dtype=torch.float32)[None, :]
    alpha = 0.6 + 0.8 * r(G, 1)
    final = place ** (-alpha)
    busts = torch.randint(1, 4, (G, 1), device=dev, generator=gen)  # the next 1-3 to bust get nothing
    # above the bust spots a top-heavy ladder, as at a final table: the chips still carry the deep run. (A nearly
    # flat ladder here, used by nm7, is a satellite bubble, whose equilibrium is to shove any two cards and never
    # call: see icm_pushfold.py.)
    bubble = torch.where(place > P - busts, torch.zeros_like(final), final)
    money = 1 + (0.3 + 0.7 * r(G, 1)) * (P - place) / P
    pay = torch.where((kind == 0)[:, None], final, torch.where((kind == 1)[:, None], bubble, money))
    # early in a big field, far from the money, chips are money: winner-take-all makes ICM equity exactly the chip share
    pay = torch.where((kind == 3)[:, None], (place == 1).float().expand_as(pay), pay)
    if short > 0:  # short-handed final tables: seat n players, the blinds and the last n-2 seats
        n = torch.where((kind == 0)[:, None] & (r(G, 1) < short),
                        torch.randint(2, P, (G, 1), device=dev, generator=gen), torch.full((G, 1), P, device=dev))  # fmt: skip
        seat = torch.arange(P, device=dev)[None, :]
        stacks = torch.where((seat < 2) | (seat >= P - (n - 2)), stacks, torch.zeros_like(stacks))
        pay = torch.where(place <= n, pay, torch.zeros_like(pay))
    pay = pay / pay.sum(1, keepdim=True)
    return stacks, pay, kind


def _brute(stacks, payouts):
    """Reference: every finishing order, by its Malmuth-Harville probability (one table, small P)."""
    P = len(stacks)
    eq = [0.0] * P
    for order in itertools.permutations(range(P)):
        pr, left = 1.0, float(sum(stacks))
        for j in order:
            pr *= stacks[j] / left if left > 0 else 0.0
            left -= stacks[j]
        for place, j in enumerate(order):
            eq[j] += pr * payouts[place]
    return eq


if __name__ == "__main__":
    torch.manual_seed(0)
    for P in (3, 5, 7):
        st = torch.rand(4, P) * 100 + 1
        pay = torch.sort(torch.rand(4, P), 1, descending=True).values
        got = icm(st, pay)
        for g in range(4):
            ref = _brute(st[g].tolist(), pay[g].tolist())
            assert max(abs(a - b) for a, b in zip(got[g].tolist(), ref)) < 1e-5, (got[g], ref)
    st, pay, kind = scenarios(2000, 9, "cpu")
    eq = icm(st.float(), pay)
    assert torch.allclose(eq.sum(1), torch.ones(2000), atol=1e-5), "equities must add up to the prize pool"
    chip = st.float() / st.sum(1, keepdim=True)  # winner-take-all ICM = chip share
    wta = torch.zeros(2000, 9)
    wta[:, 0] = 1
    assert torch.allclose(icm(st.float(), wta), chip, atol=1e-5)
    st2, pay2, kind2 = scenarios(4000, 9, "cpu", short=0.5)
    n = (st2 > 0).sum(1)
    assert bool(((n == 9) | (kind2 == 0)).all()) and bool((n[kind2 == 0] < 9).any()) and int(n.min()) == 2
    eq2 = icm(st2.float(), pay2)
    assert torch.allclose(eq2.sum(1), torch.ones(4000), atol=1e-5) and bool((eq2[st2 == 0] == 0).all())
    for g in (n < 9).nonzero().squeeze(1)[:20].tolist():  # empty seats change nothing for the seated players
        on = st2[g] > 0
        ref = _brute(st2[g][on].tolist(), pay2[g][: int(on.sum())].tolist())
        assert max(abs(a - b) for a, b in zip(eq2[g][on].tolist(), ref)) < 1e-5
    print("ICM matches brute force (3, 5, 7 players); 9-player equities sum to the pool; winner-take-all = chips;"
          " short-handed tables (empty seats) match brute force over the seated players")  # fmt: skip
    big = st.float().argmax(1)
    print("bubble tables: the chip leader's equity / chip share, mean:",
          round(float((eq / chip)[torch.arange(2000), big][kind == 1].mean()), 3))  # fmt: skip
