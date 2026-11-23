"""Check the 9-max search fixes (B3 of docs/2026-11-23-review-and-plan.md); each part fails on the search before them.

1. board   a texture network (nm41) sees each sample's board in search and the real board in match; a network
           without texture inputs (nm25) gets the same values with or without the board.
2. bounty  search scores a finished hand exactly as the trainer rewards it (play(): end_equity, knockout term
           included) on a PKO model (nm30); on a non-bounty model (nm25) the score is bit-identical to the old ICM one.
3. crn     every candidate move's rollout of a deal draws the same uniform at each step (checked on the draws), and
           the variance of Q(a) - Q(b) over repeated searches of the same spots, with the same deals (rollout noise)
           and with fresh deals (all of search's noise), against independent draws (printed, not gated: run on the
           old search, where both arms are the same algorithm, it shows how far the reading moves by chance).
4. ess     below the ESS floor the decision is the network's own policy (counted in Search.fallback); with the floor
           off it is the greedy argmax. Prints the ESS distribution, the fallback rate and how stable the argmax is
           by ESS (what the default floor rests on).

Lab, CPU (torch and the abstraction live there), from a copy of research/ with blueprint/ linked:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 .venv-gpu/bin/python scripts/ninemax_search_check.py [--spots 12]
Exits 1 on a failure.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

import ninemax_search as ns
import ninemax_selfplay as sp
from icm import end_stacks, icm
from ninemax_engine import Dealer, Menu, Tables
from ninemax_eval import load
from ninemax_selfplay import context, play, rule_probs, set_mode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILS.append(what)


def model(name, dealer, B, dev):
    run = os.path.join(ROOT, "blueprint", name)
    net, _ = load(run, B, dev)  # sets the run's mode (FEAT)
    menu = Menu(json.load(open(os.path.join(run, "config.json"))).get("menu", "medium"), dev)
    return net, menu


class Recorder:
    """Stands in for Search in match(): records every searched spot and plays a cheap search there."""

    def __init__(self, inner):
        self.inner, self.spots = inner, []

    def decide(self, stacks, pay, hists, hero, fixed):
        for k in range(len(hists)):
            cut = lambda x: None if x is None else x[k : k + 1].clone()  # noqa: E731
            self.spots.append((cut(stacks), cut(pay), list(hists[k]), hero, fixed[k : k + 1].copy()))
        return self.inner.decide(stacks, pay, hists, hero, fixed)


def spots_of(net, dealer, menu, dev, n, hands):
    """At least n post-flop spots (at most three players) from matches of the network against itself."""
    rec = Recorder(ns.Search(net, dealer, menu, 16))
    while len(rec.spots) < n:
        ns.match(net, rec, dealer, menu, hands, dev)
    return rec.spots[:n]


def repeat_search(search, spot, R):
    """R searches of one spot (fresh deals each): Q [R, A], the network's policy [R, A], the output [R, A]."""
    stacks, pay, hist, hero, fx = spot
    rep = lambda x: None if x is None else x.repeat(R, 1)  # noqa: E731
    search.trace = []
    out = search.decide(rep(stacks), rep(pay), [hist] * R, hero, np.repeat(fx, R, 0))
    Q = torch.cat([q for q, _ in search.trace])
    pn = torch.cat([p for _, p in search.trace])
    search.trace = None
    return Q, pn, out


def part_board(dealer, B, dev, a):
    print("1. board", flush=True)
    net, menu = model("nm41", dealer, B, dev)
    spots = spots_of(net, dealer, menu, dev, 4, 64)
    seen = []
    real = sp.board_texture

    def spy(board, g, street, d):
        seen.append(None if board is None else (board, street.clone()))
        return real(board, g, street, d)

    sp.board_texture = spy
    try:
        P = dealer.P
        search = ns.Search(net, dealer, menu, 64)
        for stacks, pay, hist, hero, fx in spots:
            seen.clear()
            search.decide(stacks, pay, [hist], hero, fx)
            nb = int((fx[0, 2 * P :] >= 0).sum())
            boards = [s for s in seen if s is not None]
            check(len(seen) > 0 and len(boards) == len(seen), f"search, {nb}-card board: every features call has a "
                  f"board ({len(boards)} of {len(seen)})")  # fmt: skip
            if boards:
                b = boards[0][0]
                fixed = torch.as_tensor(fx[0, 2 * P : 2 * P + nb], device=b.device)
                same = bool((b[:, :nb] == fixed).all())
                varied = len({tuple(r.tolist()) for r in b[:, nb:]}) > 1
                check(same and varied, f"search: the samples keep the real {nb} board cards and deal the rest")
        seen.clear()
        ns.match(net, None, dealer, menu, 32, dev)
        late = [s for s in seen if s is None or bool((s[1] > 0).any())]
        check(all(s is not None for s in late), f"match: the board reaches the texture inputs ({len(late)} calls)")
    finally:
        sp.board_texture = real
    # a network without texture inputs: the same values with or without the board
    net, menu = model("nm25", dealer, B, dev)
    spots = spots_of(net, dealer, menu, dev, 4, 64)
    search = ns.Search(net, dealer, menu, 64)
    real_of = getattr(ns, "board_of", None)
    qs = []
    for strip in (False, True):
        dealer.seed, search.calls = 777, 0  # the same deals and rollout draws
        if strip:
            ns.board_of = lambda cd, P, dev: None  # noqa: E731
        try:
            qs.append(torch.cat([repeat_search(search, s, 1)[0] for s in spots]))
        finally:
            ns.board_of = real_of
    check(torch.equal(qs[0], qs[1]), "nm25 (no texture inputs): Q bit-identical with and without the board")


def station_hands(dealer, menu, dev, n):
    """n hands of calling stations (deterministic): the finished tables, ranks and the context."""
    P = dealer.P
    bk, rk, _ = dealer.deal(n, cards=True)
    stacks, pay = context(n, P, dev)
    T = ns.with_pay(Tables(n, P, menu, dev, stacks=stacks), pay)
    while not bool(T.done.all()):
        g = (~T.done).nonzero().squeeze(1)
        i = T.actor[g]
        _, legal = sp.features(T, g, i)
        T.step(g, rule_probs("station", T, g, i, legal, None, dealer, bk).argmax(1))
    return T, bk, rk, stacks, pay


def old_result(T, rk):
    """Search's score before the fix: plain ICM on the payouts tensor (bounty columns ignored)."""
    u = T.payoffs(rk)
    eq0 = icm(T.stack.float(), T.pay).float()
    eq1 = icm(end_stacks(T.stack, u), T.pay).float()
    return ((eq1 - eq0) * T.stack.sum(1, keepdim=True).float()).float() / ns.BB


def part_bounty(dealer, B, dev, a):
    print("2. bounty", flush=True)
    n = 4000
    for name, pko in (("nm30", True), ("nm25", False)):
        net, menu = model(name, dealer, B, dev)
        torch.manual_seed(3)
        T, bk, rk, stacks, pay = station_hands(dealer, menu, dev, n)
        mine = ns.result(T, rk)
        u, _ = play(dealer, menu, ["station"] * dealer.P, bk, rk, dev=dev, ctx=(stacks, pay))
        train = u.float() / ns.BB
        gap = float((mine - train).abs().max())
        old = old_result(T, rk)
        old_gap = float((old - train).abs().max())
        busts = int(((T.stack > 0) & (T.stack.double() + T.payoffs(rk) <= 0.5)).any(1).sum())
        print(f"  {name}: {n} station hands ({busts} with a bust): |search - trainer| max {gap:.2e} bb; "
              f"old search score max |gap| {old_gap:.3f} bb, mean {float((old - train).abs().mean()):.4f} bb")  # fmt: skip
        check(gap < 1e-4, f"{name}: search's score equals the trainer's reward")
        if not pko:
            check(torch.equal(mine, old), f"{name} (no bounties): the score is bit-identical to the old ICM score")


def pair_vars(Qs):
    """Per move pair of one spot, over repeated searches Q [R, A]: var(Q(a) - Q(b))."""
    legal = (Qs[0] > -1e8).nonzero().squeeze(1).tolist()
    return [float((Qs[:, x] - Qs[:, y]).var()) for k, x in enumerate(legal) for y in legal[k + 1 :]]


def part_crn_ess(dealer, B, dev, a):
    print("3. crn and 4. ess", flush=True)
    net, menu = model("nm25", dealer, B, dev)
    spots = spots_of(net, dealer, menu, dev, a.spots, 128)
    N, R = a.samples, a.repeats
    var = {}
    ess_spot, agree_spot = [], []
    for crn in (False, True):
        search = ns.Search(net, dealer, menu, N)
        search.crn, search.min_ess = crn, 0.0
        fresh, fixed = [], []
        dealer.seed = 9001  # the same deals for both arms
        for k, s in enumerate(spots):
            ess0 = len(search.ess)
            Q, _, _ = repeat_search(search, s, R)  # fresh deals every search: all of search's noise
            fresh += pair_vars(Q)
            if crn:
                best = Q.argmax(1)
                agree_spot.append(float((best == best.mode().values).float().mean()))
                ess_spot.append(float(np.mean(search.ess[ess0:])) / N)
            Qf = []
            for r in range(R):  # the same deals every search: only the rollouts' draws vary
                dealer.seed = 50_000 + k
                Qf.append(repeat_search(search, s, 1)[0])
            fixed += pair_vars(torch.cat(Qf))
        var[crn] = (np.array(fresh), np.array(fixed))
    print(f"  {len(spots)} spots x {R} searches, {N} samples, {len(var[True][0])} move pairs; var of Q(a) - Q(b):")
    for k, what in ((1, "same deals (rollout noise)"), (0, "fresh deals (all noise)   ")):
        v0, v1 = var[False][k], var[True][k]
        r = v1 / np.maximum(v0, 1e-12)
        print(f"    {what}: mean independent {v0.mean():.4f}, crn {v1.mean():.4f} bb^2 (ratio of means "
              f"{v1.mean() / v0.mean():.3f}); per pair ratio median {np.median(r):.3f}, geometric mean "
              f"{np.exp(np.log(np.maximum(r, 1e-12)).mean()):.3f}, lower in {np.mean(v1 < v0):.0%} of pairs")  # fmt: skip
    # the pairing itself: at every rollout step, every candidate move's row of a deal draws the same uniform
    calls = []
    real = getattr(ns, "pick", None)

    def spy(pr, draws, key):
        calls.append((draws.clone(), key.clone()))
        return real(pr, draws, key)

    search = ns.Search(net, dealer, menu, 64)
    moves = 0
    if real is not None:
        ns.pick = spy
    try:
        for s in spots[:4]:
            Q, _, _ = repeat_search(search, s, 1)
            moves = max(moves, int((Q[0] > -1e8).sum()))
    finally:
        if real is not None:
            ns.pick = real
    shared = sum(int(len(k) - len(torch.unique(k))) for _, k in calls)  # rows that reuse another move's draw
    check(len(calls) > 0 and shared > 0 and all(len(d) == 64 for d, _ in calls),
          f"crn: one uniform per (deal, step) shared by every move ({len(calls)} steps, {shared} rows reuse a draw, "
          f"up to {moves} moves)")  # fmt: skip
    # ESS: distribution, fallback and argmax stability by ESS
    e, g = np.array(ess_spot), np.array(agree_spot)
    q = np.quantile(e, [0.1, 0.25, 0.5, 0.75])
    print(f"  ESS share of the samples per spot: p10 {q[0]:.3f}, p25 {q[1]:.3f}, median {q[2]:.3f}, p75 {q[3]:.3f}")
    for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.2), (0.2, 1.01)):
        m = (e >= lo) & (e < hi)
        if m.any():
            print(f"    ESS share {lo:.2f}-{hi:.2f}: {int(m.sum())} spots, argmax agreement across repeats "
                  f"{g[m].mean():.2f}")  # fmt: skip
    search = ns.Search(net, dealer, menu, N)
    if not hasattr(search, "fallback"):
        check(False, "ess: search has no fallback (the floor is not acted on)")
        return
    mid = float(np.median(e))  # about half of the decisions fall back
    for floor, want in ((ns.MIN_ESS, None), (mid, None), (0.0, False), (1.01, True)):
        search.min_ess, search.fallback, search.ess = floor, [], []
        dealer.seed = 31337
        Qs, pns, outs = zip(*[repeat_search(search, s, 2) for s in spots])
        Q, pn, out = torch.cat(Qs), torch.cat(pns), torch.cat(outs)
        fb = torch.tensor(search.fallback)
        greedy = torch.nn.functional.one_hot(Q.argmax(1), Q.shape[1]).float()
        right = torch.where(fb[:, None].to(out.device), pn, greedy)
        ok = torch.equal(out, right) and (want is None or bool((fb == want).all()))
        low = float(np.mean(np.array(search.ess) < floor * N))
        if not ok or abs(low - float(fb.double().mean())) >= 1e-9:
            print(f"    debug: equal {torch.equal(out, right)}, low {low}, fb {fb.tolist()}, ess {search.ess}")
        check(ok and abs(low - float(fb.double().mean())) < 1e-9,
              f"ess floor {floor}: fallback rate {float(fb.float().mean()):.3f} over {len(fb)} decisions; "
              f"fallen-back decisions play the network's policy, the rest the greedy argmax")  # fmt: skip
    # the fallback rate in play: search in match() at the default floor
    search = ns.Search(net, dealer, menu, N)
    ns.match(net, search, dealer, menu, a.hands, dev)
    e = np.array(search.ess) / N
    q = np.quantile(e, [0.01, 0.05, 0.1, 0.5])
    print(f"  match, {a.hands} deals x 9 seats: {len(e)} decisions searched; ESS share p1 {q[0]:.3f}, p5 {q[1]:.3f}, "
          f"p10 {q[2]:.3f}, median {q[3]:.3f}; below the floor {ns.MIN_ESS}: {np.mean(search.fallback):.2%} "
          f"(at 0.05: {np.mean(e < 0.05):.2%}, at 0.1: {np.mean(e < 0.1):.2%})")  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", type=int, default=16)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=16)
    ap.add_argument("--hands", type=int, default=300, help="the ESS survey's match")
    ap.add_argument("--parts", default="board,bounty,crn")
    ap.add_argument("--lib", default=os.path.join(ROOT, "blueprint", "libbp7.so"))
    ap.add_argument("--abs", default=os.path.join(ROOT, "blueprint", "out"))
    a = ap.parse_args()
    dev = "cpu"
    torch.manual_seed(0)
    B = int(open(os.path.join(a.abs, "meta.txt")).read().split()[1])
    set_mode(tourney=True)
    dealer = Dealer(a.lib, a.abs, 9, B, dev, seed=4242)
    parts = {"board": part_board, "bounty": part_bounty, "crn": part_crn_ess}
    for p in a.parts.split(","):
        parts[p](dealer, B, dev, a)
    print("all search checks pass" if not FAILS else "FAIL:\n  " + "\n  ".join(FAILS))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
