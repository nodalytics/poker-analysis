"""Search at play time for the 9-max networks (phase B, version 1: rollout search at the root).

At a decision after the flop in a pot of at most three players, the network's own choice is replaced by a search:

1. Ranges. N deals are sampled with the hero's two cards and the board so far fixed (bp_deal_fixed in libbp6.so);
   the opponents' cards and the rest of the board are random. Each sample is weighted by the probability that the
   network would have taken every action the opponents actually took in this hand with those cards (the hand is
   replayed from the start), so the samples follow the opponents' ranges as the network sees them.
2. Values. For each legal action, every sample is played to the end with the network for all players (one rollout
   each, the same samples for every action), and the hero's result is averaged with the weights: Q(action).
3. Policy. "greedy": the action with the highest Q. "mix": the network's own probabilities times exp(Q / tau), a
   one-step policy improvement that keeps the network's mixing when the values are close (tau in big blinds).

This is a one-step improvement against opponents who play the network's strategy, not a re-solve of the subgame.

Since 2026-11-23 (B3 of docs/2026-11-23-review-and-plan.md, checked by ninemax_search_check.py):
- the samples and the match tables carry their board, so a texture network (--texture) sees it in search too;
- rollouts are scored with the trainer's own reward (ninemax_selfplay.end_equity), the knockout term included;
- common random numbers: the rollouts draw one uniform per (sample, step), shared by every candidate move (and
  continuation), and sample by inverse CDF, so Q(a) - Q(b) is a paired comparison (--no-crn: the old draws);
- an ESS floor: when the weighted samples rest on fewer than --min-ess of them, the decision is the network's own
  policy (counted in ess_fallback).

--robust (version 2, after Pluribus): the opponents may respond. Every sample is rolled out four times, the
opponents continuing with the network or with the network biased towards folding, calling or raising (that
action type's probabilities times --bias, renormalised; the hero always continues with the network), and an
action is worth its worst case over the four: Q(a) = min_k Q_k(a). The hero then plays greedy or mix on that value,
so its choice holds up when the opponents adapt instead of only against the network's own play.

--opp-bias tests that: the eight opponents in the match play one of the biased strategies (fold, call, raise)
instead of the network; compare with --samples 0 under the same --opp-bias, the plain network's result there.

Evaluation (match): the searching hero in each of the nine seats in turn against eight copies of the plain network,
on the same deals. Without search the hero is the network itself and wins 0 by symmetry, so the result is the gain
from search (bb/100, 95% interval over deals). --samples 0 runs that check.

Tournament networks (ICM): the tables keep their payouts, and every rollout is scored by the change in the hero's
ICM prize equity (in chips at the table's average chip value, then big blinds), exactly as in training; the match
result is in the same unit. --preflop also searches before the flop (the push/fold decisions that dominate at
10-25bb); there every decision is searched, whatever the number of players still in. Needs libbp6.so.

Usage (lab, GPU venv, from research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_search.py blueprint/nm4 --hands 2000 --samples 256 \\
      [--mode greedy|mix] [--tau 1.0] [--out results/ninemax/search_nm4.json]
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ninemax_engine import BB, Dealer, Menu, Tables, strength
from ninemax_eval import load
from ninemax_selfplay import FEAT, board_of, context, end_equity, features, masked, start_equity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELDS = ("stack", "contrib", "folded", "allin", "acted", "bet", "last_raise", "raises", "street", "pre_raises", "hist",
          "aggressor", "to_act", "done", "showdown", "actor")  # fmt: skip
NBOARD = {0: 0, 1: 3, 2: 4, 3: 5}


def take(T, idx):
    """A new Tables holding copies of rows idx of T."""
    U = Tables(len(idx), T.P, T.menu, T.dev, stacks=T.stack[idx])
    for f in FIELDS:
        setattr(U, f, getattr(T, f)[idx].clone())
    if hasattr(T, "pay"):
        U.pay, U.eq0 = T.pay[idx], T.eq0[idx]
    if getattr(T, "board", None) is not None:  # texture runs read the board cards
        U.board = T.board[idx]
    return U


def with_pay(T, pay):
    """Attach tournament payouts (or nothing for cash) and the equity at the start of the hand, as the trainer does
    (start_equity: ICM, plus each player's chip share of the bounties in a knockout game)."""
    if pay is not None:
        T.pay, T.eq0 = pay.float(), start_equity(T.stack, pay)
    return T


def result(T, ranks):
    """Every player's result of finished hands in big blinds: chips (cash) or the prize-equity change in chips at the
    table's average chip value (tournament), exactly the trainer's reward (end_equity: ICM and, with bounties, the
    knockout term)."""
    u = T.payoffs(ranks)
    if hasattr(T, "pay"):
        u = (end_equity(T, u, ranks) - T.eq0) * T.stack.sum(1, keepdim=True).float()
    return u.float() / BB


def net_probs(net, dealer, T, g, bk):
    """The network's action probabilities for the player to act at tables g (bk: buckets of every table of T)."""
    i = T.actor[g]
    s = strength(dealer, bk[g], T.street[g]).gather(1, i[:, None]).squeeze(1)
    feat, legal = features(T, g, i)
    lg, _ = net(feat, bk[g, i], T.street[g], s)
    return F.softmax(masked(lg, legal), 1), legal


BIASES = ("net", "fold", "call", "raise")


def biased(pr, kind, factor):
    """The network's probabilities pr [n, A] tilted towards one action type (fold slot 0, call slot 1, raise slots
    2+), renormalised; kind [n] indexes BIASES (0: unchanged)."""
    m = torch.ones_like(pr)
    m[kind == 1, 0] = factor
    m[kind == 2, 1] = factor
    m[kind == 3, 2:] = factor
    q = pr * m
    return q / q.sum(1, keepdim=True).clamp(min=1e-12)


def deal_fixed(dealer, n, fixed):
    """n deals keeping the cards of fixed (np.int32 [2P + 5], -1 = deal): buckets [n, P, 4], ranks [n, P] and the
    cards [n, 2P + 5] (hole cards by seat, then the board: texture networks read the board)."""
    P = dealer.P
    bk = np.zeros((n, P, 4), np.int32)
    rk = np.zeros((n, P), np.int32)
    cd = np.zeros((n, 2 * P + 5), np.int32)
    dealer.seed += 1
    fx = np.ascontiguousarray(fixed, dtype=np.int32)
    dealer.lib.bp_deal_fixed(n, P, dealer.seed, fx.ctypes.data, bk.ctypes.data, rk.ctypes.data, cd.ctypes.data)
    t = lambda a: torch.tensor(a, device=dealer.dev, dtype=torch.long)  # noqa: E731
    return t(bk), t(rk), cd


def pick(pr, draws, key):
    """Inverse-CDF sampling: one action per row of pr [n, A] from the uniforms draws[key] (key [n]: the row's sample,
    decision * N + deal). Rows with the same key (the same deal under different candidate moves or continuations) get
    the same uniform, so their draws are coupled: that pairs the rollouts of different moves."""
    u = draws[key]
    c = pr.cumsum(1)
    a = (c <= u[:, None] * c[:, -1:]).sum(1)
    last = pr.shape[1] - 1 - (pr > 0).flip(1).float().argmax(1)  # the last action with probability > 0
    return torch.minimum(a, last)


# The ESS floor, as a share of the samples. ESS counts the deals the weighted values effectively rest on, and a value
# from k effective deals has a standard error of about sd / sqrt(k). 0.02 is 5 effective deals of 256 (20 of Kestrel's
# 1,024): conservative, it removes only values resting on a handful of deals. In a 300-deal nm25 match it sent 13.7% of
# searched decisions back to the network (0.05: 29%, 0.1: 43%; ninemax_search_check.py, docs/2026-11-23-review-and-
# plan.md B3). Its effect on play is not measured yet.
MIN_ESS = 0.02


class Search:
    def __init__(self, net, dealer, menu, samples=256, mode="greedy", tau=1.0, chunk=16, robust=False, bias=5.0,
                 crn=True, min_ess=MIN_ESS):  # fmt: skip
        self.net, self.dealer, self.menu = net, dealer, menu
        self.N, self.mode, self.tau, self.chunk = samples, mode, tau, chunk
        self.robust, self.bias = robust, bias
        self.crn = crn  # common random numbers: every candidate move's rollouts use the same draws per (deal, step)
        self.min_ess = min_ess  # below this share of the samples, the decision is the network's own policy
        self.ess = []
        self.calls = 0  # searches so far (seeds the rollout draws with the deals' seed)
        self.fallback = []  # per decision searched: True when the ESS floor returned the network's policy
        self.trace = None  # a list to receive (Q, network policy) of every decision searched (distillation data)
        import ctypes

        dealer.lib.bp_deal_fixed.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64] + [ctypes.c_void_p] * 4

    @torch.no_grad()
    def decide(self, stacks, pay, hists, hero, fixed):
        """Search policies [M, A] for M decisions: stacks [M, P] (chips) or None, payouts [M, P] or None (cash),
        hists[j] = [(seat, action), ...] from the start of the hand, hero seat, fixed [M, 2P + 5] known cards (-1
        unknown)."""
        out = []
        cut = lambda x, sl: None if x is None else x[sl]  # noqa: E731
        for c in range(0, len(hists), self.chunk):
            sl = slice(c, c + self.chunk)
            out.append(self._decide(cut(stacks, sl), cut(pay, sl), hists[sl], hero, fixed[sl]))
        return torch.cat(out)

    def _decide(self, stacks, pay, hists, hero, fixed):
        M, N, P, dev = len(hists), self.N, self.dealer.P, self.dealer.dev
        deals = [deal_fixed(self.dealer, N, fixed[j]) for j in range(M)]
        BK = torch.cat([d[0] for d in deals])  # [M*N, P, 4], rows j*N .. j*N+N-1 belong to decision j
        RK = torch.cat([d[1] for d in deals])
        st = None if stacks is None else stacks.repeat_interleave(N, 0)
        R = with_pay(Tables(M * N, P, self.menu, dev, stacks=st), None if pay is None else pay.repeat_interleave(N, 0))
        R.board = board_of(np.concatenate([d[2] for d in deals]), P, dev)  # each sample's board (texture networks)
        gen = torch.Generator(device=dev)  # the rollouts' draws: one uniform per (sample, step), shared by every move
        gen.manual_seed(self.dealer.seed * 1_000_003 + self.calls)
        self.calls += 1
        # 1. replay the hand, weighting each sample by the opponents' actions
        L = max(len(h) for h in hists)
        seat = torch.full((M, L), -1, dtype=torch.long, device=dev)
        act = torch.zeros((M, L), dtype=torch.long, device=dev)
        for j, h in enumerate(hists):
            if h:
                seat[j, : len(h)] = torch.tensor([x[0] for x in h], device=dev)
                act[j, : len(h)] = torch.tensor([x[1] for x in h], device=dev)
        logw = torch.zeros(M * N, device=dev)
        for t in range(L):
            js = (seat[:, t] >= 0).nonzero().squeeze(1)
            r = (js[:, None] * N + torch.arange(N, device=dev)).reshape(-1)
            s_t, a_t = seat[js, t].repeat_interleave(N), act[js, t].repeat_interleave(N)
            assert bool((R.actor[r] == s_t).all()), "replay diverged from the recorded hand"
            opp = s_t != hero
            if opp.any():
                ro = r[opp]
                pr, _ = net_probs(self.net, self.dealer, R, ro, BK)
                logw[ro] += torch.log(pr.gather(1, a_t[opp][:, None]).squeeze(1).clamp(min=1e-12))
            R.step(r, a_t)
        assert bool((R.actor == hero).all()), "not the hero's turn after the replay"
        w = torch.softmax(logw.view(M, N), 1)  # [M, N]
        ess = 1 / (w**2).sum(1)  # effective sample size of each decision's weighted samples
        self.ess.extend(ess.tolist())
        root = torch.arange(M, device=dev) * N
        pnet, legal = net_probs(self.net, self.dealer, R, root, BK)  # hero's own bucket is the same in every sample
        # 2. every legal action on every sample, then rollouts with the network
        pairs = legal.nonzero()  # [K, 2]: (decision, action)
        idx = (pairs[:, 0:1] * N + torch.arange(N, device=dev)).reshape(-1)
        C = len(BIASES) if self.robust else 1  # the opponents' continuation strategies
        idx = idx.repeat(C)  # block c of the rows: continuation c
        kind = torch.arange(C, device=dev).repeat_interleave(len(idx) // C)
        U = take(R, idx)
        BU, RU = BK[idx], RK[idx]
        U.step(torch.arange(len(idx), device=dev), pairs[:, 1].repeat_interleave(N).repeat(C))
        while not bool(U.done.all()):
            g = (~U.done).nonzero().squeeze(1)
            pr, _ = net_probs(self.net, self.dealer, U, g, BU)
            if self.robust:
                opp = U.actor[g] != hero
                pr[opp] = biased(pr[opp], kind[g][opp], self.bias)
            if self.crn:  # rollout row k plays sample idx[k] of R: the same uniform for every move and continuation
                draws = torch.rand(M * N, generator=gen, device=dev)
                U.step(g, pick(pr, draws, idx[g]))
            else:
                U.step(g, torch.multinomial(pr, 1).squeeze(1))
        u = result(U, RU)[:, hero].view(C, -1, N)  # [C, K, N] in big blinds (ICM equity for a tournament)
        q = (u * w[pairs[:, 0]][None]).sum(2).min(0).values  # the worst continuation for the hero
        Q = torch.full(legal.shape, -1e9, device=dev)
        Q[pairs[:, 0], pairs[:, 1]] = q
        if self.trace is not None:
            self.trace.append((Q.clone(), pnet.clone()))
        if self.mode == "greedy":
            out = F.one_hot(Q.argmax(1), legal.shape[1]).float()
        else:
            z = torch.log(pnet.clamp(min=1e-12)) + (Q - Q.max(1, keepdim=True).values) / self.tau
            out = torch.softmax(masked(z, legal), 1)
        low = ess < self.min_ess * N  # too few effective samples: the values are noise, keep the network's policy
        self.fallback.extend(low.tolist())
        return torch.where(low[:, None], pnet, out)


def match(net, search, dealer, menu, n, dev, max_live=3, first_street=1, opp_bias=0, bias=5.0):
    """Hero (search, or the network when search is None) in each seat against P-1 copies of the network, same deals.
    Returns hero's bb/100 per deal (mean over seats), and the number of searched decisions."""
    P = dealer.P
    bk, rk, cd = dealer.deal(n, cards=True)
    stacks, pay = context(n, P, dev)
    total = torch.zeros(n, device=dev)
    searched = 0
    for h in range(P):
        T = with_pay(Tables(n, P, menu, dev, stacks=stacks), pay)
        T.board = board_of(cd, P, dev)  # texture networks read the board (None of them: unused)
        hist = [[] for _ in range(n)]
        while not bool(T.done.all()):
            g = (~T.done).nonzero().squeeze(1)
            i = T.actor[g]
            with torch.no_grad():
                probs, _ = net_probs(net, dealer, T, g, bk)
            if opp_bias:  # the opponents play a biased version of the network
                o = i != h
                probs[o] = biased(probs[o], torch.full((int(o.sum()),), opp_bias, device=dev), bias)
            if search is not None:
                small = ((~T.folded[g]).sum(1) <= max_live) | (T.street[g] == 0)
                el = (i == h) & (T.street[g] >= first_street) & small
                if el.any():
                    gj = g[el].tolist()
                    fx = np.full((len(gj), 2 * P + 5), -1, np.int32)
                    for k, t in enumerate(gj):
                        fx[k, 2 * h : 2 * h + 2] = cd[t, 2 * h : 2 * h + 2]
                        nb = NBOARD[int(T.street[t])]
                        fx[k, 2 * P : 2 * P + nb] = cd[t, 2 * P : 2 * P + nb]
                    sel = lambda x: None if x is None else x[g[el]]  # noqa: E731
                    probs[el] = search.decide(sel(stacks), sel(pay), [hist[t] for t in gj], h, fx)
                    searched += len(gj)
            a = torch.multinomial(probs, 1).squeeze(1)
            for t, s, x in zip(g.tolist(), i.tolist(), a.tolist()):
                hist[t].append((s, x))
            T.step(g, a)
        total += result(T, rk)[:, h]
    per = total / P * 100  # bb/100 per deal, averaged over the hero's seats
    return per, searched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--hands", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=256, help="0: no search (the symmetry check, expect 0)")
    ap.add_argument("--mode", choices=("greedy", "mix"), default="greedy")
    ap.add_argument("--tau", type=float, default=1.0, help="mix: temperature in big blinds")
    ap.add_argument("--max-live", type=int, default=3, help="search pots with at most this many players")
    ap.add_argument("--preflop", action="store_true", help="also search before the flop")
    ap.add_argument("--chunk", type=int, default=16, help="decisions searched at once")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--robust", action="store_true", help="the opponents may switch continuation strategy (Pluribus)")
    ap.add_argument("--bias", type=float, default=5.0, help="how strongly a continuation strategy is biased")
    ap.add_argument("--opp-bias", choices=BIASES, default="net", help="the match's opponents play this strategy")
    ap.add_argument("--min-ess", type=float, default=MIN_ESS,
                    help="below this share of --samples effective samples, play the network's own policy (0: never)")  # fmt: skip
    ap.add_argument("--no-crn", action="store_true", help="independent rollout draws per move (before 2026-11-23)")
    ap.add_argument("--out")
    ap.add_argument(
        "--per-deal", help="save each deal's result (bb/100, .npy) for paired comparisons on the same deals"
    )
    ap.add_argument("--lib", default=os.path.join(ROOT, "blueprint", "libbp6.so"))
    ap.add_argument("--abs", default=os.path.join(ROOT, "blueprint", "out"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B = int(open(os.path.join(a.abs, "meta.txt")).read().split()[1])
    net, hands = load(a.run, B, dev)
    d = os.path.dirname(a.run) if a.run.endswith(".pt") else a.run
    menu = Menu(json.load(open(os.path.join(d, "config.json"))).get("menu", "medium"), dev)
    dealer = Dealer(a.lib, a.abs, 9, B, dev, seed=a.seed)
    search = None
    if a.samples:
        search = Search(net, dealer, menu, a.samples, a.mode, a.tau, a.chunk, a.robust, a.bias, not a.no_crn, a.min_ess)
    t0 = time.time()
    per, searched = match(net, search, dealer, menu, a.hands, dev, a.max_live, 0 if a.preflop else 1,
                          BIASES.index(a.opp_bias), a.bias)  # fmt: skip
    if a.per_deal:
        np.save(a.per_deal, per.cpu().numpy())
    m, sd = float(per.mean()), float(per.std())
    lo, hi = m - 1.96 * sd / math.sqrt(len(per)), m + 1.96 * sd / math.sqrt(len(per))
    res = {"run": a.run, "hands_trained": hands, "hands": a.hands, "samples": a.samples, "mode": a.mode,
           "tau": a.tau, "robust": a.robust, "crn": not a.no_crn, "min_ess": a.min_ess, "bias": a.bias, "opp_bias": a.opp_bias, "max_live": a.max_live, "preflop": a.preflop, "tourney": FEAT["tourney"], "searched_decisions": searched, "bb100": round(m, 1),
           "ci95": [round(lo, 1), round(hi, 1)], "seconds": round(time.time() - t0)}  # fmt: skip
    if search is not None and search.ess:
        e = torch.tensor(search.ess)
        res["ess_median"] = round(float(e.median()), 1)  # out of --samples
        res["ess_p10"] = round(float(torch.quantile(e, 0.1)), 1)
        res["ess_fallback"] = round(sum(search.fallback) / len(search.fallback), 4)  # share played by the network
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
