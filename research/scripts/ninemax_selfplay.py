"""The 9-max neural blueprint: one network learns full-ring no-limit hold'em by playing itself on the GPU.

The neural route (docs/2026-11-09-nine-max-plan.md): a 9-max strategy table does not fit the lab's 125 GB, so the
strategy is a network that reads the public state and the player's card buckets. The game is ninemax_engine.py
(checked against PokerKit by ninemax_check.py): 100bb, blinds 0.5/1, seats SB, BB, UTG ... button, a bet menu with
pot-fraction sizes that shrinks in multiway pots, side pots.

What the network sees at a decision (all public, plus its own cards):
  its seat (position), the street, pot, price, stack and stack-to-pot ratio, raises this street and pre-flop,
  players live / all in / still to act, for every other seat in table order after it (live, all in, chips in,
  acted this round, last aggressor), the legal actions and the size of each menu raise relative to the pot,
  and an embedding of its card bucket on every street so far (pre-flop class, flop, turn, river buckets).

Learning: the regularised actor-critic of mlp_selfplay.py (REINFORCE with a value baseline, a KL pull to a slowly
moving magnet copy, entropy bonus), one network for all seats, every decision of every seat learned from. The
exported policy is a slow moving average of the weights.

--pool-share S: population training ("learn wider"). Pure self-play learns only to play against itself: after 30
minutes the first run beat the rule-based player but barely profited from eight random players who shove 30% of the
time. With S > 0, a share S of every batch is played at tables where the net holds 1-8 random seats and one opponent
from the population holds the others: rule players (reg, tag, maniac, station, random) and the last few checkpoints. The
net learns from its own seats only.

Yardstick (every EVAL_MINUTES): 1-vs-8 duplicate matches. The same deals are replayed with the network in each of
the nine seats and eight copies of an opponent in the others, so seat and card luck cancel; bb/100 with a 95%
interval. Opponents: "tag", a rule-based tight-aggressive player on card strength and pot odds, and the network's
own previous checkpoint (progress against itself).

Usage (lab, GPU venv): .venv-gpu/bin/python scripts/ninemax_selfplay.py OUT [--players 9] [--menu medium]
    [--minutes 1440] [--eval-minutes 30] [--batch 8192]
"""

import argparse
import copy
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from icm import end_stacks, icm, scenarios
from ninemax_engine import BB, RULES, STACK, Dealer, Menu, Tables, read_buckets, strength

# What the network sees and plays for, set once per run (main, or from a run's config.json by the tools that load it):
#   stacks   None: every hand 100bb (cash). (lo, hi): each seat's stack log-uniform in lo..hi big blinds, and the
#            inputs gain every stack and the effective stack.
#   tourney  False: chips. True: tournament tables (icm.scenarios: stacks, payout ladder); the inputs gain the
#            payouts and every player's ICM equity, and the reward is the change in ICM equity (bb equivalents).
#   short    the share of tournament final tables played short-handed, 2-8 players (icm.scenarios; empty seats).
#   seats    cash tables seat at most this many players (the blinds and the last seats; the early seats stay empty),
#            so a 9-seat network learns 6-max; with short, that share of tables seats 2 to seats-1 players.
#   bounty   progressive knockout (PKO) tournaments: this share of the money at stake is in bounties on the players'
#            heads. Knocking a player out pays half their bounty in cash and adds the other half to your own; the
#            payouts tensor then carries both, [G, 2P] (split_pay). 0: no bounties.
FEAT = {"stacks": None, "tourney": False, "depths": None, "short": 0.0, "seats": None, "bounty": 0.0, "texture": False,
        "early": 0.0}  # fmt: skip
PRIORITY = {}  # prioritised scenario sampling: table cell -> weight (ninemax_leaks.py), training tables only


def split_pay(pay):
    """The regular prizes [G, P] and the bounties [G, P] (None without bounties) from a payouts tensor."""
    if pay is None or not FEAT["bounty"] or pay.shape[1] % 2:
        return pay, None
    P = pay.shape[1] // 2
    return pay[:, :P], pay[:, P:]


def start_equity(stacks, pay):
    """Each player's equity at the start of a hand, in prize-money shares: ICM on the regular prizes and, with
    bounties, its chip share of the whole bounty pool. Every bounty is eventually collected by someone (the winner
    collects its own); in proportion to chips is the simple model that keeps the money whole: the equities sum to all
    the prizes and bounties, so the reward is zero-sum. (nm29's version credited only each player's own bounty times
    its chip share, so a knockout created equity: a copy of nm29 beat itself by +90 bb/100.)"""
    reg, bty = split_pay(pay)
    eq = icm(stacks.float(), reg).float()
    if bty is not None:
        share = stacks.float() / stacks.float().sum(1, keepdim=True).clamp(min=1)
        eq = eq + share * bty.sum(1, keepdim=True)
    return eq


def board_of(cd, P, dev):
    """The board cards [G, 5] from a deal made with cards=True (None without)."""
    return None if cd is None else torch.as_tensor(cd[:, 2 * P : 2 * P + 5], dtype=torch.long, device=dev)


def board_texture(board, g, street, dev):
    """Seven numbers about the board cards visible on each table's street (zeros before the flop or without cards):
    the largest suit count, paired, trips or better, distinct ranks, the most distinct ranks in any five-rank window
    (straight potential, the ace also low), the top rank, and the broadway cards. Cards are rank * 4 + suit."""
    n = len(g)
    out = torch.zeros(n, 7, device=dev)
    if board is None:
        return out
    b = board[g]  # [n, 5]
    shown = torch.tensor([0, 3, 4, 5], device=dev)[street]  # cards visible
    vis = torch.arange(5, device=dev)[None, :] < shown[:, None]
    rank, suit = b // 4, b % 4
    rc = torch.zeros(n, 13, device=dev).scatter_add_(1, rank, vis.float())
    sc = torch.zeros(n, 4, device=dev).scatter_add_(1, suit, vis.float())
    present = (rc > 0).float()
    wheel = torch.cat([present[:, 12:13], present], 1)  # the ace low as well
    windows = wheel.unfold(1, 5, 1).sum(2).max(1).values
    top = torch.where(vis, rank, torch.full_like(rank, -1)).max(1).values.float()
    on = (shown > 0).float()
    out[:, 0] = sc.max(1).values / 5
    out[:, 1] = (rc >= 2).any(1).float()
    out[:, 2] = (rc >= 3).any(1).float()
    out[:, 3] = present.sum(1) / 5
    out[:, 4] = windows / 5
    out[:, 5] = top.clamp(min=0) / 12
    out[:, 6] = (rc[:, 8:].sum(1)) / 5
    return out * on[:, None]


def features(T, g, i):
    """Public features of tables g for actor i: [len(g), NF]."""
    P = T.P
    c = T.contrib[g].float()
    pot = c.sum(1) + T.dead[g].float()  # the big-blind ante is in the pot (0 without an ante)
    mine = c.gather(1, i[:, None]).squeeze(1)
    bet = T.bet[g].float()
    to_call = bet - mine
    stk = T.avail[g]  # chips each player can put in this hand (the start stack, less the big blind's ante)
    left = stk.gather(1, i[:, None]).squeeze(1) - mine
    live, allin, acted = ~T.folded[g], T.allin[g], T.acted[g]
    f = [
        F.one_hot(i, P).float(),
        F.one_hot(T.street[g], 4).float(),
        (pot / (P * STACK))[:, None],
        (to_call / (pot + to_call).clamp(min=1))[:, None],
        (mine / STACK)[:, None],
        (left / STACK)[:, None],
        ((left / pot.clamp(min=1)).clamp(max=20) / 20)[:, None],
        (T.raises[g].float() / 4)[:, None],
        (T.pre_raises[g].float() / 4)[:, None],
        (live.sum(1).float() / P)[:, None],
        (allin.sum(1).float() / P)[:, None],
        (T.needs()[g].sum(1).float() / P)[:, None],
    ]
    rel = (i[:, None] + 1 + torch.arange(P - 1, device=g.device)) % P  # the other seats, in table order after me
    agg = T.aggressor[g]
    for x in (live.float(), allin.float(), c / STACK, acted.float(), (rel == agg[:, None]).float()):
        f.append(x.gather(1, rel) if x.shape[1] == P else x)
    if FEAT["stacks"] or FEAT["tourney"]:  # stack depths: mine, everyone's chips behind, the effective stack
        behind = (stk - c).float()
        f.append(torch.log(stk.gather(1, i[:, None]).float() / BB) / math.log(200))
        f.append((behind.gather(1, rel) / STACK).clamp(0, 3))
        opp = torch.where(live & (F.one_hot(i, P) == 0), behind, torch.zeros_like(behind)).max(1).values
        f.append((torch.minimum(left.float(), opp) / STACK).clamp(0, 3)[:, None])
    if FEAT["tourney"]:  # the payout ladder (places 1..P) and every player's ICM equity at the start of the hand
        reg, bty = split_pay(T.pay[g])
        f.append(reg * P)
        if bty is not None:  # every seat's bounty, mine first
            f.append((bty * P).gather(1, i[:, None]))
            f.append((bty * P).gather(1, rel))
        eq = T.eq0[g] * P
        f.append(eq.gather(1, i[:, None]))
        f.append(eq.gather(1, rel))
        share = T.stack[g].float() / T.stack[g].sum(1, keepdim=True)  # chip shares at the start of the hand
        f.append((T.eq0[g] / share.clamp(min=1e-6)).gather(1, i[:, None]).clamp(0, 5))  # equity per chip
    if FEAT["texture"]:  # the betting so far, per street, and the board's texture (the network otherwise sees buckets)
        f.append((T.hist[g].float() / 4).clamp(max=1).reshape(len(g), 12))
        f.append(board_texture(getattr(T, "board", None), g, T.street[g], g.device))
    legal, t = T.legal(g, i)
    f.append(legal.float())
    S = T.menu.S
    f.append(((t - bet[:, None]) / pot.clamp(min=1)[:, None]).clamp(0, 10) / 10 * legal[:, 2 : 2 + S].float())
    return torch.cat(f, 1), legal


TEXTURE_N = 19  # the texture inputs (12 betting history + 7 board), just before the legal moves and sizes
BELIEF_BINS = 10  # the belief output's classes per seat: strength in tenths


def seat_layout(P, A, S):
    """Where each other seat's inputs sit in features() (FEAT must be set): a [P-1, d] index tensor, one row per
    other seat in table order after me (live, all-in, committed, acted, aggressor, then chips behind, bounty and
    equity when the mode has them), and the indices of everything else (the table as a whole and my own inputs)."""
    R = P - 1
    at = P + 4 + 10  # seat one-hot, street one-hot, the ten pot and price scalars
    cols = [list(range(at + k * R, at + (k + 1) * R)) for k in range(5)]
    at += 5 * R
    if FEAT["stacks"] or FEAT["tourney"]:
        cols.append(list(range(at + 1, at + 1 + R)))  # chips behind (after my log stack)
        at += 1 + R + 1
    if FEAT["tourney"]:
        at += P  # the prize ladder
        if FEAT["bounty"]:
            cols.append(list(range(at + 1, at + 1 + R)))  # bounties (after mine)
            at += 1 + R
        cols.append(list(range(at + 1, at + 1 + R)))  # start equities (after mine)
        at += 1 + R + 1
    total = at + A + S
    tok = torch.tensor(cols).T  # [R, d]
    glob = torch.tensor(sorted(set(range(total)) - set(tok.flatten().tolist())))
    return tok, glob, total


class NineNet(nn.Module):
    """arch "mlp": every input in one vector (nm1-nm30). arch "sets": the other seats as a set of tokens (Deep Sets,
    as particle-flow networks read a jet's particles): one shared network reads each seat's inputs with its position
    relative to me, the results are pooled by sum and max, so what is learnt about a seat is shared by every seat and
    tables of any size read alike."""

    def __init__(self, nfeat, A, buckets, hidden=512, blocks=4, emb=32, arch="mlp", players=9, tok_hidden=128,
                 belief=False):  # fmt: skip
        super().__init__()
        per_street = list(buckets) if isinstance(buckets, (list, tuple)) else [buckets] * 3  # flop, turn, river
        self.emb = nn.ModuleList([nn.Embedding(169, emb)] + [nn.Embedding(b, emb) for b in per_street])
        self.arch = arch
        if arch == "sets":
            tok, glob, total = seat_layout(players, A, A - 3)
            assert total == nfeat, f"seat layout covers {total} inputs, the features have {nfeat}"
            self.register_buffer("tok_idx", tok, persistent=False)
            self.register_buffer("glob_idx", glob, persistent=False)
            self.register_buffer("pos", torch.eye(players - 1), persistent=False)
            self.tok = nn.Sequential(nn.Linear(tok.shape[1] + players - 1, tok_hidden), nn.GELU(),
                                     nn.Linear(tok_hidden, tok_hidden), nn.GELU())  # fmt: skip
            nfeat = len(glob) + 2 * tok_hidden
        self.inp = nn.Linear(nfeat + 4 * emb + 1, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(blocks)
        )
        self.norm = nn.LayerNorm(hidden)
        self.pi = nn.Linear(hidden, A)
        self.v = nn.Linear(hidden, 1)
        # belief modelling (DeepStack, ReBeL): an extra output, used only in training, predicting each other seat's
        # hand strength (tenths of the strength scale) from the last hidden layer, so that layer must carry ranges
        self.belief = nn.Linear(hidden, (players - 1) * BELIEF_BINS) if belief else None
        self.last_h = None

    def forward(self, feat, bk, street, st):
        """bk: [n, 4] this player's buckets per street; streets after the current one are masked out."""
        e = [self.emb[s](bk[:, s].clamp(max=self.emb[s].num_embeddings - 1)) * (street >= s).float()[:, None]
             for s in range(4)]  # fmt: skip
        if self.arch == "sets":
            seats = feat[:, self.tok_idx]  # [n, P-1, d]
            pos = self.pos.expand(feat.shape[0], -1, -1)
            t = self.tok(torch.cat([seats, pos], 2))
            feat = torch.cat([feat[:, self.glob_idx], t.sum(1), t.max(1).values], 1)
        h = self.inp(torch.cat([feat] + e + [st[:, None]], 1))
        for blk in self.blocks:
            h = h + blk(h)
        h = self.norm(h)
        if self.belief is not None and self.training:
            self.last_h = h
        return self.pi(h), self.v(h).squeeze(1)


def masked(logits, legal):
    return logits.masked_fill(~legal, -1e9)


def tag_probs(T, g, i, legal, s, dealer):
    """A rule-based tight-aggressive player on card strength s (0..1) and pot odds; deterministic."""
    c = T.contrib[g].float()
    pot = c.sum(1) + T.dead[g].float()
    to_call = T.bet[g].float() - c.gather(1, i[:, None]).squeeze(1)
    odds = to_call / (pot + to_call).clamp(min=1)
    pre = T.street[g] == 0
    S = T.menu.S
    first_size = legal[:, 2 : 2 + S].float().argmax(1) + 2
    has_size = legal[:, 2 : 2 + S].any(1)
    raise_slot = torch.where(has_size, first_size, torch.full_like(first_size, S + 2))
    strong = torch.where(pre, s > dealer.pre_q[0], s > 0.85)
    good = torch.where(pre, s > dealer.pre_q[1], s > 0.55 + 0.4 * odds)
    a = torch.where(to_call > 0, torch.zeros_like(first_size), torch.ones_like(first_size))  # fold, or check
    a = torch.where(good, torch.ones_like(a), a)  # call / check
    a = torch.where(strong & legal.gather(1, raise_slot[:, None]).squeeze(1), raise_slot, a)
    a = torch.where(legal.gather(1, a[:, None]).squeeze(1), a, torch.ones_like(a))
    return F.one_hot(a, legal.shape[1]).float()


REG_OPEN = (0.13, 0.14, 0.16, 0.19, 0.23, 0.29, 0.45, 0.45)  # share of hands opened first in: UTG .. button, SB
REG_PUSH10 = (0.08, 0.09, 0.11, 0.15, 0.18, 0.23, 0.32, 0.59)  # open-shove share at 10bb (chip-EV equilibrium)
# with a big-blind ante (RULES["ante"] 1bb): 2.5bb in the middle instead of 1.5bb, so everything widens. Push shares
# from icm_pushfold.py --bb-ante 1 (chip EV, 9 x 10bb); opens from common big-blind-ante charts
REG_OPEN_ANTE = (0.16, 0.18, 0.21, 0.25, 0.30, 0.38, 0.55, 0.55)
REG_PUSH10_ANTE = (0.12, 0.14, 0.18, 0.21, 0.27, 0.33, 0.43, 0.75)


def pre_top(dealer):
    """[169] the share of all hands at least as strong as each pre-flop class (by pre_strength, combo-weighted),
    taken at the class's middle: 0.002 for AA, near 1 for 32o. Cached on the dealer."""
    if getattr(dealer, "_pre_top", None) is None:
        rng = np.random.default_rng(5)  # the class frequencies from a fixed sample, without moving the dealer's seed
        bk = np.zeros((1 << 15, dealer.P, 4), np.int32)
        rk = np.zeros((1 << 15, dealer.P), np.int32)
        dealer.lib.bp_deal_n(1 << 15, dealer.P, int(rng.integers(1 << 62)), bk.ctypes.data, rk.ctypes.data, None)
        freq = torch.tensor(np.bincount(bk[:, :, 0].ravel(), minlength=169)[:169], dtype=torch.float32)
        freq = (freq / freq.sum()).to(dealer.pre_strength.device)
        order = torch.argsort(dealer.pre_strength[:169], descending=True)
        cum = torch.cumsum(freq[order], 0) - freq[order] / 2
        top = torch.empty_like(cum)
        top[order] = cum
        dealer._pre_top = top
    return dealer._pre_top


def reg_probs(T, g, i, legal, bk, dealer):
    """A rule-based tournament regular, deterministic. Before the flop it plays by stack depth, position and the
    share of hands at least as strong as its own (q):
      first in   at 13bb or less: shove the push range (the 10bb equilibrium share by seat, wider when shorter),
                 else fold; deeper: open the smallest size with the seat's opening range, else fold;
      facing an  if calling commits a third of what is left (or more): call when the price is good enough for a
      all-in     hand of this strength (wider at a better price), shoving the best part of it;
      facing an  3-bet (shove at 25bb or less) the top 5% (10% in the big blind against a late open), call the
      open       next hands (the big blind defends 55% against a late open, 35% against an early one; 65% and 45%
                 with a big-blind ante), else fold;
      facing a   continue (shove) with the top 3.5%, else fold.
      3-bet
    After the flop it plays like tag. Unlike the maniac and the station, it folds to shoves that are priced wrong
    and defends its big blind, so open-shoving and a wide big blind pay against it."""
    P, S, dev = T.P, T.menu.S, g.device
    s_all = strength(dealer, bk[g], T.street[g])
    post = tag_probs(T, g, i, legal, s_all.gather(1, i[:, None]).squeeze(1), dealer)
    cls = bk[g, i, 0].clamp(max=168)
    q = pre_top(dealer)[cls]
    pos = (i - 2) % P  # 0 UTG .. P-3 button, P-2 small blind, P-1 big blind
    pos8 = (pos - (P - 9)).clamp(0, 8)  # the same seat names at shorter tables, counted from the button
    stk = T.avail[g]
    live = ~T.folded[g]
    me = stk.gather(1, i[:, None]).squeeze(1).float()
    others = torch.where(live & (F.one_hot(i, P) == 0), stk, torch.zeros_like(stk)).max(1).values.float()
    eff = torch.minimum(me, others) / BB
    c = T.contrib[g]
    mine = c.gather(1, i[:, None]).squeeze(1).float()
    to_call = T.bet[g].float() - mine
    pot = c.sum(1).float() + T.dead[g].float()
    behind = me - mine
    odds = to_call / (pot + to_call).clamp(min=1)
    nr = T.pre_raises[g]
    opener = (T.aggressor[g] - 2) % P
    late = opener >= P - 4  # cutoff, button or small blind opened
    tab = lambda v: torch.tensor(v, device=dev)[pos8.clamp(max=7)]  # noqa: E731
    first_size = legal[:, 2 : 2 + S].float().argmax(1) + 2
    has_size = legal[:, 2 : 2 + S].any(1)
    ALLIN, RAISE = S + 2, torch.where(has_size, first_size, torch.full_like(first_size, S + 2))
    fold = torch.where(to_call > 0, 0, 1)
    a = fold.clone()
    # first in (also the small blind; the big blind only checks)
    first = (nr == 0) & (pos < P - 1)
    ante = RULES["ante"] > 0
    push = (tab(REG_PUSH10_ANTE if ante else REG_PUSH10) * (10 / eff.clamp(min=1)) ** 1.2).clamp(max=1)
    a = torch.where(first & (eff <= 13) & (q < push), ALLIN, a)
    a = torch.where(first & (eff > 13) & (q < tab(REG_OPEN_ANTE if ante else REG_OPEN)), RAISE, a)
    # facing an all-in or a raise that commits
    commit = (nr >= 1) & (to_call >= behind / 3)
    cf = (1.6 * (0.62 - odds)).clamp(0.04, 0.55)
    a = torch.where(commit & (q < cf), 1, a)
    a = torch.where(commit & (q < cf / 2), ALLIN, a)
    # facing a single open that does not commit
    open1 = (nr == 1) & ~commit
    bb = pos == P - 1
    three = torch.where(bb & late, 0.10, 0.05)
    defend = torch.where(bb, torch.where(late, 0.65 if ante else 0.55, 0.45 if ante else 0.35),
                         three + torch.where(pos >= P - 3, 0.10, 0.07))  # fmt: skip
    a = torch.where(open1 & (q < defend), 1, a)
    a = torch.where(open1 & (q < three), torch.where(eff <= 25, ALLIN, RAISE), a)
    # facing a 3-bet or more
    a = torch.where((nr >= 2) & ~commit & (q < 0.035), ALLIN, a)
    a = torch.where(legal.gather(1, a[:, None]).squeeze(1), a, torch.where(legal[:, ALLIN] & (a >= 2), ALLIN, 1))
    a = torch.where(legal.gather(1, a[:, None]).squeeze(1), a, fold)
    pre = F.one_hot(a, legal.shape[1]).float()
    return torch.where((T.street[g] == 0)[:, None], pre, post)


def rule_probs(name, T, g, i, legal, s, dealer, bk=None):
    """Rule-based opponents: reg (a tournament regular, reg_probs; needs bk, every table's buckets), tag (tight-aggressive), maniac (raises and shoves most of the time), station (always
    calls or checks), random (uniform over the legal actions). Where a rule wants to call and calling is not legal
    (first in, without open limping: RULES["open_limp"] False), it opens with the first legal size instead, or shoves."""
    p = _rule_probs(name, T, g, i, legal, s, dealer, bk)
    lost = (p * ~legal).sum(1)
    if bool((lost > 0).any()):
        p = p * legal
        sizes = legal[:, 2:]
        to = torch.where(sizes.any(1), sizes.float().argmax(1) + 2, torch.zeros_like(lost, dtype=torch.long))
        p[torch.arange(len(g), device=g.device), to] += lost
    return p


def _rule_probs(name, T, g, i, legal, s, dealer, bk=None):
    if name == "reg":
        return reg_probs(T, g, i, legal, bk, dealer)
    if name == "tag":
        return tag_probs(T, g, i, legal, s, dealer)
    lf = legal.float()
    if name == "station":
        return F.one_hot(torch.ones(len(g), dtype=torch.long, device=g.device), legal.shape[1]).float()
    if name == "random":
        return lf / lf.sum(1, keepdim=True)
    if name == "maniac":  # 70% spread over the raises and all-in when there are any, else call
        w = lf.clone()
        w[:, 0] = 0
        aggr = w[:, 2:].sum(1, keepdim=True)
        w[:, 2:] = torch.where(aggr > 0, 0.7 * w[:, 2:] / aggr.clamp(min=1), w[:, 2:])
        w[:, 1] = torch.where(aggr[:, 0] > 0, 0.3, 1.0)
        return w / w.sum(1, keepdim=True)
    raise ValueError(name)


def n_features(P, menu, dev):
    """The input size in the run's mode, from a one-table probe."""
    st, pay = context(1, P, dev)
    T = Tables(1, P, menu, dev, stacks=st)
    if pay is not None:
        T.pay, T.eq0 = pay.float(), start_equity(T.stack, pay)
    return features(T, torch.zeros(1, dtype=torch.long, device=dev), T.actor[:1])[0].shape[1]


def table_cell(st, kind):
    """Each table's cell for prioritised sampling and leak maps: kind (0 final table, 1 bubble, 2 in the money), the
    seated players' average stack (under 12, 12-18, 18-25, 25+ bb) and how many are seated (2, 3-5, 6-8, 9)."""
    seated = (st > 0).sum(1)
    avg = st.float().sum(1) / seated.clamp(min=1) / BB
    depth = (avg >= 12).long() + (avg >= 18).long() + (avg >= 25).long()
    players = (seated >= 3).long() + (seated >= 6).long() + (seated >= 9).long()
    return kind.view(-1) * 16 + depth * 4 + players


def kinds():
    """The shares of tournament tables by kind (icm.scenarios): final table, bubble, in the money and, with
    FEAT["early"], early-stage tables (winner-take-all, chips are money) taking that share from the other three."""
    e = FEAT["early"]
    return (0.4 * (1 - e), 0.4 * (1 - e), 0.2 * (1 - e), e) if e else (0.4, 0.4, 0.2)


def context(G, P, dev, prioritized=False):
    """Starting stacks [G, P] and payouts (or None) for G tables in the run's mode (FEAT); None, None = 100bb cash.
    prioritized (training only): tables drawn in proportion to their cell's weight in PRIORITY (the leak map)."""
    if FEAT["tourney"]:
        if prioritized and PRIORITY:  # draw 3x and keep G by weight: the leaky cells come up more often
            st3, pay3, kind3 = scenarios(3 * G, P, dev, kinds(), avg_bb=FEAT["depths"] or (10, 60), short=FEAT["short"])
            w = torch.tensor([PRIORITY.get(int(c), 1.0) for c in table_cell(st3, kind3).tolist()], device=dev)
            keep = torch.multinomial(w, G, replacement=False)
            st, pay = st3[keep], pay3[keep]
        else:
            st, pay, _ = scenarios(G, P, dev, kinds(), avg_bb=FEAT["depths"] or (10, 60), short=FEAT["short"])
        beta = FEAT["bounty"]
        if beta:  # bounties: bigger stacks have usually collected more; empty seats have none
            share = st.float() / st.float().mean(1, keepdim=True).clamp(min=1)
            b = share.sqrt() * torch.exp(0.4 * torch.randn(G, P, device=dev)) * (st > 0).float()
            b = beta * b / b.sum(1, keepdim=True).clamp(min=1e-9)
            pay = torch.cat([pay * (1 - beta), b], 1)
        return st, pay
    st = None
    if FEAT["stacks"]:
        lo, hi = FEAT["stacks"]
        bb = lo * (hi / lo) ** torch.rand(G, P, device=dev)
        st = (bb * BB).round().long().clamp(min=2 * BB)
    k = FEAT["seats"]
    if k and (k < P or FEAT["short"] > 0):  # short-handed cash tables: the blinds and the last n-2 seats play
        if st is None:
            st = torch.full((G, P), 100 * BB, dtype=torch.long, device=dev)
        n = torch.full((G, 1), min(k, P), dtype=torch.long, device=dev)
        if FEAT["short"] > 0 and k > 2:
            fewer = torch.randint(2, min(k, P), (G, 1), device=dev)
            n = torch.where(torch.rand(G, 1, device=dev) < FEAT["short"], fewer, n)
        seat = torch.arange(P, device=dev)[None]
        st = torch.where((seat < 2) | (seat >= P - (n - 2)), st, torch.zeros_like(st))
    return st, None


def play(dealer, menu, policies, bk, rk, record=False, dev="cuda", learner=None, assign=None, ctx=None, views=None,
         board=None, chips=False):  # fmt: skip
    """Play len(bk) hands; policies[seat] is a NineNet or a rule name (rule_probs). Returns payoffs [G, P] (chips)
    and, with record, the decisions (only those of the policy `learner`, or of a list of policies, when given), each
    with the index of the policy that took it (last field).

    assign [G, P]: per-table seating instead; policies is then the list of distinct policies and assign[g, seat]
    indexes it. All tables are played in one loop, whatever their seating: the loop's length, not the number of
    tables, sets the time, so one loop over a mixed batch is several times faster than one loop per seating.

    ctx (stacks, payouts): the tables' starting stacks and tournament payouts; drawn for the run's mode if None. In
    tournament mode the returned results are ICM equity changes in chip equivalents, not chips.

    views {id(policy): (bk, dealer)}: a network that reads another abstraction sees the same cards through its own
    buckets and hand strength (ninemax_absmatch.py); the others use bk and dealer.

    chips: return the chips won even in tournament mode (the players still see the payouts and ICM)."""
    G, P = bk.shape[0], bk.shape[1]
    stacks, pay = ctx if ctx is not None else context(G, P, dev)
    T = Tables(G, P, menu, dev, stacks=stacks)
    if pay is not None:
        T.pay, T.eq0 = pay.float(), start_equity(T.stack, pay)
    T.board = board  # [G, 5] cards (texture runs; None: the texture features read zero)
    rec = []
    if assign is None:
        uniq = list({id(p): p for p in policies}.values())  # each distinct policy once; seat -> its index
        seat_pol = torch.tensor([next(k for k, q in enumerate(uniq) if q is p) for p in policies], device=dev)
        assign = seat_pol[None, :].expand(G, P)
    else:
        uniq = list(policies)
    learners = learner if isinstance(learner, (list, tuple)) else [learner]
    lids = torch.tensor([k for k, q in enumerate(uniq) if any(q is x for x in learners)] or [-1], device=dev)
    while not bool(T.done.all()):
        g = (~T.done).nonzero().squeeze(1)
        i = T.actor[g]
        st_all = strength(dealer, bk[g], T.street[g])
        s = st_all.gather(1, i[:, None]).squeeze(1)
        feat, legal = features(T, g, i)
        mybk = bk[g, i]
        a = torch.zeros(len(g), dtype=torch.long, device=dev)
        pol_of = assign[g, i]
        for k, pol in enumerate(uniq):
            m = pol_of == k
            if not m.any():
                continue
            if isinstance(pol, str):
                probs = rule_probs(pol, T, g[m], i[m], legal[m], s[m], dealer, bk)
            elif views and id(pol) in views:
                vbk, vdl = views[id(pol)]
                gm, im = g[m], i[m]
                sv = strength(vdl, vbk[gm], T.street[gm]).gather(1, im[:, None]).squeeze(1)
                with torch.no_grad():
                    lg, _ = pol(feat[m], vbk[gm, im], T.street[gm], sv)
                probs = F.softmax(masked(lg, legal[m]), 1)
            else:
                fm = feat[m]
                if FEAT["texture"] and getattr(pol, "texture", True) is False:  # a network trained without them
                    A_, S_ = menu.A, menu.S
                    cut = fm.shape[1] - A_ - S_
                    fm = torch.cat([fm[:, : cut - TEXTURE_N], fm[:, cut:]], 1)
                with torch.no_grad():
                    lg, _ = pol(fm, mybk[m], T.street[g][m], s[m])
                probs = F.softmax(masked(lg, legal[m]), 1)
            rule = getattr(pol, "early_shove_bb", 0) if not isinstance(pol, str) else 0
            if rule:  # this player follows the early-shove rule (a hero-only test of it)
                no = T.early_shove(g[m], i[m], rule, legal[m][:, 2 : 2 + menu.S].any(1))
                probs = probs.clone()
                probs[no, menu.A - 1] = 0
                probs = probs / probs.sum(1, keepdim=True)
            a[m] = torch.multinomial(probs, 1).squeeze(1)
        if record:
            keep = torch.ones_like(i, dtype=torch.bool) if learner is None else torch.isin(pol_of, lids)
            rec.append(tuple(x[keep] for x in (g, i, feat, legal, mybk, T.street[g].clone(), s, a, pol_of)))
        T.step(g, a)
    u = T.payoffs(rk)
    # tournament: the change in prize equity, in chips at the table's average chip value
    if pay is not None and not chips:
        reg, bty = split_pay(T.pay)
        end = end_stacks(T.stack, u)
        eq1 = icm(end, reg).float()
        if bty is not None:  # knockouts: each bust's bounty, half in cash, to whoever took the player's last chips
            busted, k = knockouts(T, u, rk, bty)
            left = torch.where(busted, torch.zeros_like(bty), bty) + k
            alive = torch.where(busted, torch.zeros_like(end), end).float()
            # the bounties still in play, by chip share (as start_equity): the money stays whole
            eq1 = eq1 + k + left.sum(1, keepdim=True) * alive / alive.sum(1, keepdim=True).clamp(min=1)
        u = (eq1 - T.eq0) * T.stack.sum(1, keepdim=True).float()
    return u, rec


def knockouts(T, u, ranks, bty):
    """The players a finished hand busts [G, P] bool, and the bounty cash each seat collects [G, P]: half of every
    busted player's bounty (the other half goes onto the collector's own head). A bounty belongs to whoever won the
    pot that took the player's last chips: the pot level at its contribution, contested by the live players who put
    in at least as much (as Tables.payoffs settles it), split equally between tied winners. A busted player is
    all in and live, so it contests that level itself and some player always wins it. (Until 2026-11-23 the hand's
    biggest winner, u.argmax, took every bounty: wrong on ties and when another player won a bigger side pot.)"""
    busted = (T.stack > 0) & (T.stack.double() + u <= 0.5)
    c = T.contrib
    elig = ~T.folded[:, None, :] & (c[:, None, :] >= c[:, :, None])  # [G, busted i, collector j]
    big = torch.iinfo(torch.int32).max
    r = ranks[:, None, :].expand(-1, T.P, -1).masked_fill(~elig, big)
    takers = elig & (r == r.min(2, keepdim=True).values)
    share = bty * busted.float() / takers.sum(2).clamp(min=1).float()
    k = (share[:, :, None] * takers.float()).sum(1) / 2
    return busted, k


def seats_used(P):
    """The seats players can sit in: all P, or with --seats k < P the blinds and the last k-2 (the early ones empty)."""
    k = FEAT["seats"] or P
    return [0, 1] + list(range(P - (k - 2), P)) if k < P else list(range(P))


def seated_mask(stacks, G, P, dev):
    """[G, P] bool: the seats a player sits in (stack > 0); every seat when the mode has no stacks (100bb cash)."""
    return stacks > 0 if stacks is not None else torch.ones(G, P, dtype=torch.bool, device=dev)


def split_seats(seated):
    """For population tables: per table, a random 1 .. n-1 of its n seated players for the opponent, the rest for the
    net (both always play). Returns [R, P] bool, True where the opponent sits; empty seats are never chosen."""
    R, P = seated.shape
    n = seated.sum(1, keepdim=True)
    k = torch.minimum(1 + (torch.rand(R, 1, device=seated.device) * (n - 1)).long(), (n - 1).clamp(min=1))
    order = torch.rand(R, P, device=seated.device).masked_fill(~seated, 2.0).argsort(1).argsort(1)  # rank, seated first
    return seated & (order < k)


def luck_means(dealer, n=1 << 20):
    """E[strength on street t | bucket on street t-1], t = 1..3, and E[pre-flop strength], from n independent deals
    (cached on the dealer): what the card-luck control variates subtract."""
    if getattr(dealer, "_luck", None) is None:
        bk, _, _ = dealer.deal(n)
        b = bk.reshape(-1, 4)
        s = torch.stack([strength(dealer, bk, torch.full((n,), t, device=bk.device)).reshape(-1) for t in range(4)], 1)
        tabs = []
        for t in range(1, 4):
            prev = b[:, t - 1]
            K = int(prev.max()) + 1
            tot = torch.zeros(K, device=b.device).index_add_(0, prev, s[:, t])
            cnt = torch.zeros(K, device=b.device).index_add_(0, prev, torch.ones_like(s[:, t]))
            tabs.append(tot / cnt.clamp(min=1))
        dealer._luck = (s[:, 0].mean(), tabs)
    return dealer._luck


def card_luck(dealer, bk):
    """Each seat's card luck per street, [n, P, 4]: pre-flop strength less its mean, then each street's strength less
    its expectation given the previous street's bucket. Each term has mean 0 exactly (the tower property), so any
    multiple of it can be subtracted from a result without bias: a control variate, as in physics Monte Carlo."""
    m0, tabs = luck_means(dealer)
    n, P, _ = bk.shape
    s = torch.stack([strength(dealer, bk, torch.full((n,), t, device=bk.device)) for t in range(4)], 2)  # [n, P, 4]
    x = torch.empty_like(s)
    x[..., 0] = s[..., 0] - m0
    for t in range(1, 4):
        x[..., t] = s[..., t] - tabs[t - 1][bk[..., t - 1].clamp(max=len(tabs[t - 1]) - 1)]
    return x


def duplicate(dealer, menu, hero, villain, n, dev, cv=False, chips=False, per_deal=False):
    """Hero in each seat in turn against P-1 copies of villain, on the same n deals: hero's bb/100 and 95% CI.
    cv: also the estimate with card-luck control variates (card_luck: the hero's luck in each seating, per street,
    regressed out by least squares) and its interval, as (m, lo, hi, m_cv, lo_cv, hi_cv). per_deal: also the hero's
    result on every deal (bb, a numpy array), as (m, lo, hi, x), so two heroes on the same deals can be compared deal by
    deal (a paired difference and its interval)."""
    P = dealer.P
    bk, rk, cd = dealer.deal(n, cards=FEAT["texture"])
    board = board_of(cd, P, dev)
    ctx = context(n, P, dev)  # the same stacks (and payouts) for every seating
    per = torch.zeros(n, device=dev)
    xs = []
    luck = card_luck(dealer, bk) if cv else None
    for seat in range(P):
        pols = [villain] * P
        pols[seat] = hero
        u, _ = play(dealer, menu, pols, bk, rk, dev=dev, ctx=ctx, board=board, chips=chips)
        per += u[:, seat] / BB
        if cv:
            seated = (ctx[0][:, seat] > 0).float()[:, None] if ctx[0] is not None else 1.0
            xs.append(luck[:, seat, :] * seated)  # the hero's luck in this seating (none in an empty seat)
    x = (per / P).cpu().numpy()  # hero's average over the seats, per deal
    m, se = x.mean() * 100, x.std() / math.sqrt(len(x)) * 100
    if per_deal:
        return m, m - 1.96 * se, m + 1.96 * se, x
    if not cv:
        return m, m - 1.96 * se, m + 1.96 * se
    X = torch.cat(xs, 1).cpu().double().numpy()  # [n, 4 P]: every seating's luck terms, each with mean 0
    y = x - x.mean()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = x - X @ beta  # the result less the luck the regression explains; E[X] = 0 so the mean is unbiased
    mc, sec = r.mean() * 100, r.std() / math.sqrt(len(r)) * 100
    return m, m - 1.96 * se, m + 1.96 * se, mc, mc - 1.96 * sec, mc + 1.96 * sec


def set_mode(stacks="100", tourney=False, depths=None, open_limp=True, ante=0.0, short=0.0, seats=None, bounty=0.0,
             texture=False, early=0.0):  # fmt: skip
    """Set FEAT from a run's options (also used by the tools that load a run, from its config.json). depths: the
    tournament tables' average-stack ranges, "LO,HI;LO,HI" (None: icm.scenarios' default, 10-60bb)."""
    RULES["open_limp"] = bool(open_limp)
    RULES["ante"] = float(ante or 0.0)
    FEAT["depths"] = [tuple(float(x) for x in r.split(",")) for r in depths.split(";")] if depths else None
    FEAT["stacks"] = None if str(stacks) == "100" else tuple(float(x) for x in str(stacks).split(","))
    FEAT["tourney"] = bool(tourney)
    FEAT["short"] = float(short or 0.0)
    FEAT["seats"] = int(seats) if seats else None
    FEAT["bounty"] = float(bounty or 0.0)
    FEAT["texture"] = bool(texture)
    FEAT["early"] = float(early or 0.0)


def extra_pool(args, avg, nfeat, buckets, dev, dealer=None, menu=None):
    """The population's extra members: styles (pipeline_opponents.Style around the network's running average, so a
    style is a strong player with a human-shaped leak that improves as training goes) and earlier networks, frozen."""
    out, names = [], []
    if args.pool_styles:
        from pipeline_opponents import ARCHETYPES, KEYS, Style, styles

        for part in args.pool_styles.split(","):
            kind, *rest = part.split(":")
            if kind == "archetypes":
                params = dict(ARCHETYPES)
            elif kind == "generated":
                k, seed = int(rest[0]), int(rest[1]) if len(rest) > 1 else 2026
                params = {n: p for n, p in styles(k, seed).items() if n not in ARCHETYPES}
            elif kind == "human":
                fitted = json.load(open(":".join(rest)))["styles"]
                params = {st["id"]: tuple(st["params"][key] for key in KEYS) for st in fitted}
            else:
                raise SystemExit(f"--pool-styles: unknown kind {kind!r}")
            out += [Style(avg, p) for p in params.values()]
            names += list(params)
    if args.pool_patterns:  # what people found works against the bot: it must learn to punish each of them
        from exploit_patterns import check, player, read

        pats = read(None if args.pool_patterns == "all" else args.pool_patterns.split(","))
        for p in pats:
            if check(p):
                print(f"--pool-patterns: {p.get('id')} is invalid ({'; '.join(check(p))}), left out", flush=True)
                continue
            out.append(player(avg, p, dealer, menu))
            names.append(p["id"])
    for r in [x for x in args.pool_nets.split(",") if x]:
        cfg = json.load(open(os.path.join(r, "config.json")))
        if cfg["nfeat"] != nfeat or cfg.get("buckets", buckets) != buckets:
            print(f"--pool-nets: {r} reads other inputs, left out", flush=True)
            continue
        m = NineNet(cfg["nfeat"], cfg["actions"], cfg.get("buckets", buckets), arch=cfg.get("arch", "mlp"),
                    players=args.players, belief=cfg.get("belief_weight", 0) > 0).to(dev)  # fmt: skip
        m.load_state_dict(torch.load(os.path.join(r, "checkpoint.pt"), map_location=dev, weights_only=True)["avg"])
        out.append(m.requires_grad_(False).eval())
        names.append(os.path.basename(r))
    if out:
        print(f"population extras ({len(out)}): {', '.join(names)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--players", type=int, default=9)
    ap.add_argument(
        "--arch",
        choices=("mlp", "sets"),
        default="mlp",
        help="the network: one input vector, or the other seats as a set (Deep Sets)",
    )
    ap.add_argument("--menu", default="medium")
    ap.add_argument("--abs", default="blueprint/out")
    ap.add_argument("--lib", default="blueprint/libbp5.so")
    ap.add_argument("--batch", type=int, default=8192, help="tables per training batch")
    ap.add_argument("--minutes", type=float, default=1440)
    ap.add_argument(
        "--max-hands",
        type=float,
        default=None,
        help="also stop after this many hands this session: equal training for runs compared, whatever their speed",
    )
    ap.add_argument("--eval-minutes", type=float, default=30)
    ap.add_argument("--eval-hands", type=int, default=20000, help="deals per duplicate match (x players seatings)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kl", type=float, default=0.3)
    # trainability map (docs/2026-11-09-neural-models-and-scale.md): a weak pull learns fastest early, a strong one
    # keeps long runs stable; ramp linearly from --kl-start to --kl over --kl-ramp minutes (0 = constant --kl)
    ap.add_argument("--kl-start", type=float, default=None)
    # heads-up learning-rule test (docs/2026-11-09-neural-models-and-scale.md): NeuRD was the least exploitable
    ap.add_argument("--pg", choices=("a2c", "neurd"), default="a2c", help="policy update: a2c (REINFORCE) or neurd")
    ap.add_argument("--neurd-beta", type=float, default=2.0, help="NeuRD: logits stop moving outward beyond +-beta")
    ap.add_argument("--kl-ramp", type=float, default=0.0, help="minutes to ramp the KL pull from --kl-start to --kl")
    ap.add_argument("--magnet-tau", type=float, default=3e-4)
    ap.add_argument("--avg-tau", type=float, default=1e-3)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument(
        "--keep-hours",
        type=float,
        default=0.0,
        help="also keep the moving average every N hours "
        "(avg_<hours>h.pt), so later checkpoints can be played against earlier ones",
    )
    ap.add_argument("--pool-share", type=float, default=0.0, help="share of each batch played against the population")
    ap.add_argument("--pool", default="tag,maniac,station,random", help="rule opponents in the population")
    ap.add_argument("--pool-styles", default="",
                    help="styles in the population, played by the network's own average (so they get stronger with it): "
                         "generated:K[:SEED], archetypes, human:PATH (human_styles.py fit), comma-separated")  # fmt: skip
    ap.add_argument("--pool-patterns", default="",
                    help="patterns people found (exploit_patterns.py) in the population, played around the network's "
                         "average: all (the library and the mined file), or comma-separated files")  # fmt: skip
    ap.add_argument("--pool-nets", default="", help="earlier networks in the population, frozen: comma-separated runs")
    ap.add_argument("--pool-snapshots", type=int, default=5, help="past checkpoints kept in the population")
    # the league (after AlphaStar, Vinyals et al. 2019): opponents picked by how hard they are (prioritised fictitious
    # self-play), a league exploiter that hunts the whole population, and retired exploiters kept frozen
    ap.add_argument(
        "--pfsp",
        type=float,
        default=0.0,
        help="pick pool opponents with weight (1 - p_win)^PFSP, p_win "
        "the network's running chance of beating each; 0 = uniform (as before)",
    )
    ap.add_argument(
        "--league-share",
        type=float,
        default=0.0,
        help="share of tables for a league exploiter, trained "
        "against the network and every pool member; frozen into the pool at each reset",
    )
    # expert iteration: pull the policy towards search's improved policy on stored positions (ninemax_distill_data.py)
    ap.add_argument("--distill", help="a folder of distillation shards: every step also learns pi ~ pi_net exp(Q/tau)")
    ap.add_argument("--distill-weight", type=float, default=1.0, help="weight of the distillation cross-entropy")
    ap.add_argument("--distill-tau", type=float, default=1.0, help="temperature of the target, big blinds")
    ap.add_argument("--distill-batch", type=int, default=4096, help="stored positions per step")
    ap.add_argument(
        "--belief-weight",
        type=float,
        default=0.0,
        help="belief modelling: weight of the auxiliary loss predicting each live opponent's hand strength (0 = off)",
    )
    ap.add_argument("--texture", action="store_true", help="also read the betting per street and the board's texture")
    ap.add_argument("--priority", help="a leak map (ninemax_leaks.py JSON): training tables drawn by its cell weights")
    ap.add_argument("--max-frozen", type=int, default=0, help="frozen exploiters kept in the pool (0 = all, as before)")
    ap.add_argument("--chunks", type=int, default=4, help="population chunks per batch (one opponent each)")
    # exploiters (AlphaStar's league, PSRO): a second network trains only to beat the main one. A share of each batch
    # seats it once against eight copies of the main network; both learn from those tables, so the main network has
    # to close the holes the exploiter finds. Every EXPLOIT_RESET minutes the exploiter joins the population as a
    # frozen snapshot and restarts as a copy of the main network, to hunt fresh holes. A standalone exploiter of nm3
    # won +191 bb/100 in 15 minutes (ninemax_eval.py exploit).
    ap.add_argument("--exploit-share", type=float, default=0.0, help="share of each batch at exploiter tables")
    ap.add_argument("--exploit-reset", type=float, default=60, help="minutes between exploiter restarts")
    # stack depths and tournaments (docs/2026-11-09-nine-max-plan.md, "Stacks, bubbles and final tables")
    ap.add_argument("--stacks", default="100", help='"100" (every hand 100bb) or "LO,HI": each stack log-uniform')
    ap.add_argument("--tourney", action="store_true", help="tournament tables: ICM payouts, reward = ICM equity")
    ap.add_argument("--no-open-limp", action="store_true", help="first in from UTG to the button: fold, open or shove")
    ap.add_argument("--ante", type=float, default=0.0, help="big-blind ante in big blinds (1.0 = the usual MTT ante)")
    ap.add_argument("--short", type=float, default=0.0,
                    help="share of tables played short-handed: tournament final tables 2-8 handed, or cash tables "
                         "2 to --seats - 1 handed")  # fmt: skip
    ap.add_argument("--seats", type=int, default=None, help="cash tables seat at most this many players (6: 6-max)")
    ap.add_argument("--bounty", type=float, default=0.0,
                    help="--tourney: this share of the money at stake in knockout bounties (PKO; 0.5 is usual)")  # fmt: skip
    ap.add_argument("--early", type=float, default=0.0,
                    help="--tourney: this share of tables early in the tournament, far from the money (winner-take-all "
                         "payouts: chips are money)")  # fmt: skip
    ap.add_argument("--init", help="warm start from a checkpoint.pt or avg_<h>h.pt when out has no checkpoint")
    ap.add_argument("--depths", help='tournament average-stack ranges in bb, "LO,HI;LO,HI" (default 10,60)')
    args = ap.parse_args()
    set_mode(args.stacks, args.tourney, args.depths, not args.no_open_limp, args.ante, args.short, args.seats,
             args.bounty, args.texture, args.early)  # fmt: skip
    if args.priority:  # the leak map's weights per table cell (ninemax_leaks.py)
        PRIORITY.update({int(k): float(v) for k, v in json.load(open(args.priority))["weights"].items()})
        print(f"prioritised sampling: {len(PRIORITY)} cells weighted, max {max(PRIORITY.values()):.2f}", flush=True)
    dev = "cuda"
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)
    B = int(open(os.path.join(args.abs, "meta.txt")).read().split()[1])
    dealer = Dealer(args.lib, args.abs, args.players, B, dev)
    # pre-flop thresholds for the rule-based player: top 12% of dealt hands raise, top 35% continue
    bk0, _, _ = dealer.raw(1 << 16)
    pre_s = dealer.pre_strength[torch.tensor(bk0[:, :, 0].ravel(), device=dev)]
    dealer.pre_q = (torch.quantile(pre_s, 0.88).item(), torch.quantile(pre_s, 0.65).item())
    menu = Menu(args.menu, dev)
    nfeat = n_features(args.players, menu, dev)
    buckets = list(read_buckets(args.abs))
    net = NineNet(nfeat, menu.A, buckets, arch=args.arch, players=args.players, belief=args.belief_weight > 0).to(dev)
    magnet, avg = copy.deepcopy(net), copy.deepcopy(net)
    for m in (magnet, avg):
        m.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    ck = os.path.join(args.out, "checkpoint.pt")
    hands = 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=True)
        for mod, k in ((net, "net"), (magnet, "magnet"), (avg, "avg")):
            mod.load_state_dict(st[k])
        opt.load_state_dict(st["opt"])
        hands = st["hands"]
    elif args.init:  # warm start: a run's checkpoint (net, magnet, avg) or a snapshot (avg into all three)
        st = torch.load(args.init, map_location=dev, weights_only=True)
        for mod, k in ((net, "net"), (magnet, "magnet"), (avg, "avg")):
            sd = dict(st[k] if k in st else st["avg"])
            old_w, new_w = sd["inp.weight"], mod.inp.weight
            if old_w.shape[1] < new_w.shape[1]:  # --texture from a run without it: new input columns start at zero
                pos = old_w.shape[1] - (4 * mod.emb[0].embedding_dim + 1) - (menu.A + menu.S)  # before legal and sizes
                extra_cols = torch.zeros(old_w.shape[0], new_w.shape[1] - old_w.shape[1], device=old_w.device)
                sd["inp.weight"] = torch.cat([old_w[:, :pos], extra_cols, old_w[:, pos:]], 1)
            missing, extra = mod.load_state_dict(sd, strict=False)
            # a warm start may add a training-only head the source run did not have (--belief-weight); nothing else
            assert not extra and all(m.startswith("belief.") for m in missing), (missing, extra)
        hands = st["hands"]
    prev = copy.deepcopy(avg)
    expl = copy.deepcopy(avg).requires_grad_(True) if args.exploit_share > 0 else None
    opt_x = torch.optim.Adam(expl.parameters(), lr=args.lr) if expl is not None else None
    adv_scale_x, last_reset = 1.0, time.time()
    rules = [x for x in args.pool.split(",") if x] if args.pool else []
    extra = extra_pool(
        args, avg, nfeat, buckets, dev, dealer, menu
    )  # styles and earlier networks: stronger, more varied opponents
    snaps, frozen = [], []
    lexp = copy.deepcopy(avg).requires_grad_(True) if args.league_share > 0 else None
    opt_l = torch.optim.Adam(lexp.parameters(), lr=args.lr) if lexp is not None else None
    adv_scale_l = 1.0
    score = {}  # PFSP: the network's running result (bb a hand) against each pool member, by id
    dist = None
    if args.distill:  # search-labelled positions; the target is fixed, so it is computed once
        import glob

        shards = [torch.load(f) for f in sorted(glob.glob(os.path.join(args.distill, "shard_*.pt")))]
        dist = {k: torch.cat([d[k] for d in shards]).to(dev) for k in shards[0]}
        q = dist["q"].masked_fill(~dist["legal"], -1e9)
        z = torch.log(dist["pnet"].float().clamp(min=1e-6)) + (q - q.max(1, keepdim=True).values) / args.distill_tau
        dist["target"] = torch.softmax(masked(z, dist["legal"]), 1)
        print(f"distillation: {len(q)} positions from {len(shards)} shards, tau {args.distill_tau}", flush=True)
    pool = (rules + extra or None) if args.pool_share > 0 else None
    json.dump({**vars(args), "nfeat": nfeat, "actions": menu.A, "buckets": buckets}, open(os.path.join(args.out, "config.json"), "w"),
              indent=1)  # fmt: skip
    log = open(os.path.join(args.out, "progress.log"), "a", buffering=1)
    t0 = last_eval = time.time()
    kept = 0.0  # hours of this session already kept as avg_<hours>h.pt
    h_start, adv_scale = hands, 1.0
    while time.time() - t0 < args.minutes * 60 and (args.max_hands is None or hands - h_start < args.max_hands):
        bk, rk, cd = dealer.deal(args.batch, cards=args.texture)
        # self-play on part of the batch; on the rest, chunks of tables where the net holds some seats and one
        # opponent from the population (rule players, past checkpoints) holds the others. The net learns from its
        # own seats only.
        n_pop = int(args.batch * args.pool_share) // args.chunks * args.chunks if pool else 0
        n_x = int(args.batch * args.exploit_share) if expl is not None else 0
        n_l = int(args.batch * args.league_share) // args.chunks * args.chunks if lexp is not None and pool else 0
        # one seating table for the whole batch (policy 0 = the net, 1 = the exploiter), played in a single loop:
        # rows [0, n_x) exploiter tables, then n_pop population tables, then self-play
        assign = torch.zeros(args.batch, args.players, dtype=torch.long, device=dev)
        pols = [net] + ([expl] if expl is not None else []) + ([lexp] if n_l else []) + (pool or [])
        first = len(pols) - len(pool or [])
        lid = 2 if expl is not None else 1  # the league exploiter's policy index
        # the tables' stacks first, so every seat given to the exploiters or the population is one a player sits in
        # (short tables and 6-max leave seats empty, stack 0; an empty seat never acts and wins nothing)
        ctx_t = context(args.batch, args.players, dev, prioritized=True)
        seated = seated_mask(ctx_t[0], args.batch, args.players, dev)
        if n_x:  # the exploiter in a seat where a player sits
            rows = torch.arange(n_x, device=dev)
            assign[rows, torch.multinomial(seated[rows].float(), 1).squeeze(1)] = 1
        for c in range(args.chunks if n_l else 0):  # league exploiter tables: one seat it, the rest one member or net
            rows = torch.arange(n_x + c * n_l // args.chunks, n_x + (c + 1) * n_l // args.chunks, device=dev)
            opp = int(torch.randint(len(pool) + 1, ()).item())
            assign[rows] = 0 if opp == len(pool) else first + opp
            assign[rows, torch.multinomial(seated[rows].float(), 1).squeeze(1)] = lid
        if args.pfsp > 0 and pool:  # harder opponents more often: weight (1 - p_win)^pfsp
            p_win = torch.tensor([torch.sigmoid(torch.tensor(score.get(id(m), 0.0) / 0.5)).item() for m in pool])
            weights = (1 - p_win).clamp(min=1e-3) ** args.pfsp
        else:
            weights = torch.ones(len(pool or [1]))
        chunk_opp = []
        for c in range(args.chunks if n_pop else 0):
            rows = slice(n_x + n_l + c * n_pop // args.chunks, n_x + n_l + (c + 1) * n_pop // args.chunks)
            opp = first + int(torch.multinomial(weights, 1).item())
            chunk_opp.append((rows, opp))
            assign[rows] = torch.where(split_seats(seated[rows]), opp, assign[rows])
        learner = [net] + ([expl] if expl is not None else []) + ([lexp] if n_l else [])
        learner = learner if len(learner) > 1 else net
        u, rec = play(dealer, menu, pols, bk, rk, record=True, dev=dev, learner=learner, assign=assign, ctx=ctx_t,
                      board=board_of(cd, args.players, dev))  # fmt: skip
        g, i, f, legal, mybk, street, s, a, who = (torch.cat([r[k] for r in rec]) for k in range(9))
        with torch.no_grad():  # PFSP: the network's result against each chunk's opponent
            for rows, opp in chunk_opp:
                mine = (assign[rows] == 0) & seated[rows]  # the net's seats where it plays
                if bool(mine.any()):
                    r = float((u[rows][mine] / BB).mean())
                    key = id(pols[opp])
                    score[key] = 0.9 * score.get(key, r) + 0.1 * r
        if n_l and bool((who == lid).any()):  # the league exploiter: plain policy gradient, as the exploiter
            lm_ = who == lid
            lgl, vl = lexp(f[lm_], mybk[lm_], street[lm_], s[lm_])
            Gl = u[g[lm_], i[lm_]] / BB
            logpl = F.log_softmax(masked(lgl, legal[lm_]), 1)
            entl = -(logpl.exp() * logpl).masked_fill(~legal[lm_], 0).sum(1)
            advl = Gl - vl.detach()
            adv_scale_l = 0.99 * adv_scale_l + 0.01 * advl.abs().mean().item()
            lpl = logpl.gather(1, a[lm_][:, None]).squeeze(1)
            loss_l = (-advl / max(adv_scale_l, 1e-3) * lpl + 0.5 * (vl - Gl) ** 2 / 25 - 0.003 * entl).mean()
            opt_l.zero_grad()
            loss_l.backward()
            nn.utils.clip_grad_norm_(lexp.parameters(), 1.0)
            opt_l.step()
        if expl is not None:  # the exploiter: plain policy gradient against the main network, no pull
            xm = who == 1
            lgx, vx = expl(f[xm], mybk[xm], street[xm], s[xm])
            Gx = u[g[xm], i[xm]] / BB
            logpx = F.log_softmax(masked(lgx, legal[xm]), 1)
            entx = -(logpx.exp() * logpx).masked_fill(~legal[xm], 0).sum(1)
            advx = Gx - vx.detach()
            adv_scale_x = 0.99 * adv_scale_x + 0.01 * advx.abs().mean().item()
            lpx = logpx.gather(1, a[xm][:, None]).squeeze(1)
            loss_x = (-advx / max(adv_scale_x, 1e-3) * lpx + 0.5 * (vx - Gx) ** 2 / 25 - 0.003 * entx).mean()
            opt_x.zero_grad()
            loss_x.backward()
            nn.utils.clip_grad_norm_(expl.parameters(), 1.0)
            opt_x.step()
            nm = who == 0
            g, i, f, legal, mybk, street, s, a = (x[nm] for x in (g, i, f, legal, mybk, street, s, a))
        elif n_l:  # a league exploiter without the main one: the network learns from its own seats only
            nm = who == 0
            g, i, f, legal, mybk, street, s, a = (x[nm] for x in (g, i, f, legal, mybk, street, s, a))
        G = u[g, i] / BB  # big blinds won by the player who acted
        lg, v = net(f, mybk, street, s)
        logp = F.log_softmax(masked(lg, legal), 1)
        with torch.no_grad():
            lm, _ = magnet(f, mybk, street, s)
            logm = F.log_softmax(masked(lm, legal), 1)
        p = logp.exp()
        kl = (p * (logp - logm)).masked_fill(~legal, 0).sum(1)
        ent = -(p * logp).masked_fill(~legal, 0).sum(1)
        adv = G - v.detach()
        adv_scale = 0.99 * adv_scale + 0.01 * adv.abs().mean().item()
        adv = adv / max(adv_scale, 1e-3)
        lp_a = logp.gather(1, a[:, None]).squeeze(1)
        if args.pg == "neurd":
            # Neural Replicator Dynamics (Hennes et al. 2020; R-NaD): the update acts on the logits centred over the
            # legal actions, with the sampled action's value estimate adv / pi(a); logits stop moving beyond +-beta
            y = lg.masked_fill(~legal, 0)
            y = y - y.sum(1, keepdim=True) / legal.sum(1, keepdim=True)
            y_a = y.gather(1, a[:, None]).squeeze(1)
            q_a = adv / lp_a.detach().exp().clamp(min=0.05)
            live = ((q_a > 0) & (y_a < args.neurd_beta)) | ((q_a < 0) & (y_a > -args.neurd_beta))
            pg = -(q_a * y_a) * live
        else:
            pg = -adv * lp_a
        kl_w = args.kl
        if args.kl_start is not None and args.kl_ramp > 0:
            frac = min((time.time() - t0) / (args.kl_ramp * 60), 1.0)
            kl_w = args.kl_start + frac * (args.kl - args.kl_start)
        loss = (pg + 0.5 * (v - G) ** 2 / 25 + kl_w * kl - args.entropy * ent).mean()
        if (
            args.belief_weight > 0 and net.last_h is not None
        ):  # each live opponent's hand strength, from the hidden layer
            P_ = args.players
            rel = (i[:, None] + 1 + torch.arange(P_ - 1, device=dev)) % P_
            s_all = strength(dealer, bk[g], street).gather(1, rel)  # [n, P-1] the opponents' true strength now
            tgt = (s_all * BELIEF_BINS).long().clamp(0, BELIEF_BINS - 1)
            live_opp = f[:, seat_layout(P_, menu.A, menu.S)[0][:, 0]] > 0.5  # live (not folded, seated), table order
            bl = net.belief(net.last_h).view(len(i), P_ - 1, BELIEF_BINS)
            loss_b = F.cross_entropy(bl[live_opp], tgt[live_opp]) if bool(live_opp.any()) else bl.sum() * 0
            loss = loss + args.belief_weight * loss_b
        if dist is not None:  # towards search's policy on stored positions (expert iteration)
            j = torch.randint(len(dist["q"]), (args.distill_batch,), device=dev)
            lgd, _ = net(
                dist["feat"][j].float(), dist["bk"][j].long(), dist["street"][j].long(), dist["strength"][j].float()
            )
            logpd = F.log_softmax(masked(lgd, dist["legal"][j]), 1)
            loss_d = -(dist["target"][j] * logpd.masked_fill(~dist["legal"][j], 0)).sum(1).mean()
            loss = loss + args.distill_weight * loss_d
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pm, pa, pn in zip(magnet.parameters(), avg.parameters(), net.parameters()):
                pm.lerp_(pn, args.magnet_tau)
                pa.lerp_(pn, args.avg_tau)
        hands += args.batch
        now = time.time()
        if now - last_eval >= args.eval_minutes * 60 or now - t0 >= args.minutes * 60:
            torch.save({"net": net.state_dict(), "magnet": magnet.state_dict(), "avg": avg.state_dict(),
                        "opt": opt.state_dict(), "hands": hands}, ck + ".tmp")  # fmt: skip
            os.replace(ck + ".tmp", ck)
            if args.keep_hours and (now - t0) / 3600 >= kept + args.keep_hours:
                kept += args.keep_hours
                torch.save({"avg": avg.state_dict(), "hands": hands}, os.path.join(args.out, f"avg_{kept:g}h.pt"))
            rate = (hands - h_start) / (now - t0)
            m1, lo1, hi1 = duplicate(dealer, menu, avg, "tag", args.eval_hands, dev)
            m2, lo2, hi2 = duplicate(dealer, menu, avg, prev, args.eval_hands, dev)
            line = (
                f"{(now - t0) / 60:.1f} min {hands} hands {rate:.0f} hands/s pull {kl_w:.3f} loss {loss.item():.4f} "
                f"kl {kl.mean().item():.4f} entropy {ent.mean().item():.3f}"
                + (f" distill {loss_d.item():.4f}" if dist is not None else "")
                + (f" belief {loss_b.item():.4f}" if args.belief_weight > 0 else "")
                + f"  1-vs-8 vs tag: {m1:+.1f} bb/100 "
                f"[{lo1:+.1f}, {hi1:+.1f}]  vs previous checkpoint: {m2:+.1f} [{lo2:+.1f}, {hi2:+.1f}]"
            )
            for opp in ("reg", "maniac", "station", "random"):
                m3, lo3, hi3 = duplicate(dealer, menu, avg, opp, args.eval_hands // 4, dev)
                line += f"  vs {opp}: {m3:+.1f} [{lo3:+.1f}, {hi3:+.1f}]"
            if expl is not None:  # how much the current exploiter wins: a running exploitability reading
                m4, lo4, hi4 = duplicate(dealer, menu, expl, avg, args.eval_hands // 4, dev)
                line += f"  exploiter vs 8 x net: {m4:+.1f} [{lo4:+.1f}, {hi4:+.1f}]"
            log.write(line + "\n")
            print(line, flush=True)
            prev = copy.deepcopy(avg)
            if pool is not None and args.pool_snapshots:
                snaps = (snaps + [copy.deepcopy(avg).requires_grad_(False)])[-args.pool_snapshots :]
                pool = rules + extra + frozen + snaps
            if expl is not None and time.time() - last_reset >= args.exploit_reset * 60:
                if pool is not None:  # the old exploiters stay in the population, frozen
                    frozen.append(copy.deepcopy(expl).requires_grad_(False))
                    if lexp is not None:
                        frozen.append(copy.deepcopy(lexp).requires_grad_(False))
                        lexp.load_state_dict(avg.state_dict())
                        opt_l, adv_scale_l = torch.optim.Adam(lexp.parameters(), lr=args.lr), 1.0
                    if args.max_frozen:
                        frozen = frozen[-args.max_frozen :]
                    pool = rules + extra + frozen + snaps
                expl.load_state_dict(avg.state_dict())
                opt_x = torch.optim.Adam(expl.parameters(), lr=args.lr)
                adv_scale_x, last_reset = 1.0, time.time()
            last_eval = time.time()
    # the end of training (--minutes or --max-hands): save, or everything since the last evaluation is lost
    ck = os.path.join(args.out, "checkpoint.pt")
    torch.save({"net": net.state_dict(), "magnet": magnet.state_dict(), "avg": avg.state_dict(),
                "opt": opt.state_dict(), "hands": hands}, ck + ".tmp")  # fmt: skip
    os.replace(ck + ".tmp", ck)
    with open(os.path.join(args.out, "train.log"), "a") as f:
        f.write("stopped\n")


if __name__ == "__main__":
    main()
