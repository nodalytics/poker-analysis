"""A vectorised N-player no-limit hold'em engine on the GPU, for the 9-max neural blueprint (ninemax_selfplay.py).

G tables are played at once as tensors. Cards come from libbp5's bp_deal_n (buckets per street, showdown ranks and,
optionally, the cards). Seats are fixed: 0 = small blind, 1 = big blind, 2 = UTG ... P-1 = button (P >= 2).
Pre-flop UTG acts first; after the flop the first live player from the small blind. Blinds 50/100 chips (1 bb = 100).
Every player starts with STACK chips (100bb), or with their own stack: Tables(..., stacks=[G, P] chips, at least 2bb
each), for mixed stack depths and tournaments.

Short-handed tables (final tables down to heads-up): a stack of 0 is an empty seat. It is folded from the start and
never acts. The blinds (seats 0 and 1) must be seated; empty seats are the early ones (UTG, UTG+1, ...), so the
seats that remain keep their names counted from the button, as at a real short table. Heads-up (only the blinds
seated) the small blind is the button: it acts first before the flop and the big blind first after it.

Rules implemented:
- fold, check/call, raises to the sizes of a menu (below), all-in; a raise must be at least the last raise;
- an all-in below a full raise does not reopen the betting: players who already acted may then only call or fold;
- no raise when no other player could answer it (all others folded, all in, or unable to cover the bet);
- a round ends when every live player who is not all in has acted since the last full raise and matched the bet;
  if at most one player can still act, the board runs out;
- showdown with side pots: each contribution level forms a pot shared by the best live hands that paid into it;
- big-blind ante (RULES["ante"]): dead money posted by the big blind before the blind, at most what leaves it the
  blind; it counts toward the pot (and its size) but not toward the bet, and goes to the main pot.

Menu (MENUS): pre-flop open sizes in bb and re-raise multiples of the current bet by raise level; post-flop bet and
raise sizes as pot fractions, smaller once 3+ players are in the pot; caps on raises per street. Action slots are
fixed: 0 fold, 1 check/call, 2..2+S-1 the menu sizes, last all-in; illegal slots are masked.
"""

import ctypes
import json
import os

import numpy as np
import torch

BB = 100
SB = 50
# Game rules a run can change (set by ninemax_selfplay.set_mode from the run's config, so every tool that loads the
# run plays the same game). open_limp False: a player first in before the flop, from UTG to the button, cannot just
# call the big blind: fold, open or shove (the small blind may still complete). Limping strong hands to trap was a
# habit learned against the rule-based maniacs and stations in the training pool (2026-11-11).
# ante: the big-blind ante in big blinds (1.0 = the big blind also posts one big blind of dead money, the format of
# almost every tournament today); 0 = no ante (every run before nm15, 2026-11-13).
RULES = {"open_limp": True, "ante": 0.0}
STACK = 100 * BB

# open sizes (bb), re-raise multiples by level, max pre-flop raises, heads-up bet / raise fractions,
# multiway bet / raise fractions, max post-flop raises per street
MENUS = {
    "lean": ((2.5,), ((3.0,), (2.3,)), 3, (0.5, 1.0), (1.0,), (0.66,), (), 2),
    "medium": ((2.5,), ((3.0,), (2.3,)), 4, (0.33, 0.75, 1.5), (1.0,), (0.5, 1.0), (1.0,), 3),
    # tournaments (2026-11-11): the medium menu with the opens real players use at 10-25bb, 2x, 2.5x and 3x (the
    # partner: the open need not be 2.5x); the same number of action slots as medium (3 sizes)
    "mtt": ((2.0, 2.5, 3.0), ((3.0,), (2.3,)), 4, (0.33, 0.75, 1.5), (1.0,), (0.5, 1.0), (1.0,), 3),
    "wide": (
        (2.0, 2.5, 3.5),
        ((2.5, 3.5), (2.2, 3.0)),
        4,
        (0.25, 0.5, 0.75, 1.25, 2.0),
        (0.75, 1.5),
        (0.33, 0.75),
        (1.0,),
        3,
    ),  # fmt: skip
}


def read_buckets(absdir):
    """The abstraction's bucket counts (flop, turn, river) from its meta.txt: per street when it gives them
    (abstraction.cpp OUT FLOP TURN RIVER), else the one count for all three."""
    meta = {}
    for line in open(os.path.join(absdir, "meta.txt")):
        k, _, v = line.partition(" ")
        meta[k] = v.strip()
    b = int(meta["buckets"])
    return tuple(int(meta.get(f"buckets_{s}", b)) for s in ("flop", "turn", "river"))


class Dealer:
    def __init__(self, lib, absdir, players, buckets, dev, seed=1):
        self.lib = ctypes.CDLL(lib)
        if self.lib.bp_init(absdir.encode()) != 0:
            raise RuntimeError(f"no buckets in {absdir}")
        self.lib.bp_deal_n.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64] + [ctypes.c_void_p] * 3
        self.P, self.dev, self.seed = players, dev, seed
        self.Bs = read_buckets(absdir)  # flop, turn, river
        self.B = self.Bs[2]
        buckets = self.B
        # pre-flop class strength: the mean final (river) bucket of each of the 169 classes, scaled to 0..1
        bk, _, _ = self.raw(1 << 18)
        fin = bk[:, :, 3].ravel() / (buckets - 1)
        cls = bk[:, :, 0].ravel()
        self.pre_strength = torch.tensor(
            np.bincount(cls, weights=fin, minlength=169) / np.maximum(np.bincount(cls, minlength=169), 1),
            dtype=torch.float32,
            device=dev,
        )

    def raw(self, n, cards=False):
        bk = np.zeros((n, self.P, 4), np.int32)
        rk = np.zeros((n, self.P), np.int32)
        cd = np.zeros((n, 2 * self.P + 5), np.int32) if cards else None
        self.seed += 1
        self.lib.bp_deal_n(n, self.P, self.seed, bk.ctypes.data, rk.ctypes.data, cd.ctypes.data if cards else None)
        return bk, rk, cd

    def buckets_for(self, cards):
        """This abstraction's buckets and the showdown ranks for cards dealt elsewhere: cards [n, 2P + 5] (numpy)."""
        n = cards.shape[0]
        c = np.ascontiguousarray(cards, dtype=np.int32)
        bk = np.zeros((n, self.P, 4), np.int32)
        rk = np.zeros((n, self.P), np.int32)
        self.lib.bp_bucket_cards.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 3
        self.lib.bp_bucket_cards(n, self.P, c.ctypes.data, bk.ctypes.data, rk.ctypes.data)
        t = lambda a: torch.tensor(a, device=self.dev, dtype=torch.long)  # noqa: E731
        return t(bk), t(rk)

    def deal(self, n, cards=False):
        bk, rk, cd = self.raw(n, cards)
        t = lambda a: torch.tensor(a, device=self.dev, dtype=torch.long)  # noqa: E731
        return t(bk), t(rk), (cd if cards else None)


class Menu:
    def __init__(self, name, dev):
        opens, rer, self.max_pre, hb, hr, mb, mr, self.max_post = MENUS[name]
        self.S = max(len(opens), *(len(x) for x in rer), len(hb), len(hr), len(mb), len(mr))
        pad = lambda x: torch.tensor(list(x) + [float("nan")] * (self.S - len(x)), device=dev)  # noqa: E731
        self.opens = pad([o * BB for o in opens])
        self.rer = torch.stack([pad(x) for x in rer])  # [levels, S]
        self.hb, self.hr, self.mb, self.mr = pad(hb), pad(hr), pad(mb), pad(mr)
        self.A = self.S + 3  # fold, call, sizes, all-in


class Tables:
    """The state of G tables. step() applies one action per table at the tables still in play."""

    def __init__(self, G, P, menu, dev, stacks=None):
        self.G, self.P, self.menu, self.dev = G, P, menu, dev
        z = lambda dt=torch.long: torch.zeros(G, P, dtype=dt, device=dev)  # noqa: E731
        # chips each player starts the hand with (contributions can never exceed them)
        self.stack = torch.full((G, P), STACK, dtype=torch.long, device=dev) if stacks is None else stacks.long()
        self.seated = self.stack > 0  # a stack of 0 is an empty seat
        assert bool(((self.stack >= 2 * BB) | ~self.seated).all()), "every stack must be at least 2bb (0: empty)"
        assert bool(self.seated[:, :2].all()), "the blinds must be seated"
        self.hu = self.seated.sum(1) == 2  # heads-up: the big blind acts first after the flop
        self.contrib = z()
        self.contrib[:, 0], self.contrib[:, 1] = SB, BB
        self.folded, self.allin, self.acted = ~self.seated, z(torch.bool), z(torch.bool)
        # the big-blind ante: dead money [G], and the chips each player can still put into the pot [G, P]
        self.dead = torch.clamp(self.stack[:, 1] - BB, max=int(round(RULES["ante"] * BB)))
        self.avail = self.stack.clone()
        self.avail[:, 1] -= self.dead
        self.allin[:, 1] = self.avail[:, 1] <= BB
        g = lambda v: torch.full((G,), v, dtype=torch.long, device=dev)  # noqa: E731
        self.bet, self.last_raise, self.raises, self.street = g(BB), g(BB), g(0), g(0)
        self.pre_raises = g(0)  # complete raises pre-flop: 1 = opened, 2 = 3-bet pot, ...
        self.aggressor = g(-1)
        self.to_act = g(1)  # the scan for the next actor starts after this seat
        self.done = torch.zeros(G, dtype=torch.bool, device=dev)
        self.showdown = torch.zeros(G, dtype=torch.bool, device=dev)
        self.ar = torch.arange(P, device=dev)
        self.actor = g(-1)
        self.advance()

    # ---- whose turn ----
    def needs(self):
        return ~self.folded & ~self.allin & (~self.acted | (self.contrib < self.bet[:, None]))

    def advance(self):
        """Find each live table's next actor; end betting rounds, deal the next street or finish the hand."""
        for _ in range(5):  # at most 4 street changes in one call
            live = ~self.done
            alone = (~self.folded).sum(1) <= 1  # everyone else folded: the hand is over, whoever was still to act
            need = self.needs() & (live & ~alone)[:, None]
            order = (self.to_act[:, None] + 1 + self.ar) % self.P
            need_o = need.gather(1, order)
            has = need_o.any(1)
            first = need_o.float().argmax(1)
            self.actor = torch.where(has, order.gather(1, first[:, None]).squeeze(1), torch.full_like(first, -1))
            over = live & ~has  # round over at these tables
            if not over.any():
                return
            n_live = (~self.folded).sum(1)
            can_act = (~self.folded & ~self.allin).sum(1)
            finish = over & ((n_live <= 1) | (self.street == 3) | (can_act <= 1))
            self.showdown |= finish & (n_live > 1)
            self.done |= finish
            nxt = over & ~finish
            if not nxt.any():
                return
            self.street = torch.where(nxt, self.street + 1, self.street)
            self.raises = torch.where(nxt, 0, self.raises)
            self.last_raise = torch.where(nxt, BB, self.last_raise)
            self.acted &= ~nxt[:, None]
            # the scan starts at the small blind (heads-up: at the big blind, the small blind being the button)
            self.to_act = torch.where(nxt, torch.where(self.hu, 0, self.P - 1), self.to_act)

    # ---- the action menu ----
    def targets(self, g, i):
        """Raise-to totals of the menu sizes for tables g with actor i: [len(g), S] (nan where unused)."""
        m = self.menu
        bet, pot = self.bet[g], self.contrib[g].sum(1) + self.dead[g]
        to_call = bet - self.contrib[g, i]
        pre = self.street[g] == 0
        lvl = (self.raises[g] - 1).clamp(0, len(m.rer) - 1)
        pre_t = torch.where((self.raises[g] == 0)[:, None], m.opens[None, :], m.rer[lvl] * bet[:, None])
        multi = (~self.folded[g]).sum(1) > 2
        unopened = (self.raises[g] == 0)[:, None]
        frac = torch.where(multi[:, None], torch.where(unopened, m.mb, m.mr), torch.where(unopened, m.hb, m.hr))
        post_t = bet[:, None] + frac * (pot + to_call)[:, None]
        t = torch.where(pre[:, None], pre_t, post_t).floor()
        return t

    def legal(self, g, i):
        m = self.menu
        bet = self.bet[g]
        to_call = bet - self.contrib[g, i]
        t = self.targets(g, i)
        cap = torch.where(self.street[g] == 0, m.max_pre, m.max_post)
        reopened = ~self.acted[g, i]  # False after an incomplete all-in raise: call or fold only
        # no raise when no other player could answer it: every other player has folded, is all in, or has no more
        # chips than the current bet (calling is all they can do). Only possible with unequal stacks.
        can_answer = ~self.folded[g] & ~self.allin[g] & (self.avail[g] > bet[:, None])
        can_answer[torch.arange(len(g), device=self.dev), i] = False
        others = can_answer.any(1)
        reopened = reopened & others
        stk = self.avail[g, i]
        ok_size = (
            ~torch.isnan(t)
            & (t - bet[:, None] >= self.last_raise[g][:, None])
            & (t < stk[:, None])
            & (self.raises[g] < cap)[:, None]
            & reopened[:, None]
        )
        fold = to_call > 0
        call = torch.ones_like(fold)
        allin = (stk > bet) & reopened  # shoving more than the bet (with less, calling is the all-in)
        if not RULES["open_limp"]:  # first in from UTG to the button: no limping, when there is a way in by raising
            first_in = (self.street[g] == 0) & (self.pre_raises[g] == 0) & (bet == BB) & (i >= 2)
            call = call & ~(first_in & (ok_size.any(1) | allin))
        return torch.cat([fold[:, None], call[:, None], ok_size, allin[:, None]], 1), torch.nan_to_num(t, nan=0)

    def step(self, g, a):
        """Tables g (their current actors) take action slots a."""
        i = self.actor[g]
        legal, t = self.legal(g, i)
        assert bool(legal.gather(1, a[:, None]).all()), "illegal action"
        S = self.menu.S
        bet = self.bet[g]
        fold = a == 0
        call = a == 1
        size = (a >= 2) & (a < 2 + S)
        stk = self.avail[g, i]
        total = torch.where(size, t.gather(1, (a - 2).clamp(0, S - 1)[:, None]).squeeze(1).long(), stk)
        total = torch.where(call, torch.minimum(bet, stk), total)
        total = torch.where(fold, self.contrib[g, i], total)
        inc = total - bet
        full = ~fold & ~call & (inc >= self.last_raise[g])
        self.folded[g, i] |= fold
        self.contrib[g, i] = total
        self.allin[g, i] |= total >= stk
        # a full raise reopens the betting: everyone else must act again
        reset = full[:, None] & (self.ar[None, :] != i[:, None])
        self.acted[g] = torch.where(reset, False, self.acted[g])
        self.acted[g, i] = True
        self.last_raise[g] = torch.where(full, inc, self.last_raise[g])
        self.raises[g] = self.raises[g] + full.long()
        self.pre_raises[g] = self.pre_raises[g] + (full & (self.street[g] == 0)).long()
        self.aggressor[g] = torch.where(full | (inc > 0) & ~fold & ~call, i, self.aggressor[g])
        self.bet[g] = torch.maximum(bet, total)
        self.to_act[g] = i
        self.advance()

    # ---- payoffs ----
    def payoffs(self, ranks):
        """Chips won minus chips paid, per seat, for finished tables [G, P] (float)."""
        c = self.contrib.float()
        live = ~self.folded
        win = torch.zeros_like(c)
        levels = torch.sort(self.contrib, 1).values.float()
        prev = torch.zeros(self.G, device=self.dev)
        big = torch.iinfo(torch.int32).max
        for k in range(self.P):
            L = levels[:, k]
            part = torch.minimum(c, L[:, None]) - torch.minimum(c, prev[:, None])
            pot = part.sum(1)
            elig = live & (c >= L[:, None]) & (pot > 0)[:, None]
            r = torch.where(elig, ranks, torch.full_like(ranks, big))
            best = r.min(1).values
            winners = elig & (ranks == best[:, None])
            nw = winners.sum(1)
            # no eligible live player at this level (cannot happen in legal play): the best live hand takes it
            fallback = (nw == 0) & (pot > 0)
            if fallback.any():
                rl = torch.where(live, ranks, torch.full_like(ranks, big))
                winners = torch.where(fallback[:, None], live & (ranks == rl.min(1).values[:, None]), winners)
                nw = winners.sum(1)
            win += winners.float() * (pot / nw.clamp(min=1))[:, None]
            prev = L
        if bool((self.dead > 0).any()):  # the ante goes to the main pot: every live player is eligible
            rl = torch.where(live, ranks, torch.full_like(ranks, big))
            main = live & (ranks == rl.min(1).values[:, None])
            win += main.float() * (self.dead.float() / main.sum(1).clamp(min=1))[:, None]
            c = c.clone()
            c[:, 1] += self.dead.float()
        return win - c


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_dealer(run, lib, absdir, players, B, dev, seed, tmp):
    """A Dealer for one run, through its own copy of the bucket library: the library keeps the first abstraction it
    loads for the whole process, so runs with different abstractions in one process need one copy each."""
    import shutil

    d = os.path.dirname(run) if run.endswith(".pt") else run
    if absdir is None:
        absdir = os.path.join(ROOT, json.load(open(os.path.join(d, "config.json"))).get("abs", "blueprint/out"))
    copy = os.path.join(tmp, f"libbp_{len(os.listdir(tmp))}.so")
    shutil.copy(lib, copy)
    return Dealer(copy, absdir, players, B, dev, seed=seed)


def strength(dealer, bk, street):
    """0..1 strength of each player's current bucket: pre-flop class strength, else bucket / (B - 1)."""
    b = bk.gather(2, street[:, None, None].expand(-1, bk.shape[1], 1)).squeeze(2)
    pre = dealer.pre_strength[b.clamp(max=168)]
    denom = torch.tensor([1.0, *(x - 1.0 for x in dealer.Bs)], device=b.device)[street][:, None]
    return torch.where(street[:, None] == 0, pre, b.float() / denom)
