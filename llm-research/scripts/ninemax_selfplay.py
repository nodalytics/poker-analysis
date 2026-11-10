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
from the population holds the others: rule players (tag, maniac, station, random) and the last few checkpoints. The
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ninemax_engine import BB, STACK, Dealer, Menu, Tables, strength


def features(T, g, i):
    """Public features of tables g for actor i: [len(g), NF]."""
    P = T.P
    c = T.contrib[g].float()
    pot = c.sum(1)
    mine = c.gather(1, i[:, None]).squeeze(1)
    bet = T.bet[g].float()
    to_call = bet - mine
    left = STACK - mine
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
    legal, t = T.legal(g, i)
    f.append(legal.float())
    S = T.menu.S
    f.append(((t - bet[:, None]) / pot.clamp(min=1)[:, None]).clamp(0, 10) / 10 * legal[:, 2 : 2 + S].float())
    return torch.cat(f, 1), legal


class NineNet(nn.Module):
    def __init__(self, nfeat, A, buckets, hidden=512, blocks=4, emb=32):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(169, emb)] + [nn.Embedding(buckets, emb) for _ in range(3)])
        self.inp = nn.Linear(nfeat + 4 * emb + 1, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(blocks)
        )
        self.norm = nn.LayerNorm(hidden)
        self.pi = nn.Linear(hidden, A)
        self.v = nn.Linear(hidden, 1)

    def forward(self, feat, bk, street, st):
        """bk: [n, 4] this player's buckets per street; streets after the current one are masked out."""
        e = [self.emb[s](bk[:, s].clamp(max=self.emb[s].num_embeddings - 1)) * (street >= s).float()[:, None]
             for s in range(4)]  # fmt: skip
        h = self.inp(torch.cat([feat] + e + [st[:, None]], 1))
        for blk in self.blocks:
            h = h + blk(h)
        h = self.norm(h)
        return self.pi(h), self.v(h).squeeze(1)


def masked(logits, legal):
    return logits.masked_fill(~legal, -1e9)


def tag_probs(T, g, i, legal, s, dealer):
    """A rule-based tight-aggressive player on card strength s (0..1) and pot odds; deterministic."""
    c = T.contrib[g].float()
    pot = c.sum(1)
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


def rule_probs(name, T, g, i, legal, s, dealer):
    """Rule-based opponents: tag (tight-aggressive), maniac (raises and shoves most of the time), station (always
    calls or checks), random (uniform over the legal actions)."""
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


def play(dealer, menu, policies, bk, rk, record=False, dev="cuda", learner=None, assign=None):
    """Play len(bk) hands; policies[seat] is a NineNet or a rule name (rule_probs). Returns payoffs [G, P] (chips)
    and, with record, the decisions (only those of the policy `learner` when it is given).

    assign [G, P]: per-table seating instead; policies is then the list of distinct policies and assign[g, seat]
    indexes it. All tables are played in one loop, whatever their seating: the loop's length, not the number of
    tables, sets the time, so one loop over a mixed batch is several times faster than one loop per seating."""
    G, P = bk.shape[0], bk.shape[1]
    T = Tables(G, P, menu, dev)
    rec = []
    if assign is None:
        uniq = list({id(p): p for p in policies}.values())  # each distinct policy once; seat -> its index
        seat_pol = torch.tensor([next(k for k, q in enumerate(uniq) if q is p) for p in policies], device=dev)
        assign = seat_pol[None, :].expand(G, P)
    else:
        uniq = list(policies)
    lid = next((k for k, q in enumerate(uniq) if q is learner), -1)
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
                probs = rule_probs(pol, T, g[m], i[m], legal[m], s[m], dealer)
            else:
                with torch.no_grad():
                    lg, _ = pol(feat[m], mybk[m], T.street[g][m], s[m])
                probs = F.softmax(masked(lg, legal[m]), 1)
            a[m] = torch.multinomial(probs, 1).squeeze(1)
        if record:
            keep = torch.ones_like(i, dtype=torch.bool) if learner is None else pol_of == lid
            rec.append(tuple(x[keep] for x in (g, i, feat, legal, mybk, T.street[g].clone(), s, a)))
        T.step(g, a)
    return T.payoffs(rk), rec


def duplicate(dealer, menu, hero, villain, n, dev):
    """Hero in each seat in turn against P-1 copies of villain, on the same n deals: hero's bb/100 and 95% CI."""
    P = dealer.P
    bk, rk, _ = dealer.deal(n)
    per = torch.zeros(n, device=dev)
    for seat in range(P):
        pols = [villain] * P
        pols[seat] = hero
        u, _ = play(dealer, menu, pols, bk, rk, dev=dev)
        per += u[:, seat] / BB
    x = (per / P).cpu().numpy()  # hero's average over the seats, per deal
    m, se = x.mean() * 100, x.std() / math.sqrt(len(x)) * 100
    return m, m - 1.96 * se, m + 1.96 * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--players", type=int, default=9)
    ap.add_argument("--menu", default="medium")
    ap.add_argument("--abs", default="blueprint/out")
    ap.add_argument("--lib", default="blueprint/libbp5.so")
    ap.add_argument("--batch", type=int, default=8192, help="tables per training batch")
    ap.add_argument("--minutes", type=float, default=1440)
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
    ap.add_argument("--pool-share", type=float, default=0.0, help="share of each batch played against the population")
    ap.add_argument("--pool", default="tag,maniac,station,random", help="rule opponents in the population")
    ap.add_argument("--pool-snapshots", type=int, default=5, help="past checkpoints kept in the population")
    ap.add_argument("--chunks", type=int, default=4, help="population chunks per batch (one opponent each)")
    args = ap.parse_args()
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
    T0 = Tables(1, args.players, menu, dev)
    nfeat = features(T0, torch.zeros(1, dtype=torch.long, device=dev), T0.actor[:1])[0].shape[1]
    net = NineNet(nfeat, menu.A, B).to(dev)
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
    prev = copy.deepcopy(avg)
    rules = [x for x in args.pool.split(",") if x] if args.pool else []
    snaps = []
    pool = (rules or None) if args.pool_share > 0 else None
    json.dump({**vars(args), "nfeat": nfeat, "actions": menu.A}, open(os.path.join(args.out, "config.json"), "w"),
              indent=1)  # fmt: skip
    log = open(os.path.join(args.out, "progress.log"), "a", buffering=1)
    t0 = last_eval = time.time()
    h_start, adv_scale = hands, 1.0
    while time.time() - t0 < args.minutes * 60:
        bk, rk, _ = dealer.deal(args.batch)
        # self-play on part of the batch; on the rest, chunks of tables where the net holds some seats and one
        # opponent from the population (rule players, past checkpoints) holds the others. The net learns from its
        # own seats only.
        n_pop = int(args.batch * args.pool_share) // args.chunks * args.chunks if pool else 0
        # one seating table for the whole batch (policy 0 = the net), played in a single loop
        assign = torch.zeros(args.batch, args.players, dtype=torch.long, device=dev)
        pols = [net] + (pool or [])
        for c in range(args.chunks if n_pop else 0):
            rows = slice(c * n_pop // args.chunks, (c + 1) * n_pop // args.chunks)
            opp = 1 + int(torch.randint(len(pool), ()).item())
            k = int(torch.randint(1, args.players, ()).item())  # the net holds k seats, chosen at random
            theirs = torch.randperm(args.players, device=dev)[k:]
            assign[rows, theirs] = opp
        u, rec = play(dealer, menu, pols, bk, rk, record=True, dev=dev, learner=net, assign=assign)
        g, i, f, legal, mybk, street, s, a = (torch.cat([r[k] for r in rec]) for k in range(8))
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
            rate = (hands - h_start) / (now - t0)
            m1, lo1, hi1 = duplicate(dealer, menu, avg, "tag", args.eval_hands, dev)
            m2, lo2, hi2 = duplicate(dealer, menu, avg, prev, args.eval_hands, dev)
            line = (
                f"{(now - t0) / 60:.1f} min {hands} hands {rate:.0f} hands/s pull {kl_w:.3f} loss {loss.item():.4f} "
                f"kl {kl.mean().item():.4f} entropy {ent.mean().item():.3f}  1-vs-8 vs tag: {m1:+.1f} bb/100 "
                f"[{lo1:+.1f}, {hi1:+.1f}]  vs previous checkpoint: {m2:+.1f} [{lo2:+.1f}, {hi2:+.1f}]"
            )
            for opp in ("maniac", "station", "random"):
                m3, lo3, hi3 = duplicate(dealer, menu, avg, opp, args.eval_hands // 4, dev)
                line += f"  vs {opp}: {m3:+.1f} [{lo3:+.1f}, {hi3:+.1f}]"
            log.write(line + "\n")
            print(line, flush=True)
            prev = copy.deepcopy(avg)
            if pool is not None and args.pool_snapshots:
                snaps = (snaps + [copy.deepcopy(avg).requires_grad_(False)])[-args.pool_snapshots :]
                pool = rules + snaps
            last_eval = time.time()
    with open(os.path.join(args.out, "train.log"), "a") as f:
        f.write("stopped\n")


if __name__ == "__main__":
    main()
