"""An MLP that learns heads-up no-limit hold'em from scratch by playing itself, millions of hands a second, on the GPU.

It plays the blueprint's simplified game (the betting tree in RUN/tree.tsv and the card buckets of ABS, read by
libbp's bp_deal), so everything that measures the blueprint measures it too: training exports its policy in the
blueprint's table layout (OUT/avg.f32 + tree.tsv + iterations.txt), and blueprint/lbr_trend.sh, blueprint_bot.py,
the Slumbot client and bot_sim play it unchanged.

Learning: one network for both seats, an actor-critic with a KL pull toward a slowly moving "magnet" copy of itself
(regularised Nash dynamics, as in R-NaD/DeepNash and magnetic mirror descent; the KL penalty is the same device as
RLHF's pull toward a reference model). Plain self-play policy gradients cycle in zero-sum games; the magnet damps the
cycling. The exported policy is a slow moving average of the weights.

Dealing: half of every batch is fair (uniform random deals). The other half is spread evenly over the nine pairings
of strong / medium / weak final hands for the two players (coolers, strong against weak, weak against weak), so rare
spots are seen often. Every deal carries the importance weight p_fair / p_dealt of its pairing, so the gradient is
the fair game's gradient and the strategy learned is for the real deal distribution, not the skewed one.

Every EVAL_MINUTES it also plays the current policy against a blueprint table (--vs RUN, both seats, fair deals) in
the same simplified game: an exact head-to-head with no bet mapping, in bb/100 with a 95% interval.

--distill RUN: first copy RUN's table for --distill-minutes (the MLP + blueprint start), then fine-tune by
self-play with the KL reference set to that copy; with --magnet-tau 0 the reference stays the blueprint copy for
good, as RLHF keeps its reference model.

Usage (lab, GPU venv): .venv-gpu/bin/python scripts/mlp_selfplay.py OUT [--tree blueprint/run2] [--abs blueprint/out]
    [--lib blueprint/libbp3.so] [--batch 65536] [--minutes 1440] [--vs blueprint/run2] [--eval-minutes 30]
"""

import argparse
import copy
import ctypes
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAXA = 6
STACK = 20000


class Tree:
    def __init__(self, path, dev):
        rows = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(path, "tree.tsv"))][1:]
        n = len(rows)
        kind, player, street = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64)
        c = np.zeros((n, 2), np.int64)
        offset = np.zeros(n, np.int64)
        child = np.full((n, MAXA), -1, np.int64)
        labels = [r[7].split(",") if r[7] else [] for r in rows]
        self.vocab = sorted({x for ls in labels for x in ls})
        lid = {x: i for i, x in enumerate(self.vocab)}
        lab = np.full((n, MAXA), -1, np.int64)
        for r in rows:
            i = int(r[0])
            kind[i], player[i], street[i] = int(r[1]), int(r[2]), int(r[3])
            c[i] = int(r[4]), int(r[5])
            offset[i] = int(r[6])
            if r[8]:
                ch = [int(x) for x in r[8].split(",")]
                child[i, : len(ch)] = ch
                lab[i, : len(ch)] = [lid[x] for x in labels[i]]
        # who folded at each fold node, the label of the edge into each node, raises so far this street
        folder = np.full(n, -1, np.int64)
        into = np.full(n, len(self.vocab), np.int64)
        raises = np.zeros(n, np.int64)
        for i in range(n):  # parents precede children in the file (built depth first)
            for k in range(MAXA):
                j = child[i, k]
                if j < 0:
                    continue
                into[j] = lab[i, k]
                if kind[j] == 1:
                    folder[j] = player[i]
                name = self.vocab[lab[i, k]]
                aggressive = name not in ("fold", "call", "check")
                raises[j] = 0 if street[j] != street[i] else raises[i] + int(aggressive)
        pot = c.sum(1)
        feat = np.zeros((n, 11 + len(self.vocab) + 1), np.float32)
        feat[np.arange(n), street] = 1
        feat[:, 4] = player
        feat[:, 5] = pot / (2 * STACK)
        mine = c[np.arange(n), player]
        theirs = c[np.arange(n), 1 - player]
        feat[:, 6] = np.maximum(theirs - mine, 0) / np.maximum(pot, 1)
        feat[:, 7] = (theirs > mine).astype(np.float32)
        feat[:, 8] = (STACK - c.max(1)) / STACK
        feat[:, 9] = np.minimum((STACK - c.max(1)) / np.maximum(pot, 1), 50) / 50
        feat[:, 10] = np.minimum(raises, 4) / 4
        feat[np.arange(n), 11 + into] = 1
        t = lambda a: torch.tensor(a, device=dev)  # noqa: E731
        self.n, self.kind, self.player, self.street, self.c = n, t(kind), t(player), t(street), t(c)
        self.child, self.lab, self.folder, self.feat, self.offset = t(child), t(lab), t(folder), t(feat), offset
        self.na = (child >= 0).sum(1)
        self.rows = rows


class Net(nn.Module):
    def __init__(self, nfeat, nvocab, buckets, hidden=256, emb=32):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(169, emb)] + [nn.Embedding(buckets, emb) for _ in range(3)])
        self.body = nn.Sequential(
            nn.Linear(nfeat + emb + 1, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )  # fmt: skip
        self.pi = nn.Linear(hidden, nvocab)
        self.v = nn.Linear(hidden, 1)

    def forward(self, feat, street, bucket, strength):
        e = torch.zeros(len(street), self.emb[0].embedding_dim, device=feat.device)
        for s in range(4):
            m = street == s
            if m.any():
                e[m] = self.emb[s](bucket[m])
        h = self.body(torch.cat([feat, e, strength.unsqueeze(1)], 1))
        return self.pi(h), self.v(h).squeeze(1)


def masked_logits(logits, labs):
    """Gather the vocabulary logits of each node's actions; -1e9 where the node has fewer actions (finite, so that
    p * log p is 0 there and the KL and entropy gradients do not become 0 * inf = NaN)."""
    g = logits.gather(1, labs.clamp(min=0))
    return g.masked_fill(labs < 0, -1e9)


class Dealer:
    def __init__(self, lib, absdir, buckets, dev, seed=1):
        self.lib = ctypes.CDLL(lib)
        if self.lib.bp_init(absdir.encode()) != 0:
            raise RuntimeError(f"no buckets in {absdir}")
        self.lib.bp_deal.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
        self.B, self.dev, self.seed = buckets, dev, seed
        # pre-flop class strength (mean final strength) and the fair frequency of each strength pairing
        bk, _ = self.raw(1 << 20)
        fin = bk[:, :, 3] / (buckets - 1)
        cls_sum = np.bincount(bk[:, :, 0].ravel(), weights=fin.ravel(), minlength=169)
        cls_n = np.bincount(bk[:, :, 0].ravel(), minlength=169)
        self.pre_strength = torch.tensor(cls_sum / np.maximum(cls_n, 1), dtype=torch.float32, device=dev)
        cell = self.cells(bk)
        self.p_fair = np.bincount(cell, minlength=9) / len(cell)

    def raw(self, n):
        bk = np.zeros((n, 2, 4), np.int32)
        rk = np.zeros((n, 2), np.int32)
        self.seed += 1
        self.lib.bp_deal(n, self.seed, bk.ctypes.data, rk.ctypes.data)
        return bk, rk

    def cells(self, bk):
        """Strength pairing of each deal: tier of each player's final hand, 0 weak (<1/3), 1 medium, 2 strong (>=0.8)."""
        s = bk[:, :, 3] / (self.B - 1)
        tier = np.where(s < 1 / 3, 0, np.where(s < 0.8, 1, 2))
        return tier[:, 0] * 3 + tier[:, 1]

    def deal(self, n, mixed=True):
        """n deals: half fair, half spread evenly over the 9 pairings, each with weight p_fair / p_dealt."""
        if not mixed:
            bk, rk = self.raw(n)
            w = np.ones(n, np.float32)
        else:
            pool_bk, pool_rk = self.raw(4 * n)
            cell = self.cells(pool_bk)
            half = n // 2
            pick = [np.arange(half)]
            per = (n - half) // 9
            rng = np.random.default_rng(self.seed)
            for c in range(9):
                idx = np.where(cell[half:] == c)[0] + half
                pick.append(rng.choice(idx, per, replace=len(idx) < per))
            pick = np.concatenate(pick)
            bk, rk = pool_bk[pick], pool_rk[pick]
            q = 0.5 * self.p_fair + 0.5 / 9
            w = (self.p_fair / q)[self.cells(bk)].astype(np.float32)
        t = lambda a: torch.tensor(a, device=self.dev, dtype=torch.long)  # noqa: E731
        return t(bk), t(rk), torch.tensor(w, device=self.dev)


def obs(tree, dealer, nodes, games, bk):
    p, s = tree.player[nodes], tree.street[nodes]
    b = bk[games, p, s]
    strength = torch.where(s == 0, dealer.pre_strength[b.clamp(max=168)], b.float() / (dealer.B - 1))
    return tree.feat[nodes], s, b, strength


def play(tree, dealer, policies, bk, rk, record=True):
    """Play one batch to the end. policies[seat] is a net or a table; returns decisions and payoff to seat 0 (chips)."""
    B = len(bk)
    node = torch.zeros(B, dtype=torch.long, device=bk.device)
    rec = []
    while True:
        dec = tree.kind[node] == 0
        if not dec.any():
            break
        games = dec.nonzero().squeeze(1)
        nd = node[games]
        f, s, b, st = obs(tree, dealer, nd, games, bk)
        labs = tree.lab[nd]
        seat = tree.player[nd]
        a = torch.zeros(len(games), dtype=torch.long, device=bk.device)
        for k, pol in enumerate(policies):
            m = seat == k
            if not m.any():
                continue
            if isinstance(pol, nn.Module):
                with torch.no_grad():
                    lg, _ = pol(f[m], s[m], b[m], st[m])
                probs = F.softmax(masked_logits(lg, labs[m]), 1)
            else:  # a blueprint table: average strategy at offset + bucket * na
                probs = pol.probs(nd[m], b[m])
            a[m] = torch.multinomial(probs, 1).squeeze(1)
        if record:
            rec.append((games, nd, f, s, b, st, labs, a))
        node[games] = tree.child[nd, a]
    c0, c1 = tree.c[node, 0], tree.c[node, 1]
    fold = tree.kind[node] == 1
    loser = torch.where(tree.folder[node] == 1, c1, c0)
    u_fold = torch.where(tree.folder[node] == 0, -loser, loser)
    paid = torch.minimum(c0, c1)
    u_show = paid * torch.sign(rk[:, 1] - rk[:, 0])  # lower rank is stronger
    return rec, torch.where(fold, u_fold, u_show).float()


class Table:
    """A blueprint run's average strategy, on the GPU, for the head-to-head check."""

    def __init__(self, run, tree, dev):
        avg = np.fromfile(os.path.join(run, "avg.f32"), dtype=np.float32)
        self.avg = torch.tensor(avg, device=dev)
        self.offset = torch.tensor(tree.offset, device=dev)
        self.na = torch.tensor(tree.na, device=dev)
        self.ar = torch.arange(MAXA, device=dev)

    def probs(self, nd, b):
        na = self.na[nd]
        idx = self.offset[nd].unsqueeze(1) + (b * na).unsqueeze(1) + self.ar
        ok = self.ar.unsqueeze(0) < na.unsqueeze(1)
        s = torch.where(ok, self.avg[idx.clamp(max=len(self.avg) - 1)], torch.zeros((), device=nd.device))
        tot = s.sum(1, keepdim=True)
        uni = ok.float() / na.unsqueeze(1)
        return torch.where(tot > 0, s / tot.clamp(min=1e-30), uni)


def export(net, tree, dealer, out, hands):
    """Write the policy in the blueprint's layout: OUT/avg.f32 (+ tree.tsv, iterations.txt) for lbr_trend.sh etc."""
    total = int(max(tree.offset[i] + (169 if tree.rows[i][3] == "0" else dealer.B) * tree.na[i]
                    for i in range(tree.n) if tree.rows[i][1] == "0"))  # fmt: skip
    table = np.zeros(total, np.float32)
    dec = np.where(tree.kind.cpu().numpy() == 0)[0]
    dev = tree.feat.device
    for chunk in np.array_split(dec, max(1, len(dec) // 1500)):
        nd = torch.tensor(chunk, device=dev)
        st = tree.street[nd]
        nb = torch.where(st == 0, 169, dealer.B)
        reps = nb.max().item()
        ndr = nd.repeat_interleave(reps)
        b = torch.arange(reps, device=dev).repeat(len(nd))
        keep = b < nb.repeat_interleave(reps)
        ndr, b = ndr[keep], b[keep]
        s = tree.street[ndr]
        strength = torch.where(s == 0, dealer.pre_strength[b.clamp(max=168)], b.float() / (dealer.B - 1))
        with torch.no_grad():
            lg, _ = net(tree.feat[ndr], s, b, strength)
            p = F.softmax(masked_logits(lg, tree.lab[ndr]), 1).cpu().numpy()
        ndn, bn = ndr.cpu().numpy(), b.cpu().numpy()
        na = tree.na[ndn]
        for a in range(MAXA):
            m = a < na
            table[tree.offset[ndn[m]] + bn[m] * na[m] + a] = p[m, a]
    table.tofile(os.path.join(out, "avg.f32.tmp"))
    os.replace(os.path.join(out, "avg.f32.tmp"), os.path.join(out, "avg.f32"))
    with open(os.path.join(out, "iterations.txt"), "w") as f:
        f.write(f"{hands}\n")


def h2h(tree, dealer, net, table, hands, batch):
    """Policy vs blueprint table, both seats, fair deals: bb/100 for the policy with a 95% interval."""
    res = []
    for k in range(max(1, hands // (2 * batch))):
        bk, rk, _ = dealer.deal(batch, mixed=False)
        _, u = play(tree, dealer, [net, table], bk, rk, record=False)  # policy is seat 0
        _, v = play(tree, dealer, [table, net], bk, rk, record=False)  # same cards, seats swapped
        res.append(((u - v) / 2 / 100).cpu().numpy())  # bb per hand, duplicate pairs
    x = np.concatenate(res)
    m, se = x.mean() * 100, x.std() / math.sqrt(len(x)) * 100
    return m, m - 1.96 * se, m + 1.96 * se, len(x) * 2


def distill(net, opt, tree, dealer, teacher, args, table):
    """Phase 1: the teacher table plays itself; the net learns its probabilities at every spot reached (cross-entropy
    to the table's distribution) and its value head learns the hands' results."""
    log = open(os.path.join(args.out, "progress.log"), "a", buffering=1)
    t0, last, n = time.time(), time.time(), 0
    while time.time() - t0 < args.distill_minutes * 60:
        bk, rk, _ = dealer.deal(args.batch, mixed=False)
        rec, u0 = play(tree, dealer, [teacher, teacher], bk, rk)
        games, nd = torch.cat([r[0] for r in rec]), torch.cat([r[1] for r in rec])
        f, s, b, st = (torch.cat([r[k] for r in rec]) for k in (2, 3, 4, 5))
        labs = torch.cat([r[6] for r in rec])
        target = teacher.probs(nd, b)
        G = torch.where(tree.player[nd] == 0, u0[games], -u0[games]) / 100
        lg, v = net(f, s, b, st)
        logp = F.log_softmax(masked_logits(lg, labs), 1)
        ce = -(target * logp).sum(1).mean()
        loss = ce + 0.5 * ((v - G) ** 2).mean() / 25
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        n += args.batch
        if time.time() - last > 600 or time.time() - t0 >= args.distill_minutes * 60:
            ent = -(target * target.clamp(min=1e-12).log()).sum(1).mean().item()
            line = f"distill {(time.time() - t0) / 60:.1f} min {n} hands cross-entropy {ce.item():.4f} (teacher entropy {ent:.4f})"
            if table is not None:
                m, lo, hi, k = h2h(tree, dealer, net, table, 2_000_000, args.batch)
                line += f"  vs {os.path.basename(args.vs)}: {m:+.1f} bb/100 [{lo:+.1f}, {hi:+.1f}] over {k} hands"
            log.write(line + "\n")
            print(line, flush=True)
            last = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tree", default="blueprint/run2")
    ap.add_argument("--abs", default="blueprint/out")
    ap.add_argument("--lib", default="blueprint/libbp3.so")
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--minutes", type=float, default=1440)
    ap.add_argument("--vs", default="blueprint/run2")
    ap.add_argument("--eval-minutes", type=float, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kl", type=float, default=0.05)
    ap.add_argument("--magnet-tau", type=float, default=0.002)
    ap.add_argument("--avg-tau", type=float, default=0.001)
    ap.add_argument("--entropy", type=float, default=0.003)
    ap.add_argument("--distill", default=None, help="start by copying this blueprint run's table (e.g. blueprint/run2)")
    ap.add_argument("--distill-minutes", type=float, default=60)
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    tree = Tree(args.tree, dev)
    buckets = int(open(os.path.join(args.abs, "meta.txt")).read().split()[1])
    dealer = Dealer(args.lib, args.abs, buckets, dev)
    shutil.copy(os.path.join(args.tree, "tree.tsv"), os.path.join(args.out, "tree.tsv"))
    net = Net(tree.feat.shape[1], len(tree.vocab), buckets).to(dev)
    ck = os.path.join(args.out, "checkpoint.pt")
    hands = 0
    magnet, avg = copy.deepcopy(net), copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=True)
        net.load_state_dict(st["net"])
        magnet.load_state_dict(st["magnet"])
        avg.load_state_dict(st["avg"])
        opt.load_state_dict(st["opt"])
        hands = st["hands"]
    for m in (magnet, avg):
        m.requires_grad_(False)
    table = Table(args.vs, tree, dev) if args.vs else None
    if args.distill and hands == 0:
        distill(net, opt, tree, dealer, Table(args.distill, tree, dev), args, table)
        magnet.load_state_dict(net.state_dict())  # the KL reference is now the blueprint copy
        avg.load_state_dict(net.state_dict())
    log = open(os.path.join(args.out, "progress.log"), "a", buffering=1)
    meta = {k: v for k, v in vars(args).items()}
    meta.update({"p_fair_cells": dealer.p_fair.tolist(), "vocab": tree.vocab})
    json.dump(meta, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    t0 = last_eval = time.time()
    h_start, adv_scale = hands, 1.0
    while time.time() - t0 < args.minutes * 60:
        bk, rk, w = dealer.deal(args.batch)
        rec, u0 = play(tree, dealer, [net, net], bk, rk)
        games = torch.cat([r[0] for r in rec])
        seat = tree.player[torch.cat([r[1] for r in rec])]
        G = torch.where(seat == 0, u0[games], -u0[games]) / 100  # big blinds won by the player who acted
        wg = w[games]
        f, s, b, st = (torch.cat([r[k] for r in rec]) for k in (2, 3, 4, 5))
        labs, a = torch.cat([r[6] for r in rec]), torch.cat([r[7] for r in rec])
        lg, v = net(f, s, b, st)
        logp = F.log_softmax(masked_logits(lg, labs), 1)
        with torch.no_grad():
            lm, _ = magnet(f, s, b, st)
            logm = F.log_softmax(masked_logits(lm, labs), 1)
        p = logp.exp()
        valid = labs >= 0
        kl = (p * (logp - logm)).masked_fill(~valid, 0).sum(1)
        ent = -(p * logp).masked_fill(~valid, 0).sum(1)
        adv = G - v.detach()
        adv_scale = 0.99 * adv_scale + 0.01 * adv.abs().mean().item()
        adv = adv / max(adv_scale, 1e-3)
        lp_a = logp.gather(1, a.unsqueeze(1)).squeeze(1)
        loss = (wg * (-adv * lp_a + 0.5 * (v - G) ** 2 / 25 + args.kl * kl - args.entropy * ent)).sum() / wg.sum()
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
            rate = (hands - h_start) / (now - t0)
            export(avg, tree, dealer, args.out, hands)
            torch.save(
                {"net": net.state_dict(), "magnet": magnet.state_dict(), "avg": avg.state_dict(),
                 "opt": opt.state_dict(), "hands": hands},
                ck + ".tmp",
            )  # fmt: skip
            os.replace(ck + ".tmp", ck)
            line = f"{(now - t0) / 60:.1f} min {hands} hands {rate:.0f} hands/s loss {loss.item():.4f} kl {kl.mean().item():.4f} entropy {ent.mean().item():.3f}"
            if table is not None:
                m, lo, hi, n = h2h(tree, dealer, avg, table, 2_000_000, args.batch)
                line += f"  vs {os.path.basename(args.vs)}: {m:+.1f} bb/100 [{lo:+.1f}, {hi:+.1f}] over {n} hands"
            log.write(line + "\n")
            print(line, flush=True)
            last_eval = time.time()
    with open(os.path.join(args.out, "train.log"), "a") as f:
        f.write("stopped\n")


if __name__ == "__main__":
    main()
