"""Deep CFR on the GPU, in the blueprint's simplified game: a network learns CFR's regrets instead of a table.

Deep CFR (Brown, Lerer, Gross & Sandholm 2019) replaces the regret table with a network trained on sampled regrets,
and the average-strategy table with a second network. The sampling here is the GPU-friendly form used by the
self-play trainer, with action values from rollouts (as in ESCHER, McAleer et al. 2023, values are estimated rather
than importance-weighted):

  1. Deal a batch and play it with the current strategy sigma_t (regret matching on the regret network's outputs),
     with EPS uniform exploration so that every action is reached.
  2. Pick decisions from those games. For each legal action a, play the same deal on from the child of a with
     sigma_t: the payoff is a one-sample estimate of the action value Q(a) for the player to act.
  3. Instantaneous regret r(a) = Q(a) - sum_b sigma_t(b) Q(b). Store (decision, r, t) in a reservoir buffer, and
     (decision, sigma_t, t) in a second one.
  4. Train the regret network on the buffer with weights t (linear CFR); its output is proportional to the
     cumulative regret, which is all regret matching needs. Train the average-strategy network by weighted
     cross-entropy on the strategy buffer.

The average network is what plays and what is measured: against run 2's table every EVAL_MINUTES (exact, both seats)
and, exported in the table layout, by lbr_trend.sh. Networks: any --arch of mlp_selfplay.py.

Caveat: decisions are sampled on-policy, so a holding's regrets are also weighted by the acting player's own reach,
which external sampling avoids. Exploration (EPS) bounds that weight away from zero.

Usage (lab, GPU venv): .venv-gpu/bin/python scripts/deep_cfr.py OUT [--minutes 120] [--arch mlp] [--eval-minutes 30]
"""

import argparse
import json
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp_selfplay import MAXA, Dealer, Table, Tree, export, h2h, make_net, masked_logits, play


class RegretMatching(nn.Module):
    """sigma_t from a regret network: proportional to the positive regrets; near the arg-max when none is positive.
    Returns logits, so play() and h2h() use it like any policy network."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, feat, street, bucket, strength):
        r, v = self.net(feat, street, bucket, strength)
        return torch.where(r > 0, r.clamp(min=1e-9).log(), -30 + 20 * r), v


class Reservoir:
    """A fixed-size uniform sample of everything ever added (reservoir sampling), on the GPU."""

    def __init__(self, cap, width, dev):
        self.cap, self.seen = cap, 0
        self.node = torch.zeros(cap, dtype=torch.long, device=dev)
        self.bucket = torch.zeros(cap, dtype=torch.long, device=dev)
        self.t = torch.zeros(cap, device=dev)
        self.y = torch.zeros(cap, width, device=dev)

    def add(self, node, bucket, t, y):
        j = self.seen + torch.arange(len(node), device=node.device)
        slot = torch.where(j < self.cap, j, (torch.rand(len(j), device=j.device) * (j + 1)).long())
        keep = (slot < self.cap).nonzero().squeeze(1)
        # one write per slot: with two items for the same slot, separate scatters of node and bucket could otherwise
        # keep one item's node and the other's bucket
        uniq, inv = torch.unique(slot[keep], return_inverse=True)
        first = torch.full((len(uniq),), len(keep), device=slot.device).scatter_reduce(
            0, inv, torch.arange(len(keep), device=slot.device), "amin"
        )
        keep = keep[first]
        sl = slot[keep]
        self.node[sl], self.bucket[sl], self.y[sl] = node[keep], bucket[keep], y[keep]
        self.t[sl] = t
        self.seen += len(node)

    def sample(self, n):
        i = torch.randint(min(self.seen, self.cap), (n,), device=self.t.device)
        return self.node[i], self.bucket[i], self.t[i], self.y[i]


def inputs(tree, dealer, node, bucket):
    s = tree.street[node]
    strength = torch.where(s == 0, dealer.pre_strength[bucket.clamp(max=168)], bucket.float() / (dealer.B - 1))
    return tree.featx[node], s, bucket, strength, tree.lab[node]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tree", default="blueprint/run2")
    ap.add_argument("--abs", default="blueprint/out")
    ap.add_argument("--lib", default="blueprint/libbp3.so")
    ap.add_argument("--vs", default="blueprint/run2")
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--eval-minutes", type=float, default=30)
    ap.add_argument("--batch", type=int, default=16384, help="games dealt per iteration")
    ap.add_argument("--decisions", type=int, default=16384, help="decisions expanded per iteration")
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--buffer", type=int, default=20_000_000)
    ap.add_argument("--steps", type=int, default=8, help="SGD steps per network per iteration")
    ap.add_argument("--train-batch", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    tree = Tree(args.tree, dev)
    B = int(open(os.path.join(args.abs, "meta.txt")).read().split()[1])
    dealer = Dealer(args.lib, args.abs, B, dev)
    shutil.copy(os.path.join(args.tree, "tree.tsv"), os.path.join(args.out, "tree.tsv"))
    regret, avg = make_net(args.arch, tree, B).to(dev), make_net(args.arch, tree, B).to(dev)
    sigma = RegretMatching(regret)
    ropt = torch.optim.Adam(regret.parameters(), lr=args.lr)
    aopt = torch.optim.Adam(avg.parameters(), lr=args.lr)
    rbuf, sbuf = Reservoir(args.buffer, MAXA, dev), Reservoir(args.buffer, MAXA, dev)
    table = Table(args.vs, tree, dev)
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"), indent=1)
    log = open(os.path.join(args.out, "progress.log"), "a", buffering=1)
    t0 = last_eval = time.time()
    it = hands = 0
    while time.time() - t0 < args.minutes * 60:
        it += 1
        # 1. trajectories with sigma_t and exploration
        bk, rk, _ = dealer.deal(args.batch, mixed=False)
        rec, _ = play(tree, dealer, [sigma, sigma], bk, rk, eps=args.eps)
        games, nd = torch.cat([r[0] for r in rec]), torch.cat([r[1] for r in rec])
        bks = torch.cat([r[4] for r in rec])
        pick = torch.randperm(len(nd), device=dev)[: args.decisions]
        g, n, b = games[pick], nd[pick], bks[pick]
        seat = tree.player[n]
        na = (tree.lab[n] >= 0).sum(1)
        # 2. one rollout per legal action, from that action's child, with sigma_t
        rows = torch.arange(len(n), device=dev).repeat_interleave(na)
        act = torch.arange(len(rows), device=dev) - (torch.cumsum(na, 0) - na)[rows]  # 0..na-1 within each decision
        start = tree.child[n[rows], act]
        _, u0 = play(tree, dealer, [sigma, sigma], bk[g[rows]], rk[g[rows]], record=False, start=start)
        q = torch.zeros(len(n), MAXA, device=dev)
        q[rows, act] = torch.where(seat[rows] == 0, u0, -u0) / 100  # big blinds for the player to act
        # 3. instantaneous regrets against sigma_t
        f, s, bb, st, labs = inputs(tree, dealer, n, b)
        with torch.no_grad():
            p = F.softmax(masked_logits(sigma(f, s, bb, st)[0], labs), 1)
        valid = labs >= 0
        r = (q - (p * q).sum(1, keepdim=True)).masked_fill(~valid, 0)
        rbuf.add(n, b, float(it), r)
        sbuf.add(n, b, float(it), p.masked_fill(~valid, 0))
        hands += args.batch + len(rows)
        # 4. train both networks on their buffers, weighted by iteration (linear CFR)
        for _ in range(args.steps):
            n2, b2, t2, y = rbuf.sample(args.train_batch)
            f, s, bb, st, labs = inputs(tree, dealer, n2, b2)
            out = masked_logits(regret(f, s, bb, st)[0], labs).masked_fill(labs < 0, 0)
            loss = (t2 * ((out - y) ** 2).sum(1)).sum() / t2.sum()
            ropt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(regret.parameters(), 1.0)
            ropt.step()
            n2, b2, t2, y = sbuf.sample(args.train_batch)
            f, s, bb, st, labs = inputs(tree, dealer, n2, b2)
            logp = F.log_softmax(masked_logits(avg(f, s, bb, st)[0], labs), 1)
            aloss = (t2 * -(y * logp.masked_fill(labs < 0, 0)).sum(1)).sum() / t2.sum()
            aopt.zero_grad()
            aloss.backward()
            nn.utils.clip_grad_norm_(avg.parameters(), 1.0)
            aopt.step()
        now = time.time()
        if now - last_eval >= args.eval_minutes * 60 or now - t0 >= args.minutes * 60:
            export(avg, tree, dealer, args.out, hands)
            m, lo, hi, k = h2h(tree, dealer, avg, table, 2_000_000, 65536)
            mc, loc, hic, _ = h2h(tree, dealer, sigma, table, 2_000_000, 65536)
            line = (
                f"{(now - t0) / 60:.1f} min iteration {it} {hands} hands  regret loss {loss.item():.4f}  "
                f"average-net loss {aloss.item():.4f}  vs {os.path.basename(args.vs)}: average {m:+.1f} bb/100 "
                f"[{lo:+.1f}, {hi:+.1f}], current {mc:+.1f} [{loc:+.1f}, {hic:+.1f}] over {k} hands"
            )
            log.write(line + "\n")
            print(line, flush=True)
            torch.save(
                {"regret": regret.state_dict(), "avg": avg.state_dict(), "it": it}, os.path.join(args.out, "nets.pt")
            )
            last_eval = time.time()
    with open(os.path.join(args.out, "train.log"), "a") as f:
        f.write("stopped\n")


if __name__ == "__main__":
    main()
