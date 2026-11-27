"""Beliefs for search: what each opponent holds, from the public state (plan: search with beliefs, 2026-11-27).

Search (ninemax_search.py) values a move by playing the hand out on N guesses of the cards it cannot see, each guess
weighted by how likely the network was to take the opponents' actions with those cards: the posterior over their
cards, by importance sampling from a uniform proposal. Uniform guesses are mostly cards the opponents would not have
played this way, so the weights pile onto a few of them (the ESS floor exists for that). A better proposal changes
nothing in what is estimated (each guess is weighted by likelihood / proposal, so the estimate stays the posterior) but
spends the samples where the posterior is:

  BeliefNet  for one opponent, a distribution over its 1,326 two-card combinations, from the public state at the
             searching player's decision (the network's own inputs, from that player's seat), the opponent's seat
             relative to it, the board so far and the searching player's own cards (both only to rule combinations
             out, and as context). Trained on the network's self-play: at every decision after the flop in a pot of
             at most three, each live opponent's real cards are the target (cross-entropy over the possible ones).
  sample     M x K proposals for the opponents still in, one after the other so their cards never clash, with the log
             proposal probability of each; the folded players' cards and the rest of the board are dealt uniformly
             from what is left, which is the same for every guess and so cancels out of the weights.

Usage (lab, GPU, from research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_belief.py train blueprint/nm25 OUT [--deals 400000]
writes OUT/belief.pt and OUT/belief.json (the held-out log-likelihood per opponent against the uniform guess, in bits).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ninemax_eval import load, setup
from ninemax_selfplay import FEAT, board_of, context, play

COMBOS = torch.tensor([(a, b) for a in range(52) for b in range(a + 1, 52)])  # [1326, 2], a < b
NBOARD = {0: 0, 1: 3, 2: 4, 3: 5}


def combo_index(c1, c2):
    """The index in COMBOS of the two cards (any order), vectorised (tensors)."""
    a, b = torch.minimum(c1, c2), torch.maximum(c1, c2)
    return a * 51 - a * (a - 1) // 2 + (b - a - 1)


def card_mask(known):
    """[n, 1326] True where a combination uses none of the known cards (known [n, 52] bool)."""
    c = COMBOS.to(known.device)
    return ~(known[:, c[:, 0]] | known[:, c[:, 1]])


class BeliefNet(nn.Module):
    def __init__(self, nfeat, players=9, hidden=512):
        super().__init__()
        self.P = players
        self.body = nn.Sequential(
            nn.Linear(nfeat + (players - 1) + 52 + 52, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(hidden, 1326)

    def forward(self, feat, rel, board, mine):
        """feat [n, F] public inputs at the decision (the searcher's view), rel [n] the opponent's seat after the
        searcher (0..P-2), board and mine [n, 52] the board so far and the searcher's cards. Logits [n, 1326], the
        impossible combinations at -1e9."""
        x = torch.cat([feat, F.one_hot(rel, self.P - 1).float(), board.float(), mine.float()], 1)
        lg = self.out(self.body(x))
        return lg.masked_fill(~card_mask(board | mine), -1e9)


def examples(net, dealer, menu, n, dev, max_live=3):
    """Self-play of the network on n deals: for each decision after the flop in a pot of at most max_live, and each
    live opponent of the player to act, (features, opponent's relative seat, board so far, own cards, the opponent's
    real combination)."""
    P = dealer.P
    bk, rk, cd = dealer.deal(n, cards=True)
    _, rec = play(dealer, menu, [net] * P, bk, rk, record=True, dev=dev, ctx=context(n, P, dev),
                  board=board_of(cd, P, dev) if FEAT["texture"] else None)  # fmt: skip
    cards = torch.tensor(cd, device=dev, dtype=torch.long)
    out = {k: [] for k in ("feat", "rel", "board", "mine", "y")}
    live0 = P + 4 + 10  # the others' live flags, in table order after the actor (features())
    for g, i, feat, legal, mybk, street, s, a, pol in rec:
        live = feat[:, live0 : live0 + P - 1] > 0.5
        keep = (street >= 1) & (live.sum(1) + 1 <= max_live)
        if not keep.any():
            continue
        g, i, feat, street, live = g[keep], i[keep], feat[keep], street[keep], live[keep]
        nb = torch.tensor([NBOARD[int(t)] for t in street.tolist()], device=dev)
        board = torch.zeros(len(g), 52, dtype=torch.bool, device=dev)
        for k in range(5):
            on = nb > k
            board[on.nonzero().squeeze(1), cards[g[on], 2 * P + k]] = True
        mine = torch.zeros_like(board)
        ar = torch.arange(len(g), device=dev)
        mine[ar, cards[g, 2 * i]] = True
        mine[ar, cards[g, 2 * i + 1]] = True
        for r in range(P - 1):
            on = live[:, r]
            if not on.any():
                continue
            j = (i[on] + 1 + r) % P
            out["feat"].append(feat[on])
            out["rel"].append(torch.full((int(on.sum()),), r, device=dev))
            out["board"].append(board[on])
            out["mine"].append(mine[on])
            out["y"].append(combo_index(cards[g[on], 2 * j], cards[g[on], 2 * j + 1]))
    return {k: torch.cat(v) for k, v in out.items()}


def nll_bits(model, d, batch=8192):
    """Mean negative log-likelihood of the real combinations, and of the uniform guess over the possible ones, in bits."""
    tot = uni = 0.0
    with torch.no_grad():
        for k in range(0, len(d["y"]), batch):
            sl = slice(k, k + batch)
            lg = model(d["feat"][sl], d["rel"][sl], d["board"][sl], d["mine"][sl])
            tot += float(F.cross_entropy(lg, d["y"][sl], reduction="sum"))
            uni += float(torch.log((lg > -1e8).sum(1).float()).sum())
    n = len(d["y"])
    return tot / n / math.log(2), uni / n / math.log(2)


def train(a):
    dev = "cuda"
    torch.manual_seed(3)
    dealer, menu, B = setup(a, dev)
    net, _ = load(a.run, B, dev)
    assert not getattr(net, "size_head", False), "beliefs read the slot layout (a size-head network: not yet)"
    t0 = time.time()
    held = examples(net, dealer, menu, a.deals // 20, dev)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train.log"), "a", buffering=1)
    model = BeliefNet(held["feat"].shape[1], dealer.P).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    seen = 0
    while seen < a.deals:
        d = examples(net, dealer, menu, a.chunk, dev)
        seen += a.chunk
        perm = torch.randperm(len(d["y"]), device=dev)
        for k in range(0, len(perm), a.batch):
            ix = perm[k : k + a.batch]
            lg = model(d["feat"][ix], d["rel"][ix], d["board"][ix], d["mine"][ix])
            loss = F.cross_entropy(lg, d["y"][ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
        h, u = nll_bits(model.eval(), held)
        model.train()
        log.write(f"{seen} deals {time.time() - t0:.0f}s held-out {h:.3f} bits per opponent, uniform {u:.3f}, "
                  f"gain {u - h:.3f}\n")  # fmt: skip
    torch.save(model.state_dict(), os.path.join(a.out, "belief.pt"))
    h, u = nll_bits(model.eval(), held)
    json.dump({"run": a.run, "nfeat": int(held["feat"].shape[1]), "players": dealer.P, "deals": seen,
               "held_out_examples": len(held["y"]), "bits": round(h, 4), "uniform_bits": round(u, 4),
               "gain_bits": round(u - h, 4)}, open(os.path.join(a.out, "belief.json"), "w"), indent=1)  # fmt: skip


def load_belief(path, dev):
    meta = json.load(open(os.path.join(path, "belief.json")))
    m = BeliefNet(meta["nfeat"], meta["players"]).to(dev)
    m.load_state_dict(torch.load(os.path.join(path, "belief.pt"), map_location=dev, weights_only=True))
    return m.eval().requires_grad_(False)


def propose(belief, feat, hero, live, fixed, K, gen=None):
    """K guesses of the hidden cards for each of M decisions, the live opponents' from the belief and the rest
    uniform. feat [M, F] the searcher's inputs, hero its seat, live [M, P] bool (opponents still in), fixed [M, 2P + 5]
    numpy (-1 unknown). Returns cards [M*K, 2P + 5] numpy and the log proposal probability [M*K] of the live
    opponents' combinations (the rest is uniform over the same number of cards in every guess: a constant)."""
    dev = feat.device
    M, W = fixed.shape
    P = (W - 5) // 2
    fx = torch.tensor(fixed, device=dev, dtype=torch.long)
    cards = fx.repeat_interleave(K, 0).clone()  # [M*K, W]
    used = torch.zeros(M * K, 52, dtype=torch.bool, device=dev)
    kn = cards >= 0
    used[torch.arange(M * K, device=dev)[:, None].expand(-1, W)[kn], cards[kn]] = True
    board = torch.zeros(M, 52, dtype=torch.bool, device=dev)
    mine = torch.zeros(M, 52, dtype=torch.bool, device=dev)
    for k in range(5):
        on = fx[:, 2 * P + k] >= 0
        board[on.nonzero().squeeze(1), fx[on, 2 * P + k]] = True
    mine[torch.arange(M, device=dev), fx[:, 2 * hero]] = True
    mine[torch.arange(M, device=dev), fx[:, 2 * hero + 1]] = True
    logq = torch.zeros(M * K, device=dev)
    c = COMBOS.to(dev)
    for r in range(P - 1):  # the live opponents in table order after the searcher, each conditioned on the earlier
        j = (hero + 1 + r) % P
        on = live[:, j]
        if not on.any():
            continue
        lg = belief(feat[on], torch.full((int(on.sum()),), r, device=dev), board[on], mine[on])  # [m, 1326]
        rows = (on.nonzero().squeeze(1)[:, None] * K + torch.arange(K, device=dev)).reshape(-1)
        lgk = lg.repeat_interleave(K, 0).masked_fill(~card_mask(used[rows]), -1e9)  # not the cards already given
        lp = torch.log_softmax(lgk, 1)
        pick = torch.multinomial(lp.exp(), 1, generator=gen).squeeze(1)
        logq[rows] += lp.gather(1, pick[:, None]).squeeze(1)
        cards[rows, 2 * j], cards[rows, 2 * j + 1] = c[pick, 0], c[pick, 1]
        used[rows, c[pick, 0]] = True
        used[rows, c[pick, 1]] = True
    # the rest (folded players, empty seats are dealt too; the board to come) uniformly from the cards left
    for col in range(W):
        need = cards[:, col] < 0
        if not need.any():
            continue
        rows = need.nonzero().squeeze(1)
        u = torch.rand(len(rows), 52, device=dev, generator=gen).masked_fill(used[rows], 2.0)
        pick = u.argmin(1)
        cards[rows, col] = pick
        used[rows, pick] = True
    return cards.cpu().numpy().astype(np.int32), logq


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("run")
    t.add_argument("out")
    t.add_argument("--deals", type=int, default=400000)
    t.add_argument("--chunk", type=int, default=8192)
    t.add_argument("--batch", type=int, default=4096)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--players", type=int, default=9)
    t.add_argument("--menu", default="mtt")
    t.add_argument("--abs", default="blueprint/out")
    t.add_argument("--lib", default="blueprint/libbp7.so")
    t.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)


if __name__ == "__main__":
    main()
