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

--opponent TABLE --opponent-share P: phase B, exploiting a known opponent. A share P of every batch is played
against a fixed table (a clone of a human, clone_table.py), the rest is self-play, and the KL pull keeps the policy
near its reference (with --distill and --magnet-tau 0, the blueprint copy): a restricted Nash response with a KL
anchor, which wins from the opponent while staying hard to exploit. --eval-opponent reports a second table (a clone
of held-out hands) so the gain is measured on an opponent the net did not train against.

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
        # the betting history of each node as tokens (street * vocab + action label), for the transformer; pad 4*vocab
        nv = len(self.vocab)
        hist = [[] for _ in range(n)]
        for i in range(n):
            for k in range(MAXA):
                j = child[i, k]
                if j >= 0:
                    hist[j] = hist[i] + [street[i] * nv + lab[i, k]]
        L = max(map(len, hist))
        htok = np.full((n, max(L, 1)), 4 * nv, np.int64)
        for i, h in enumerate(hist):
            htok[i, : len(h)] = h
        featx = np.concatenate([feat, np.arange(n, dtype=np.float32)[:, None]], 1)  # last column: the node id
        t = lambda a: torch.tensor(a, device=dev)  # noqa: E731
        self.hist, self.featx = t(htok), t(featx)
        self.n, self.kind, self.player, self.street, self.c = n, t(kind), t(player), t(street), t(c)
        self.child, self.lab, self.folder, self.feat, self.offset = t(child), t(lab), t(folder), t(feat), offset
        self.na = (child >= 0).sum(1)
        self.rows = rows


class Net(nn.Module):
    """The default network: card-bucket embedding per street + node features -> 3 GELU layers -> policy and value.
    The input row's last column is the node id (Tree.featx); this network does not use it."""

    def __init__(self, nfeat, nvocab, buckets, hidden=256, emb=32):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(169, emb)] + [nn.Embedding(buckets, emb) for _ in range(3)])
        self.body = nn.Sequential(
            nn.Linear(nfeat + emb + 1, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )  # fmt: skip
        self.pi = nn.Linear(hidden, nvocab)
        self.v = nn.Linear(hidden, 1)

    def cards(self, street, bucket):
        e = torch.zeros(len(street), self.emb[0].embedding_dim, device=bucket.device)
        for s in range(4):
            m = street == s
            if m.any():
                e[m] = self.emb[s](bucket[m])
        return e

    def forward(self, feat, street, bucket, strength):
        h = self.body(torch.cat([feat[:, :-1], self.cards(street, bucket), strength.unsqueeze(1)], 1))
        return self.pi(h), self.v(h).squeeze(1)


class ResNet(Net):
    """Deeper and wider: an input layer, BLOCKS pre-norm residual blocks of width HIDDEN, then the heads."""

    def __init__(self, nfeat, nvocab, buckets, hidden=512, blocks=6, emb=64, extra=0):
        nn.Module.__init__(self)
        self.emb = nn.ModuleList([nn.Embedding(169, emb)] + [nn.Embedding(buckets, emb) for _ in range(3)])
        self.inp = nn.Linear(nfeat + emb + 1 + extra, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            for _ in range(blocks)
        )
        self.norm = nn.LayerNorm(hidden)
        self.pi = nn.Linear(hidden, nvocab)
        self.v = nn.Linear(hidden, 1)

    def trunk(self, x):
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.norm(h)

    def node_extra(self, feat):
        return feat[:, :0]

    def forward(self, feat, street, bucket, strength):
        x = torch.cat([feat[:, :-1], self.cards(street, bucket), strength.unsqueeze(1), self.node_extra(feat)], 1)
        h = self.trunk(x)
        return self.pi(h), self.v(h).squeeze(1)


class NodeEmbNet(ResNet):
    """The residual network plus a learned vector per betting node: the table's memory with the network's sharing
    across card buckets."""

    def __init__(self, nfeat, nvocab, buckets, nodes, node_emb=64, **kw):
        super().__init__(nfeat, nvocab, buckets, extra=node_emb, **kw)
        self.node = nn.Embedding(nodes, node_emb)

    def node_extra(self, feat):
        return self.node(feat[:, -1].long())


class HistoryNet(ResNet):
    """A transformer reads each node's betting history (one token per action, tagged with its street); its summary
    joins the residual network's input. The history depends on the node alone, so it is encoded once per distinct
    node in the batch."""

    def __init__(self, nfeat, nvocab, buckets, hist, d=128, layers=3, heads=4, **kw):
        super().__init__(nfeat, nvocab, buckets, extra=d, **kw)
        self.register_buffer("hist", hist, persistent=False)
        self.tok = nn.Embedding(4 * nvocab + 1, d, padding_idx=4 * nvocab)
        self.pos = nn.Embedding(hist.shape[1] + 1, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.pad = 4 * nvocab

    def node_extra(self, feat):
        nodes, inv = torch.unique(feat[:, -1].long(), return_inverse=True)
        tok = self.hist[nodes]
        x = torch.cat([self.cls.expand(len(nodes), 1, -1), self.tok(tok)], 1)
        x = x + self.pos(torch.arange(x.shape[1], device=x.device))
        mask = torch.cat([torch.zeros(len(nodes), 1, dtype=torch.bool, device=x.device), tok == self.pad], 1)
        return self.enc(x, src_key_padding_mask=mask)[:, 0][inv]


class RecurrentNet(ResNet):
    """Like HistoryNet, with a recurrent reader instead of the transformer: KIND is rnn (Elman), lstm or gru. The
    reader's last hidden state, after the node's final action, joins the residual network's input."""

    def __init__(self, nfeat, nvocab, buckets, hist, kind="gru", d=128, layers=2, **kw):
        super().__init__(nfeat, nvocab, buckets, extra=d, **kw)
        self.register_buffer("hist", hist, persistent=False)
        self.tok = nn.Embedding(4 * nvocab + 1, d, padding_idx=4 * nvocab)
        self.rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind](d, d, layers, batch_first=True)
        self.pad = 4 * nvocab

    def node_extra(self, feat):
        nodes, inv = torch.unique(feat[:, -1].long(), return_inverse=True)
        tok = self.hist[nodes]
        n = (tok != self.pad).sum(1).clamp(min=1).cpu()  # the root has no history: one padding token (zeros)
        packed = nn.utils.rnn.pack_padded_sequence(self.tok(tok), n, batch_first=True, enforce_sorted=False)
        out = self.rnn(packed)[1]
        h = out[0] if isinstance(out, tuple) else out  # LSTM returns (h, c)
        return h[-1][inv]


class SwiGLU(nn.Module):
    def __init__(self, d, mult=2):
        super().__init__()
        self.norm, self.w1, self.w3 = nn.LayerNorm(d), nn.Linear(d, mult * d), nn.Linear(d, mult * d)
        self.w2 = nn.Linear(mult * d, d)

    def forward(self, x):
        h = self.norm(x)
        return self.w2(F.silu(self.w1(h)) * self.w3(h))


class SwiGLUNet(ResNet):
    """The residual network with the gated feed-forward blocks of current LLMs (SwiGLU) instead of GELU MLPs."""

    def __init__(self, nfeat, nvocab, buckets, hidden=384, blocks=6, **kw):
        super().__init__(nfeat, nvocab, buckets, hidden=hidden, blocks=0, **kw)
        self.blocks = nn.ModuleList(SwiGLU(hidden) for _ in range(blocks))


class SoftMoE(nn.Module):
    """A block of E expert MLPs mixed by a learned softmax gate per input (dense: every expert runs)."""

    def __init__(self, d, experts=4):
        super().__init__()
        self.norm, self.gate = nn.LayerNorm(d), nn.Linear(d, experts)
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)) for _ in range(experts))

    def forward(self, x):
        h = self.norm(x)
        g = F.softmax(self.gate(h), 1)
        return sum(g[:, k : k + 1] * e(h) for k, e in enumerate(self.experts))


class MoENet(ResNet):
    """The residual network with mixture-of-experts blocks (4 experts of width 256, 4 blocks)."""

    def __init__(self, nfeat, nvocab, buckets, hidden=256, blocks=4, **kw):
        super().__init__(nfeat, nvocab, buckets, hidden=hidden, blocks=0, **kw)
        self.blocks = nn.ModuleList(SoftMoE(hidden) for _ in range(blocks))


class KANLayer(nn.Module):
    """A Kolmogorov-Arnold layer in the FastKAN form: each input passes through G Gaussian bumps on a grid (a learned
    curve per input-output pair), plus a SiLU-linear base path."""

    def __init__(self, i, o, grid=8):
        super().__init__()
        self.norm = nn.LayerNorm(i)
        self.register_buffer("centres", torch.linspace(-2, 2, grid))
        self.spline = nn.Linear(i * grid, o)
        self.base = nn.Linear(i, o)

    def forward(self, x):
        h = self.norm(x)
        phi = torch.exp(-(((h.unsqueeze(-1) - self.centres) / 0.5) ** 2)).flatten(1)
        return self.spline(phi) + self.base(F.silu(h))


class KANNet(Net):
    def __init__(self, nfeat, nvocab, buckets, hidden=256, emb=32):
        nn.Module.__init__(self)
        self.emb = nn.ModuleList([nn.Embedding(169, emb)] + [nn.Embedding(buckets, emb) for _ in range(3)])
        self.body = nn.Sequential(KANLayer(nfeat + emb + 1, hidden), KANLayer(hidden, hidden))
        self.pi = nn.Linear(hidden, nvocab)
        self.v = nn.Linear(hidden, 1)


class TCNNet(ResNet):
    """A temporal convolution network reads the betting history: dilated 1-D convolutions (kernel 3, dilation 1, 2,
    4), causal, read out at the node's last action."""

    def __init__(self, nfeat, nvocab, buckets, hist, d=128, **kw):
        super().__init__(nfeat, nvocab, buckets, extra=d, **kw)
        self.register_buffer("hist", hist, persistent=False)
        self.tok = nn.Embedding(4 * nvocab + 1, d, padding_idx=4 * nvocab)
        self.convs = nn.ModuleList(nn.Conv1d(d, d, 3, dilation=r) for r in (1, 2, 4))
        self.pad = 4 * nvocab

    def node_extra(self, feat):
        nodes, inv = torch.unique(feat[:, -1].long(), return_inverse=True)
        tok = self.hist[nodes]
        x = self.tok(tok).transpose(1, 2)
        for conv in self.convs:
            x = x + F.gelu(conv(F.pad(x, (2 * conv.dilation[0], 0))))
        last = ((tok != self.pad).sum(1) - 1).clamp(min=0)
        return x[torch.arange(len(nodes), device=x.device), :, last][inv]


class TextCNNNet(ResNet):
    """A convolutional network over the betting history in the text-CNN form (Kim 2014): parallel convolutions of
    width 2, 3 and 4 over the action tokens, max-pooled over the history, then into the residual network."""

    def __init__(self, nfeat, nvocab, buckets, hist, d=128, maps=64, **kw):
        super().__init__(nfeat, nvocab, buckets, extra=3 * maps, **kw)
        self.register_buffer("hist", hist, persistent=False)
        self.tok = nn.Embedding(4 * nvocab + 1, d, padding_idx=4 * nvocab)
        self.convs = nn.ModuleList(nn.Conv1d(d, maps, k, padding=k - 1) for k in (2, 3, 4))
        self.pad = 4 * nvocab

    def node_extra(self, feat):
        nodes, inv = torch.unique(feat[:, -1].long(), return_inverse=True)
        x = self.tok(self.hist[nodes]).transpose(1, 2)
        return torch.cat([F.gelu(c(x)).amax(2) for c in self.convs], 1)[inv]


class FeatureTokenNet(Net):
    """Attention across the inputs (the FT-Transformer for tables): each node feature, the card embedding and the
    strength become tokens, a transformer attends among them, and the policy and value read a CLS token."""

    def __init__(self, nfeat, nvocab, buckets, d=64, layers=3, heads=4, emb=32):
        nn.Module.__init__(self)
        self.emb = nn.ModuleList([nn.Embedding(169, emb)] + [nn.Embedding(buckets, emb) for _ in range(3)])
        self.fw = nn.Parameter(torch.randn(nfeat + 1, d) * 0.1)  # one token per scalar input: value * w + b
        self.fb = nn.Parameter(torch.zeros(nfeat + 1, d))
        self.card = nn.Linear(emb, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.pi = nn.Linear(d, nvocab)
        self.v = nn.Linear(d, 1)

    def forward(self, feat, street, bucket, strength):
        x = torch.cat([feat[:, :-1], strength.unsqueeze(1)], 1)
        tok = x.unsqueeze(-1) * self.fw + self.fb
        tok = torch.cat([self.cls.expand(len(x), 1, -1), self.card(self.cards(street, bucket)).unsqueeze(1), tok], 1)
        h = self.norm(self.enc(tok)[:, 0])
        return self.pi(h), self.v(h).squeeze(1)


class JointNet(FeatureTokenNet):
    """One attention sequence per decision: CLS, the card token, a token of the node features, then the betting
    history. Unlike HistoryNet, the cards and the history attend to each other directly."""

    def __init__(self, nfeat, nvocab, buckets, hist, d=64, layers=3, heads=4, emb=32):
        super().__init__(nfeat, nvocab, buckets, d, layers, heads, emb)
        self.register_buffer("hist", hist, persistent=False)
        self.tok = nn.Embedding(4 * nvocab + 1, d, padding_idx=4 * nvocab)
        self.pos = nn.Embedding(hist.shape[1] + 3, d)
        self.feat_tok = nn.Linear(nfeat + 1, d)
        self.pad = 4 * nvocab

    def forward(self, feat, street, bucket, strength):
        hist = self.hist[feat[:, -1].long()]
        head = torch.stack(
            [self.cls[0, 0].expand(len(feat), -1), self.card(self.cards(street, bucket)),
             self.feat_tok(torch.cat([feat[:, :-1], strength.unsqueeze(1)], 1))], 1,
        )  # fmt: skip
        x = torch.cat([head, self.tok(hist)], 1)
        x = x + self.pos(torch.arange(x.shape[1], device=x.device))
        mask = torch.cat([torch.zeros(len(feat), 3, dtype=torch.bool, device=x.device), hist == self.pad], 1)
        h = self.norm(self.enc(x, src_key_padding_mask=mask)[:, 0])
        return self.pi(h), self.v(h).squeeze(1)


ARCHS = (
    "mlp", "wide", "res", "nodeemb", "history", "rnn", "lstm", "gru", "swiglu", "moe", "kan", "tcn", "cnn", "ftt",
    "joint",
)  # fmt: skip


def make_net(arch, tree, buckets):
    nf, nv = tree.feat.shape[1], len(tree.vocab)
    if arch == "mlp":
        return Net(nf, nv, buckets)
    if arch == "wide":
        return Net(nf, nv, buckets, hidden=1024, emb=64)
    if arch == "res":
        return ResNet(nf, nv, buckets)
    if arch == "nodeemb":
        return NodeEmbNet(nf, nv, buckets, tree.n)
    if arch == "history":
        return HistoryNet(nf, nv, buckets, tree.hist)
    if arch in ("rnn", "lstm", "gru"):
        return RecurrentNet(nf, nv, buckets, tree.hist, kind=arch)
    if arch == "swiglu":
        return SwiGLUNet(nf, nv, buckets)
    if arch == "moe":
        return MoENet(nf, nv, buckets)
    if arch == "kan":
        return KANNet(nf, nv, buckets)
    if arch == "tcn":
        return TCNNet(nf, nv, buckets, tree.hist)
    if arch == "cnn":
        return TextCNNNet(nf, nv, buckets, tree.hist)
    if arch == "ftt":
        return FeatureTokenNet(nf, nv, buckets)
    if arch == "joint":
        return JointNet(nf, nv, buckets, tree.hist)
    raise ValueError(arch)


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
    return tree.featx[nodes], s, b, strength


def play(tree, dealer, policies, bk, rk, record=True, start=None, eps=0.0):
    """Play one batch to the end. policies[seat] is a net or a table; returns decisions and payoff to seat 0 (chips).
    start: each game's first node (default the root). eps: mix that share of uniform play into every decision."""
    B = len(bk)
    node = torch.zeros(B, dtype=torch.long, device=bk.device) if start is None else start.clone()
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
            if eps:
                legal = (labs[m] >= 0).float()
                probs = (1 - eps) * probs + eps * legal / legal.sum(1, keepdim=True)
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
    for chunk in np.array_split(dec, max(1, len(dec) // 250)):
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
            lg, _ = net(tree.featx[ndr], s, b, strength)
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
        opt.zero_grad()
        ce_sum = 0.0
        for part in torch.arange(len(f), device=f.device).chunk(args.micro):  # gradient accumulation: same batch
            lg, v = net(f[part], s[part], b[part], st[part])
            logp = F.log_softmax(masked_logits(lg, labs[part]), 1)
            ce_part = -(target[part] * logp).sum(1).sum()
            loss = (ce_part + 0.5 * ((v - G[part]) ** 2).sum() / 25) / len(f)
            loss.backward()
            ce_sum += ce_part.item()
        ce = torch.tensor(ce_sum / len(f))
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
    ap.add_argument(
        "--arch",
        default="mlp",
        choices=ARCHS,
        help="network: mlp (default), wide, res (residual), nodeemb (+ a vector per node), history (+ a "
        "transformer over the betting history), rnn / lstm / gru (+ a recurrent reader of it), swiglu (gated "
        "LLM-style blocks), moe (mixture of experts), kan (Kolmogorov-Arnold), tcn (+ a convolutional history "
        "reader), cnn (text-CNN over the history), ftt (attention across the inputs), joint (one attention sequence of cards, features and history)",
    )
    ap.add_argument(
        "--micro",
        type=int,
        default=1,
        help="split each distillation batch into this many chunks "
        "(gradient accumulation, for networks that do not fit the GPU at the full batch)",
    )
    ap.add_argument(
        "--pg",
        default="a2c",
        choices=("a2c", "ppo", "neurd", "qpg", "qlearn", "qmc"),
        help="policy update: a2c "
        "(REINFORCE + value baseline, default), ppo (clipped ratio), neurd (logit-space replicator "
        "dynamics), qpg (all-actions gradient with a learned Q critic), qlearn (Q-learning, bootstrapped "
        "from the next decision's best action, Boltzmann play), qmc (Monte Carlo Q)",
    )
    ap.add_argument("--q-temp", type=float, default=1.0, help="Boltzmann temperature in big blinds (qlearn, qmc)")
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--ppo-clip", type=float, default=0.2)
    ap.add_argument("--neurd-beta", type=float, default=2.0)
    ap.add_argument(
        "--reward",
        default="chips",
        choices=("chips", "duplicate", "tanh"),
        help="chips (default); duplicate (minus the mirrored deal's result: card luck removed); tanh (bounded utility)",
    )
    ap.add_argument("--league", type=float, default=0.0, help="share of each batch played against past snapshots")
    ap.add_argument("--league-every", type=float, default=3.0, help="minutes between snapshots")
    ap.add_argument("--kl", type=float, default=0.05)
    ap.add_argument("--magnet-tau", type=float, default=0.002)
    ap.add_argument("--avg-tau", type=float, default=0.001)
    ap.add_argument("--entropy", type=float, default=0.003)
    ap.add_argument("--distill", default=None, help="start by copying this blueprint run's table (e.g. blueprint/run2)")
    ap.add_argument("--distill-minutes", type=float, default=60)
    ap.add_argument("--opponent", default=None, help="a fixed table to exploit (e.g. a clone_table.py table)")
    ap.add_argument("--opponent-share", type=float, default=0.5, help="share of each batch played against --opponent")
    ap.add_argument("--eval-opponent", default=None, help="a second table to report the head-to-head against")
    args = ap.parse_args()
    if args.reward == "duplicate" and (args.opponent or args.league):
        ap.error("--reward duplicate pairs every deal with its mirror inside self-play; not with --opponent/--league")
    dev = "cuda"
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    tree = Tree(args.tree, dev)
    buckets = int(open(os.path.join(args.abs, "meta.txt")).read().split()[1])
    dealer = Dealer(args.lib, args.abs, buckets, dev)
    shutil.copy(os.path.join(args.tree, "tree.tsv"), os.path.join(args.out, "tree.tsv"))
    net = make_net(args.arch, tree, buckets).to(dev)
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
    opponent = Table(args.opponent, tree, dev) if args.opponent else None
    eval_opp = Table(args.eval_opponent, tree, dev) if args.eval_opponent else None
    if args.distill and hands == 0:
        distill(net, opt, tree, dealer, Table(args.distill, tree, dev), args, table)
        magnet.load_state_dict(net.state_dict())  # the KL reference is now the blueprint copy
        avg.load_state_dict(net.state_dict())
    if args.distill and args.minutes == 0:  # distillation only (the architecture bake-off): export and stop
        export(net, tree, dealer, args.out, hands)
        with open(os.path.join(args.out, "train.log"), "a") as f:
            f.write("stopped\n")
        return
    log = open(os.path.join(args.out, "progress.log"), "a", buffering=1)
    meta = {k: v for k, v in vars(args).items()}
    meta.update({"p_fair_cells": dealer.p_fair.tolist(), "vocab": tree.vocab})
    json.dump(meta, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    t0 = last_eval = time.time()
    h_start, adv_scale = hands, 1.0
    qnet = qopt = None
    if args.pg == "qpg":  # a separate critic with one value per action (its "policy" outputs are read as Q values)
        qnet = make_net("mlp", tree, buckets).to(dev)
        qopt = torch.optim.Adam(qnet.parameters(), lr=args.lr)
    snapshots, last_snap = [], time.time()
    while time.time() - t0 < args.minutes * 60:
        if args.reward == "duplicate":  # every deal twice, the second with the players' cards swapped
            bk, rk, w = dealer.deal(args.batch // 2)
            bk, rk, w = torch.cat([bk, bk[:, [1, 0]]]), torch.cat([rk, rk[:, [1, 0]]]), torch.cat([w, w])
        else:
            bk, rk, w = dealer.deal(args.batch)
        # self-play, plus (with --opponent) games against the fixed table in either seat, or (with --league) against a
        # past snapshot of the net; the net learns only from its own seats' decisions
        opp, share = opponent, args.opponent_share
        if opp is None and args.league and snapshots:
            opp, share = snapshots[int(torch.randint(len(snapshots), ()).item())], args.league
        k = int(args.batch * share) // 2 if opp is not None else 0
        parts = [(slice(2 * k, None), [net, net], (0, 1))]
        if k:
            parts += [(slice(0, k), [net, opp], (0,)), (slice(k, 2 * k), [opp, net], (1,))]
        cols = [[] for _ in range(9)]
        for sl, pols, mine in parts:
            rec, u0 = play(tree, dealer, pols, bk[sl], rk[sl])
            games = torch.cat([r[0] for r in rec])
            seat = tree.player[torch.cat([r[1] for r in rec])]
            keep = (seat == mine[0]) if len(mine) == 1 else torch.ones_like(seat, dtype=torch.bool)
            G = torch.where(seat == 0, u0[games], -u0[games]) / 100  # big blinds won by the player who acted
            if args.reward == "duplicate":
                # baseline: what the other seat won with these same cards against the same cards in the mirrored deal;
                # it does not depend on this game's actions, so the gradient stays unbiased
                half = len(u0) // 2
                mirror = (games + half) % len(u0)
                G = G - torch.where(seat == 1, u0[mirror], -u0[mirror]) / 100
            elif args.reward == "tanh":  # bounded, risk-averse utility: no longer the zero-sum chip objective
                G = 10 * torch.tanh(G / 10)
            gid = games + (sl.start or 0)  # game id within the whole batch
            for j, x in enumerate(
                [G, w[sl][games]] + [torch.cat([r[q] for r in rec]) for q in (2, 3, 4, 5, 6, 7)] + [gid * 2 + seat]
            ):
                cols[j].append(x[keep])
        G, wg, f, s, b, st, labs, a, key = (torch.cat(c) for c in cols)
        valid = labs >= 0
        with torch.no_grad():
            lm, _ = magnet(f, s, b, st)
            logm = F.log_softmax(masked_logits(lm, labs), 1)
        if qnet is not None:
            qv = masked_logits(qnet(f, s, b, st)[0], labs).masked_fill(~valid, 0)
            qloss = (wg * (qv.gather(1, a.unsqueeze(1)).squeeze(1) - G) ** 2).sum() / wg.sum()
            qopt.zero_grad()
            qloss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), 1.0)
            qopt.step()
            qv = qv.detach()
        if args.pg in ("qlearn", "qmc"):
            # value-based: the policy head's outputs are action values in units of the temperature (Q = tau * logit),
            # and the net plays the softmax over them (Boltzmann exploration)
            lg, v = net(f, s, b, st)
            q = args.q_temp * masked_logits(lg, labs)
            q_a = q.gather(1, a.unsqueeze(1)).squeeze(1)
            target = G
            if args.pg == "qlearn":  # Q-learning: bootstrap from the best action at this player's next decision
                idx = torch.arange(len(key), device=key.device)
                order = torch.argsort(key * len(key) + idx)  # decisions grouped by (game, seat), in play order
                nxt = torch.full_like(idx, -1)
                same = key[order[1:]] == key[order[:-1]]
                nxt[order[:-1][same]] = order[1:][same]
                with torch.no_grad():
                    qmax = q.masked_fill(~valid, -1e9).max(1).values
                target = torch.where(nxt >= 0, qmax[nxt.clamp(min=0)], G)
            loss = (wg * ((q_a - target) ** 2 / 25 + 0.5 * (v - G) ** 2 / 25)).sum() / wg.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            p = F.softmax(masked_logits(lg.detach(), labs), 1)
            kl = (p * (p.clamp(min=1e-12).log() - logm)).masked_fill(~valid, 0).sum(1)
            ent = -(p * p.clamp(min=1e-12).log()).masked_fill(~valid, 0).sum(1)
        logp_old = None
        for epoch in range(0 if args.pg in ("qlearn", "qmc") else args.ppo_epochs if args.pg == "ppo" else 1):
            lg, v = net(f, s, b, st)
            logp = F.log_softmax(masked_logits(lg, labs), 1)
            p = logp.exp()
            kl = (p * (logp - logm)).masked_fill(~valid, 0).sum(1)
            ent = -(p * logp).masked_fill(~valid, 0).sum(1)
            lp_a = logp.gather(1, a.unsqueeze(1)).squeeze(1)
            if epoch == 0:
                adv = G - v.detach()
                adv_scale = 0.99 * adv_scale + 0.01 * adv.abs().mean().item()
                adv = adv / max(adv_scale, 1e-3)
                logp_old = lp_a.detach()
            if args.pg == "a2c":  # REINFORCE with the value baseline (the default)
                pg = -adv * lp_a
            elif args.pg == "ppo":  # clipped probability ratio, several passes over the batch (Schulman et al. 2017)
                ratio = (lp_a - logp_old).exp()
                pg = -torch.minimum(ratio * adv, ratio.clamp(1 - args.ppo_clip, 1 + args.ppo_clip) * adv)
            elif args.pg == "neurd":
                # Neural Replicator Dynamics (Hennes et al. 2020; R-NaD): the update acts on the centred logits, with
                # the sampled action's value estimate adv / pi(a); logits stop moving outward beyond +-beta
                y = masked_logits(lg, labs)
                y = y - (y.masked_fill(~valid, 0).sum(1, keepdim=True) / valid.sum(1, keepdim=True))
                y_a = y.gather(1, a.unsqueeze(1)).squeeze(1)
                q_a = adv / lp_a.detach().exp().clamp(min=0.05)
                live = ((q_a > 0) & (y_a < args.neurd_beta)) | ((q_a < 0) & (y_a > -args.neurd_beta))
                pg = -(q_a * y_a) * live
            else:  # qpg: the all-actions policy gradient with the learned Q (Srinivasan et al. 2018)
                base = (p * qv).sum(1, keepdim=True)
                pg = -(p * (qv - base) / max(adv_scale, 1e-3)).masked_fill(~valid, 0).sum(1)
            loss = (wg * (pg + 0.5 * (v - G) ** 2 / 25 + args.kl * kl - args.entropy * ent)).sum() / wg.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if args.league and time.time() - last_snap >= args.league_every * 60:  # keep the 10 latest snapshots
            snapshots = (snapshots + [copy.deepcopy(net).requires_grad_(False)])[-10:]
            last_snap = time.time()
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
            for name, tb in ((args.opponent, opponent), (args.eval_opponent, eval_opp)):
                if tb is not None:
                    m, lo, hi, n = h2h(tree, dealer, avg, tb, 2_000_000, args.batch)
                    line += f"  vs {os.path.basename(name)}: {m:+.1f} [{lo:+.1f}, {hi:+.1f}]"
            log.write(line + "\n")
            print(line, flush=True)
            last_eval = time.time()
    with open(os.path.join(args.out, "train.log"), "a") as f:
        f.write("stopped\n")


if __name__ == "__main__":
    main()
