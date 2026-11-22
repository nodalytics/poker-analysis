"""Turn an MLP clone of a human (human_clone.py) into a strategy table in the blueprint's simplified game, so the GPU
trainer (mlp_selfplay.py --opponent) can play millions of hands against it and br_vec can best-respond to it.

At every decision node and card bucket, the clone's inputs are built from the node and the bucket:
    hand strength  the mean expected strength of the hands in the bucket (out/{flop,turn}_ehs.f32, out/river_hs.f32;
                   pre-flop: the mean flop strength of each starting-hand class over dealt hands)
    price, pot, stack, raises this street, position: from the node (abstract player 0 is the button)
Its seven action classes go to the node's actions: fold, check/call, a raise class to the raise of the nearest size,
all-in to all-in. Writes OUT/avg.f32 and OUT/tree.tsv in the blueprint's layout.

Usage (GPU venv, lab): python scripts/clone_table.py CLONE.npz OUT [--tree blueprint/run2] [--abs blueprint/out]
    [--lib blueprint/libbp3.so]
"""

import argparse
import os
import shutil

import numpy as np

from mlp_selfplay import STACK, Dealer, Tree
from opp_mlp_prep import NFEAT, RAISE_FRACTION


def forward(z, f):
    """The clone MLP (Linear-GELU-Linear-GELU-Linear) on the GPU when there is one; returns class probabilities."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len([k for k in z.files if k.startswith("w")])
    with torch.no_grad():
        h = torch.from_numpy(f).to(dev)
        for k in range(n):
            h = h @ torch.from_numpy(z[f"w{k}"]).to(dev).T + torch.from_numpy(z[f"b{k}"]).to(dev)
            if k < n - 1:
                h = torch.nn.functional.gelu(h)
        return torch.softmax(h, 1).cpu().numpy()


def bucket_strength(absdir, B, dealer):
    """Mean expected strength per bucket on each street; pre-flop per starting-hand class."""
    out = np.zeros((4, max(B, 169)), np.float32)
    for st, name, key in ((1, "flop", "ehs"), (2, "turn", "ehs"), (3, "river", "hs")):
        v = np.fromfile(os.path.join(absdir, f"{name}_{key}.f32"), dtype=np.float32)
        bp = os.path.join(absdir, f"{name}_buckets.u8")
        b = np.fromfile(bp, dtype=np.uint8) if os.path.exists(bp) else np.fromfile(bp[:-2] + "u16", dtype=np.uint16)
        out[st, :B] = np.bincount(b, weights=v, minlength=B) / np.maximum(np.bincount(b, minlength=B), 1)
    bk, _ = dealer.raw(1 << 21)
    pre, flop = bk[:, :, 0].ravel(), bk[:, :, 1].ravel()
    out[0, :169] = np.bincount(pre, weights=out[1, flop], minlength=169) / np.maximum(
        np.bincount(pre, minlength=169), 1
    )
    return out


def node_actions(tree, c, i, probs):
    """The clone's seven classes (rows of probs, one per bucket) moved to node i's actions."""
    p, na = int(tree.player[i]), int(tree.na[i])
    mine, theirs = c[i, p], c[i, 1 - p]
    pot, to_call = c[i].sum(), max(theirs - mine, 0)
    labels = [tree.vocab[x] for x in tree.lab[i, :na].numpy()]
    passive = next((k for k, lb in enumerate(labels) if lb in ("check", "call")), 0)
    fold = labels.index("fold") if "fold" in labels else passive
    allin = labels.index("allin") if "allin" in labels else None
    sized = []
    for k, lb in enumerate(labels):
        if lb in ("fold", "check", "call", "allin"):
            continue
        to = c[tree.child[i, k].item(), p]
        sized.append(((to - theirs) / max(pot + to_call, 1), k))
    target = {0: fold, 1: passive}
    for cls, frac in RAISE_FRACTION.items():
        if sized:
            target[cls] = min(sized, key=lambda x: abs(np.log(max(x[0], 1e-3) / frac)))[1]
        else:
            target[cls] = allin if allin is not None else passive
    target[6] = allin if allin is not None else (max(sized)[1] if sized else passive)
    s = np.zeros((len(probs), na), np.float32)
    for cls in range(7):
        s[:, target[cls]] += probs[:, cls]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clone")
    ap.add_argument("out")
    ap.add_argument("--tree", default="blueprint/run2")
    ap.add_argument("--abs", default="blueprint/out")
    ap.add_argument("--lib", default="blueprint/libbp3.so")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tree = Tree(args.tree, "cpu")
    B = int(open(os.path.join(args.abs, "meta.txt")).read().split()[1])
    dealer = Dealer(args.lib, args.abs, B, "cpu")
    hs = bucket_strength(args.abs, B, dealer)
    z = np.load(args.clone)
    kind, player, street = tree.kind.numpy(), tree.player.numpy(), tree.street.numpy()
    c = tree.c.numpy()
    raises_f = tree.feat[:, 10].numpy()
    total = int(
        max(tree.offset[i] + (169 if street[i] == 0 else B) * tree.na[i] for i in range(tree.n) if kind[i] == 0)
    )
    table = np.zeros(total, np.float32)
    dec = np.where(kind == 0)[0]
    for chunk in np.array_split(dec, max(1, len(dec) // 4000)):
        # every (node, bucket) of the chunk as one batch through the clone, then each node's classes to its actions
        nbs = np.where(street[chunk] == 0, 169, B)
        node_of = np.repeat(chunk, nbs)
        bucket = np.concatenate([np.arange(nb) for nb in nbs])
        p, st = player[node_of], street[node_of]
        mine, theirs = c[node_of, p], c[node_of, 1 - p]
        pot = c[node_of].sum(1)
        to_call = np.maximum(theirs - mine, 0)
        f = np.zeros((len(node_of), NFEAT), np.float32)
        f[np.arange(len(node_of)), st] = 1
        f[:, 4] = f[:, 5] = hs[st, bucket]
        f[:, 6] = to_call / np.maximum(pot, 1)
        f[:, 7] = np.minimum(pot / np.maximum(STACK - mine, 1), 5) / 5
        f[:, 8] = np.minimum((STACK - mine) / np.maximum(pot, 1), 50) / 50
        f[:, 9] = raises_f[node_of]
        f[:, 10] = to_call > 0
        f[:, 11] = 2 / 9
        seat = 1 - p  # abstract player 0 is the button: PokerKit heads-up seat 1
        f[:, 12] = seat
        f[:, 13] = seat == 1
        probs = forward(z, f)
        at = 0
        for i, nb in zip(chunk, nbs):
            table[tree.offset[i] : tree.offset[i] + nb * tree.na[i]] = node_actions(
                tree, c, i, probs[at : at + nb]
            ).ravel()
            at += nb
    table.tofile(os.path.join(args.out, "avg.f32"))
    shutil.copy(os.path.join(args.tree, "tree.tsv"), os.path.join(args.out, "tree.tsv"))
    print(f"{args.out}: {total} entries; pre-flop strength range {hs[0, :169].min():.3f}-{hs[0, :169].max():.3f}")


if __name__ == "__main__":
    main()
