"""Does a player's own history help the opponent model? A paired test (2026-11-28).

The bins of fm_train's score compare different hands (a player's 20th hand in a sequence is not a random hand), so they
confound history with the hands' make-up. Here the same hand is scored twice: the focus player's actions in a target
hand, after L of that player's own earlier hands (A), and after L hands drawn at random from other sequences (B), and
also with no context at all. A − B, paired by action, is the value of the player's history.

Usage: fm_paired.py DATA OUT_DIR_OF_MODEL [--L 5 20] [--val-tokens 4000000]
"""

import argparse
import json
import math
import os

import numpy as np
import torch

from fm_train import ACTION, GPT, POS, MambaLM
from poker_tokens import ID, VOCAB


def focus_actions(hand):
    """Positions (within the hand) of the focus player's action tokens: the token before is their position token."""
    out, fpos, last_pos, st = [], -1, -1, 0
    for j, t in enumerate(hand):
        if t in (ID["FLOP"], ID["TURN"], ID["RIVER"]):
            st = 1
        if POS[t] and st == 0 and fpos == -1:
            last_pos = t
        if t == ID["FOCUS"]:
            fpos = last_pos
        if j >= 1 and ACTION[t] and POS[hand[j - 1]] and hand[j - 1] == fpos:
            out.append(j)
    return out


def bits(model, ctx_hands, target, dev):
    """The focus player's bits an action in target, after the context hands."""
    pre = np.concatenate(ctx_hands) if ctx_hands else np.zeros(0, np.uint16)
    w = np.concatenate([pre, target]).astype(np.int64)
    x = torch.from_numpy(w[:-1])[None].to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lp = torch.log_softmax(model(x).float()[0], -1).cpu().numpy()
    off = len(pre)
    return [-lp[off + j - 1, w[off + j]] / math.log(2) for j in focus_actions(target)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("model_dir")
    ap.add_argument("--L", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--val-tokens", type=int, default=4_000_000)
    a = ap.parse_args()
    dev = "cuda"
    ck = torch.load(os.path.join(a.model_dir, "model.pt"), map_location=dev)
    g = ck["args"]
    if g["arch"] == "gpt":
        model = GPT(len(VOCAB), g["d"], g["layers"], g["heads"], g["ctx"])
    else:
        model = MambaLM(len(VOCAB), g["d"], g["layers"], g["heads"], g["attn_every"] if g["arch"] == "hybrid" else 0)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    arr = np.load(os.path.join(a.data, "players_000.npy"), mmap_mode="r")
    seq = np.asarray(arr[: min(a.val_tokens, len(arr) // 5)])  # the held-out part, as in training
    seps = np.nonzero(seq == ID["SEP"])[0]
    seqs, s0 = [], 0
    for e in seps:
        part = seq[s0:e]
        b = np.nonzero(part == ID["BOS"])[0]
        seqs.append([part[b[k] : (b[k + 1] if k + 1 < len(b) else len(part))] for k in range(len(b))])
        s0 = e + 1
    pool = [(i, h) for i, s in enumerate(seqs) for h in s]
    rng = np.random.default_rng(1)
    res = {"model": g["arch"], "sequences": len(seqs)}
    for L in a.L:
        own, rnd, none = [], [], []
        for i, s in enumerate(seqs):
            if len(s) < L + 1:
                continue
            target, mine = s[L], s[:L]
            other = []
            while len(other) < L:
                j, h = pool[rng.integers(len(pool))]
                if j != i:
                    other.append(h)
            n_own = sum(len(h) for h in mine) + len(target)
            n_rnd = sum(len(h) for h in other) + len(target)
            if max(n_own, n_rnd) > g["ctx"] + 1 or not focus_actions(target):
                continue
            own += bits(model, mine, target, dev)
            rnd += bits(model, other, target, dev)
            none += bits(model, [], target, dev)
        own, rnd, none = map(np.array, (own, rnd, none))
        d, d0 = own - rnd, own - none
        res[f"L{L}"] = {"actions": len(own), "own_bits": float(own.mean()), "random_bits": float(rnd.mean()),
                        "no_context_bits": float(none.mean()),
                        "own_minus_random": [float(d.mean()), float(d.std() / math.sqrt(len(d)))],
                        "own_minus_none": [float(d0.mean()), float(d0.std() / math.sqrt(len(d0)))]}  # fmt: skip
        print(json.dumps({f"L{L}": res[f"L{L}"]}), flush=True)
    json.dump(res, open(os.path.join(a.model_dir, "paired.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
