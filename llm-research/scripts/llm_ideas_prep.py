"""Turn the Slumbot hand logs into next-action examples for the LLM-style model study (llm_ideas_train.py).

Every Slumbot decision becomes one example: the hand so far as a token sequence (Slumbot's hole cards, the board as it
is dealt, every action with its size bucket), a few hand-made features (street, pot, price, hand strength against a
random hand), and the target, Slumbot's action class:
    0 fold, 1 check/call, 2 bet <= 0.4 pot, 3 <= 0.8, 4 <= 1.2, 5 bigger, 6 all-in.
Slumbot always shows its cards in the API logs, so every decision is labelled with the cards it held.

Usage: python scripts/llm_ideas_prep.py OUT.npz [--workers 16]    (from llm-research, on the lab)
"""

import argparse
import glob
import json
import random
import re
import zlib
from multiprocessing import Pool

import eval7
import numpy as np

STACK, SB, BB = 20000, 50, 100
RANKS, SUITS = "23456789TJQKA", "cdhs"
DECK = [r + s for r in RANKS for s in SUITS]
CARD = {c: i for i, c in enumerate(DECK)}
# token ids: 0 pad, 1-52 cards, 53 hole, 54 flop, 55 turn, 56 river, 57+ actions (actor 0/1 x class 0-6), 71 query
PAD, HOLE, STREET0, ACT0, QUERY = 0, 53, 54, 57, 71
VOCAB = 72
MAXLEN = 48
NFEAT = 12


def size_class(add, pot, allin):
    if allin:
        return 6
    f = add / max(pot, 1)
    return 2 if f <= 0.4 else 3 if f <= 0.8 else 4 if f <= 1.2 else 5


def strength(hole, board, n=150, rng=None):
    """Probability of beating a random hand on a random run-out (ties count half)."""
    rng = rng or random
    hc = [eval7.Card(c) for c in hole]
    bc = [eval7.Card(c) for c in board]
    rest = [eval7.Card(c) for c in DECK if c not in hole and c not in board]
    win = 0.0
    for _ in range(n):
        draw = rng.sample(rest, 2 + 5 - len(bc))
        full = bc + draw[2:]
        a, b = eval7.evaluate(hc + full), eval7.evaluate(draw[:2] + full)
        win += 1.0 if a > b else 0.5 if a == b else 0.0
    return win / n


def examples(row):
    """All Slumbot decisions in one hand: (tokens, features, target)."""
    rng = random.Random(zlib.crc32((row["action"] + "".join(row["slumbot_hole"])).encode()))
    me = 1 - row["pos"]  # Slumbot's seat; seat 1 is the small blind / button
    hole, board = row["slumbot_hole"], row["board"]
    toks = [HOLE, 1 + CARD[hole[0]], 1 + CARD[hole[1]]]
    out = []
    carry = 0  # chips each player put in on earlier streets
    for st, text in enumerate(row["action"].split("/")):
        if st > 0:
            toks.append(STREET0 + st - 1)
            new = board[:3] if st == 1 else board[1 + st : 2 + st]
            toks += [1 + CARD[c] for c in new]
        put = [0, 0]
        if st == 0:
            put = [BB, SB]  # seat 0 big blind, seat 1 small blind
        actor = 1 if st == 0 else 0
        raises = 0
        hs = None
        for a in re.findall(r"b\d+|[kcf]", text):
            pot = 2 * carry + put[0] + put[1]
            to_call = put[1 - actor] - put[actor]
            if a == "f":
                cls = 0
            elif a in "kc":
                cls = 1
                put[actor] = put[1 - actor]
            else:
                amt = int(a[1:])
                add = amt - put[actor]
                cls = size_class(add - to_call, pot + to_call, carry + amt >= STACK)
                put[actor] = amt
                raises += 1
            if actor == me:
                if hs is None:
                    vis = [] if st == 0 else (board[:3] if st == 1 else board[: 2 + st])
                    hs = strength(hole, vis, rng=rng)
                feat = np.zeros(NFEAT, np.float32)
                feat[st] = 1
                feat[4] = hs
                feat[5] = to_call / max(pot, 1)
                feat[6] = pot / STACK
                feat[7] = (STACK - carry - max(put)) / STACK
                feat[8] = min(raises, 4) / 4
                feat[9] = float(to_call > 0)
                feat[10] = float(me == 1)  # Slumbot on the button
                feat[11] = hs * hs
                out.append((toks + [QUERY], feat, cls))
            toks.append(ACT0 + 7 * int(actor == me) + cls)
            actor = 1 - actor
        carry += max(put)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    files = sorted(glob.glob("slumbot_*/hands_*.jsonl"))
    rows, groups = [], []
    for g, f in enumerate(sorted({x.split("/")[0] for x in files})):
        for p in files:
            if p.startswith(f + "/"):
                for line in open(p):
                    r = json.loads(line)
                    if r.get("slumbot_hole"):
                        rows.append(r)
                        groups.append(g)
    with Pool(args.workers) as pool:
        per = pool.map(examples, rows, chunksize=200)
    T, F, Y, H = [], [], [], []
    for h, (ex, g) in enumerate(zip(per, groups)):
        for toks, feat, cls in ex:
            t = toks if len(toks) <= MAXLEN else toks[:3] + toks[-(MAXLEN - 3) :]
            t = t + [PAD] * (MAXLEN - len(t))
            T.append(t)
            F.append(feat)
            Y.append(cls)
            H.append(h)
    np.savez_compressed(
        args.out,
        tokens=np.array(T, np.int16),
        feats=np.array(F, np.float32),
        y=np.array(Y, np.int8),
        hand=np.array(H, np.int32),
    )
    print(f"{len(rows)} hands, {len(Y)} Slumbot decisions, classes {np.bincount(Y, minlength=7).tolist()}")


if __name__ == "__main__":
    main()
