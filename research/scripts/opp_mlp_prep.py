"""Next-action examples for every player in our self-play logs (session*/hands.jsonl), for opp_mlp_train.py.

The logs record each decision with the player's own cards, so every decision is labelled with what the player held.
Up to CAP decisions a player are taken, whole hands at a time. Features (the same as llm_ideas_prep.py, extended to
several players): street, hand strength against one random hand and against the number of opponents still in,
price, pot and stack ratios, raises this street, facing a bet, players in, position. Target, the action class:
    0 fold, 1 check/call, 2 raise <= 0.4 pot, 3 <= 0.8, 4 <= 1.2, 5 bigger, 6 all-in.

Usage: python scripts/opp_mlp_prep.py OUT.npz [--cap 40000] [--workers 24]    (from research, on the lab)
"""

import argparse
import glob
import json
import random
import zlib
from collections import defaultdict
from multiprocessing import Pool

import eval7
import numpy as np

PLAYERS = [
    "solverbot", "blueprintbot", "rlbot", "hedgebot", "adaptbot", "sol", "monte", "tina", "tinaplus",
    "bob", "carl", "max", "nora", "style_9000001", "shift_9100001", "shift_9100003",
    "qwen2.5vl:32b#plain", "hf.co/TheDrummer/Tiger-Gemma-9B-v3-GGUF:Q4_K_M#plain",
    "hf.co/TheDrummer/Big-Tiger-Gemma-27B-v3-GGUF:Q4_K_M#plain",
]  # fmt: skip
NFEAT = 15
DECK = [r + s for r in "23456789TJQKA" for s in "cdhs"]


def strength(hole, board, rng, n=120):
    hc = [eval7.Card(c) for c in hole]
    bc = [eval7.Card(c) for c in board]
    rest = [eval7.Card(c) for c in DECK if c not in hole and c not in board]
    win = 0.0
    for _ in range(n):
        draw = rng.sample(rest, 7 - len(bc))
        full = bc + draw[2:]
        a, b = eval7.evaluate(hc + full), eval7.evaluate(draw[:2] + full)
        win += 1.0 if a > b else 0.5 if a == b else 0.0
    return win / n


def split_cards(s):
    s = (s or "").replace(" ", "").replace(",", "")
    return [] if s in ("", "-") else [s[k : k + 2] for k in range(0, len(s), 2)]


STREETS = ["preflop", "flop", "turn", "river"]
# mid-points of the raise classes, as a fraction of the pot after calling (mlp_agent.py plays these)
RAISE_FRACTION = {2: 0.33, 3: 0.66, 4: 1.0, 5: 1.6}


def features(st, hs, to_call, pot, stack, raises, n_in, seat, nseat):
    """The model inputs for one decision; shared by the log extraction here and live play in mlp_agent.py."""
    f = np.zeros(NFEAT, np.float32)
    f[st] = 1
    f[4] = hs
    f[5] = hs ** max(n_in - 1, 1)
    f[6] = to_call / max(pot, 1)
    f[7] = min(pot / max(stack, 1), 5) / 5
    f[8] = min(stack / max(pot, 1), 50) / 50
    f[9] = min(raises, 4) / 4
    f[10] = float(to_call > 0)
    f[11] = n_in / 9
    f[12] = seat / max(nseat - 1, 1)
    f[13] = float(seat == nseat - 1)
    f[14] = float(nseat == 2)
    return f


def hand_examples(rows, players=None, payoffs=None):
    """rows: the decision rows of one hand, in order. Returns [(model, features, class, seat's payoff in chips)]."""
    players = PLAYERS if players is None else players
    out = []
    # blinds in the pot before the first action: 1/2 in our logs; other sources (human_clone.py) say so per row
    pre_pot, street, raises, prev = rows[0].get("start_pot", 3), None, 0, None
    folded = set()
    nseat = len(rows[0]["seats"])
    for r in rows:
        if r["street"] != street:
            street, raises = r["street"], 0
        to_call = r["to_call"] or 0
        added = r["pot"] - pre_pot
        act = r["action"]
        # the hand's last row is logged after the pot is paid out, so read stacks from the row before
        if prev is not None:
            stack_before = prev["stacks"][r["seat"]]
        else:
            stack_before = r["stacks"][r["seat"]] + (0 if act.startswith("folds") else added)
        if act.startswith("folds"):
            cls = 0
        elif act.startswith(("calls", "checks")):
            cls = 1
        else:
            extra = added - to_call
            if r["stacks"][r["seat"]] == 0:
                cls = 6
            else:
                f = extra / max(pre_pot + to_call, 1)
                cls = 2 if f <= 0.4 else 3 if f <= 0.8 else 4 if f <= 1.2 else 5
        if r["model"] in players and not r.get("invalid"):
            st = STREETS.index(street)
            hole, board = split_cards(r["hole"]), split_cards(r["board"])
            rng = random.Random(zlib.crc32((r["hole"] + str(r["board"]) + act).encode()))
            hs = strength(hole, board, rng) if len(hole) == 2 else 0.5
            f = features(st, hs, to_call, pre_pot, stack_before, raises, nseat - len(folded), r["seat"], nseat)
            out.append((r["model"], f, cls, payoffs[r["seat"]] if payoffs else 0))
        if cls == 0:
            folded.add(r["seat"])
        if cls >= 2:
            raises += 1
        pre_pot, prev = r["pot"], r
    return out


def read_hands(path):
    """[(path, hand, decision rows, payoffs per seat or None)] for one hands.jsonl."""
    hands, pay = defaultdict(list), {}
    for line in open(path):
        if '"model"' in line:
            r = json.loads(line)
            hands[r["hand"]].append(r)
        elif '"end"' in line:
            r = json.loads(line)
            pay[r["hand"]] = r.get("payoffs")
    return [(path, h, rows, pay.get(h)) for h, rows in hands.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cap", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    files = [p for p in glob.glob("**/hands.jsonl", recursive=True) if ".venv" not in p]
    random.Random(0).shuffle(files)
    count = defaultdict(int)
    chosen = []
    with Pool(args.workers) as pool:
        for hands in pool.imap_unordered(read_hands, files, chunksize=20):
            for path, h, rows, _ in hands:
                ms = {r["model"] for r in rows if r["model"] in PLAYERS and count[r["model"]] < args.cap}
                if not ms:
                    continue
                for r in rows:
                    if r["model"] in ms:
                        count[r["model"]] += 1
                chosen.append(rows)
            if all(count[p] >= args.cap for p in PLAYERS):
                pool.terminate()
                break
        print("decisions chosen:", dict(count), flush=True)
    with Pool(args.workers) as pool:
        per = pool.map(hand_examples, chosen, chunksize=50)
    M, F, Y, H = [], [], [], []
    for h, ex in enumerate(per):
        for m, f, c, _ in ex:
            M.append(PLAYERS.index(m))
            F.append(f)
            Y.append(c)
            H.append(h)
    np.savez_compressed(
        args.out, player=np.array(M, np.int8), feats=np.array(F), y=np.array(Y, np.int8), hand=np.array(H, np.int32)
    )
    print(f"{len(Y)} decisions from {len(chosen)} hands")


if __name__ == "__main__":
    main()
