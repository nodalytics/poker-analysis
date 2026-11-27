"""Training data for a poker foundation model (stage 0): hands as token sequences (poker_tokens.py), in shards.

  selfplay RUN OUT --hands N    our network playing itself (every card known): each hand in both layouts
  phh DIR... OUT --source S     PHH hand histories (HandHQ: human, cards mostly unknown; Pluribus: all known)
  stats OUT                     tokens, hands, mean length, the tokens never used, and three hands decoded

A shard is a flat uint16 array of token ids, hands one after another (each from BOS to EOS), OUT/<name>_NNN.npy, with
OUT/<name>_NNN.json counting its hands and tokens. Self-play hands are written in the "public" layout (cards only
when shown, then REVEAL) and the "full" layout (cards first), so a model sees both; human hands only in "public".

Usage (lab, from research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/fm_data.py selfplay blueprint/nm25 data/fm --hands 1000000
  PYTHONPATH=scripts .venv/bin/python scripts/fm_data.py phh ~/datasets/phh-dataset/data/handhq/*NLH* data/fm \
      --source human --workers 16
  PYTHONPATH=scripts .venv/bin/python scripts/fm_data.py phh ~/datasets/phh-dataset/data/pluribus data/fm --source pluribus
  PYTHONPATH=scripts .venv/bin/python scripts/fm_data.py stats data/fm
"""

import argparse
import glob
import json
import os
import random
from multiprocessing import Pool

import numpy as np

from poker_tokens import ID, VOCAB, decode, encode, from_phh

SHARD = 50_000_000  # tokens a shard, at most


class Writer:
    def __init__(self, out, name):
        self.out, self.name, self.k = out, name, 0
        self.buf, self.n, self.hands = [], 0, 0
        os.makedirs(out, exist_ok=True)

    def add(self, ids):
        self.buf.append(np.asarray(ids, np.uint16))
        self.n += len(ids)
        self.hands += 1
        if self.n >= SHARD:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        a = np.concatenate(self.buf)
        base = os.path.join(self.out, f"{self.name}_{self.k:03d}")
        np.save(base + ".npy", a)
        json.dump({"hands": self.hands, "tokens": int(len(a))}, open(base + ".json", "w"))
        self.k += 1
        self.buf, self.n, self.hands = [], 0, 0


def cmd_selfplay(a):
    import torch

    from ninemax_eval import load, setup
    from ninemax_selfplay import FEAT, board_of, context, play
    from poker_tokens import from_tables

    dev = "cuda"
    torch.manual_seed(a.seed)
    dealer, menu, B = setup(a, dev)
    net, _ = load(a.run, B, dev)
    ante = 1.0 if json.load(open(os.path.join(a.run, "config.json"))).get("ante", 0) else 0.0
    w = Writer(a.out, f"self_{os.path.basename(a.run.rstrip('/'))}")
    P, done, bad = dealer.P, 0, 0
    while done < a.hands:
        n = min(a.batch, a.hands - done)
        bk, rk, cd = dealer.deal(n, cards=True)
        tabs = []
        play(dealer, menu, [net] * P, bk, rk, dev=dev, ctx=context(n, P, dev),
             board=board_of(cd, P, dev) if FEAT["texture"] else None, tables_out=tabs)  # fmt: skip
        T, u = tabs[0]
        u = u.cpu().numpy()
        for g in range(n):
            try:
                h = from_tables(T, g, cd[g], u[g], ante_bb=ante)
            except (IndexError, KeyError, ValueError):
                bad += 1
                continue
            w.add(encode(h, "public"))
            w.add(encode(h, "full"))
        done += n
    w.flush()
    print(json.dumps({"run": a.run, "hands": done, "unreadable": bad}), flush=True)


def _phh_file(args):
    path, source = args
    import re

    from hand_histories import parse_phh

    text = open(path, errors="replace").read()
    parts = re.split(r"(?m)^\[\d+\]\s*$", text) if path.endswith(".phhs") else [text]  # .phhs: many hands
    out, bad = [], 0
    for part in parts:
        if not part.strip():
            continue
        try:
            h = from_phh(parse_phh(part), source)
        except (KeyError, ValueError, IndexError, TypeError):
            h = None
        if h is None:
            bad += 1
            continue
        out.append(encode(h, "public"))
        if source != "human" and all(x is not None for x in h.holes):
            out.append(encode(h, "full"))
    return out, bad


def cmd_phh(a):
    files = []
    for d in a.dirs:
        files += sorted(glob.glob(os.path.join(d, "**", "*.phh*"), recursive=True)) if os.path.isdir(d) else [d]
    random.Random(1).shuffle(files)
    if a.max_files:
        files = files[: a.max_files]
    w = Writer(a.out, a.source)
    bad = 0
    with Pool(a.workers) as pool:
        for seqs, b in pool.imap_unordered(_phh_file, [(f, a.source) for f in files], chunksize=4):
            bad += b
            for s in seqs:
                w.add(s)
    w.flush()
    print(json.dumps({"source": a.source, "files": len(files), "unreadable": bad}), flush=True)


def cmd_stats(a):
    res, used = {}, np.zeros(len(VOCAB), np.int64)
    for f in sorted(glob.glob(os.path.join(a.out, "*.npy"))):
        x = np.load(f, mmap_mode="r")
        meta = json.load(open(f[:-4] + ".json"))
        src = os.path.basename(f).rsplit("_", 1)[0]
        r = res.setdefault(src, {"hands": 0, "tokens": 0})
        r["hands"] += meta["hands"]
        r["tokens"] += meta["tokens"]
        used += np.bincount(np.asarray(x[:20_000_000]), minlength=len(VOCAB))
    for r in res.values():
        r["mean_tokens_a_hand"] = round(r["tokens"] / max(r["hands"], 1), 1)
    res["vocab"] = len(VOCAB)
    res["never_used"] = [VOCAB[k] for k in range(len(VOCAB)) if used[k] == 0]
    examples = {}
    for f in sorted(glob.glob(os.path.join(a.out, "*_000.npy"))):
        x = np.load(f, mmap_mode="r")[:4000]
        ends = np.nonzero(np.asarray(x) == ID["EOS"])[0]
        if len(ends):
            examples[os.path.basename(f)] = decode(list(np.asarray(x[: ends[0] + 1])))
    res["examples"] = examples
    json.dump(res, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))
    for k, v in examples.items():
        print(f"--- {k}\n{v}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selfplay")
    s.add_argument("run")
    s.add_argument("out")
    s.add_argument("--hands", type=int, default=100000)
    s.add_argument("--batch", type=int, default=8192)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--players", type=int, default=9)
    s.add_argument("--menu", default="mtt")
    s.add_argument("--abs", default="blueprint/out")
    s.add_argument("--lib", default="blueprint/libbp7.so")
    p = sub.add_parser("phh")
    p.add_argument("dirs", nargs="+")
    p.add_argument("out")
    p.add_argument("--source", choices=("human", "pluribus"), required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-files", type=int, default=0)
    t = sub.add_parser("stats")
    t.add_argument("out")
    a = ap.parse_args()
    {"selfplay": cmd_selfplay, "phh": cmd_phh, "stats": cmd_stats}[a.cmd](a)


if __name__ == "__main__":
    main()
