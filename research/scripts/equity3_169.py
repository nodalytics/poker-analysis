"""Three-way all-in equities for the 169 pre-flop classes, for the two-caller push/fold solver (icm_pushfold.py).

For every unordered triple of classes (a <= b <= c in results/preflop/equity169.json's order; 818,805 triples) a Monte
Carlo over real cards: each sample deals one combo of each class with card removal (the three hands never share a
card; a conflicting draw is redrawn), then a board of five from the 46 cards left, and ranks the three hands with
the blueprint library's 7-card evaluator. A tie is broken at random, so the six strength orders (who has the best,
second and worst hand) carry the split pots in expectation: exact for chips, close for ICM. Stored per triple: the
probability of each order, as uint16 (p x 65535), in an .npz of about 10 MB. Triples no deal can produce (three
classes that need more than four aces, say AA, AA, AKs) are flagged and read as equal odds; the solver weighs
classes without card removal, so it gives them a tiny weight.

Order k of ORDERS lists the triple's sorted positions from best hand to worst: (0, 1, 2) means a beat b beat c.

Usage (lab, CPU: the library needs the abstraction for its bucket tables):
  PYTHONPATH=scripts OMP_NUM_THREADS=4 python scripts/equity3_169.py --samples 2000 --workers 8 \
      --lib blueprint/libbp7.so --abs blueprint/out --out results/preflop/equity3_169.npz
"""

import argparse
import ctypes
import json
import os
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATRIX = os.path.join(ROOT, "results", "preflop", "equity169.json")
RANKS = "23456789TJQKA"
ORDERS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def class_combos(name):
    """The combos of a class name ('AA', 'AKs', 'T9o') as card pairs, card = rank * 4 + suit."""
    r1, r2 = RANKS.index(name[0]), RANKS.index(name[1])
    if len(name) == 2:
        return [(r1 * 4 + s, r1 * 4 + t) for s in range(4) for t in range(s + 1, 4)]
    if name[2] == "s":
        return [(r1 * 4 + s, r2 * 4 + s) for s in range(4)]
    return [(r1 * 4 + s, r2 * 4 + t) for s in range(4) for t in range(4) if s != t]


def combo_table(classes):
    """[169, 12, 2] combos (padded) and [169] counts."""
    tab = np.zeros((len(classes), 12, 2), np.int32)
    cnt = np.zeros(len(classes), np.int64)
    for i, c in enumerate(classes):
        cc = class_combos(c)
        tab[i, : len(cc)] = cc
        cnt[i] = len(cc)
    return tab, cnt


def triples(k=169):
    """Every sorted triple a <= b <= c, [T, 3]."""
    a, b, c = np.meshgrid(np.arange(k), np.arange(k), np.arange(k), indexing="ij")
    keep = (a <= b) & (b <= c)
    return np.stack([a[keep], b[keep], c[keep]], 1).astype(np.int32)


_W = {}


def _init(lib, absdir, classes):
    L = ctypes.CDLL(lib)
    if L.bp_init(absdir.encode()) != 0:
        raise RuntimeError(f"no buckets in {absdir}")
    L.bp_bucket_cards.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 3
    _W["lib"] = L
    _W["tab"], _W["cnt"] = combo_table(classes)


def _chunk(job):
    """Order counts [len(tri), 6] and accepted samples [len(tri)] for one chunk of triples."""
    tri, S, seed = job
    tab, cnt, L = _W["tab"], _W["cnt"], _W["lib"]
    rng = np.random.default_rng(seed)
    T = len(tri)
    N = T * S
    cls = np.repeat(tri, S, axis=0)  # [N, 3]
    cards = np.zeros((N, 6), np.int32)
    todo = np.arange(N)
    for _ in range(60):  # rejection: redraw every sample whose three hands share a card
        if len(todo) == 0:
            break
        c = cls[todo]
        pick = (rng.random((len(todo), 3)) * cnt[c]).astype(np.int64)
        h = tab[c, pick]  # [m, 3, 2]
        cards[todo] = h.reshape(-1, 6)
        srt = np.sort(cards[todo], 1)
        bad = (srt[:, 1:] == srt[:, :-1]).any(1)
        todo = todo[bad]
    ok = np.ones(N, bool)
    ok[todo] = False  # never dealt without a conflict: impossible (or vanishingly rare) triples
    cards[todo] = np.arange(6, dtype=np.int32)  # a dummy deal, discarded
    mask = np.ones((N, 52), bool)
    mask[np.arange(N)[:, None], cards] = False
    rem = np.nonzero(mask)[1].reshape(N, 46).astype(np.int32)
    for j in range(5):  # partial Fisher-Yates: the board
        r = j + (rng.random(N) * (46 - j)).astype(np.int64)
        a, b = rem[:, j].copy(), rem[np.arange(N), r]
        rem[:, j], rem[np.arange(N), r] = b, a
    deal = np.ascontiguousarray(np.concatenate([cards, rem[:, :5]], 1), dtype=np.int32)  # [N, 2 x 3 + 5]
    bk = np.zeros((N, 3, 4), np.int32)
    rk = np.zeros((N, 3), np.int32)
    L.bp_bucket_cards(N, 3, deal.ctypes.data, bk.ctypes.data, rk.ctypes.data)
    key = rk + rng.random((N, 3))  # lower rank is stronger; ties broken at random
    order = np.argsort(key, 1)  # best first: positions
    code = order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]
    lut = np.full(27, -1)
    for i, o in enumerate(ORDERS):
        lut[o[0] * 9 + o[1] * 3 + o[2]] = i
    k = lut[code]
    row = np.repeat(np.arange(T), S)
    counts = np.zeros((T, 6), np.int64)
    np.add.at(counts, (row[ok], k[ok]), 1)
    return counts, counts.sum(1)


def build(lib, absdir, samples, workers, out, chunk=200, seed=1):
    classes = json.load(open(MATRIX))["classes"]
    tri = triples(len(classes))
    jobs = [(tri[s : s + chunk], samples, seed * 1_000_003 + s) for s in range(0, len(tri), chunk)]
    t0 = time.time()
    counts = np.zeros((len(tri), 6), np.int64)
    with Pool(workers, initializer=_init, initargs=(lib, absdir, classes)) as pool:
        for j, (c, _) in enumerate(pool.imap(_chunk, jobs, chunksize=1)):
            s = j * chunk
            counts[s : s + len(c)] = c
            if j % 200 == 0:
                print(f"{s + len(c):,} / {len(tri):,} triples, {time.time() - t0:.0f}s", flush=True)
    n = counts.sum(1)
    impossible = n == 0
    p = np.where(impossible[:, None], 1 / 6, counts / np.maximum(n, 1)[:, None])
    np.savez_compressed(
        out,
        probs=np.round(p * 65535).astype(np.uint16),
        impossible=impossible,
        classes=np.array(classes),
        samples=np.int64(samples),
        orders=np.array(ORDERS),
    )
    print(f"done: {len(tri):,} triples x {samples} samples in {time.time() - t0:.0f}s; impossible {impossible.sum()};"
          f" fewest accepted samples {n[~impossible].min()}; wrote {out}", flush=True)  # fmt: skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=2000, help="Monte Carlo deals per triple")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lib", default=os.path.join(ROOT, "blueprint", "libbp7.so"))
    ap.add_argument("--abs", default=os.path.join(ROOT, "blueprint", "out"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "preflop", "equity3_169.npz"))
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first LIMIT chunks (a timing test)")
    a = ap.parse_args()
    if a.limit:
        classes = json.load(open(MATRIX))["classes"]
        _init(a.lib, a.abs, classes)
        tri = triples(len(classes))
        t0 = time.time()
        for j in range(a.limit):
            _chunk((tri[j * a.chunk : (j + 1) * a.chunk], a.samples, j))
        dt = (time.time() - t0) / a.limit
        print(f"{dt:.2f}s a chunk of {a.chunk} triples x {a.samples}: the full table needs "
              f"{dt * len(tri) / a.chunk / a.workers / 60:.0f} min on {a.workers} workers")  # fmt: skip
        return
    build(a.lib, a.abs, a.samples, a.workers, a.out, a.chunk, a.seed)


if __name__ == "__main__":
    main()
