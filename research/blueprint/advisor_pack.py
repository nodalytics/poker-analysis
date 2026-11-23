"""Build the heads-up blueprint data pack Kestrel reads (--pack): about 85 MB instead of 0.8 GB.

    tree.tsv, meta.txt                     the betting tree and abstraction size
    avg.u8                                 the average strategy, each (node, bucket) normalised and stored in 8 bits
    flop_buckets.u8, turn_buckets.u8       the flop and turn bucket tables (15 MB; .u16 when the abstraction has them)
    river_cuts.f32                         one upper hand-strength edge per river bucket, instead of the 123 MB river
                                           table: libbp computes the hand's strength (990 opponent hands) and cuts it

Checks, printed: the largest probability change from 8-bit storage, and river-bucket agreement between the cut
points and the full table on random hands (the two libraries side by side, in separate processes).

Only hand-strength abstractions (abstraction.cpp) can be packed: the river cut points need buckets ordered by hand
strength. A distribution-aware abstraction (abstraction_pa.cpp, OCHS river labels) is refused with a message.

Usage (lab, from research): .venv/bin/python blueprint/advisor_pack.py blueprint/run2 blueprint/out blueprint/pack
"""

import os
import shutil
import subprocess
import sys

import numpy as np


def quantise(run, absdir, out):
    rows = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(run, "tree.tsv"))][1:]
    buckets = int(open(os.path.join(absdir, "meta.txt")).read().split()[1])
    avg = np.fromfile(os.path.join(run, "avg.f32"), dtype=np.float32)
    q = np.zeros(len(avg), np.uint8)
    worst = 0.0
    for r in rows:
        if r[1] != "0":
            continue
        na = len(r[8].split(","))
        nb = 169 if r[3] == "0" else buckets
        off = int(r[6])
        s = avg[off : off + nb * na].reshape(nb, na).astype(np.float64)
        tot = s.sum(1, keepdims=True)
        p = np.where(tot > 0, np.maximum(s, 0) / np.maximum(tot, 1e-30), 1.0 / na)
        qq = np.round(p * 255).astype(np.uint8)
        back = qq / np.maximum(qq.sum(1, keepdims=True), 1)
        worst = max(worst, float(np.abs(back - p).max()))
        q[off : off + nb * na] = qq.ravel()
    q.tofile(os.path.join(out, "avg.u8"))
    print(f"avg.u8: {len(q)} bytes; largest probability change from 8-bit storage {worst:.4f}")


def bucket_table(absdir, street):
    """(path, dtype) of a street's bucket table: the 16-bit one if present (more than 255 buckets, or abstraction_pa),
    else the 8-bit one; None if neither exists. Kestrel's buckets.cpp reads them in the same order."""
    for ext, dt in (("u16", np.uint16), ("u8", np.uint8)):
        p = os.path.join(absdir, f"{street}_buckets.{ext}")
        if os.path.exists(p):
            return p, dt
    return None


def river_cuts(absdir, out):
    """One hand-strength upper edge per river bucket. Valid only when the river buckets are ordered by hand strength
    (abstraction.cpp, metric river=HS). A distribution-aware abstraction (abstraction_pa.cpp: OCHS k-means labels in
    *_buckets.u16) has no such order, and cut points would silently re-bucket every river hand: refused."""
    meta = open(os.path.join(absdir, "meta.txt")).read()
    if "OCHS" in meta:
        sys.exit(
            f"advisor_pack: {absdir} is a distribution-aware abstraction (abstraction_pa: river buckets are OCHS "
            "k-means labels, not ordered by hand strength), so its river cannot be packed as hand-strength cut points. "
            "Pack only hand-strength abstractions (abstraction.cpp, meta 'river=HS')."
        )
    table = bucket_table(absdir, "river")
    if table is None:
        sys.exit(f"advisor_pack: no river_buckets.u16 or river_buckets.u8 in {absdir}")
    hs = np.fromfile(os.path.join(absdir, "river_hs.f32"), dtype=np.float32)
    rb = np.fromfile(table[0], dtype=table[1])
    if len(rb) != len(hs):
        sys.exit(f"advisor_pack: {table[0]} has {len(rb)} entries, river_hs.f32 {len(hs)}")
    order = np.argsort(rb, kind="stable")
    starts = np.searchsorted(rb[order], np.arange(int(rb.max()) + 1))
    top = np.maximum.reduceat(hs[order], starts).astype(np.float32)
    if np.any(np.diff(top) < -1e-6):  # a bucket whose strongest hand is weaker than the previous bucket's
        sys.exit(
            f"advisor_pack: the river buckets in {table[0]} are not ordered by hand strength (bucket edges fall at "
            f"{int(np.sum(np.diff(top) < -1e-6))} places), so cut points cannot reproduce them; refusing to pack"
        )
    cuts = np.maximum.accumulate(top)  # monotone upper edges
    cuts.tofile(os.path.join(out, "river_cuts.f32"))
    print(f"river_cuts.f32: {len(cuts)} edges, {cuts[0]:.4f} .. {cuts[-1]:.4f}")


CHECK = r"""
import ctypes, random, sys
lib = ctypes.CDLL(sys.argv[1]); assert lib.bp_init(sys.argv[2].encode()) == 0
rng = random.Random(5); out = []
for _ in range(int(sys.argv[3])):
    c = rng.sample(range(52), 7); arr = (ctypes.c_int * 7)(*c); out.append(lib.bp_bucket(arr, 7))
print(" ".join(map(str, out)))
"""


def check_river(absdir, out, n=20000):
    full = subprocess.run([sys.executable, "-c", CHECK, "blueprint/libbp.so", absdir, str(n)], capture_output=True)
    cut = subprocess.run([sys.executable, "-c", CHECK, "blueprint/libbp4.so", out, str(n)], capture_output=True)
    a = np.array(full.stdout.split(), dtype=int)
    b = np.array(cut.stdout.split(), dtype=int)
    print(
        f"river buckets, cut points vs the full table on {n} random hands: {np.mean(a == b):.4f} identical, "
        f"{np.mean(np.abs(a - b) <= 1):.4f} within one bucket"
    )


def main():
    run, absdir, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    river_cuts(absdir, out)  # first: it refuses an abstraction a pack cannot hold before the slow steps
    for street in ("flop", "turn"):
        table = bucket_table(absdir, street)
        if table is None:
            sys.exit(f"advisor_pack: no {street}_buckets.u16 or {street}_buckets.u8 in {absdir}")
        shutil.copy(table[0], os.path.join(out, os.path.basename(table[0])))
    shutil.copy(os.path.join(absdir, "meta.txt"), os.path.join(out, "meta.txt"))
    shutil.copy(os.path.join(run, "tree.tsv"), os.path.join(out, "tree.tsv"))
    quantise(run, absdir, out)
    check_river(absdir, out)


if __name__ == "__main__":
    main()
