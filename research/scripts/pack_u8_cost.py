"""What 8-bit storage (blueprint/advisor_pack.py, Kestrel's pack) changes in a heads-up run's average strategy: per
(node, bucket) row, the avg.f32 policy against the 8-bit one. Usage: pack_u8_cost.py blueprint/run2 blueprint/out"""

import os
import sys

import numpy as np

run, absdir = sys.argv[1], sys.argv[2]
rows = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(run, "tree.tsv"))][1:]
B = int(open(os.path.join(absdir, "meta.txt")).read().split()[1])
avg = np.fromfile(os.path.join(run, "avg.f32"), dtype=np.float32)
diffs, tv, dropped_mass, n_pos, n_drop, nrows = [], [], 0.0, 0, 0, 0
for r in rows:
    if r[1] != "0":
        continue
    na, nb, off = len(r[8].split(",")), (169 if r[3] == "0" else B), int(r[6])
    s = avg[off : off + nb * na].reshape(nb, na).astype(np.float64)
    tot = s.sum(1, keepdims=True)
    p = np.where(tot > 0, np.maximum(s, 0) / np.maximum(tot, 1e-30), 1.0 / na)
    qq = np.round(p * 255)
    back = qq / np.maximum(qq.sum(1, keepdims=True), 1)
    d = np.abs(back - p)
    diffs.append(d.max(1))
    tv.append(0.5 * d.sum(1))
    gone = (p > 0) & (qq == 0)
    n_pos += int((p > 0).sum())
    n_drop += int(gone.sum())
    dropped_mass += float(p[gone].sum())
    nrows += nb
mx, t = np.concatenate(diffs), np.concatenate(tv)
print(
    f"rows {nrows}: largest change per row: max {mx.max():.4f}, mean {mx.mean():.5f}, p99 {np.quantile(mx, 0.99):.4f}"
)
print(f"total variation per row: mean {t.mean():.5f}, p99 {np.quantile(t, 0.99):.4f}, max {t.max():.4f}")
print(
    f"actions with p > 0 rounded to 0: {n_drop} of {n_pos} ({n_drop / n_pos:.2%}), mean mass per row {dropped_mass / nrows:.5f}"
)
