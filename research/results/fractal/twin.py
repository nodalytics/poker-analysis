import numpy as np, sys
a, b, absdir = sys.argv[1:4]
B = int(open(absdir + "/meta.txt").read().split()[1])
rows = [l.rstrip("\n").split("\t") for l in open(a + "/tree.tsv")][1:]
rb = [l.rstrip("\n").split("\t") for l in open(b + "/tree.tsv")][1:]
assert [r[6] for r in rows] == [r[6] for r in rb]
A = np.fromfile(a + "/avg.f32", np.float32); C = np.fromfile(b + "/avg.f32", np.float32)
for st in (0, 1, 2, 3):
    num = den = 0.0
    for r in rows:
        if r[1] != "0" or int(r[3]) != st: continue
        na = len(r[7].split(",")); nb = 169 if st == 0 else B; o = int(r[6])
        x = A[o:o+nb*na].astype(np.float64).reshape(nb, na); y = C[o:o+nb*na].astype(np.float64).reshape(nb, na)
        wx, wy = x.sum(1), y.sum(1)
        px = np.where(wx[:, None] > 0, np.maximum(x, 0) / np.maximum(wx[:, None], 1e-30), 1 / na)
        py = np.where(wy[:, None] > 0, np.maximum(y, 0) / np.maximum(wy[:, None], 1e-30), 1 / na)
        w = np.sqrt(np.maximum(wx, 0) / max(wx.sum(), 1e-30) * np.maximum(wy, 0) / max(wy.sum(), 1e-30)) * np.sqrt(max(wx.sum(),0) * max(wy.sum(),0))
        num += (w * 0.5 * np.abs(px - py).sum(1)).sum(); den += w.sum()
    print(st, "reach-weighted TV between the two runs:", round(num / den, 4), flush=True)
