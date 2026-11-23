"""Check advisor_pack's river step (B8 of docs/2026-11-23-review-and-plan.md) on small synthetic abstractions:

1. a distribution-aware abstraction (abstraction_pa: OCHS labels in river_buckets.u16) is refused with a message
   naming it, not a FileNotFoundError for river_buckets.u8;
2. a hand-strength abstraction stored in 16 bits (more than 255 river buckets) packs, and its cut points give back
   the table's buckets;
3. 16-bit river labels that are not ordered by hand strength are refused, whatever meta.txt says.

python3 scripts/advisor_pack_check.py          (numpy only, under a second; exits 1 on a failure)
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blueprint"))
import advisor_pack  # noqa: E402

N, K = 60000, 300


def write(d, meta, labels, hs):
    open(os.path.join(d, "meta.txt"), "w").write(meta)
    labels.astype(np.uint16).tofile(os.path.join(d, "river_buckets.u16"))
    hs.astype(np.float32).tofile(os.path.join(d, "river_hs.f32"))


def run(d, out):
    """'ok', or the refusal message, or the name of any other exception."""
    try:
        advisor_pack.river_cuts(d, out)
        return "ok"
    except SystemExit as e:
        return f"refused: {e.code}"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def main():
    rng = np.random.default_rng(3)
    hs = rng.random(N).astype(np.float32)
    ranks = np.argsort(np.argsort(hs, kind="stable"), kind="stable")
    ordered = (ranks * K // N).astype(np.uint16)  # equal-population hand-strength buckets
    fails = []
    with tempfile.TemporaryDirectory() as tmp:
        # 1. distribution-aware
        d, out = os.path.join(tmp, "pa"), os.path.join(tmp, "pa_out")
        os.makedirs(d), os.makedirs(out)
        meta = "buckets 300\nmetric flop,turn=CDF of next-street strength (k-means) river=OCHS(8) k-means\n"
        write(d, meta, rng.integers(0, K, N), hs)
        got = run(d, out)
        print(f"distribution-aware abstraction: {got}")
        if not (got.startswith("refused") and "distribution-aware" in got):
            fails.append("a distribution-aware abstraction is not refused with a clear message")

        # 2. hand strength in 16 bits
        d, out = os.path.join(tmp, "hs16"), os.path.join(tmp, "hs16_out")
        os.makedirs(d), os.makedirs(out)
        write(d, "buckets 300\nmetric flop,turn=EHS2 river=HS\n", ordered, hs)
        got = run(d, out)
        agree = float("nan")
        if got == "ok":
            cuts = np.fromfile(os.path.join(out, "river_cuts.f32"), dtype=np.float32)
            back = np.minimum(np.searchsorted(cuts, hs, side="left"), len(cuts) - 1)  # buckets.cpp's lower_bound
            agree = float(np.mean(back == ordered))
        print(f"hand-strength abstraction in river_buckets.u16 ({K} buckets): {got}, cut points agree {agree:.4f}")
        if got != "ok" or not agree >= 0.999:
            fails.append("a 16-bit hand-strength river table does not pack exactly")

        # 3. unordered labels claiming hand strength
        d, out = os.path.join(tmp, "bad"), os.path.join(tmp, "bad_out")
        os.makedirs(d), os.makedirs(out)
        write(d, "buckets 300\nmetric flop,turn=EHS2 river=HS\n", rng.permutation(K)[ordered], hs)
        got = run(d, out)
        print(f"unordered labels with meta river=HS: {got[:120]}")
        if not (got.startswith("refused") and "not ordered" in got):
            fails.append("river labels not ordered by hand strength are packed anyway")
    for f in fails:
        print("FAIL:", f)
    print("PASS" if not fails else "FAIL")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
