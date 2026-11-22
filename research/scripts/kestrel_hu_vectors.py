"""Reference data for Kestrel's heads-up engine: the Python blueprint bot (blueprint_bot.py, the advisor's data pack)
at every decision of ours in logged Slumbot hands.

For each logged hand (results/slumbot/<run>/hands_*.jsonl) and each point where it was our turn, the Slumbot response
is rebuilt from the action string up to that point (slumbot_client.replay) and the blueprint is asked. Recorded: the
action prefix, our position, cards, Slumbot's legal raise range (slumbot_client.legal_range), the abstract node the hand
walks to, our bucket, and the distribution [(kind, raise-to in chips, prob)]; null when the hand has left the tree.

Made deterministic for comparison: an off-tree raise maps to the nearer abstract size by the pseudo-harmonic weight
(the size with weight >= 0.5) instead of at random, and the minimum-defence floor is off (BLUEPRINT_MDF_FROM=0), unless --floor.
Kestrel's self-test does the same.

Usage (lab, from research, the CPU venv with pokerkit):
  BLUEPRINT_RUN=blueprint/pack BLUEPRINT_ABS=blueprint/pack BLUEPRINT_LIB=blueprint/pack/libbp.so \\
  BLUEPRINT_MDF_FROM=0 PYTHONPATH=scripts .venv/bin/python scripts/kestrel_hu_vectors.py \\
      results/slumbot/slumbot_blueprint_final [--hands 400]
"""

import argparse
import glob
import json
import os
import random

import blueprint_bot as bb
import slumbot_client as sc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def map_raise_det(self, node, x):
    n = self.nodes[node]
    p = n["player"]
    cands = []
    for lab, ch in zip(n["labels"], n["children"]):
        if lab in ("fold", "check", "call"):
            continue
        c = self.nodes[ch]["c"]
        cands.append((self.frac(c[p], n["c"][1 - p], n["c"][p], n["c"][0] + n["c"][1]), ch))
    if not cands:
        return None
    cands.sort()
    if x <= cands[0][0]:
        return cands[0][1]
    if x >= cands[-1][0]:
        return cands[-1][1]
    for (a, ca), (b, cb) in zip(cands, cands[1:]):
        if a <= x <= b:
            pa = ((b - x) * (1 + a)) / max((b - a) * (1 + x), 1e-9)
            return ca if pa >= 0.5 else cb
    return cands[-1][1]


def prefixes(action, pos):
    """The action-string prefixes at which it is position `pos`'s turn (Slumbot: 0 big blind, 1 small blind)."""
    out, streets = [], sc.tokens(action)
    text = ""
    for si, st in enumerate(streets):
        if si:
            text += "/"
        actor = 1 if si == 0 else 0
        for a, amt in st:
            if actor == pos:
                out.append((text, si))
            text += a if a != "b" else f"b{amt}"
            actor = 1 - actor
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--hands", type=int, default=400)
    ap.add_argument(
        "--floor", action="store_true", help="keep the minimum-defence floor on (vectors_headsup_floor.json)"
    )
    ap.add_argument(
        "--texassolver",
        action="store_true",
        help="turn and river decisions, with TexasSolver's strategy at our hand from the blueprint ranges "
        "(vectors_headsup_texassolver.json; SOLVER_THREADS=1 for a deterministic solve)",
    )
    ap.add_argument(
        "--resolve",
        action="store_true",
        help="river decisions only, with the safe re-solve's strategy at our hand (vectors_headsup_resolve.json; "
        "needs a training run with avg.f32 and libresolve.so)",
    )
    a = ap.parse_args()
    bb.Blueprint.map_raise = map_raise_det
    if not a.floor:
        bb.Blueprint.MDF_FROM = 0.0
    bp = bb.get()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.run, "hands_*.jsonl"))):
        rows += [json.loads(x) for x in open(f)]
    rows = rows[: a.hands]
    cases = []
    for r in rows:
        for text, si in prefixes(r["action"], r["pos"]):
            board = r["board"][: (0, 3, 4, 5)[si]]
            resp = {"action": text, "client_pos": r["pos"], "hole_cards": r["hole"], "board": board}
            state = sc.replay(resp, random.Random(0))[0]
            i = r["pos"]
            if state.actor_index != i:
                continue
            scale = 100 / max(state.blinds_or_straddles)
            w = bp.walk(state, scale)
            if a.resolve and len(board) < 5 or a.texassolver and len(board) < 4:
                continue
            dist = bp.distribution(state, i, scale) if w else None
            res = None
            if a.resolve and w:
                all_s = bp.safe_strategies(state, i, scale)
                deck = [r + s for r in "23456789TJQKA" for s in "cdhs"]
                hole = sorted(r["hole"], key=deck.index)
                mine = all_s.get(hole[0] + hole[1]) if all_s else None
                res = [round(float(x), 6) for x in mine] if mine is not None else None
            if a.texassolver and w:
                ts = bp.resolve(state, i, scale)
                res = {k: round(float(v), 6) for k, v in ts.items()} if ts else None
            lo, hi, facing, last = sc.legal_range(sc.tokens(text))
            cases.append({
                "action": text, "pos": i, "hole": r["hole"], "board": board,
                "legal": {"min_to": lo, "max_to": hi, "facing": facing, "last": last},
                "node": w[0] if w else None,
                "bucket": bp.bucket(r["hole"] + board),
                "dist": [[k, None if t is None else int(t * sc.SCALE), round(float(p), 6)] for k, t, p in dist]
                if dist else None,
                "resolve": res,
            })  # fmt: skip
    name = "vectors_headsup_floor.json" if a.floor else "vectors_headsup.json"
    if a.resolve:
        name = "vectors_headsup_resolve.json"
    if a.texassolver:
        name = "vectors_headsup_texassolver.json"
    out = os.path.join(ROOT, "..", "kestrel", "testdata", name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"source": a.run, "hands": len(rows), "cases": cases}, open(out, "w"))
    off = sum(c["node"] is None for c in cases)
    print(f"{len(rows)} hands, {len(cases)} decisions ({off} off the tree) -> {out}")


if __name__ == "__main__":
    main()
