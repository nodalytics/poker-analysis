"""Read a blueprint run: the average strategy at a node for every pre-flop class (or bucket).

Usage: python inspect_blueprint.py RUN_DIR [NODE_ID]   (default: node 0, the small blind's first decision)
"""

import sys

import numpy as np


def load(run):
    rows = [ln.rstrip("\n").split("\t") for ln in open(f"{run}/tree.tsv")][1:]
    nodes = {
        int(r[0]): {
            "kind": int(r[1]),
            "player": int(r[2]),
            "street": int(r[3]),
            "c0": int(r[4]),
            "c1": int(r[5]),
            "offset": int(r[6]),
            "labels": r[7].split(",") if r[7] else [],
            "children": [int(x) for x in r[8].split(",")] if r[8] else [],
        }
        for r in rows
    }
    avg = np.fromfile(f"{run}/avg.f32", dtype=np.float32)
    return nodes, avg


def strategy(nodes, avg, node, bucket):
    n = nodes[node]
    k = len(n["labels"])
    s = avg[n["offset"] + bucket * k : n["offset"] + (bucket + 1) * k].astype(float)
    return s / s.sum() if s.sum() > 0 else np.full(k, 1 / k)


if __name__ == "__main__":
    run = sys.argv[1]
    node = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    nodes, avg = load(run)
    classes = [ln.split("\t")[1].strip() for ln in open(f"{run}/classes.tsv")] if node == 0 or nodes[node]["street"] == 0 else None
    n = nodes[node]
    print("node", node, "player", n["player"], "street", n["street"], "actions", n["labels"])
    nb = 169 if n["street"] == 0 else 200
    for b in range(nb):
        s = strategy(nodes, avg, node, b)
        name = classes[b] if classes else f"bucket {b}"
        print(f"{name:6}", " ".join(f"{lab}:{p:.2f}" for lab, p in zip(n["labels"], s)))
