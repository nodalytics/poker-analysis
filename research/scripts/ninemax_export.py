"""Export a 9-max network for the repo: the playing weights only (the moving average, "avg"), in half precision,
with the run's config.json, into models/ninemax/<name>/ (avg_fp16.pt about 4.5 MB instead of a 45 MB training
checkpoint). Adds or replaces the run's entry in models/manifest.json with what it is, how long it trained, its
results and the sha256 of each file.

Every tool that loads a run also loads an export: give it models/ninemax/<name>/avg_fp16.pt (ninemax_eval.load reads
the config next to the file; the half-precision weights are cast back to float32 on load).

Usage (lab, GPU venv, from research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_export.py blueprint/nm4 [--name nm4] \\
      [--note "cash 100bb, NeuRD; fresh exploiter +44 at 30 min"]
"""

import argparse
import hashlib
import json
import os
import re
import shutil

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--name")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "models"))
    a = ap.parse_args()
    run = a.run.rstrip("/")
    name = a.name or os.path.basename(run)
    st = torch.load(os.path.join(run, "checkpoint.pt"), map_location="cpu", weights_only=True)
    avg = {k: (v.half() if v.is_floating_point() else v) for k, v in st["avg"].items()}
    d = os.path.join(a.out, "ninemax", name)
    os.makedirs(d, exist_ok=True)
    torch.save({"avg": avg, "hands": st["hands"]}, os.path.join(d, "avg_fp16.pt"))
    shutil.copy(os.path.join(run, "config.json"), os.path.join(d, "config.json"))
    last = ""
    prog = os.path.join(run, "progress.log")
    if os.path.exists(prog):
        lines = [x for x in open(prog) if " min " in x]
        last = lines[-1].strip() if lines else ""
    m = re.match(r"([\d.]+) min", last)
    tag = re.search(r"vs tag: ([-+\d.]+)", last)
    man_path = os.path.join(a.out, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    man.setdefault("ninemax", {})[name] = {
        "path": f"ninemax/{name}/avg_fp16.pt",
        "source": run,
        "hands_trained": int(st["hands"]),
        "minutes_trained": float(m.group(1)) if m else None,
        "vs_tag_bb100": float(tag.group(1)) if tag else None,
        "note": a.note,
        "sha256": {f: sha256(os.path.join(d, f)) for f in ("avg_fp16.pt", "config.json")},
        "bytes": os.path.getsize(os.path.join(d, "avg_fp16.pt")),
    }
    json.dump(man, open(man_path, "w"), indent=1, sort_keys=True)
    print(f"{name}: {man['ninemax'][name]['bytes'] / 1e6:.1f} MB -> {d}")


if __name__ == "__main__":
    main()
