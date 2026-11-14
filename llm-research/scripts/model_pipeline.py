"""The model pipeline: from a recipe to a model Kestrel can install. Four steps, each its own command:

  train RECIPE --run blueprint/nmXX [--gpu N]
      Launch the 9-max trainer (ninemax_selfplay.py) on the lab with the recipe's settings, detached. The recipe is
      copied into the run directory (pipeline.yaml), so every model can be retrained exactly.
  gate RUN [--snapshot avg_12h.pt] [--recipe R]
      The quality gates, with the recipe's thresholds: the poker-sense checks, push/fold agreement with the ICM charts,
      a duplicate match against the current champion of the game (models/manifest.json "champions"), and Kestrel's
      C++ self-test on this network. Writes results/models/<name>_gate.json; a model that fails a gate does not ship.
  package RUN --name NAME --version V [--snapshot ...] [--force]
      A versioned archive dist/models/<name>-<version>.tar.gz (and .sha256): the fp16 weights, Kestrel's net.bin and
      net.json, its test vectors, and model.json (recipe, training, gate results, every file's sha256). Refused
      unless the gate passed.
  publish PACKAGE   (on your machine: needs gh)
      A GitHub Release model-<name>-v<version> with the archive and its checksum; the release is recorded in
      models/manifest.json. Kestrel installs it with: kestreld models install <name>-<version>.tar.gz

Recipes are YAML files in pipeline/recipes/ (see nm17.yaml). Usage (lab, from llm-research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/model_pipeline.py train pipeline/recipes/mtt9.yaml --run blueprint/nm21 --gpu 1
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/model_pipeline.py gate blueprint/nm21 --snapshot avg_12h.pt
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/model_pipeline.py package blueprint/nm21 --name mtt9 --version 2
  uv run python scripts/model_pipeline.py publish dist/models/mtt9-2.tar.gz          # on your machine
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
ENV = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "scripts"), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
RES = os.path.join(ROOT, "results", "models")
KESTREL = os.path.join(os.path.expanduser("~"), "kestrel")  # the lab's core-only build of Kestrel (--kestrel)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, env=kw.pop("env", ENV), **kw)


def recipe_of(run_dir, path=None):
    p = path or os.path.join(run_dir, "pipeline.yaml")
    if not os.path.exists(p):
        sys.exit(f"no recipe: give --recipe or train the run with this pipeline ({p} is missing)")
    return yaml.safe_load(open(p))


# ---- train ----


def train(a):
    r = yaml.safe_load(open(a.recipe))
    if os.path.exists(os.path.join(a.run, "checkpoint.pt")):
        sys.exit(f"{a.run} already has a checkpoint; pick a new run directory")
    os.makedirs(a.run, exist_ok=True)
    cmd = [PY, os.path.join(ROOT, "scripts", "ninemax_selfplay.py"), a.run]
    for k, v in r["train"].items():
        if v is True:
            cmd.append(f"--{k}")
        elif v not in (False, None):
            cmd += [f"--{k}", str(v)]
    if r.get("init"):
        cmd += ["--init", r["init"]]
    shutil.copy(a.recipe, os.path.join(a.run, "pipeline.yaml"))
    env = {**ENV, "CUDA_VISIBLE_DEVICES": str(a.gpu), "OMP_NUM_THREADS": "1"}
    log = open(os.path.join(a.run, "stdout.log"), "a")
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"training {r['name']} in {a.run} on GPU {a.gpu} (pid {p.pid}); {r['train'].get('minutes')} minutes")


# ---- gate ----


def snapshot_path(run_dir, snap):
    return os.path.join(run_dir, snap) if snap else os.path.join(run_dir, "checkpoint.pt")


def gate(a):
    r = recipe_of(a.run, a.recipe)
    g = r.get("gates", {})
    snap = snapshot_path(a.run, a.snapshot)
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    name = os.path.basename(a.run.rstrip("/"))
    os.makedirs(RES, exist_ok=True)
    out = {"run": a.run, "snapshot": snap, "recipe": r["name"], "time": time.strftime("%Y-%m-%d %H:%M"), "gates": []}

    def add(gate_name, ok, value, need):
        out["gates"].append({"gate": gate_name, "passed": bool(ok), "value": value, "need": need})
        print(f"  {'PASS' if ok else 'FAIL'}  {gate_name}: {value} (need {need})", flush=True)

    gpu = {**ENV, "CUDA_VISIBLE_DEVICES": str(a.gpu)}
    with tempfile.TemporaryDirectory() as tmp:
        # poker sense: the reference bands (premiums raised, trash folded, opening widths, short-stack and BB play)
        js = os.path.join(tmp, "sense.json")
        run([PY, "scripts/ninemax_sense.py", snap, "--out", js], env=gpu, check=True, capture_output=True)
        checks = next(iter(json.load(open(js)).values()))["checks"]
        n_ok = sum(c["ok"] for c in checks)
        add("poker-sense checks within their bands", n_ok >= g.get("sense_min_ok", 7), f"{n_ok}/{len(checks)}",
            f">= {g.get('sense_min_ok', 7)}")  # fmt: skip
        prem = all(c["ok"] for c in checks if c["check"].startswith("premiums"))
        add("premium hands raised first in", prem, prem, True)
        # push/fold: agreement with the ICM charts at 10bb (the chip-EV table, all positions)
        jp = os.path.join(tmp, "pushfold.json")
        extra = (
            ["--charts", os.path.join(ROOT, "results", "ninemax", "icm_pushfold_ante.json")] if cfg.get("ante") else []
        )
        run([PY, "scripts/ninemax_pushfold_check.py", snap, "--out", jp, *extra], env=gpu, check=True,
            capture_output=True)  # fmt: skip
        tables = json.load(open(jp))
        first = next(k for k in tables if "chip EV" in k)
        agree = sum(v["agree"] for v in tables[first].values()) / len(tables[first])
        add("push/fold agreement with the ICM chart (chip EV)", agree >= g.get("pushfold_agree_min", 0.85),
            round(agree, 3), f">= {g.get('pushfold_agree_min', 0.85)}")  # fmt: skip
        # the champion: a duplicate 1-vs-8 match; the new model must not be clearly worse
        man = json.load(open(os.path.join(ROOT, "models", "manifest.json")))
        champ = man.get("champions", {}).get(r["game"])
        if champ and g.get("not_worse_than_champion", True):
            cp = os.path.join(ROOT, "models", "ninemax", champ, "avg_fp16.pt")
            jc = os.path.join(tmp, "cross.json")
            run([PY, "scripts/ninemax_eval.py", "cross", snap, cp, "--hands", str(g.get("champion_hands", 20000)),
                 "--menu", cfg.get("menu", "medium"), "--out", jc], env=gpu, check=True, capture_output=True)  # fmt: skip
            res = json.load(open(jc))
            key = next(k for k in res if k.endswith(f"vs {os.path.relpath(cp, ROOT).removesuffix('.pt')}")
                       or k.endswith(f"vs models/ninemax/{champ}/avg_fp16"))  # fmt: skip
            m, lo, hi = res[key]
            add(f"against the champion ({champ}), 1-vs-8 duplicate", hi >= 0, f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}] bb/100",
                "not clearly worse: interval reaches 0")  # fmt: skip
        else:
            add("against the champion", True, "no champion for this game yet", "-")
        # Kestrel: the C++ build must reproduce this network (features, logits, game, ICM)
        sel = os.path.join(a.kestrel, "build", "kestrel_selftest")
        if os.path.exists(sel):
            vec = os.path.join(tmp, f"vectors_{name}.json")
            run([PY, "scripts/advisor_vectors.py", snap, "--out", tmp, "--vectors-out", vec], check=True,
                capture_output=True)  # fmt: skip
            p = run([sel, "--vectors", os.path.join(a.kestrel, "testdata"), "--nine-max", vec, "--model", tmp,
                     "--pack", os.path.join(ROOT, "blueprint", "pack")], capture_output=True, text=True)  # fmt: skip
            add("Kestrel's C++ build reproduces it (self-test)", p.returncode == 0 and "PASS" in p.stdout,
                p.stdout.strip().splitlines()[-1] if p.stdout else p.stderr[-200:], "PASS")  # fmt: skip
        else:
            add("Kestrel's C++ build reproduces it (self-test)", False, f"no {sel}", "a Kestrel build (--kestrel)")
    out["passed"] = all(x["passed"] for x in out["gates"])
    path = os.path.join(RES, f"{name}_gate.json")
    json.dump(out, open(path, "w"), indent=1)
    print(("PASSED" if out["passed"] else "FAILED") + f" -> {path}")
    return 0 if out["passed"] else 1


# ---- package ----


def package(a):
    name_run = os.path.basename(a.run.rstrip("/"))
    gate_path = os.path.join(RES, f"{name_run}_gate.json")
    gres = json.load(open(gate_path)) if os.path.exists(gate_path) else None
    if not a.force and not (gres and gres["passed"]):
        sys.exit(f"the gate has not passed for {a.run} ({gate_path}); run `gate` first, or --force")
    r = recipe_of(a.run, a.recipe)
    snap = snapshot_path(a.run, a.snapshot)
    stem = f"{a.name}-{a.version}"
    dist = os.path.join(ROOT, "dist", "models")
    work = os.path.join(dist, stem)
    shutil.rmtree(work, ignore_errors=True)
    mdir = os.path.join(work, a.name)
    os.makedirs(mdir)
    # the fp16 weights (ninemax_export.py's format), the config, Kestrel's flat weights and its test vectors
    import torch

    st = torch.load(snap, map_location="cpu", weights_only=True)
    avg = {k: (v.half() if v.is_floating_point() else v) for k, v in st["avg"].items()}
    torch.save({"avg": avg, "hands": st["hands"]}, os.path.join(mdir, "avg_fp16.pt"))
    shutil.copy(os.path.join(a.run, "config.json"), os.path.join(mdir, "config.json"))
    vec = os.path.join(mdir, f"vectors_{a.name}.json")
    run([PY, "scripts/advisor_vectors.py", os.path.join(mdir, "avg_fp16.pt"), "--out", mdir, "--vectors-out", vec],
        check=True)  # fmt: skip
    cfg = json.load(open(os.path.join(a.run, "config.json")))
    manifest = {
        "format": 1, "name": a.name, "version": str(a.version), "game": r["game"],
        "players": cfg.get("players", 9), "about": r.get("about", ""),
        "trained": {"run": a.run, "snapshot": os.path.basename(snap), "hands": int(st["hands"]),
                    "recipe": r},
        "gates": gres, "created": time.strftime("%Y-%m-%d %H:%M"),
        "files": {f: sha256(os.path.join(mdir, f)) for f in sorted(os.listdir(mdir))},
    }  # fmt: skip
    json.dump(manifest, open(os.path.join(mdir, "model.json"), "w"), indent=1)
    tgz = os.path.join(dist, f"{stem}.tar.gz")
    with tarfile.open(tgz, "w:gz") as t:
        t.add(mdir, arcname=a.name)
    open(tgz + ".sha256", "w").write(f"{sha256(tgz)}  {os.path.basename(tgz)}\n")
    shutil.rmtree(work)
    print(f"{tgz} ({os.path.getsize(tgz) / 1e6:.1f} MB)")


# ---- publish ----


def publish(a):
    tgz = a.package
    with tarfile.open(tgz) as t:
        m = next(x for x in t.getmembers() if x.name.endswith("/model.json"))
        manifest = json.load(t.extractfile(m))
    want = open(tgz + ".sha256").read().split()[0]
    if sha256(tgz) != want:
        sys.exit("the archive does not match its .sha256")
    tag = f"model-{manifest['name']}-v{manifest['version']}"
    gl = "\n".join(f"- {'pass' if x['passed'] else 'FAIL'}: {x['gate']}: {x['value']}"
                   for x in (manifest.get("gates") or {}).get("gates", []))  # fmt: skip
    notes = (f"{manifest['name']} {manifest['version']}: {manifest.get('about', '')}\n\n"
             f"Game: {manifest['game']}. Trained {manifest['trained']['hands'] / 1e9:.1f} billion hands "
             f"({manifest['trained']['run']}, {manifest['trained']['snapshot']}).\n\nGates:\n{gl}\n\n"
             f"Install: kestreld models install {os.path.basename(tgz)}  (checked against the .sha256)")  # fmt: skip
    run(["gh", "release", "create", tag, tgz, tgz + ".sha256", "--title", f"Model {manifest['name']} "
         f"{manifest['version']}", "--notes", notes], check=True, env=os.environ)  # fmt: skip
    mp = os.path.join(ROOT, "models", "manifest.json")
    man = json.load(open(mp))
    man.setdefault("releases", {})[tag] = {"name": manifest["name"], "version": manifest["version"],
                                            "game": manifest["game"], "asset": os.path.basename(tgz),
                                            "sha256": want, "gates_passed": (manifest.get("gates") or {}).get("passed")}  # fmt: skip
    json.dump(man, open(mp, "w"), indent=1, sort_keys=True)
    print(f"published {tag}; recorded in models/manifest.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("recipe")
    t.add_argument("--run", required=True)
    t.add_argument("--gpu", type=int, default=0)
    g = sub.add_parser("gate")
    g.add_argument("run")
    g.add_argument("--snapshot")
    g.add_argument("--recipe")
    g.add_argument("--gpu", type=int, default=0)
    g.add_argument("--kestrel", default=KESTREL)
    p = sub.add_parser("package")
    p.add_argument("run")
    p.add_argument("--name", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--snapshot")
    p.add_argument("--recipe")
    p.add_argument("--force", action="store_true")
    u = sub.add_parser("publish")
    u.add_argument("package")
    a = ap.parse_args()
    sys.exit({"train": train, "gate": gate, "package": package, "publish": publish}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()
