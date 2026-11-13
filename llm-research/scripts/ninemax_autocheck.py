"""Check every new snapshot of the tournament runs as it is saved, so the watch page's Results tab fills itself.

Every --every minutes, for each blueprint/nm*/ run whose config says "tourney", and each avg_<h>h.pt snapshot that has
no result yet, it runs the poker-sense checks (ninemax_sense.py -> results/ninemax/sense_<run>_<h>h.json) and the ICM
push/fold check (ninemax_pushfold_check.py -> pushfold_check_<run>_<h>h.json; against icm_pushfold_ante.json for
runs trained with a big-blind ante), one at a time, each in its own
process. A check that fails leaves a .failed marker so it is not retried every pass; delete the marker to retry.
The watch page reads both files on its next refresh (every minute).

Usage (lab, from llm-research; detached, one copy only):
  (setsid nohup env PYTHONPATH=scripts CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \\
      .venv-gpu/bin/python scripts/ninemax_autocheck.py > results/ninemax/autocheck.log 2>&1 &)
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "ninemax")
JOBS = (("sense", "ninemax_sense.py", "sense_{run}_{h}.json"),
        ("pushfold", "ninemax_pushfold_check.py", "pushfold_check_{run}_{h}.json"))  # fmt: skip


def pending():
    for run in sorted(glob.glob(os.path.join(ROOT, "blueprint", "nm*"))):
        try:
            if not json.load(open(os.path.join(run, "config.json"))).get("tourney"):
                continue
        except (OSError, ValueError):
            continue
        name = os.path.basename(run)
        ante = json.load(open(os.path.join(run, "config.json"))).get("ante", 0) > 0
        for snap in sorted(glob.glob(os.path.join(run, "avg_*h.pt")), key=os.path.getmtime):
            if time.time() - os.path.getmtime(snap) < 60:  # still being written
                continue
            h = os.path.basename(snap)[4:-3]
            for kind, script, pattern in JOBS:
                out = os.path.join(RES, pattern.format(run=name, h=h))
                if not os.path.exists(out) and not os.path.exists(out + ".failed"):
                    extra = (
                        ["--charts", os.path.join(RES, "icm_pushfold_ante.json")] if ante and kind == "pushfold" else []
                    )
                    yield kind, script, os.path.relpath(snap, ROOT), out, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=10, help="minutes between passes")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    while True:
        for kind, script, snap, out, extra in pending():
            t0 = time.time()
            print(f"{time.strftime('%F %T')} {kind} {snap}", flush=True)
            r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), snap, "--out", out, *extra],
                               cwd=ROOT, capture_output=True, text=True)  # fmt: skip
            log = out[:-5] + ".log"
            open(log, "w").write(r.stdout + r.stderr)
            if r.returncode != 0 or not os.path.exists(out):
                open(out + ".failed", "w").write(r.stderr[-4000:])
                print(f"  FAILED ({r.returncode}), see {log}", flush=True)
            else:
                print(f"  done in {time.time() - t0:.0f} s", flush=True)
        if a.once:
            return
        time.sleep(a.every * 60)


if __name__ == "__main__":
    main()
