"""Does Kestrel's opponent model help? Kestrel itself, with and without it, against the same named bots.

Kestrel tracks every opponent it meets (VPIP, PFR, 3-bet, fold to a c-bet, aggression, showdowns, kept per name in
its database) and, with the search-exploit solver, plays each one as its profile says: the search's opponents are
tilted towards folding, calling or raising by how far that player's numbers stand from the population's, shrunk by
how many hands back them. The model is a separate solver, off by default; this measures whether it earns its place.

Three arms, the same everything else: Kestrel (the image the lab runs, kestrel:local) plays the 9-max test platform
(kestrel/testplatform/ninemax_platform.py) whose bots keep their names and styles hand after hand (a calling
station, limpers, a tilter, a player who switches style, two allies who soft-play each other, odd sizers):
  none            the network alone (the baseline)
  search          search at play time, every opponent assumed to play like the network
  search-exploit  search with the opponent model
Each arm gets its own platform on its own port and its own fresh database, so profiles are built from scratch in the
session, as at a new site. Every arm plays the same seeds. The result is each arm's bb/100 with its interval and
tournament places, and what the model adds over search and over none.

Search is slow: Kestrel runs the network on the CPU for every rollout, about 40 s a hand at 64 samples on 4 CPUs
(the network alone: well under a second). 500 hands an arm and seed take about 3 hours with 8 CPUs each, and leave
intervals of 50-100 bb/100: enough to see a large effect of the model, not a small one.

Usage (lab, from research; stdlib, needs docker and the kestrel:local image):
  python3 scripts/kestrel_opponent_check.py --model ../models/... (a folder with Kestrel's net.bin and net.json) \
      --pack blueprint/pack --hands 2000 --out results/kestrel_opponents/nm25.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(HERE)
# Kestrel's source: beside research/ in a checkout, or the lab's deployment copy
KESTREL = next((d for d in (os.path.join(REPO, "kestrel"), os.path.expanduser("~/kestrel-server/repo/kestrel"))
                if os.path.exists(os.path.join(d, "testplatform", "ninemax_platform.py"))), os.path.join(REPO, "kestrel"))  # fmt: skip
PLATFORM = os.path.join(KESTREL, "testplatform", "ninemax_platform.py")
CONNECTOR = os.path.join(KESTREL, "connectors", "testplatform.yaml")
BOTS = "tag,station,limper,tilter,switcher,odd,ally,ally"


def mount(src, dst, ro=False):
    """A bind mount that fails when src is missing (docker -v would create it, owned by root)."""
    return ["--mount", f"type=bind,source={os.path.abspath(src)},target={dst}" + (",readonly" if ro else "")]


def arm(solver, seed, port, a, work):
    """One session: a platform on port, Kestrel against it with solver; returns the session summary."""
    d = os.path.join(work, f"{solver}-{seed}")
    os.makedirs(d)
    os.chmod(d, 0o777)  # the image runs as its own user
    yaml = open(CONNECTOR).read().replace("http://127.0.0.1:8793/", f"http://127.0.0.1:{port}/")
    assert f":{port}/" in yaml, "the connector's address changed: update kestrel_opponent_check.py"
    open(os.path.join(d, "connector.yaml"), "w").write(yaml)
    plat = subprocess.Popen([a.python, PLATFORM, "--port", str(port), "--seed", str(seed), "--bots", a.bots],
                            stdout=open(os.path.join(d, "platform.log"), "w"), stderr=subprocess.STDOUT)  # fmt: skip
    time.sleep(3)
    cmd = ["docker", "run", "--rm", "--network", "host", "--name", f"kestrel-oppcheck-{solver}-{seed}-{os.getpid()}",
           "--cpus", str(a.cpus), *mount(a.model, "/model", True), *mount(a.pack, "/pack", True), *mount(d, "/o"), a.image, "play", "/o/connector.yaml", "--model", "/model", "--pack", "/pack",
           "--hands", str(a.hands), "--tournaments", "100000", "--solver", solver, "--search-samples",
           str(a.samples), "--seed", str(seed), "--out", "/o/session", "--db", "/o/kestrel.db"]  # fmt: skip
    try:
        with open(os.path.join(d, "kestrel.log"), "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout_min * 60, check=True)
    finally:
        plat.terminate()
    return json.load(open(os.path.join(d, "session", "summary.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a Kestrel model folder (net.bin, net.json)")
    ap.add_argument("--pack", default=os.path.join(HERE, "blueprint", "pack"))
    ap.add_argument("--hands", type=int, default=500, help="hands per arm and seed")
    ap.add_argument("--seeds", default="1,2", help="comma-separated platform seeds, the same for every arm")
    ap.add_argument("--arms", default="none,search,search-exploit")
    ap.add_argument("--bots", default=BOTS)
    ap.add_argument("--samples", type=int, default=64, help="search samples a decision")
    ap.add_argument("--cpus", type=float, default=8, help="CPUs for each Kestrel container")
    ap.add_argument("--image", default="kestrel:local")
    ap.add_argument("--python", default=sys.executable, help="a Python with pokerkit for the platform")
    ap.add_argument("--port", type=int, default=8811, help="the first platform port (one per session)")
    ap.add_argument("--timeout-min", type=float, default=360)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not os.path.exists(os.path.join(a.model, "net.json")):
        raise SystemExit(f"{a.model} has no net.json: export the network for Kestrel first (advisor_vectors.py --out)")
    work = tempfile.mkdtemp(prefix="oppcheck-")
    seeds = [int(s) for s in a.seeds.split(",")]
    arms = a.arms.split(",")
    from concurrent.futures import ThreadPoolExecutor

    jobs = {}
    keep = os.path.splitext(a.out)[0] + "_sessions"  # every session's logs, kept even when one fails
    try:
        with ThreadPoolExecutor(len(arms) * len(seeds)) as ex:
            port = a.port
            for s in seeds:
                for solver in arms:
                    jobs[(solver, s)] = ex.submit(arm, solver, s, port, a, work)
                    port += 1
            got = {k: f.result() for k, f in jobs.items()}
    finally:
        os.makedirs(os.path.dirname(keep) or ".", exist_ok=True)
        shutil.copytree(work, keep, dirs_exist_ok=True)
        shutil.rmtree(work, ignore_errors=True)
        if os.path.exists(work):  # files Kestrel's container wrote as its own user: removed through the image
            subprocess.run(["docker", "run", "--rm", "--user", "0", "--entrypoint", "sh", *mount(work, "/x"), a.image,
                            "-c", "rm -rf /x/*"], capture_output=True)  # fmt: skip
            shutil.rmtree(work, ignore_errors=True)
    res = {"model": a.model, "hands": a.hands, "seeds": seeds, "bots": a.bots, "samples": a.samples, "arms": {}}
    for solver in arms:
        runs = [got[(solver, s)] for s in seeds]
        n = sum(r["hands"] for r in runs)
        bb = sum(r["bb100"] * r["hands"] for r in runs) / max(n, 1)
        # the interval from each session's own, combined as independent (hands-weighted)
        half = (sum(((r["ci95"][1] - r["ci95"][0]) / 2 * r["hands"]) ** 2 for r in runs) ** 0.5) / max(n, 1)
        places = [p for r in runs for p in (r.get("places") or [t.get("place") for t in r.get("tournaments", [])])]
        res["arms"][solver] = {"hands": n, "bb100": round(bb, 1), "ci95": [round(bb - half, 1), round(bb + half, 1)],
                               "half_width": round(half, 1), "places": places}  # fmt: skip
    A = res["arms"]
    if "search-exploit" in A:
        for other in ("search", "none"):
            if other in A:
                d = A["search-exploit"]["bb100"] - A[other]["bb100"]
                hw = (A["search-exploit"]["half_width"] ** 2 + A[other]["half_width"] ** 2) ** 0.5
                res[f"model_over_{other.replace('-', '_')}"] = {"bb100": round(d, 1), "ci95": [round(d - hw, 1),
                                                                                                round(d + hw, 1)]}  # fmt: skip
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for solver, v in A.items():
        print(
            f"{solver:15} {v['bb100']:+8.1f} bb/100 [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] over {v['hands']} hands"
        )
    for k in ("model_over_search", "model_over_none"):
        if k in res:
            print(f"{k.replace('_', ' ')}: {res[k]['bb100']:+.1f} [{res[k]['ci95'][0]:+.1f}, {res[k]['ci95'][1]:+.1f}]")


if __name__ == "__main__":
    main()
