"""The research pipeline: a recipe merged into main is trained, judged and reported on the lab, with no one logging in.

A recipe (research/pipeline/runs/<name>.json, format in research/pipeline/README.md) says what to train and how
the result is judged. It is the pre-registration: the merge commit that adds it is the record, and the checks cannot
change after the run starts. The lab never takes commands from GitHub; it pulls the public main branch:

  validate RECIPE...   check recipes (standard library only: runs in CI on every pull request and push)
  poll                 the lab's service: every --every seconds fetch main, and run each merged recipe that has no
                       state yet, one at a time (training, then every check, then the verdict and a results PR)
  run RECIPE           one recipe now (the same steps; for a manual rerun of a failed stage use --from STAGE)
  status               every recipe's state, as the dashboard shows it

Each run keeps results/pipeline/<name>/state.json (the dashboard's Pipeline panel reads it) and its logs and check
outputs beside it; the verdict goes to results/prereg/<name>.result.json. With a token in
~/.config/research-pipeline/token (a fine-grained token: this repo, contents and pull requests read/write), the lab
opens a pull request with the results; without one the results stay on the lab and the state says so.

Stop a run: touch results/pipeline/<name>/STOP on the lab (checked every 30 s; the training process is ended).

Usage (lab, from research):
  python3 scripts/pipeline.py validate pipeline/runs/*.json
  PIPELINE_REPO=~/pipeline/poker-analysis python3 scripts/pipeline.py poll
"""

import argparse
import base64
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # research/
REPO_SLUG = "nodalytics/poker-analysis"
PY = os.path.join(HERE, ".venv-gpu", "bin", "python")
MIN_FREE_GB = 25  # refuse to start a run with less disk than this (checkpoints and snapshots take ~5-10 GB)
MAX_MINUTES = 720
NAME = re.compile(r"^nm[0-9]{2,4}$")  # the dashboard shows blueprint/nm* runs
SAFE = re.compile(r"^[A-Za-z0-9_.,:;/+=-]*$")

# nm25's training command (as nm41-nm43 ran it); a recipe's train.args override or add to it
BASE_TRAIN = {
    "minutes": 360, "eval-minutes": 30, "batch": 16384, "pool-share": 0.5, "pool": "reg,reg,tag,maniac,station",
    "chunks": 4, "exploit-share": 0.25, "exploit-reset": 60, "entropy": 0.04, "pg": "neurd", "neurd-beta": 4,
    "kl": 0.3, "keep-hours": 2, "tourney": True, "no-open-limp": True, "menu": "mtt", "depths": "10,25", "ante": 1,
    "short": 0.5, "lib": "blueprint/libbp7.so", "abs": "blueprint/out",
}  # fmt: skip
# the fields each check type takes; consistency and opponent-model are reports (they always pass)
CHECKS = {
    "seqtest": {"against", "early", "pass", "max", "looks", "alpha", "margin"},
    "exploit": {"minutes", "limit"},
    "bluffers": {"early", "unit", "bluffers", "pass"},
    "opponents": {"against", "hands", "baseline", "tolerance"},
    "styles": {"count", "seed", "hands", "baseline", "worst_tolerance", "mean_tolerance", "human"},
    "consistency": {"reference", "deals"},
    "exploiters": {"profiles", "minutes", "limit"},
    "patterns": {"baseline", "hands", "limit", "mine"},
    "opponent-model": {"hands", "seeds", "samples"},
}
SEQ_PASS = {"ahead", "not_behind", "equivalent_or_ahead"}
BLUFFERS = {"jam-any", "jam-40", "raise-any"}
RULE_PLAYERS = {"reg", "tag", "maniac", "station", "random"}
HUMAN = os.path.join("results", "human_styles", "styles.json")
# exploiters that attack differently: a copy of the target (the standard reading), one that keeps trying odd moves,
# one that learns fast and commits, and one that starts from an older, different network and searches elsewhere
EXPLOITERS = [
    {"id": "copy", "lr": 3e-4, "entropy": 0.003},
    {"id": "explore", "lr": 3e-4, "entropy": 0.02},
    {"id": "fast", "lr": 1e-3, "entropy": 0.003},
    {"id": "outsider", "lr": 3e-4, "entropy": 0.003, "init": "nm17"},
]
EXPLOITER_KEYS = {"id", "lr", "entropy", "init", "minutes"}  # opponents fitted to real player types
OPPONENTS = {"versions", "generated", "seed", "archetypes", "human", "patterns"}


def auto_versions(init, keep=10):
    """Earlier networks that read init's inputs, newest first: nm17 onwards, with a checkpoint (stdlib only)."""
    ref = json.load(open(os.path.join(HERE, "blueprint", init, "config.json")))
    same = lambda c: all(c.get(k) == ref.get(k) for k in ("nfeat", "actions", "abs", "menu")) and all(  # noqa: E731
        bool(c.get(k)) == bool(ref.get(k)) for k in ("tourney", "texture", "bounty"))  # fmt: skip
    found = []
    for d in glob.glob(os.path.join(HERE, "blueprint", "nm*")):
        n = os.path.basename(d)
        if not NAME.match(n) or int(n[2:]) < 17 or n == init or not os.path.exists(os.path.join(d, "checkpoint.pt")):
            continue
        try:
            if same(json.load(open(os.path.join(d, "config.json")))):
                found.append(n)
        except (OSError, ValueError):
            pass
    return sorted(found, key=lambda n: -int(n[2:]))[:keep]


def seq_pass(verdict, bb100, rule):
    """Whether a seqtest reading (verdict difference / equivalent / inconclusive, and the last mean) meets a rule."""
    ahead = verdict == "difference" and bb100 > 0
    behind = verdict == "difference" and bb100 < 0
    return {
        "ahead": ahead,
        "equivalent_or_ahead": verdict == "equivalent" or ahead,
        "not_behind": not behind and not (verdict == "inconclusive" and bb100 < -0.25),
    }[rule]


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def trainer_flags():
    """The trainer's options, read from its source (no torch needed)."""
    src = open(os.path.join(HERE, "scripts", "ninemax_selfplay.py")).read()
    return set(re.findall(r'add_argument\(\s*"--([a-z0-9-]+)"', src))


def known_runs():
    """Run names already used: models, pre-registrations and other recipes."""
    names = {os.path.basename(d) for d in glob.glob(os.path.join(HERE, "models", "ninemax", "nm*"))}
    for f in glob.glob(os.path.join(HERE, "results", "prereg", "*.json")):
        names.update(re.findall(r"nm[0-9]{2,4}", os.path.basename(f)))
    return names


# ---------------------------------------------------------------------------------------------------- validation


def validate(path, flags=None):
    """The problems with one recipe ([] if none)."""
    flags = flags if flags is not None else trainer_flags()
    try:
        r = json.load(open(path))
    except (OSError, ValueError) as e:
        return [f"not readable JSON: {e}"]
    errs = []
    name = r.get("name", "")
    if not NAME.match(str(name)):
        errs.append(f"name {name!r}: must be nm followed by digits (the dashboard shows blueprint/nm*)")
    if os.path.basename(path) != f"{name}.json":
        errs.append(f"file must be named {name}.json")
    for k in ("question", "why"):
        if not isinstance(r.get(k), str) or len(r[k]) < 20:
            errs.append(f"{k}: a sentence or more is required")
    t = r.get("train", {})
    if not isinstance(t, dict):
        return errs + ["train must be an object"]
    init = t.get("init", "nm25")
    if not NAME.match(str(init)):
        errs.append(f"train.init {init!r}: a run name such as nm25")
    m = t.get("minutes", BASE_TRAIN["minutes"])
    if not isinstance(m, (int, float)) or not 10 <= m <= MAX_MINUTES:
        errs.append(f"train.minutes {m!r}: 10 to {MAX_MINUTES}")
    o = t.get("opponents")
    if o is not None:
        if not isinstance(o, dict) or set(o) - OPPONENTS:
            errs.append(f"train.opponents: an object with some of {sorted(OPPONENTS)}")
        else:
            v = o.get("versions", [])
            if not (v == "auto" or (isinstance(v, list) and all(NAME.match(str(x)) for x in v))):
                errs.append('train.opponents.versions: "auto" or a list of run names')
            if not isinstance(o.get("generated", 0), int) or not 0 <= o.get("generated", 0) <= 40:
                errs.append("train.opponents.generated: 0 to 40 generated styles")
            for k in ("archetypes", "human", "patterns"):
                if not isinstance(o.get(k, False), bool):
                    errs.append(f"train.opponents.{k}: true or false")
    if t.get("gpu", "any") not in ("any", "pro6000", "4090", "3090"):
        errs.append("train.gpu: any, pro6000, 4090 or 3090")
    for k, v in (t.get("args") or {}).items():
        if k not in flags:
            errs.append(f"train.args.{k}: not an option of ninemax_selfplay.py")
        elif k in ("init", "out") or k == "minutes":
            errs.append(f"train.args.{k}: set by the pipeline (use train.init / train.minutes)")
        elif isinstance(v, str) and (not SAFE.match(v) or ".." in v or v.startswith("/")):
            errs.append(f"train.args.{k}: {v!r} has characters or a path that are not allowed")
        elif not isinstance(v, (str, int, float, bool)):
            errs.append(f"train.args.{k}: a string, number or true/false")
    checks = r.get("checks")
    if not isinstance(checks, list) or not checks:
        errs.append("checks: at least one")
        checks = []
    ids = set()
    for i, c in enumerate(checks):
        where = f"checks[{i}]"
        if not isinstance(c, dict) or c.get("type") not in CHECKS:
            errs.append(f"{where}: type must be one of {sorted(CHECKS)}")
            continue
        cid = c.get("id", "")
        if not re.match(r"^[a-z0-9-]{2,30}$", str(cid)) or cid in ids:
            errs.append(f"{where}: a unique id (lower case, digits, -)")
        ids.add(cid)
        extra = set(c) - CHECKS[c["type"]] - {"type", "id", "why"}
        if extra:
            errs.append(f"{where}: unknown fields {sorted(extra)}")
        if c["type"] == "seqtest" and not NAME.match(str(c.get("against"))):
            errs.append(f"{where}.against: a run name such as nm25")
        if c["type"] == "seqtest" and c.get("pass") not in SEQ_PASS:
            errs.append(f"{where}.pass: one of {sorted(SEQ_PASS)}")
        if c["type"] == "seqtest" and not 2000 <= c.get("max", 40000) <= 200000:
            errs.append(f"{where}.max: 2,000 to 200,000 deals")
        if "early" in c and not (isinstance(c["early"], (int, float)) and 0 <= c["early"] <= 1):
            errs.append(f"{where}.early: 0 to 1")
        if c["type"] == "exploit":
            if not isinstance(c.get("limit"), (int, float)):
                errs.append(f"{where}.limit: the largest exploiter result allowed, bb/100")
            if not 5 <= c.get("minutes", 30) <= 120:
                errs.append(f"{where}.minutes: 5 to 120")
        if c["type"] == "bluffers":
            if not set(c.get("bluffers", [])) <= BLUFFERS or not c.get("bluffers"):
                errs.append(f"{where}.bluffers: some of {sorted(BLUFFERS)}")
            if c.get("unit", "chips") not in ("chips", "equity"):
                errs.append(f"{where}.unit: chips or equity")
            if c.get("pass", "all_lose") != "all_lose":
                errs.append(f"{where}.pass: all_lose")
        if c["type"] == "opponents":
            if not set(c.get("against", [])) <= RULE_PLAYERS or not c.get("against"):
                errs.append(f"{where}.against: some of {sorted(RULE_PLAYERS)}")
            if not NAME.match(str(c.get("baseline", "nm25"))):
                errs.append(f"{where}.baseline: a run name such as nm25")
            if not isinstance(c.get("tolerance"), (int, float)) or c["tolerance"] < 0:
                errs.append(f"{where}.tolerance: how far below the baseline it may fall against each, bb/100")
        if c["type"] == "exploiters":
            if not isinstance(c.get("limit"), (int, float)):
                errs.append(f"{where}.limit: the most any exploiter may win, bb/100")
            if not 5 <= c.get("minutes", 30) <= 120:
                errs.append(f"{where}.minutes: 5 to 120")
            for j, x in enumerate(c.get("profiles", EXPLOITERS)):
                if (
                    not isinstance(x, dict)
                    or set(x) - EXPLOITER_KEYS
                    or not re.match(r"^[a-z0-9-]{2,20}$", str(x.get("id")))
                ):
                    errs.append(f"{where}.profiles[{j}]: an id and some of {sorted(EXPLOITER_KEYS - {'id'})}")
                elif "init" in x and not NAME.match(str(x["init"])):
                    errs.append(f"{where}.profiles[{j}].init: a run name such as nm17")
                elif not 1e-5 <= x.get("lr", 3e-4) <= 1e-2 or not 0 <= x.get("entropy", 0.003) <= 0.1:
                    errs.append(f"{where}.profiles[{j}]: lr 1e-5 to 1e-2, entropy 0 to 0.1")
        if c["type"] == "consistency" and not NAME.match(str(c.get("reference", "nm25"))):
            errs.append(f"{where}.reference: a run name such as nm25")
        if c["type"] == "styles":
            if not isinstance(c.get("count", 12), int) or not 0 <= c.get("count", 12) <= 60:
                errs.append(f"{where}.count: 0 to 60 generated styles (the 5 archetypes are always played)")
            if not NAME.match(str(c.get("baseline", "nm25"))):
                errs.append(f"{where}.baseline: a run name such as nm25")
            for k in ("worst_tolerance", "mean_tolerance"):
                if not isinstance(c.get(k), (int, float)) or c[k] < 0:
                    errs.append(f"{where}.{k}: how far below the baseline it may fall, bb/100")
    if r.get("adopt", "all") != "all":
        errs.append("adopt: all (every check must pass)")
    return errs


def cmd_validate(a):
    flags, bad = trainer_flags(), 0
    seen = known_runs()
    for p in a.recipes:
        errs = validate(p, flags)
        name = os.path.basename(p)[:-5]
        if a.new and name in seen:
            errs.append(f"{name} is already used (a model or a pre-registration has that name)")
        print(f"{p}: {'ok' if not errs else 'INVALID'}")
        for e in errs:
            print(f"  - {e}")
        bad += bool(errs)
    sys.exit(1 if bad else 0)


# ---------------------------------------------------------------------------------------------------- running


class Stopped(Exception):
    pass


class Run:
    def __init__(self, recipe_path, commit=None):
        self.r = json.load(open(recipe_path))
        self.name = self.r["name"]
        self.dir = os.path.join(HERE, "results", "pipeline", self.name)
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, "state.json")
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {}
        self.state.setdefault("name", self.name)
        self.state.setdefault("question", self.r["question"])
        self.state.setdefault("checks", {})
        if commit:
            self.state["commit"] = commit

    def save(self, **kw):
        self.state.update(kw, updated=now())
        tmp = self.state_path + ".tmp"
        json.dump(self.state, open(tmp, "w"), indent=1)
        os.replace(tmp, self.state_path)

    def stopped(self):
        return os.path.exists(os.path.join(self.dir, "STOP"))

    def sh(self, args, log, gpu=None, timeout_min=None):
        """Run one step; raises Stopped when STOP appears, RuntimeError when it fails."""
        env = dict(os.environ, PYTHONPATH="scripts", OMP_NUM_THREADS="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with open(os.path.join(self.dir, log), "a") as out:
            out.write(f"\n# {now()} {' '.join(args)}\n")
            out.flush()
            p = subprocess.Popen(args, cwd=HERE, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            t0 = time.time()
            while p.poll() is None:
                time.sleep(30)
                if self.stopped() or (timeout_min and time.time() - t0 > timeout_min * 60):
                    p.terminate()
                    try:
                        p.wait(60)
                    except subprocess.TimeoutExpired:
                        p.kill()
                    raise Stopped("stopped" if self.stopped() else f"over its {timeout_min} minutes")
        if p.returncode:
            raise RuntimeError(f"{args[1]} exited {p.returncode} (see {log})")

    def pick_gpu(self):
        want = {"pro6000": "PRO 6000", "4090": "4090", "3090": "3090"}.get(self.r["train"].get("gpu", "any"))
        while True:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout  # fmt: skip
            free = []
            for line in q.strip().splitlines():
                i, nm, mem, util = [x.strip() for x in line.split(",")]
                if int(mem) < 2000 and int(util) < 10 and (want is None or want in nm):
                    free.append((0 if "PRO 6000" in nm else 1 if "4090" in nm else 2, int(i), nm))
            if free:
                return min(free)[1:]
            self.save(status="waiting", stage="waiting for a free GPU")
            if self.stopped():
                raise Stopped("stopped")
            time.sleep(120)

    def mine_patterns(self):
        """Re-mine the practice tables' hands for patterns that win against the bot (best effort: no Kestrel, no mining)."""
        tmp = os.path.join(self.dir, "practice_copy")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        try:
            for src, dst in (("/data/sessions", "sessions"), ("/data/kestrel.db", "kestrel.db")):
                subprocess.run(["docker", "cp", "-q", f"kestrel-kestrel-1:{src}", os.path.join(tmp, dst)], check=True,
                               capture_output=True, timeout=300)  # fmt: skip
            self.sh([sys.executable, "scripts/exploit_patterns.py", "mine", "--sessions", f"{tmp}/sessions", "--db",
                     f"{tmp}/kestrel.db", "--out", "results/exploit_patterns/mined.json"], "patterns.log", timeout_min=30)  # fmt: skip
        except (subprocess.SubprocessError, OSError, RuntimeError) as e:
            self.save(patterns_mined=f"not refreshed: {type(e).__name__}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def human_styles(self):
        """Opponents fitted to real player types (human_styles.py), built from the hand histories when missing."""
        if os.path.exists(os.path.join(HERE, HUMAN)):
            return
        R = os.path.join("results", "human_styles")
        os.makedirs(os.path.join(HERE, R), exist_ok=True)
        dirs = sorted(glob.glob(os.path.expanduser("~/datasets/phh-dataset/data/handhq/*NLH*")))
        if not dirs:
            raise RuntimeError("no hand histories in ~/datasets/phh-dataset/data/handhq for the human styles")
        self.save(status="preparing", stage="human styles: reading the hand histories")
        self.sh([sys.executable, "scripts/human_styles.py", "players", *dirs, "--out", f"{R}/players.json"],
                "human_styles.log", timeout_min=240)  # fmt: skip
        self.sh([PY, "scripts/human_styles.py", "types", "--players", f"{R}/players.json", "--out", f"{R}/types.json"],
                "human_styles.log", timeout_min=30)  # fmt: skip
        gpu, gname = self.pick_gpu()
        self.save(status="preparing", stage=f"human styles: fitting a style to each player type on the {gname}")
        self.sh([PY, "scripts/human_styles.py", "fit", "--types", f"{R}/types.json", "--out", HUMAN],
                "human_styles.log", gpu, timeout_min=360)  # fmt: skip

    def train(self):
        t = self.r["train"]
        args = dict(BASE_TRAIN, minutes=t.get("minutes", BASE_TRAIN["minutes"]))
        for k, v in (t.get("args") or {}).items():
            args[k] = v
        o = t.get("opponents") or {}
        styles = []
        if o.get("generated"):
            styles.append(f"generated:{o['generated']}:{o.get('seed', 2026)}")
        if o.get("archetypes"):
            styles.append("archetypes")
        if o.get("human"):
            self.human_styles()
            styles.append(f"human:{HUMAN}")
        if styles:
            args["pool-styles"] = ",".join(styles)
        versions = auto_versions(t.get("init", "nm25")) if o.get("versions") == "auto" else o.get("versions", [])
        if versions:
            args["pool-nets"] = ",".join(f"blueprint/{v}" for v in versions)
        if o.get("patterns"):  # what people found: the library and, freshly mined, the practice tables
            self.mine_patterns()
            args["pool-patterns"] = "all"
        if o:
            self.save(opponents={"styles": styles, "versions": versions})
        out = os.path.join("blueprint", self.name)
        if os.path.exists(os.path.join(HERE, out, "checkpoint.pt")) and self.state.get("trained"):
            return
        if os.path.exists(os.path.join(HERE, out)) and not self.state.get("training_started"):
            raise RuntimeError(f"{out} already exists and is not this pipeline's run")
        cmd = [PY, "scripts/ninemax_selfplay.py", out, "--init", f"blueprint/{t.get('init', 'nm25')}/checkpoint.pt"]
        for k, v in args.items():
            if v is True:
                cmd.append(f"--{k}")
            elif v is not False and v is not None:
                cmd += [f"--{k}", str(v)]
        gpu, gname = self.pick_gpu()
        self.save(status="training", stage=f"training on the {gname}", gpu=gname, training_started=now())
        self.sh(cmd, "train.log", gpu, timeout_min=args["minutes"] + 60)
        self.save(trained=now())

    def check(self, c):
        cid, typ, run = c["id"], c["type"], f"blueprint/{self.name}"
        gpu, gname = self.pick_gpu()
        self.save(status="checking", stage=f"check {cid} ({typ}) on the {gname}")
        out = os.path.join(self.dir, f"{cid}.json")
        if typ == "seqtest":
            cmd = [PY, "scripts/seqtest.py", "match", run, f"blueprint/{c['against']}", "--max", str(c.get("max", 40000)),
                   "--looks", str(c.get("looks", 8)), "--alpha", str(c.get("alpha", 0.025)),
                   "--margin", str(c.get("margin", 0.5)), "--out", out]  # fmt: skip
            if "early" in c:
                cmd += ["--early", str(c["early"])]
            self.sh(cmd, f"{cid}.log", gpu, timeout_min=240)
            res = json.load(open(out))
            last = res["looks"][-1]["bb100"]
            v, p = res["verdict"], c["pass"]
            if v in ("no difference at the end", "inconclusive"):  # seqtest's word for "ran out of deals"
                v = "inconclusive"
            ok = seq_pass(v, last, p)
            return ok, f"{v} after {res['deals']:,} deals: {last:+.2f} bb/100 (needs {p.replace('_', ' ')})"
        if typ == "exploit":
            d = os.path.join(self.dir, f"{cid}_exploiter")
            cmd = [PY, "scripts/ninemax_eval.py", "exploit", run, "--out", d, "--minutes", str(c.get("minutes", 30)),
                   "--eval-minutes", "10", "--menu", "mtt", "--lib", "blueprint/libbp7.so"]  # fmt: skip
            self.sh(cmd, f"{cid}.log", gpu, timeout_min=c.get("minutes", 30) + 30)
            lines = [x for x in open(os.path.join(d, "exploit.log")) if "learner vs 8 x target" in x]
            m = float(re.search(r"target: ([+-][0-9.]+)", lines[-1]).group(1))
            json.dump({"final_bb100": m, "log": lines}, open(out, "w"), indent=1)
            return m <= c["limit"], f"exploiter wins {m:+.1f} bb/100 (limit {c['limit']:+.1f})"
        if typ == "bluffers":
            cmd = [PY, "scripts/ninemax_hybrid.py", "--post", run, "--early", str(c.get("early", 0)), "--out", out]
            self.sh(cmd, f"{cid}.log", gpu, timeout_min=120)
            res = json.load(open(out))
            table = res["bluffers_chips" if c.get("unit", "chips") == "chips" else "bluffers"]
            got = {b: table[f"{b} vs 8 x {self.name}"] for b in c["bluffers"]}
            ok = all(v[2] < 0 for v in got.values())
            return ok, ", ".join(f"{b} {v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]" for b, v in got.items())
        if typ == "opponents":  # the network against tables of rule players that do not play like it
            base = c.get("baseline", "nm25")
            cmd = [PY, "scripts/pipeline_opponents.py", run, "--baseline", f"blueprint/{base}",
                   "--against", ",".join(c["against"]), "--hands", str(c.get("hands", 20000)), "--out", out]  # fmt: skip
            self.sh(cmd, f"{cid}.log", gpu, timeout_min=180)
            res = json.load(open(out))
            ok = all(v["difference"] >= -c["tolerance"] for v in res.values())
            return ok, ", ".join(
                f"vs {k} {v['run'][0]:+.1f} ({base} {v['baseline'][0]:+.1f})" for k, v in res.items()
            ) + (f"; each within {c['tolerance']:.1f} of {base}")
        if typ == "opponent-model":  # Kestrel itself with the run: none, search, search with the opponent model
            kdir = os.path.join(self.dir, "kestrel_model")
            if not os.path.exists(os.path.join(kdir, "net.json")):
                self.save(status="checking", stage=f"check {cid}: exporting {self.name} for Kestrel")
                os.makedirs(kdir, exist_ok=True)
                self.sh([PY, "scripts/advisor_vectors.py", os.path.join(run, "checkpoint.pt"), "--out", kdir,
                         "--vectors-out", os.path.join(kdir, "vectors.json"), "--hands", "4", "--icm", "10"],
                        f"{cid}.log", timeout_min=30)  # fmt: skip
            self.save(status="checking", stage=f"check {cid}: Kestrel against the test platform's bots, three ways")
            cmd = [PY, "scripts/kestrel_opponent_check.py", "--model", kdir, "--hands", str(c.get("hands", 2000)),
                   "--seeds", ",".join(str(x) for x in c.get("seeds", [1, 2])), "--samples", str(c.get("samples", 256)),
                   "--python", PY, "--out", out]  # fmt: skip
            self.sh(cmd, f"{cid}.log", timeout_min=600)
            res = json.load(open(out))
            arms = ", ".join(f"{k} {v['bb100']:+.1f}" for k, v in res["arms"].items())
            gain = res.get("model_over_search", {})
            g = (
                f"; model over search {gain['bb100']:+.1f} [{gain['ci95'][0]:+.1f}, {gain['ci95'][1]:+.1f}]"
                if gain
                else ""
            )
            return True, f"report: {arms} bb/100{g}"
        if typ == "patterns":  # what people found works against the bot: each must win at most limit against the run
            if c.get("mine", True):
                self.mine_patterns()
            gpu, gname = self.pick_gpu()
            self.save(status="checking", stage=f"check {cid} (patterns people found) on the {gname}")
            base = c.get("baseline", "nm25")
            cmd = [PY, "scripts/exploit_patterns.py", "test", run, "--baseline", f"blueprint/{base}",
                   "--hands", str(c.get("hands", 10000)), "--out", out]  # fmt: skip
            self.sh(cmd, f"{cid}.log", gpu, timeout_min=240)
            res = json.load(open(out))
            lim = c.get("limit", 0.0)
            still = [k for k, v in res.items() if v["run"][0] > lim]
            text = ", ".join(f"{k} {v['run'][0]:+.2f} ({base} {v['baseline'][0]:+.2f})" for k, v in res.items())
            tail = f"; still winning: {', '.join(still)}" if still else ""
            return not still, f"{len(res)} patterns: {text}; each at most {lim:+.1f}{tail}"
        # a battery: different ways of attacking the run, each a lower bound on its exploitability
        if typ == "exploiters":
            got = {}
            for x in c.get("profiles", EXPLOITERS):
                gpu, gname = self.pick_gpu()
                self.save(status="checking", stage=f"check {cid}: exploiter {x['id']} on the {gname}")
                d = os.path.join(self.dir, f"{cid}_{x['id']}")
                mins = x.get("minutes", c.get("minutes", 30))
                cmd = [PY, "scripts/ninemax_eval.py", "exploit", run, "--out", d, "--minutes", str(mins),
                       "--eval-minutes", "10", "--menu", "mtt", "--lib", "blueprint/libbp7.so",
                       "--lr", str(x.get("lr", 3e-4)), "--entropy", str(x.get("entropy", 0.003))]  # fmt: skip
                if x.get("init"):
                    cmd += ["--init", f"blueprint/{x['init']}"]
                self.sh(cmd, f"{cid}.log", gpu, timeout_min=mins + 30)
                lines = [y for y in open(os.path.join(d, "exploit.log")) if "learner vs 8 x target" in y]
                got[x["id"]] = float(re.search(r"target: ([+-][0-9.]+)", lines[-1]).group(1))
            json.dump(got, open(out, "w"), indent=1)
            worst = max(got, key=got.get)
            return got[worst] <= c["limit"], (", ".join(f"{k} {v:+.1f}" for k, v in got.items())
                                              + f"; the best ({worst}) at most {c['limit']:+.1f}")  # fmt: skip
        if typ == "consistency":  # a report, never a gate: where the run decides differently from the reference
            ref = c.get("reference", "nm25")
            cmd = [PY, "scripts/ninemax_consistency.py", "--reference", f"blueprint/{ref}", "--runs", run,
                   "--deals", str(c.get("deals", 4000)), "--out", out]  # fmt: skip
            self.sh(cmd, f"{cid}.log", gpu, timeout_min=60)
            res = json.load(open(out))
            d = res["vs_reference"]["distance"][self.name]
            flips = res["vs_reference"]["different_top_action"][self.name]
            by = {g: max(rows, key=lambda r: r["vs_reference"].get(self.name, 0)) for g, rows in res["groups"].items()}
            most = ", ".join(f"{g} {r['group']} {r['vs_reference'][self.name]:.3f}" for g, r in by.items())
            return (
                True,
                f"report: distance from {ref} {d:.3f}, a different top action in {flips:.1%} of spots; most in {most}",
            )
        if typ == "styles":  # generated opponents: the baseline with its tendencies pushed, many times over
            base = c.get("baseline", "nm25")
            cmd = [PY, "scripts/pipeline_opponents.py", run, "--baseline", f"blueprint/{base}", "--no-rules",
                   "--styles", str(c.get("count", 12)), "--style-seed", str(c.get("seed", 2026)),
                   "--hands", str(c.get("hands", 5000)), "--out", out]  # fmt: skip
            if c.get("human", False):
                self.human_styles()
                cmd += ["--human", HUMAN]
            self.sh(cmd, f"{cid}.log", gpu, timeout_min=240)
            res = json.load(open(out))
            diffs = {k: v["difference"] for k, v in res.items()}
            worst = min(diffs, key=diffs.get)
            mean = sum(diffs.values()) / len(diffs)
            ok = diffs[worst] >= -c["worst_tolerance"] and mean >= -c["mean_tolerance"]
            return ok, (f"{len(diffs)} styles: mean {mean:+.2f} vs {base}, worst {worst} {diffs[worst]:+.2f} "
                        f"(within {c['worst_tolerance']:.1f} at worst and {c['mean_tolerance']:.1f} on average)")  # fmt: skip
        raise ValueError(typ)

    def go(self, start="train"):
        if shutil.disk_usage(HERE).free < MIN_FREE_GB * 2**30:
            self.save(status="failed", stage=f"refused: under {MIN_FREE_GB} GB of disk free")
            return
        self.save(status="started", started=self.state.get("started") or now())
        try:
            if start == "train":
                self.train()
            for c in self.r["checks"]:
                if c["id"] in self.state["checks"] and self.state["checks"][c["id"]].get("pass") is not None:
                    continue
                ok, text = self.check(c)
                self.state["checks"][c["id"]] = {"type": c["type"], "pass": bool(ok), "result": text, "at": now()}
                self.save()
            passed = all(v["pass"] for v in self.state["checks"].values())
            verdict = "adopted" if passed else "not adopted"
            self.save(status="done", stage="done", verdict=verdict, finished=now())
            self.write_result()
            self.report()
            self.refresh_consistency()
            self.save(status="done", stage="done")  # waiting for a GPU for the report must not read as unfinished
        except Stopped as e:
            self.save(status="stopped", stage=str(e))
        except Exception as e:  # noqa: BLE001  (any failure is recorded, and the poller goes on to the next recipe)
            self.save(status="failed", stage=f"{type(e).__name__}: {e}")

    def refresh_consistency(self):
        """After every run: every version against the same spots again, the new one included (a report; best effort)."""
        try:
            gpu, _ = self.pick_gpu()
            self.sh([PY, "scripts/ninemax_consistency.py", "--out", "results/ninemax/consistency.json"],
                    "consistency_all.log", gpu, timeout_min=60)  # fmt: skip
        except Exception as e:  # noqa: BLE001
            self.save(consistency=f"not refreshed: {type(e).__name__}")

    def write_result(self):
        res = {
            "name": self.name,
            "registered": f"research/pipeline/runs/{self.name}.json (merge commit {self.state.get('commit', '?')})",
            "question": self.r["question"],
            "train": self.r["train"],
            "checks": self.state["checks"],
            "verdict": self.state["verdict"],
            "ran": {"started": self.state.get("started"), "finished": self.state.get("finished"),
                    "gpu": self.state.get("gpu")},
        }  # fmt: skip
        json.dump(res, open(os.path.join(HERE, "results", "prereg", f"{self.name}.result.json"), "w"), indent=1)

    def report(self):
        """A pull request with the results (needs the token and the repo clone; otherwise recorded and skipped)."""
        tok_file = os.path.expanduser("~/.config/research-pipeline/token")
        repo = os.path.expanduser(os.environ.get("PIPELINE_REPO", "~/pipeline/poker-analysis"))
        if not os.path.exists(tok_file) or not os.path.isdir(os.path.join(repo, ".git")):
            self.save(report="no pull request: the lab has no token or repo clone yet; results are on the lab")
            return
        token = open(tok_file).read().strip()
        try:
            self._report(repo, token)
        except subprocess.CalledProcessError as e:  # the token is never in argv, but keep any error text out anyway
            self.save(report=f"no pull request: git {e.cmd[3] if len(e.cmd) > 3 else ''} failed (exit {e.returncode})")
        except Exception as e:  # noqa: BLE001
            self.save(report=f"no pull request: {type(e).__name__}: {str(e).replace(token, '***')[:200]}")

    def _report(self, repo, token):
        branch = f"pipeline/{self.name}-results"
        g = ["git", "-C", repo, "-c", "user.name=research-pipeline",
             "-c", "user.email=research-pipeline@users.noreply.github.com"]  # fmt: skip
        # the token reaches git as an HTTP header through the environment: never on a command line, in a URL, a log or
        # an exception's text (which the state file, the dashboard and the results pull request would show)
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        genv = dict(os.environ, GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
                    GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {basic}", GIT_TERMINAL_PROMPT="0")  # fmt: skip
        url = f"https://github.com/{REPO_SLUG}.git"
        run = lambda *a, **kw: subprocess.run(g + list(a), check=True, capture_output=True, **kw)  # noqa: E731
        run("fetch", "-q", url, "main", env=genv)
        run("checkout", "-q", "-B", branch, "FETCH_HEAD")
        dst = os.path.join(repo, "research", "results", "pipeline", self.name)
        os.makedirs(dst, exist_ok=True)
        for f in glob.glob(os.path.join(self.dir, "*.json")) + glob.glob(os.path.join(self.dir, "*.log")):
            if os.path.getsize(f) < 2_000_000:
                shutil.copy(f, dst)
        shutil.copy(os.path.join(HERE, "results", "prereg", f"{self.name}.result.json"),
                    os.path.join(repo, "research", "results", "prereg"))  # fmt: skip
        run("add", "-f", "research/results")
        run("commit", "-q", "-m", f"{self.name}: {self.state['verdict']} (pipeline results)")
        run("push", "-q", "-f", url, f"{branch}:{branch}", env=genv)
        lines = [f"**{self.name}: {self.state['verdict']}**", "", self.r["question"], "", "| Check | Result | Pass |",
                 "|---|---|---|"]  # fmt: skip
        lines += [
            f"| {k} | {v['result']} | {'yes' if v['pass'] else '**no**'} |" for k, v in self.state["checks"].items()
        ]
        lines += ["", f"Run file `research/pipeline/runs/{self.name}.json`, merge commit {self.state.get('commit')}."]
        if self.state["verdict"] == "adopted":
            lines += [
                "",
                "Adopted: package it for Kestrel with the model pipeline (pipeline/README.md) in a follow-up.",
            ]
        body = json.dumps({"title": f"{self.name}: {self.state['verdict']}", "head": branch, "base": "main",
                           "body": "\n".join(lines)}).encode()  # fmt: skip
        req = urllib.request.Request(f"https://api.github.com/repos/{REPO_SLUG}/pulls", body, method="POST",
                                     headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})  # fmt: skip
        try:
            pr = json.load(urllib.request.urlopen(req, timeout=60))
            self.save(report=pr.get("html_url", "pull request opened"))
        except urllib.error.HTTPError as e:
            self.save(report=f"pull request not opened: HTTP {e.code}")


def cmd_run(a):
    errs = validate(a.recipe)
    run = Run(a.recipe, commit=a.commit)
    if errs:
        run.save(status="failed", stage="invalid run file: " + "; ".join(errs))
        return
    run.go(a.start)


def cmd_poll(a):
    """Fetch main, sync research/ from it, run every merged recipe without a state; one at a time, forever."""
    repo = os.path.expanduser(os.environ.get("PIPELINE_REPO", "~/pipeline/poker-analysis"))
    if not os.path.isdir(os.path.join(repo, ".git")):
        subprocess.run(["git", "clone", "-q", f"https://github.com/{REPO_SLUG}.git", repo], check=True)
    log = open(os.path.join(HERE, "results", "pipeline", "poll.log"), "a", buffering=1)
    while True:
        try:
            subprocess.run(["git", "-C", repo, "fetch", "-q", "origin", "main"], check=True, timeout=300)
            sha = subprocess.run(["git", "-C", repo, "rev-parse", "origin/main"], capture_output=True, text=True,
                                 check=True).stdout.strip()  # fmt: skip
            names = subprocess.run(["git", "-C", repo, "ls-tree", "--name-only", "origin/main",
                                    "research/pipeline/runs/"], capture_output=True, text=True).stdout.split()  # fmt: skip
            todo = [os.path.basename(n)[:-5] for n in names if n.endswith(".json")]
            todo = [n for n in todo if not os.path.exists(os.path.join(HERE, "results", "pipeline", n, "state.json"))]
            if todo:
                # the code that runs is main's: research/ (tracked files) from the fetched commit, over the lab's copy
                subprocess.run(["git", "-C", repo, "checkout", "-q", "--detach", sha], check=True)
                # only code (scripts, the pipeline's files, the dashboard page): never results the lab has not pushed
                code = ["--include=/scripts/***", "--include=/pipeline/***", "--include=/web/***", "--exclude=*"]
                subprocess.run(["rsync", "-a", *code, os.path.join(repo, "research") + "/", HERE + "/"], check=True)
                name = todo[0]
                path = os.path.join(HERE, "pipeline", "runs", f"{name}.json")
                log.write(f"{now()} start {name} at {sha[:12]}\n")
                # a fresh process, so the run uses the code just merged, not the code this service started with
                subprocess.run([sys.executable, os.path.join(HERE, "scripts", "pipeline.py"), "run", path,
                                "--commit", sha[:12]], cwd=HERE)  # fmt: skip
                st = os.path.join(HERE, "results", "pipeline", name, "state.json")
                s = json.load(open(st)) if os.path.exists(st) else {}
                log.write(f"{now()} {name}: {s.get('status')} {s.get('verdict', '')}\n")
                continue
        except Exception as e:  # noqa: BLE001  (a network blip must not end the service)
            log.write(f"{now()} poll error: {type(e).__name__}: {e}\n")
        time.sleep(a.every)


def cmd_status(a):
    for f in sorted(glob.glob(os.path.join(HERE, "results", "pipeline", "*", "state.json"))):
        s = json.load(open(f))
        print(f"{s['name']:>6}  {s.get('status', '?'):9}  {s.get('stage', '')}  {s.get('verdict', '')}")
        for k, v in s.get("checks", {}).items():
            print(f"          {k}: {'pass' if v['pass'] else 'FAIL'}  {v['result']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("recipes", nargs="+")
    v.add_argument("--new", action="store_true", help="also require names not used before (for added recipes)")
    r = sub.add_parser("run")
    r.add_argument("recipe")
    r.add_argument("--from", dest="start", choices=("train", "checks"), default="train")
    r.add_argument("--commit", default=None, help="the merge commit it runs at (set by poll)")
    p = sub.add_parser("poll")
    p.add_argument("--every", type=int, default=120, help="seconds between fetches")
    sub.add_parser("status")
    a = ap.parse_args()
    {"validate": cmd_validate, "run": cmd_run, "poll": cmd_poll, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
