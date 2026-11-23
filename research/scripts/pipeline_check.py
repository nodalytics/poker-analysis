"""Check the pipeline's checks (B4 of docs/2026-11-23-review-and-plan.md), standard library only (CI runs it):
  - the exploit and exploiters checks play the run's own game: menu, bucket library and abstraction from the run's
    config.json, nm25's (mtt, libbp7, blueprint/out) when it says nothing; seqtest takes the library the same way;
  - validate() checks the values of the patterns and opponent-model checks, not only their field names;
  - the exploit check's seeds: one exploiter per seed (7, 8, ...), judged on the mean of their final readings, each
    reading, the mean and the SD in the result; without seeds, one exploiter and the result exactly as before.
No exploiter runs: Run.sh is replaced by a stand-in that writes the exploiter's log.

  python3 scripts/pipeline_check.py          (seconds; exits 1 on a failure)
"""

import copy
import json
import os
import sys
import tempfile

import pipeline

FAILS = []


def expect(ok, what):
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        FAILS.append(what)


def recipe(tmp, checks, name="nm98"):
    r = json.load(open(os.path.join(pipeline.HERE, "pipeline", "example.json")))
    r["name"], r["checks"] = name, checks
    path = os.path.join(tmp, f"{name}.json")
    json.dump(r, open(path, "w"))
    return path


def check_validate(tmp, flags):
    ex = json.load(open(os.path.join(pipeline.HERE, "pipeline", "example.json")))
    errs = pipeline.validate(recipe(tmp, ex["checks"]), flags)
    expect(not errs, f"the example recipe validates ({errs})")
    good = [{"id": "pat", "type": "patterns", "baseline": "nm25", "hands": 10000, "limit": 0.0, "mine": False},
            {"id": "om", "type": "opponent-model", "hands": 500, "seeds": [1, 2], "samples": 64},
            {"id": "ex", "type": "exploit", "minutes": 30, "limit": 1.0, "seeds": 4},
            {"id": "ex1", "type": "exploit", "minutes": 30, "limit": 1.0}]  # fmt: skip
    errs = pipeline.validate(recipe(tmp, good), flags)
    expect(not errs, f"good patterns, opponent-model and exploit seeds validate ({errs})")
    bad = [
        ("patterns", "baseline", "../nm25"),
        ("patterns", "hands", "lots"),
        ("patterns", "hands", 10),
        ("patterns", "limit", "zero"),
        ("patterns", "mine", "yes"),
        ("opponent-model", "hands", -5),
        ("opponent-model", "hands", 1.5),
        ("opponent-model", "seeds", "1,2"),
        ("opponent-model", "seeds", []),
        ("opponent-model", "seeds", [1, "x"]),
        ("opponent-model", "samples", 0),
        ("opponent-model", "samples", True),
        ("exploit", "seeds", 0),
        ("exploit", "seeds", 2.5),
        ("exploit", "seeds", 100),
        ("exploit", "seeds", True),
    ]
    base = {g["type"]: g for g in good[:3]}
    for typ, k, v in bad:
        c = dict(copy.deepcopy(base[typ]), **{k: v})
        errs = pipeline.validate(recipe(tmp, [c]), flags)
        expect(any(f".{k}:" in e for e in errs), f"{typ}.{k} = {v!r} is refused ({errs})")


class FakeRun(pipeline.Run):
    """A Run whose steps write an exploiter log with given readings instead of training an exploiter."""

    def __init__(self, d, name, readings):
        self.dir, self.name, self.state, self.r = d, name, {}, {}
        self.readings, self.cmds = readings, []

    def save(self, **kw):
        self.state.update(kw)

    def pick_gpu(self):
        return 0, "stand-in GPU"

    def sh(self, args, log, gpu=None, timeout_min=None):
        self.cmds.append(list(args))
        out = args[args.index("--out") + 1]
        seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 7
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "exploit.log"), "w") as f:
            for t, x in enumerate(self.readings[seed]):
                f.write(f"{10.0 * t:.1f} min {t} hands  learner vs 8 x target: {x:+.1f} bb/100 [-1.0, +1.0]\n")


def opt(cmd, k):
    return cmd[cmd.index(k) + 1] if k in cmd else None


def check_game_and_seeds(tmp):
    here = pipeline.HERE
    pipeline.HERE = tmp  # runs under tmp/blueprint
    try:
        os.makedirs(os.path.join(tmp, "blueprint", "nm97"))
        json.dump({"menu": "wide", "lib": "blueprint/libbp9.so", "abs": "blueprint/abs_x"},
                  open(os.path.join(tmp, "blueprint", "nm97", "config.json"), "w"))  # fmt: skip
        os.makedirs(os.path.join(tmp, "blueprint", "nm96"))  # no config.json: nm25's game
        readings = {7: [3.0, 0.4], 8: [1.0, 1.2], 9: [0.5, 1.4]}  # finals 0.4, 1.2, 1.4: mean 1.0, SD 0.529
        for name, game in (("nm97", ("wide", "blueprint/libbp9.so", "blueprint/abs_x")),
                           ("nm96", ("mtt", "blueprint/libbp7.so", "blueprint/out"))):  # fmt: skip
            d = os.path.join(tmp, "results", name)
            os.makedirs(d)
            # no seeds: one exploiter, seed 7 (ninemax_eval's default, not passed), the result as before
            r = FakeRun(d, name, readings)
            ok, text = r.check({"id": "ex", "type": "exploit", "limit": 0.5})
            cmd = r.cmds[-1]
            expect((opt(cmd, "--menu"), opt(cmd, "--lib"), opt(cmd, "--abs")) == game,
                   f"{name} exploit plays the run's game {game} (got {opt(cmd, '--menu')}, {opt(cmd, '--lib')}, "
                   f"{opt(cmd, '--abs')})")  # fmt: skip
            res = json.load(open(os.path.join(d, "ex.json")))
            expect(len(r.cmds) == 1 and "--seed" not in cmd and opt(cmd, "--out") == os.path.join(d, "ex_exploiter"),
                   f"{name} exploit without seeds runs one exploiter as before")  # fmt: skip
            expect(set(res) == {"final_bb100", "log"} and res["final_bb100"] == 0.4 and ok,
                   f"{name} exploit without seeds: the last reading, +0.4 <= 0.5, passes ({text})")  # fmt: skip
            # three seeds: 7, 8, 9, judged on the mean +1.0
            for limit, want in ((1.0, True), (0.9, False)):
                r = FakeRun(d, name, readings)
                ok, text = r.check({"id": f"ex3-{int(limit * 10)}", "type": "exploit", "limit": limit, "seeds": 3})
                seeds = [opt(c, "--seed") for c in r.cmds]
                outs = {opt(c, "--out") for c in r.cmds}
                res = json.load(open(os.path.join(d, f"ex3-{int(limit * 10)}.json")))
                expect(
                    seeds == ["7", "8", "9"] and len(outs) == 3, f"{name} exploit seeds 3 runs seeds 7, 8, 9 ({seeds})"
                )
                expect(all((opt(c, "--menu"), opt(c, "--lib"), opt(c, "--abs")) == game for c in r.cmds),
                       f"{name} exploit seeds play the run's game")  # fmt: skip
                mean, sd, runs = res.get("mean_bb100", 0.0), res.get("sd_bb100", 0.0), res.get("runs", [])
                expect(abs(mean - 1.0) < 1e-9 and abs(sd - 0.5291503) < 1e-6
                       and [x["final_bb100"] for x in runs] == [0.4, 1.2, 1.4] and ok == want,
                       f"{name} exploit seeds 3, limit {limit}: mean +1.0, SD 0.53, each run recorded, "
                       f"{'passes' if want else 'fails'} ({text})")  # fmt: skip
            # the exploiters battery plays the run's game too
            r = FakeRun(d, name, readings)
            r.check({"id": "bat", "type": "exploiters", "limit": 2.0, "profiles": [{"id": "copy"}]})
            cmd = r.cmds[-1]
            expect((opt(cmd, "--menu"), opt(cmd, "--lib"), opt(cmd, "--abs")) == game,
                   f"{name} exploiters battery plays the run's game")  # fmt: skip
        try:
            import seqtest
        except ImportError as e:  # numpy: absent on CI's bare Python
            print(f"skip  seqtest's library ({e})")
            return
        want = {"nm97": "blueprint/libbp9.so", "nm96": "blueprint/libbp7.so"}
        for name, lib in want.items():
            got = seqtest.run_lib(os.path.join(tmp, "blueprint", name))
            expect(got == os.path.join(seqtest.ROOT, lib), f"seqtest plays {name} with {lib} (got {got})")
    finally:
        pipeline.HERE = here


def main():
    flags = pipeline.trainer_flags()
    with tempfile.TemporaryDirectory() as tmp:
        check_validate(tmp, flags)
    with tempfile.TemporaryDirectory() as tmp:
        check_game_and_seeds(tmp)
    print("all pipeline checks pass" if not FAILS else f"FAIL: {len(FAILS)} checks")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
