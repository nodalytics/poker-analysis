"""Watch the 9-max networks learn and play: a small web page served on the lab (127.0.0.1 only; reach it through an
SSH tunnel). The page is web/index.html; this is its server. The training runs need no change: everything is
read from their run directories.

  1 Training (on by default)  charts of every run's progress.log: 1-vs-8 results against the rule players, against
                              its previous checkpoint, the exploiter's winnings (exploitability), speed, entropy.
  2 Replayer (on by default)  a background thread notices each new checkpoint.pt, plays sample 9-handed hands with
                              it (all cards dealt, every action, the showdown) and keeps them, so the hands of
                              every checkpoint since the watcher started can be replayed and compared.
  3 Real tables (from the page) hands from bot_sim tables (results/ninemax/table*/session*/table.log): the network
                              against the solver bot, the current bot, the professional clones and the personas.
  4 Live (from the page)      fresh hands from the newest checkpoint, dealt continuously at watching speed.

Usage (lab, from research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_watch.py [--port 8790] [--runs blueprint/nm3,...]
then on your machine: ssh -N -L 8790:127.0.0.1:8790 "$LAB" and open http://localhost:8790
"""

import argparse
import glob
import json
import os
import re
import shutil
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch
import torch.nn.functional as F

from ninemax_engine import BB, Dealer, Menu, Tables, strength
from icm import icm
from ninemax_selfplay import NineNet, context, masked, rule_probs, set_mode

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POS = ["SB", "BB", "UTG", "UTG+1", "MP", "LJ", "HJ", "CO", "BTN"]
RANKS, SUITS = "23456789TJQKA", "cdhs"
LOCK = threading.Lock()


def card(c):
    return RANKS[c // 4] + SUITS[c % 4]


# ---------------------------------------------------------------- 1: training curves


NUM = r"([-+]?\d+(?:\.\d+)?)"
FIELDS = {
    "tag": r"vs tag: " + NUM,
    "previous": r"vs previous checkpoint: " + NUM,
    "reg": r"vs reg: " + NUM,
    "maniac": r"vs maniac: " + NUM,
    "station": r"vs station: " + NUM,
    "random": r"vs random: " + NUM,
    "exploiter": r"exploiter vs 8 x net: " + NUM,
    "rate": NUM + r" hands/s",
    "entropy": r"entropy " + NUM,
    "pull": r"pull " + NUM,
    "loss": r"loss " + NUM,
    "kl": r" kl " + NUM,
}


def curves(run):
    """Every evaluation line of RUN/progress.log as points keyed by the hands played (monotonic across resumes)."""
    pts = []
    path = os.path.join(run, "progress.log")
    if not os.path.exists(path):
        return pts
    for line in open(path):
        m = re.match(r"\s*" + NUM + r" min (\d+) hands", line)
        if not m:
            continue
        p = {"minutes": float(m.group(1)), "hands": int(m.group(2))}
        for k, rx in FIELDS.items():
            f = re.search(rx, line)
            if f:
                p[k] = float(f.group(1))
        pts.append(p)
    return pts


def _last(path, rx):
    """The last match of rx in a text file, or None."""
    if not os.path.exists(path):
        return None
    found = None
    for line in open(path, errors="ignore"):
        m = re.search(rx, line)
        if m:
            found = m
    return found


def results():
    """The evaluation results kept in results/ninemax: search at play time (search*_*.json), the poker-sense
    checks (sense_*.json) and the ICM push/fold agreement (pushfold_check*.json)."""
    res = os.path.join(ROOT, "results", "ninemax")
    out = {"search": [], "sense": {}, "pushfold": [], "ladder": None, "prereg": [], "ratings": None}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "prereg", "*.result.json"))):  # pre-registered tests
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue
        out["prereg"].append({"name": os.path.basename(f)[: -len(".result.json")], "family": r.get("family", {}),
                              "decision": r.get("decision", ""), "deviations": r.get("deviations", [])})  # fmt: skip
    try:  # the Bayesian ratings (ratings_bayes.py): P(best) and the fit of the ladder
        out["ratings"] = json.load(open(os.path.join(res, "ratings_bayes.json")))
    except (OSError, ValueError):
        pass
    try:  # the version ladder (ninemax_ladder.py): every version against every other, one rating each
        out["ladder"] = json.load(open(os.path.join(res, "ladder.json")))
    except (OSError, ValueError):
        pass
    for f in sorted(glob.glob(os.path.join(res, "search*_*.json"))):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue
        out["search"].append({k: r.get(k) for k in ("run", "samples", "mode", "tau", "bb100", "ci95", "ess_median",
                                                     "searched_decisions", "seconds", "hands", "preflop", "tourney")})  # fmt: skip
    sense_files = glob.glob(os.path.join(res, "sense*_*.json")) + glob.glob(os.path.join(res, "postflop*_*.json"))
    for f in sorted(sense_files, key=os.path.getmtime):
        try:
            d = json.load(open(f))
            if os.path.basename(f).startswith("postflop"):  # after-flop checks: their own rows
                d = {f"{k} (after the flop)": v for k, v in d.items()}
            out["sense"].update(d)  # later files replace earlier readings of the same run
        except (OSError, ValueError):
            continue
    pf = {}
    for f in sorted(glob.glob(os.path.join(res, "pushfold_check*.json")), key=os.path.getmtime):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        for key, seats in d.items():
            run, table = key.split(" | ", 1)
            agree = sum(v["agree"] for v in seats.values()) / len(seats)
            bb = seats.get("bb_vs_sb", {})
            row = {
                "run": run,
                "table": table,
                "agree": round(agree, 3),
                "bb_calls": bb.get("in"),
                "bb_chart": bb.get("chart"),
            }
            pf[(run, table)] = row  # later files replace earlier readings of the same run and table
    out["pushfold"] = list(pf.values())
    out["search"].sort(key=lambda r: (r["run"] or "", r["samples"] or 0, r["mode"] or "", r["tau"] or 0))
    return out


def info(run):
    """What a run is and how it stands: its recipe (config.json), whether it is still training, its latest
    evaluation, the fresh exploiter test and the real-table result when they exist, and the kept snapshots."""
    name = os.path.basename(run)
    try:
        cfg = json.load(open(os.path.join(run, "config.json")))
    except (OSError, ValueError):
        cfg = {}
    if cfg.get("tourney"):
        d = cfg.get("depths") or "10,60"
        parts = d.split(";")
        uniq = list(dict.fromkeys(parts))
        share = lambda r: "" if len(uniq) == 1 else f" ({parts.count(r)} in {len(parts)} tables)"  # noqa: E731
        mode = "tournament (ICM), table averages " + " or ".join(r.replace(",", "-") + "bb" + share(r) for r in uniq)
    elif str(cfg.get("stacks", "100")) != "100":
        mode = "cash, stacks " + str(cfg["stacks"]).replace(",", "-") + "bb"
    else:
        mode = "cash, 100bb"
    recipe = ["NeuRD" if cfg.get("pg") == "neurd" else "policy gradient"]
    if cfg.get("pool_share"):
        recipe.append(f"population {cfg['pool_share']:.0%}")
    if cfg.get("exploit_share"):
        recipe.append(f"exploiters {cfg['exploit_share']:.0%}")
    if cfg.get("kl_start") is not None:
        recipe.append("ramped pull")
    if cfg.get("no_open_limp"):
        recipe.append("no open-limping")
    if cfg.get("ante"):
        recipe.append(f"big-blind ante {cfg['ante']:g}bb")
    if cfg.get("short"):
        recipe.append(f"{cfg['short']:.0%} of final tables 2-8 handed")
    if cfg.get("init"):
        src = os.path.basename(cfg["init"])  # run/checkpoint.pt or run/avg_4h.pt: the run; a copied file: its name
        recipe.append(
            "from "
            + (
                os.path.basename(os.path.dirname(cfg["init"]))
                if src == "checkpoint.pt" or src.startswith("avg_")
                else src[:-3]
            )
        )
    pts = curves(run)
    last = pts[-1] if pts else {}
    # training while anything in the run is still being written (a new run has no checkpoint for its first 30 min)
    files = [os.path.join(run, f) for f in ("checkpoint.pt", "progress.log", "stdout.log")]
    training = any(os.path.exists(f) and time.time() - os.path.getmtime(f) < 45 * 60 for f in files)
    res = os.path.join(ROOT, "results", "ninemax")
    ex = _last(os.path.join(res, f"exploit_{name}.stdout"), NUM + r" min .*learner vs 8 x target: " + NUM + r" bb/100")
    tb = _last(os.path.join(res, f"table2k_{name}.log"), r"nine#" + re.escape(name) + r"\s+" + NUM)
    snaps = [os.path.basename(f)[4:-3] for f in glob.glob(os.path.join(run, "avg_*h.pt"))]  # avg_4h.pt -> 4h
    return {
        "mode": mode,
        "recipe": ", ".join(recipe),
        "status": "training" if training else "finished",
        "hours": round(last.get("minutes", 0) / 60, 1),
        # a run that has not reached its first evaluation: minutes since it started (its config.json is written then)
        "running_minutes": round((time.time() - os.path.getmtime(os.path.join(run, "config.json"))) / 60)
        if training and not pts and os.path.exists(os.path.join(run, "config.json"))
        else None,
        "hands": last.get("hands", 0),
        "tag": last.get("tag"),
        "exploiter": last.get("exploiter"),
        "fresh_exploit": {"minutes": float(ex.group(1)), "bb100": float(ex.group(2))} if ex else None,
        "tables": float(tb.group(1)) if tb else None,
        "snapshots": sorted(snaps, key=lambda s: float(s[:-1])),
    }


# ---------------------------------------------------------------- 2 and 4: sample hands from a checkpoint


class Lab:
    """The engine, the dealer and the networks on the CPU, for sample hands."""

    def __init__(self, absdir, lib):
        torch.set_num_threads(2)
        self.lib, self.dealers, self.tmp = lib, {}, tempfile.mkdtemp(prefix="watch-")
        self.dealer = self.dealer_for(absdir)
        self.B = self.dealer.B
        self.menus = {}

    def dealer_for(self, absdir):
        """A dealer for one card abstraction, through its own copy of the bucket library (one loaded library holds
        one abstraction): every run is dealt through the buckets it was trained on."""
        if absdir not in self.dealers:
            copy = os.path.join(self.tmp, f"libbp_{len(self.dealers)}.so")
            shutil.copy(self.lib, copy)
            B = int(open(os.path.join(absdir, "meta.txt")).read().split()[1])
            d = Dealer(copy, absdir, 9, B, "cpu", seed=int(time.time()) % 100000)
            bk0, _, _ = d.raw(1 << 16)
            pre_s = d.pre_strength[torch.tensor(bk0[:, :, 0].ravel())]
            d.pre_q = (torch.quantile(pre_s, 0.88).item(), torch.quantile(pre_s, 0.65).item())
            self.dealers[absdir] = d
        return self.dealers[absdir]

    def load(self, run):
        cfg = json.load(open(os.path.join(run, "config.json")))
        self.dealer = self.dealer_for(cfg.get("abs", "blueprint/out"))  # under LOCK, until the hands are played
        self.B = self.dealer.B
        set_mode(  # under LOCK, until the hands are played
            cfg.get("stacks", "100"),
            cfg.get("tourney", False),
            cfg.get("depths"),
            not cfg.get("no_open_limp", False),
            cfg.get("ante", 0.0),
            cfg.get("short", 0.0),
            cfg.get("seats"),
            cfg.get("bounty", 0.0),
            cfg.get("texture", False),
        )
        menu = self.menus.setdefault(cfg.get("menu", "medium"), Menu(cfg.get("menu", "medium"), "cpu"))
        net = NineNet(
            cfg["nfeat"],
            cfg["actions"],
            cfg.get("buckets", self.B),
            arch=cfg.get("arch", "mlp"),
            belief=cfg.get("belief_weight", 0) > 0,
        )
        st = torch.load(os.path.join(run, "checkpoint.pt"), map_location="cpu", weights_only=True)
        net.load_state_dict(st["avg"])
        return net.eval(), menu, int(st["hands"])

    def hands(self, net, menu, n=48, keep=8, opp="self"):
        """Play n deals; keep the `keep` most eventful ones (plus one quiet one) as replayable hands."""
        from ninemax_selfplay import features

        P = 9
        bk, rk, cd = self.dealer.deal(n, cards=True)
        stacks, pay = context(n, P, "cpu")  # the run's mode: 100bb, mixed stacks, or tournament tables
        T = Tables(n, P, menu, "cpu", stacks=stacks)
        if pay is not None:
            T.pay, T.eq0 = pay.float(), icm(T.stack.float(), pay).float()
        # with a rule opponent: the network's seat at each table, a random seated one (short tables have empty seats)
        hero = torch.multinomial((T.stack > 0).float(), 1).squeeze(1)
        steps = [[] for _ in range(n)]
        while not bool(T.done.all()):
            g = (~T.done).nonzero().squeeze(1)
            i = T.actor[g]
            s = strength(self.dealer, bk[g], T.street[g]).gather(1, i[:, None]).squeeze(1)
            feat, legal = features(T, g, i)
            with torch.no_grad():
                lg, _ = net(feat, bk[g, i], T.street[g], s)
            probs = F.softmax(masked(lg, legal), 1)
            if opp != "self":
                rule = rule_probs(opp, T, g, i, legal, s, self.dealer, bk)
                probs = torch.where((i == hero[g])[:, None], probs, rule)
            a = torch.multinomial(probs, 1).squeeze(1)
            to_call = T.bet[g] - T.contrib[g, i]
            street = T.street[g].clone()
            _, targets = T.legal(g, i)
            avail = T.avail[g, i]
            pr = probs.tolist()
            T.step(g, a)
            for k in range(len(g)):
                t, seat, act = int(g[k]), int(i[k]), int(a[k])
                total = int(T.contrib[t, seat])
                if act == 0:
                    text = "folds"
                elif act == 1:
                    text = f"calls {int(to_call[k]) / BB:g}" if int(to_call[k]) else "checks"
                elif act == menu.A - 1:
                    text = f"all-in {total / BB:g}"
                else:
                    text = f"raises to {total / BB:g}"
                # the decision: every legal action with its probability (the network's, or the rule player's)
                opts = []
                for slot, p in enumerate(pr[k]):
                    if p < 0.005:
                        continue
                    if slot == 0:
                        lab = "fold"
                    elif slot == 1:
                        lab = f"call {int(to_call[k]) / BB:g}" if int(to_call[k]) else "check"
                    elif slot == menu.A - 1:
                        lab = f"all-in {int(avail[k]) / BB:g}"
                    else:
                        lab = f"raise to {float(targets[k, slot - 2]) / BB:g}"
                    opts.append([lab, round(p, 3), slot == act])
                net_seat = opp == "self" or seat == int(hero[t])
                steps[t].append({
                    "street": int(street[k]), "seat": seat, "text": text, "options": opts,
                    "by": "network" if net_seat else opp,
                    "contrib": [round(x / BB, 2) for x in T.contrib[t].tolist()],
                    "folded": T.folded[t].tolist(),
                })  # fmt: skip
        u = T.payoffs(rk) / BB
        names = lambda t: [("net" if opp == "self" or s == int(hero[t]) else opp) for s in range(P)]  # noqa: E731
        out = []
        for t in range(n):
            out.append({
                "seats": [{"name": nm, "pos": POS[s], "hole": [card(cd[t][2 * s]), card(cd[t][2 * s + 1])],
                           "stack": int(T.stack[t, s]) / BB} for s, nm in enumerate(names(t))],
                "pay": [round(x, 3) for x in pay[t].tolist()] if pay is not None else None,
                "equity": [round(x, 3) for x in T.eq0[t].tolist()] if pay is not None else None,
                "board": [card(c) for c in cd[t][2 * P :]],
                "steps": steps[t],
                "result": [round(x, 2) for x in u[t].tolist()],
                "showdown": bool(T.showdown[t]),
                "flop": any(st["street"] > 0 for st in steps[t]) or bool(T.showdown[t]),
            })  # fmt: skip
        busy = sorted(range(n), key=lambda t: -len(out[t]["steps"]))
        quiet = [t for t in range(n) if not out[t]["flop"]][:1]
        return [out[t] for t in sorted(set(busy[:keep] + quiet))]


class Recorder(threading.Thread):
    """2: on every new checkpoint of every run, play sample hands and store them under OUT/<run>/<hands>.json."""

    def __init__(self, lab, runs, out, every=60):
        super().__init__(daemon=True)
        self.lab, self.runs, self.out, self.every = lab, runs, out, every
        self.seen = {}

    def run(self):
        while True:
            for run in self.runs():
                ck = os.path.join(run, "checkpoint.pt")
                if not os.path.exists(ck):
                    continue
                mt = os.path.getmtime(ck)
                if self.seen.get(run) == mt:
                    continue
                try:
                    with LOCK:
                        net, menu, hands = self.lab.load(run)
                        sample = self.lab.hands(net, menu)
                    d = os.path.join(self.out, os.path.basename(run))
                    os.makedirs(d, exist_ok=True)
                    json.dump({"run": run, "hands_trained": hands, "time": time.strftime("%Y-%m-%d %H:%M"),
                               "hands": sample}, open(os.path.join(d, f"{hands:012d}.json"), "w"))  # fmt: skip
                    self.seen[run] = mt
                except Exception as e:  # noqa: BLE001  (a checkpoint being written; try again next round)
                    print(f"recorder {run}: {e}", flush=True)
            time.sleep(self.every)


# ---------------------------------------------------------------- 3: real tables (bot_sim logs)


ACT = re.compile(r"\s*\[(\w+)\s*\]\s+(\S+)\s+(\S\S \S\S)\s+(.+?)\s+\([\d.]+s\)$")
SBC, BBC = 1, 2  # bot_sim blinds in chips


def real_hands(path):
    """Hands of one bot_sim session (table.log) in the replayer's format; chips become big blinds."""
    hands, cur = [], None
    for line in open(path):
        if line.startswith("--- Hand"):
            names = re.findall(r"(\S+) \d+ \(net [-+]\d+\)", line)
            cur = {"seats": [{"name": nm, "pos": POS[k] if len(names) == 9 else str(k), "hole": None}
                             for k, nm in enumerate(names)],
                   "board": [], "steps": [], "result": [0] * len(names), "showdown": False}  # fmt: skip
            contrib = [0.0] * len(names)
            contrib[0], contrib[1] = SBC, BBC
            street_in = list(contrib)
            folded = [False] * len(names)
            street = 0
            hands.append(cur)
            continue
        if cur is None:
            continue
        m = ACT.match(line)
        if m:
            st = ["preflop", "flop", "turn", "river"].index(m.group(1))
            if st != street:
                street, street_in = st, list(contrib)
            seat = [s["name"] for s in cur["seats"]].index(m.group(2))
            cur["seats"][seat]["hole"] = m.group(3).split()
            text = m.group(4)
            if text.startswith("folds"):
                folded[seat] = True
            elif text.startswith("calls"):
                contrib[seat] += float(text.split()[1])
            elif text.startswith("raises to"):
                contrib[seat] = street_in[seat] + float(text.split()[2])
            shown = re.sub(r"(\d+(?:\.\d+)?)", lambda x: f"{float(x.group(1)) / BBC:g}", text)
            cur["steps"].append({"street": st, "seat": seat, "text": shown,
                                 "contrib": [c / BBC for c in contrib], "folded": list(folded)})  # fmt: skip
        elif line.strip().startswith("board"):
            parts = [p.strip() for p in line.strip()[6:].split("|")]
            cur["board"] = [] if parts[0] == "-" else parts[0].split()
            for chunk in re.findall(r"(\S+) (\S\S \S\S)", parts[1] if len(parts) > 1 else ""):
                names = [s["name"] for s in cur["seats"]]
                if chunk[0] in names:
                    cur["seats"][names.index(chunk[0])]["hole"] = chunk[1].split()
            for nm, v in re.findall(r"(\S+) ([-+]\d+)", parts[2] if len(parts) > 2 else ""):
                names = [s["name"] for s in cur["seats"]]
                if nm in names:
                    cur["result"][names.index(nm)] = int(v) / BBC
            cur["showdown"] = len(cur["board"]) == 5
    return hands


# ---------------------------------------------------------------- the server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--runs", default="", help="comma-separated run dirs (default: every blueprint/nm*)")
    ap.add_argument("--out", default="results/ninemax/watch")
    ap.add_argument("--abs", default="blueprint/out")
    ap.add_argument("--lib", default="blueprint/libbp5.so")
    args = ap.parse_args()
    if args.port == 8000:
        raise SystemExit("port 8000 is the another service tunnel")

    def runs():
        if args.runs:
            return [r for r in args.runs.split(",") if r]
        found = [d for d in glob.glob("blueprint/nm*") if os.path.exists(os.path.join(d, "progress.log"))]
        return sorted(found, key=lambda d: int(re.sub(r"\D", "", os.path.basename(d)) or 0))  # nm2 before nm10

    lab = Lab(args.abs, args.lib)
    Recorder(lab, runs, args.out).start()
    page = os.path.join(ROOT, "web", "index.html")  # read on every request: edits show on reload

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, obj, ctype="application/json"):
            body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self.send(open(page, "rb").read(), "text/html; charset=utf-8")
                if u.path == "/api/results":  # search, poker-sense and push/fold results
                    return self.send(results())
                if u.path == "/api/info":  # what each run is and where it stands
                    return self.send({os.path.basename(r): info(r) for r in runs()})
                if u.path == "/api/runs":
                    return self.send({os.path.basename(r): curves(r) for r in runs()})
                if u.path == "/api/snapshots":  # the recorded checkpoints of a run
                    d = os.path.join(args.out, os.path.basename(q["run"]))
                    return self.send(sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(d, "*.json"))))
                if u.path == "/api/snapshot":
                    d = os.path.join(args.out, os.path.basename(q["run"]), os.path.basename(q["id"]) + ".json")
                    return self.send(json.load(open(d)))
                if u.path == "/api/live":  # 4: fresh hands from the newest checkpoint
                    run = [r for r in runs() if os.path.basename(r) == q["run"]][0]
                    opp = q.get("opp", "self")
                    if opp not in ("self", "reg", "tag", "maniac", "station", "random"):
                        opp = "self"
                    with LOCK:
                        net, menu, hands = lab.load(run)
                        sample = lab.hands(net, menu, n=24, keep=4, opp=opp)
                    return self.send({"run": run, "hands_trained": hands, "hands": sample})
                if u.path == "/api/tables":  # 3: bot_sim result directories and their sessions
                    out = {}
                    for d in sorted(glob.glob("results/ninemax/table*/")):
                        n = len(glob.glob(os.path.join(d, "session*/table.log")))
                        if n:
                            out[os.path.basename(d.rstrip("/"))] = n
                    return self.send(out)
                if u.path == "/api/table":
                    d = os.path.basename(q["dir"])
                    k = int(q.get("session", 1))
                    return self.send({"hands": real_hands(f"results/ninemax/{d}/session{k:04d}/table.log")})
            except (KeyError, IndexError, FileNotFoundError, ValueError) as e:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(str(e).encode())
                return
            self.send_response(404)
            self.end_headers()

    print(f"watching on http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
