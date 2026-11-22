"""Reinforcement learning: poker agents that learn from the chips they win, trained at scale against many styles.

Usage: uv run python scripts/rl_agent.py OUTDIR ITERATIONS [--short] [--workers N] [--sessions-per-worker K]
       [--league] [--vs SEAT] [--rnr P --target SEAT] [--belief] [--boltzmann] [--kelly]
Tabular every-visit Monte Carlo control with epsilon-greedy exploration. State: street, hand strength (Monte Carlo
equity against the opponents with chips in the pot, 10 bands), facing a bet, the price of calling (4 bands),
position (last to act or not) and stack depth (3 bands). Actions: fold, check/call, bet half the pot, bet the pot,
all-in. An action's value is the mean big blinds the hand finally won after taking it.

Plain mode: three learning seats (rlbot#1..3) with monte, tina, tinaplus, bob, max and carl.
--league: every session draws six opponents afresh: random-style players (poker_selfplay.style_<seed>, a random point
in the personality space: raising and calling thresholds, limping, bluffing, stickiness, bet size), the named bots,
and frozen earlier copies of the agent (rlsnap#k, saved every 25 iterations). It also keeps an average policy, the
count of each action the learner chose greedily over all of training (neural fictitious self-play's average-policy
idea): self-play that averages converges where greedy self-play cycles (docs/2026-11-06-alphapoker.md).

--vs SEAT: an exploiter. One learning seat plays SEAT heads-up (positions alternate) and learns only to beat it. If
an exploiter trained on a large sample cannot beat a system, that system has no leak this learner can find; if it
can, the states where it gains show the leak (a cheap stand-in for a best response in a game too big to compute).

--rnr P --target SEAT: RL robust exploitation (the restricted Nash response with learners). Heads-up, the opponent
is SEAT (the player to exploit) with probability P, otherwise rlx#1, an adversary trained at the same time only to
beat the learner (values in qx.json). P = 1 learns pure exploitation, P = 0 learns not to be beaten.

Outputs in OUTDIR: q.json (values), avg.json (average policy), snapshots/, curve.json. Seats in poker_selfplay:
`rlbot` plays the greedy policy (RL_Q, default results/rl/q.json), `rlavg` samples the average policy (RL_AVG,
default avg.json next to RL_Q), `rlsnap#k` a saved snapshot.
"""

import glob
import json
import math
import os
import random
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACTIONS = ["fold", "passive", "half", "pot", "allin"]
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q_PATH = os.environ.get("RL_Q", os.path.join(HERE, "results", "rl", "q.json"))
AVG_PATH = os.environ.get("RL_AVG", os.path.join(os.path.dirname(Q_PATH), "avg.json"))
NAMED = ["monte", "tina", "tinaplus", "bob", "max", "carl", "nora"]
_Q, _AVG, _SNAPS, _QX = {}, {}, {}, {}
EPSILON = [0.0]
# combinations with ideas from other fields, chosen by RL_MODE (comma-separated) or the matching command-line flags:
#   belief     opponent modelling: the state also holds the opponents' type read from their play this session
#   boltzmann  statistical physics: explore by sampling actions from exp(value / temperature), the temperature
#              annealed from 2 big blinds down to 0.2 over training, instead of epsilon-greedy
#   kelly      information theory / finance: reward is 100 x the log growth of a 100 big blind bankroll, not chips
#              (concave: a loss hurts more than an equal win helps)
KELLY_BANKROLL_BB = 100
MODE = {m for m in os.environ.get("RL_MODE", "").split(",") if m}
BUFFER = []  # (hand, seat name, state key, action index, greedy?) while a session plays
_RNG = random.Random()


def _read(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {}


def load(qpath=None, avgpath=None, snapdir=None):
    _Q.clear()
    _Q.update(_read(qpath or Q_PATH))
    _QX.clear()
    _QX.update(_read(os.path.join(os.path.dirname(qpath or Q_PATH), "qx.json")))
    _AVG.clear()
    _AVG.update(_read(avgpath or AVG_PATH))
    _SNAPS.clear()
    for f in glob.glob(os.path.join(snapdir, "q_*.json")) if snapdir else []:
        _SNAPS[os.path.basename(f)[2:-5]] = _read(f)


def key(state, i, ctx):
    import poker_agents as pa

    board = [repr(c) for c in state.get_board_cards(0)]
    street = {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]
    live = [j for j, s in enumerate(state.statuses) if s]
    opp = sum(1 for j in live if j != i and (board or state.bets[j]))
    eq = pa.equity_vs_random(pa._cards(state.hole_cards[i]), pa._cards(state.get_board_cards(0)), max(1, opp), 400)
    to_call, pot = state.checking_or_calling_amount, state.total_pot_amount
    price = to_call / (pot + to_call) if to_call else 0.0
    price_b = 0 if not to_call else 1 if price < 0.2 else 2 if price < 0.34 else 3
    bb = max(getattr(state, "blinds_or_straddles", (1, 2)))
    eff = min(state.stacks[i] + state.bets[i], max(state.stacks[j] + state.bets[j] for j in live if j != i)) / bb
    depth = 0 if eff <= 10 else 1 if eff <= 25 else 2
    order = [j for j in live if j > 1] + [j for j in live if j <= 1] if not board else sorted(live)
    last = order[-1] == i
    k = f"{street}|{min(9, int(eq * 10))}|{int(bool(to_call))}|{price_b}|{int(last)}|{depth}"
    if "belief" in MODE:
        read = pa.opponent_read(ctx, i)
        if read is None:
            typ = "unknown"
        elif read["fold_to_bet"] < 0.35 and read["vpip"] > 0.4:
            typ = "station"
        elif read["fold_to_bet"] > 0.6:
            typ = "folder"
        elif read["aggression"] > 0.6:
            typ = "maniac"
        else:
            typ = "balanced"
        k += "|" + typ
    return k


def legal(state):
    ok = [state.can_fold() and state.checking_or_calling_amount > 0, True]
    r = state.can_complete_bet_or_raise_to()
    return ok + [r, r, r]


def reply(state, i, ctx, name):
    base, _, tag = name.partition("#")
    if not _Q and os.path.exists(Q_PATH):
        load()
    if not _Q and not os.path.exists(Q_PATH) and EPSILON[0] == 0:
        # evaluation with no policy file would play an empty table (fold everything) and look like a weak agent;
        # on 2026-11-07 that silently produced -27 bb/100 in all 36 environments
        raise RuntimeError(f"RL policy missing at {Q_PATH}; set RL_Q (e.g. ~/research/rl_league_short/q.json)")
    k = key(state, i, ctx)
    ok = legal(state)
    choices = [a for a in range(5) if ok[a]]
    if base == "rlx":  # robust-exploitation adversary: learns only to beat the learner
        q = _QX.get(k, {"v": [0.0] * 5})["v"]
        best = max(choices, key=lambda x: q[x])
        explore = _RNG.random() < EPSILON[0]
        a = _RNG.choice(choices) if explore else best
        BUFFER.append((ctx["hand"], name, k, a, not explore))
    elif base == "rlsnap":
        q = _SNAPS.get(tag, {}).get(k, {"v": [0.0] * 5})["v"]
        a = max(choices, key=lambda x: q[x])
    elif base == "rlavg" and k in _AVG and sum(_AVG[k][x] for x in choices) > 0:
        a = _RNG.choices(choices, weights=[_AVG[k][x] for x in choices])[0]
    else:
        q = _Q.get(k, {"v": [0.0] * 5})["v"]
        best = max(choices, key=lambda x: q[x])
        if base == "rlbot" and "boltzmann" in MODE and EPSILON[0] > 0:
            temp = max(0.2, 4 * EPSILON[0])  # epsilon 0.5 -> 2 bb, annealing to 0.2 bb
            mx = q[best]
            a = _RNG.choices(choices, weights=[math.exp((q[x] - mx) / temp) for x in choices])[0]
            explore = a != best
        else:
            explore = base == "rlbot" and _RNG.random() < EPSILON[0]
            a = _RNG.choice(choices) if explore else best
        if base == "rlbot":
            BUFFER.append((ctx["hand"], name, k, a, not explore))
    to_call, pot = state.checking_or_calling_amount, state.total_pot_amount
    if a == 0:
        return json.dumps({"action": "fold"})
    if a == 1:
        return json.dumps({"action": "call" if to_call else "check"})
    lo = state.min_completion_betting_or_raising_to_amount
    hi = state.max_completion_betting_or_raising_to_amount
    target = {2: 0.5, 3: 1.0}.get(a)
    amt = hi if target is None else state.bets[i] + to_call + int(target * (pot + to_call))
    return json.dumps({"action": "raise", "amount": int(max(lo, min(hi, amt)))})


def table(rng, league, vs=None, rnr=None):
    if rnr:  # robust exploitation: the target with probability p, else the co-trained adversary
        p, target = rnr
        seats = ["rlbot#1", target if rng.random() < p else "rlx#1"]
        rng.shuffle(seats)
        return seats
    if vs:  # exploiter: one learning seat heads-up against a fixed system, positions alternating
        seats = ["rlbot#1", vs]
        rng.shuffle(seats)
        return seats
    if not league:
        return ["rlbot#1", "monte", "rlbot#2", "tina", "tinaplus", "rlbot#3", "bob", "max", "carl"]
    import rl_agent as me

    opp, named = [], rng.sample(NAMED, len(NAMED))
    while len(opp) < 6:
        u = rng.random()
        if u < 0.45:
            opp.append(f"style_{rng.randrange(1_000_000)}")
        elif u < 0.8 and named:
            opp.append(named.pop())
        elif me._SNAPS:
            s = f"rlsnap#{rng.choice(sorted(me._SNAPS))}"
            opp.append(s if s not in opp else f"style_{rng.randrange(1_000_000)}")
        else:
            opp.append(f"style_{rng.randrange(1_000_000)}")
    seats = ["rlbot#1", "rlbot#2", "rlbot#3"] + opp
    rng.shuffle(seats)
    return seats


def play(args):
    """One worker: some sessions with the current values; returns (state, action, reward in bb, greedy?)."""
    qpath, avgpath, snapdir, eps, sessions, short, league, seed, vs, rnr = args
    import poker_selfplay as ps
    import rl_agent as me  # the module poker_selfplay calls, not this file run as __main__

    me.load(qpath, avgpath, snapdir)
    me.EPSILON[0] = eps
    me._RNG.seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    out = []
    tmp = tempfile.mkdtemp(prefix="rl-")
    try:
        for s in range(sessions):
            seats = table(rng, league, vs, rnr)
            me.BUFFER.clear()
            d = os.path.join(tmp, f"s{s}")
            ps.main(d, 24, seats, short_stack=(5, 20) if short else None)
            pay = {}
            for line in open(os.path.join(d, "hands.jsonl")):
                r = json.loads(line)
                if r.get("end"):
                    for name, p in zip(r["seats"], r["payoffs"]):
                        if "kelly" in me.MODE:  # log growth of a 100-big-blind bankroll, x100
                            pay[(r["hand"], name)] = 100 * math.log(max(1 + (p / 2) / KELLY_BANKROLL_BB, 1e-6))
                        else:
                            pay[(r["hand"], name)] = p / 2
            out += [(k_, a, pay.get((h, n), 0.0), g, n) for h, n, k_, a, g in me.BUFFER]
            shutil.rmtree(d, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def main(outdir, iterations, short, workers, per_worker, league, vs=None, rnr=None):
    os.makedirs(os.path.join(outdir, "snapshots"), exist_ok=True)
    qpath, avgpath = os.path.join(outdir, "q.json"), os.path.join(outdir, "avg.json")
    q, avg = _read(qpath), _read(avgpath)
    qxpath = os.path.join(outdir, "qx.json")
    qx = _read(qxpath)
    curve = _read(os.path.join(outdir, "curve.json")) or []
    for it in range(len(curve) + 1, iterations + 1):
        eps = max(0.05, 0.5 * (1 - it / iterations))
        json.dump(q, open(qpath, "w"))
        json.dump(avg, open(avgpath, "w"))
        snapdir = os.path.join(outdir, "snapshots") if league else None
        json.dump(qx, open(qxpath, "w"))
        jobs = [
            (qpath, avgpath, snapdir, eps, per_worker, short, league, it * 1000 + w, vs, rnr) for w in range(workers)
        ]
        with ProcessPoolExecutor(workers) as pool:
            batches = list(pool.map(play, jobs))
        n, tot = 0, 0.0
        for batch in batches:
            for k, a, r, greedy, who in batch:
                if who.startswith("rlx"):
                    e = qx.setdefault(k, {"v": [0.0] * 5, "n": [0] * 5})
                    e["n"][a] += 1
                    e["v"][a] += (r - e["v"][a]) / e["n"][a]
                    continue
                e = q.setdefault(k, {"v": [0.0] * 5, "n": [0] * 5})
                e["n"][a] += 1
                e["v"][a] += (r - e["v"][a]) / e["n"][a]
                if greedy:
                    avg.setdefault(k, [0] * 5)[a] += 1
                n += 1
                tot += r
        curve.append(
            {
                "iteration": it,
                "epsilon": round(eps, 3),
                "decisions": n,
                "mean_reward_bb": round(tot / max(n, 1), 3),
                "states": len(q),
            }
        )
        print(curve[-1], flush=True)
        json.dump(q, open(qpath, "w"))
        json.dump(avg, open(avgpath, "w"))
        json.dump(curve, open(os.path.join(outdir, "curve.json"), "w"), indent=1)
        json.dump(qx, open(qxpath, "w"))
        if league and it % 25 == 0:
            json.dump(q, open(os.path.join(outdir, "snapshots", f"q_{it}.json"), "w"))


if __name__ == "__main__":
    a = sys.argv[1:]
    opts = {"--workers": 16, "--sessions-per-worker": 4, "--vs": None, "--rnr": None, "--target": None}
    pos, k = [], 0
    while k < len(a):
        if a[k] in opts:
            opts[a[k]] = a[k + 1] if a[k] in ("--vs", "--rnr", "--target") else int(a[k + 1])
            k += 2
            continue
        if a[k] in ("--belief", "--boltzmann", "--kelly"):
            MODE.add(a[k][2:])
            os.environ["RL_MODE"] = ",".join(sorted(MODE))  # worker processes and their imports see it
        elif a[k] not in ("--short", "--league"):
            pos.append(a[k])
        k += 1
    main(
        pos[0],
        int(pos[1]),
        "--short" in a,
        opts["--workers"],
        opts["--sessions-per-worker"],
        "--league" in a,
        opts["--vs"],
        (float(opts["--rnr"]), opts["--target"]) if opts["--rnr"] is not None else None,
    )
