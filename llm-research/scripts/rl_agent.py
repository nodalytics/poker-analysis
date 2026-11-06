"""Reinforcement learning: a poker agent that learns from the chips it wins, playing the bots in poker_selfplay.

Usage: uv run python scripts/rl_agent.py OUTDIR ITERATIONS [--short] [--workers N] [--sessions-per-worker K]
Tabular every-visit Monte Carlo control with epsilon-greedy exploration. State: street, hand strength (Monte Carlo
equity against the opponents with chips in the pot, 10 bands), facing a bet, the price of calling (4 bands),
position (last to act or not) and stack depth (3 bands). Actions: fold, check/call, bet half the pot, bet the pot,
all-in. The value of an action is the mean big blinds the hand finally won after taking it.

Training tables: three learning seats (rlbot#1..3, sharing one table of values) with monte, tina, tinaplus, bob, max
and carl, 24-hand sessions in parallel workers; values are merged after every iteration and written to
OUTDIR/q.json with the learning curve in OUTDIR/curve.json. The trained agent plays greedily as seat `rlbot` in
poker_selfplay (reads RL_Q, default results/rl/q.json), e.g. in bot_sim.py against solverbot.
"""

import json
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
_Q = {}
EPSILON = [0.0]
BUFFER = []  # (hand, seat name, state key, action index) while a session plays
_RNG = random.Random()


def load(path=None):
    _Q.clear()
    try:
        _Q.update(json.load(open(path or Q_PATH)))
    except OSError:
        pass


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
    return f"{street}|{min(9, int(eq * 10))}|{int(bool(to_call))}|{price_b}|{int(last)}|{depth}"


def legal(state):
    ok = [state.can_fold() and state.checking_or_calling_amount > 0, True]
    r = state.can_complete_bet_or_raise_to()
    return ok + [r, r, r]


def reply(state, i, ctx, name):
    if not _Q and os.path.exists(Q_PATH):
        load()
    k = key(state, i, ctx)
    ok = legal(state)
    q = _Q.get(k, {"v": [0.0] * 5})["v"]
    choices = [a for a in range(5) if ok[a]]
    a = _RNG.choice(choices) if _RNG.random() < EPSILON[0] else max(choices, key=lambda x: q[x])
    BUFFER.append((ctx["hand"], name, k, a))
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


def play(args):
    """One worker: some sessions with the current values; returns (state key, action, reward in big blinds)."""
    qpath, eps, sessions, short, seed = args
    import poker_selfplay as ps
    import rl_agent as me  # the module poker_selfplay calls, not this file run as __main__

    me.load(qpath)
    me.EPSILON[0] = eps
    me._RNG.seed(seed)
    random.seed(seed)
    out = []
    tmp = tempfile.mkdtemp(prefix="rl-")
    try:
        for s in range(sessions):
            seats = ["rlbot#1", "monte", "rlbot#2", "tina", "tinaplus", "rlbot#3", "bob", "max", "carl"]
            k = _RNG.randrange(len(seats))
            seats = seats[k:] + seats[:k]
            me.BUFFER.clear()
            d = os.path.join(tmp, f"s{s}")
            ps.main(d, 24, seats, short_stack=(5, 20) if short else None)
            pay = {}
            for line in open(os.path.join(d, "hands.jsonl")):
                r = json.loads(line)
                if r.get("end"):
                    for name, p in zip(r["seats"], r["payoffs"]):
                        pay[(r["hand"], name)] = p / 2
            out += [(k_, a, pay.get((h, n), 0.0)) for h, n, k_, a in me.BUFFER]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def main(outdir, iterations, short, workers, per_worker):
    os.makedirs(outdir, exist_ok=True)
    qpath = os.path.join(outdir, "q.json")
    q = json.load(open(qpath)) if os.path.exists(qpath) else {}
    curve = []
    for it in range(1, iterations + 1):
        eps = max(0.05, 0.5 * (1 - it / iterations))
        json.dump(q, open(qpath, "w"))
        with ProcessPoolExecutor(workers) as pool:
            batches = list(pool.map(play, [(qpath, eps, per_worker, short, it * 1000 + w) for w in range(workers)]))
        n, tot = 0, 0.0
        for batch in batches:
            for k, a, r in batch:
                e = q.setdefault(k, {"v": [0.0] * 5, "n": [0] * 5})
                e["n"][a] += 1
                e["v"][a] += (r - e["v"][a]) / e["n"][a]
                n += 1
                tot += r
        curve.append(
            {"iteration": it, "epsilon": round(eps, 3), "decisions": n, "mean_reward_bb": round(tot / max(n, 1), 3)}
        )
        print(curve[-1], f"states {len(q)}", flush=True)
        json.dump(q, open(qpath, "w"))
        json.dump(curve, open(os.path.join(outdir, "curve.json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    opts = {"--workers": 16, "--sessions-per-worker": 4}
    pos, k = [], 0
    while k < len(a):
        if a[k] in opts:
            opts[a[k]] = int(a[k + 1])
            k += 2
            continue
        if a[k] != "--short":
            pos.append(a[k])
        k += 1
    main(pos[0], int(pos[1]), "--short" in a, opts["--workers"], opts["--sessions-per-worker"])
