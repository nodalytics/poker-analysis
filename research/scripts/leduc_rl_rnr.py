"""RL with robust exploitation, and RL against robust exploitation, measured exactly on Leduc poker.

Usage: .venv-alphapoker/bin/python scripts/leduc_rl_rnr.py OUTDIR [EPISODES]   (open_spiel environment)

1. RL + robust exploitation. A Monte Carlo control learner plays, seat alternating, an opponent who is the known
   weak player with probability p and otherwise a second learner trained at the same time only to beat it (the
   reinforcement-learning version of the restricted Nash response). The learner's average policy (how often it chose
   each action greedily over training, fictitious-play style) is scored exactly: profit against the weak player and
   exploitability, next to the CFR frontier (leduc_rnr.py).
2. RL against robust exploitation. For each CFR robust-exploitation strategy (p = 0, 0.25, 0.5, 1), a learner trains
   against it; its greedy policy's exact value against the strategy is recorded as training grows, next to the most
   any opponent could win (the strategy's exploitability). How fast does an adaptive opponent find the leak?
3. Teaching (--teach-only): the learner first copies a teacher (CFR equilibrium, CFR robust exploitation, or the
   pure best response, or AlphaPoker's network distilled from the robust strategy) and weans off it, against the
   same p = 0.5 mixture; compared with learning from scratch.
4. AlphaPoker (--alphapoker): each CFR robust-exploitation strategy compressed into AlphaPoker's policy network and
   scored exactly, and an RL learner trained against the network.
Milli-chips per hand, averaged over seats where it applies.
"""

import json

import numpy as np
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leduc_rnr as L  # noqa: E402

GAME = L.GAME


class Learner:
    """Every-visit Monte Carlo control over information sets, epsilon-greedy, with an average policy."""

    def __init__(self, seed):
        self.q, self.n, self.avg = {}, {}, {}
        self.rng = random.Random(seed)

    def act(self, s, eps):
        key, legal = s.information_state_string(), s.legal_actions()
        q = self.q.get(key, {})
        best = max(legal, key=lambda a: q.get(a, 0.0))
        if self.rng.random() < eps:
            return self.rng.choice(legal), key, False
        return best, key, True

    def update(self, visits, ret):
        for key, a, greedy in visits:
            n = self.n.setdefault(key, {})
            q = self.q.setdefault(key, {})
            n[a] = n.get(a, 0) + 1
            q[a] = q.get(a, 0.0) + (ret - q.get(a, 0.0)) / n[a]
            if greedy:
                av = self.avg.setdefault(key, {})
                av[a] = av.get(a, 0) + 1

    def greedy_fn(self):
        def fn(s):
            q = self.q.get(s.information_state_string(), {})
            legal = s.legal_actions()
            return {max(legal, key=lambda a: q.get(a, 0.0)): 1.0}

        return fn

    def average_fn(self):
        def fn(s):
            av = self.avg.get(s.information_state_string())
            legal = s.legal_actions()
            if not av:
                return {a: 1 / len(legal) for a in legal}
            tot = sum(av.get(a, 0) for a in legal) or 1
            return {a: av.get(a, 0) / tot for a in legal}

        return fn


def alphapoker_net(fn, hidden=(128, 64), seed=0):
    """AlphaPoker's policy network (alphapoker.py): one MLP over the information-state tensor, trained on a strategy."""
    from sklearn.neural_network import MLPRegressor

    sets = {}

    def walk(s):
        if s.is_terminal():
            return
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                walk(s.child(a))
            return
        sets.setdefault(s.information_state_string(), s.clone())
        for a in s.legal_actions():
            walk(s.child(a))

    walk(GAME.new_initial_state())
    na = GAME.num_distinct_actions()
    X = np.array([s.information_state_tensor() for s in sets.values()])
    Y = np.array([[fn(s).get(a, 0.0) for a in range(na)] for s in sets.values()])
    net = MLPRegressor(hidden_layer_sizes=hidden, max_iter=3000, random_state=seed).fit(X, Y)

    def policy(s):
        pr = net.predict([s.information_state_tensor()])[0]
        legal = s.legal_actions()
        v = np.clip([pr[a] for a in legal], 1e-6, None)
        return dict(zip(legal, v / v.sum()))

    return policy


def alphapoker_jobs(args):
    """AlphaPoker versions of the three tests, for one p: the CFR robust-exploitation strategy compressed into
    AlphaPoker's policy network (one net per seat) and scored exactly; an RL learner trained against that network;
    and the network as the teacher of an RL learner."""
    opp, p, episodes, seed, gv = args
    weak = L.weak(opp)
    sols = [L.RNR(seat, weak, p) for seat in (0, 1)]
    for s_ in sols:
        s_.run(600)
    nets = {seat: alphapoker_net(sols[seat].policy(), seed=seed) for seat in (0, 1)}
    out = {"opponent": opp, "p": p}
    for name, pols in (("cfr", {st: sols[st].policy() for st in (0, 1)}), ("alphapoker_net", nets)):
        e = pf = 0.0
        for st in (0, 1):
            ee, pp = L.evaluate(st, pols[st], gv)
            e += ee / 2
            pf += pp[opp] / 2
        out[name] = {"profit": round(pf, 1), "exploitability": round(e, 1)}
    # an RL learner against the AlphaPoker network
    from open_spiel.python.algorithms import expected_game_score

    rng = random.Random(seed)
    learner = Learner(seed)
    curve = []
    marks = {int(episodes * f) for f in (0.01, 0.1, 0.3, 1.0)}
    for t in range(1, episodes + 1):
        seat = t % 2
        visits, ret = episode(
            {seat: learner, 1 - seat: nets[1 - seat]},
            None,
            {seat: max(0.05, 0.3 * (1 - t / episodes)), 1 - seat: 0},
            rng,
        )
        learner.update(visits[seat], ret[seat])
        if t in marks:
            won = 0.0
            for sl in (0, 1):
                lp, rp = L.tabular(learner.greedy_fn()), L.tabular(nets[1 - sl])
                v = expected_game_score.policy_value(GAME.new_initial_state(), [lp, rp] if sl == 0 else [rp, lp])[sl]
                won += 1000 * (v - gv[sl]) / 2
            curve.append({"episodes": t, "learner_wins_above_game_value": round(won, 1)})
    out["rl_against_alphapoker_net"] = curve
    return out


def teach(args):
    """3. Teaching: the learner copies a teacher's action with probability beta (falling from 1 to 0 over the first
    half of training, DAgger-style) and learns every action's value as it goes; it trains against the same
    p-mixture of weak player and co-trained adversary as rl_rnr. Teachers: 'nash' (CFR+ equilibrium, the robust
    strategy), 'rnr' (CFR robust exploitation at the same p), 'br' (pure best response to the weak player), 'none'."""
    opp, p, teacher, episodes, seed, gv = args
    rng = random.Random(seed)
    weak = L.weak(opp)
    tp = {"nash": 0.0, "rnr": p, "br": 1.0, "alphapoker": p}.get(teacher)
    tfn = None
    if tp is not None:
        sols = [L.RNR(seat, weak, tp) for seat in (0, 1)]
        for s_ in sols:
            s_.run(600)
        tfn = {seat: sols[seat].policy() for seat in (0, 1)}
        if teacher == "alphapoker":  # AlphaPoker's network, distilled from the robust strategy, is the teacher
            tfn = {seat: alphapoker_net(tfn[seat], seed=seed) for seat in (0, 1)}
    me, adv = Learner(seed), Learner(seed + 1)
    curve = []
    marks = {int(episodes * f) for f in (0.05, 0.25, 0.5, 1.0)}
    for t in range(1, episodes + 1):
        frac = t / episodes
        beta = max(0.0, 1 - 2 * frac) if tfn else 0.0
        eps = max(0.05, 0.3 * (1 - frac))
        seat = t % 2
        other = adv if rng.random() >= p else weak
        s = GAME.new_initial_state()
        mine, theirs = [], []
        while not s.is_terminal():
            if s.is_chance_node():
                outs = s.chance_outcomes()
                s.apply_action(rng.choices([a for a, _ in outs], weights=[x for _, x in outs])[0])
                continue
            pl = s.current_player()
            if pl == seat:
                if tfn and rng.random() < beta:
                    pr = tfn[seat](s)
                    a = rng.choices(list(pr), weights=list(pr.values()))[0]
                    mine.append((s.information_state_string(), a, True))  # imitation counts toward the policy
                else:
                    a, key, g = me.act(s, eps)
                    mine.append((key, a, g))
            elif other is adv:
                a, key, g = adv.act(s, eps)
                theirs.append((key, a, g))
            else:
                pr = other(s)
                a = rng.choices(list(pr), weights=list(pr.values()))[0]
            s.apply_action(a)
        ret = s.returns()
        me.update(mine, ret[seat])
        if other is adv:
            adv.update(theirs, ret[1 - seat])
        if t in marks:
            row = {"episodes": t}
            for name, fn in (("average", me.average_fn()), ("greedy", me.greedy_fn())):
                e = pf = 0.0
                for st in (0, 1):
                    ee, pp = L.evaluate(st, fn, gv)
                    e += ee / 2
                    pf += pp[opp] / 2
                row[name] = {"profit": round(pf, 1), "exploitability": round(e, 1)}
            curve.append(row)
    return {"opponent": opp, "p": p, "teacher": teacher, "curve": curve}


def episode(players, seat_of, eps, rng):
    """One hand. players: dict seat -> (learner or fixed fn); returns visits per learner seat and returns."""
    s = GAME.new_initial_state()
    visits = {0: [], 1: []}
    while not s.is_terminal():
        if s.is_chance_node():
            outs = s.chance_outcomes()
            s.apply_action(rng.choices([a for a, _ in outs], weights=[p for _, p in outs])[0])
            continue
        p = s.current_player()
        agent = players[p]
        if isinstance(agent, Learner):
            a, key, g = agent.act(s, eps[p])
            visits[p].append((key, a, g))
        else:
            probs = agent(s)
            a = rng.choices(list(probs), weights=list(probs.values()))[0]
        s.apply_action(a)
    return visits, s.returns()


def rl_rnr(args):
    opp, p, episodes, seed, gv = args
    rng = random.Random(seed)
    me, adv = Learner(seed), Learner(seed + 1)
    weak = L.weak(opp)
    for t in range(episodes):
        eps = max(0.05, 0.3 * (1 - t / episodes))
        seat = t % 2
        other = adv if rng.random() >= p else weak
        players = {seat: me, 1 - seat: other}
        visits, ret = episode(players, None, {seat: eps, 1 - seat: eps}, rng)
        me.update(visits[seat], ret[seat])
        if other is adv:
            adv.update(visits[1 - seat], ret[1 - seat])
    out = {"opponent": opp, "p": p, "episodes": episodes}
    for name, fn in (("average", me.average_fn()), ("greedy", me.greedy_fn())):
        expl, profit = 0.0, 0.0
        for seat in (0, 1):
            e, pr = L.evaluate(seat, fn, gv)
            expl += e / 2
            profit += pr[opp] / 2
        out[name] = {"profit": round(profit, 1), "exploitability": round(expl, 1)}
    return out


def rl_vs(args):
    p, opp, episodes, seed, gv = args
    sols = [L.RNR(seat, L.weak(opp), p) for seat in (0, 1)]
    for s_ in sols:
        s_.run(600)
    fixed = {seat: sols[seat].policy() for seat in (0, 1)}
    expl = sum(L.evaluate(seat, fixed[seat], gv)[0] for seat in (0, 1)) / 2
    rng = random.Random(seed)
    learner = Learner(seed)
    curve = []
    checkpoints = {int(episodes * f) for f in (0.01, 0.03, 0.1, 0.3, 1.0)}
    from open_spiel.python.algorithms import expected_game_score

    for t in range(1, episodes + 1):
        seat = t % 2  # learner's seat; the robust strategy plays the other seat
        players = {seat: learner, 1 - seat: fixed[1 - seat]}
        eps = max(0.05, 0.3 * (1 - t / episodes))
        visits, ret = episode(players, None, {seat: eps, 1 - seat: 0}, rng)
        learner.update(visits[seat], ret[seat])
        if t in checkpoints:
            won = 0.0
            for seat_l in (0, 1):
                lp = L.tabular(learner.greedy_fn())
                rp = L.tabular(fixed[1 - seat_l])
                pols = [lp, rp] if seat_l == 0 else [rp, lp]
                v = expected_game_score.policy_value(GAME.new_initial_state(), pols)[seat_l]
                won += 1000 * (v - gv[seat_l]) / 2
            curve.append({"episodes": t, "learner_wins_above_game_value": round(won, 1)})
    return {"rnr_p": p, "built_against": opp, "exploitability_max_winnable": round(expl, 1), "learner_curve": curve}


def main(outdir, episodes=300000):
    from open_spiel.python.algorithms import cfr, expected_game_score

    os.makedirs(outdir, exist_ok=True)
    nash = cfr.CFRPlusSolver(GAME)
    for _ in range(1000):
        nash.evaluate_and_update_policy()
    gv0 = expected_game_score.policy_value(GAME.new_initial_state(), [nash.average_policy()] * 2)[0]
    gv = [gv0, -gv0]
    a_jobs = [
        (o, p, episodes, 11 + k, gv)
        for k, (o, p) in enumerate((o, p) for o in ("maniac", "station") for p in (1.0, 0.75, 0.5, 0.25, 0.0))
    ]
    b_jobs = [(p, "maniac", episodes, 101 + k, gv) for k, p in enumerate((0.0, 0.25, 0.5, 1.0))]
    if "--alphapoker" in sys.argv:
        jobs = [
            (o, p, episodes, 301 + k, gv)
            for k, (o, p) in enumerate((o, p) for o in ("maniac", "station") for p in (0.0, 0.25, 0.5, 1.0))
        ]
        with ProcessPoolExecutor(8) as pool:
            res = list(pool.map(alphapoker_jobs, jobs))
        out = {"units": "milli-chips per hand", "episodes": episodes, "alphapoker": res}
        json.dump(out, open(os.path.join(outdir, "alphapoker.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))
        return
    if "--teach-only" in sys.argv:
        c_jobs = [
            (o, 0.5, t, episodes, 201 + k, gv)
            for k, (o, t) in enumerate(
                (o, t) for o in ("maniac", "station") for t in ("none", "nash", "rnr", "br", "alphapoker")
            )
        ]
        with ProcessPoolExecutor(8) as pool:
            c = list(pool.map(teach, c_jobs))
        out = {"units": "milli-chips per hand", "episodes": episodes, "p": 0.5, "teaching": c}
        json.dump(out, open(os.path.join(outdir, "teaching.json"), "w"), indent=1)
        print(json.dumps(out, indent=1))
        return
    with ProcessPoolExecutor(14) as pool:
        a = list(pool.map(rl_rnr, a_jobs))
        b = list(pool.map(rl_vs, b_jobs))
    rep = {
        "units": "milli-chips per hand",
        "episodes": episodes,
        "rl_robust_exploitation": a,
        "rl_against_robust_exploitation": b,
    }
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], *(int(x) for x in sys.argv[2:3] if x.isdigit()))
