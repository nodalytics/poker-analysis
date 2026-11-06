"""AlphaPoker: does the AlphaGo recipe (search + self-play + networks) work for poker? Measured exactly on Leduc.

Usage: uv run --extra alphapoker python scripts/alphapoker.py OUTDIR [--quick]
Leduc hold'em (6 cards, 2 betting rounds, a public card) is small enough to compute exact exploitability: how much
a best-responding opponent wins per hand against a strategy (0 = unbeatable). Reported in milli-chips per hand
(the ante is 1 chip). Every method is turned into a full strategy table and scored the same way (OpenSpiel).

Methods
  random                 uniform over legal actions (baseline)
  AlphaGo search (PIMC)  sample the hidden cards, run perfect-information MCTS in each sampled world, vote:
                         AlphaGo's search applied directly to a game with hidden information
  AlphaGo search (IS-MCTS) Monte Carlo tree search over information sets, the standard imperfect-info variant
  self-play RL (Q-learning) AlphaZero-style learning by self-play without search: two tabular Q-learners
  fictitious self-play   each player best-responds to the other's average strategy (XFP)
  CFR+                   counterfactual regret minimisation, the solver family behind Libratus and Pluribus
  MCCFR (sampled self-play) outcome-sampling CFR: learns from sampled self-play games like AlphaZero, but averages
  AlphaPoker v0          MCCFR self-play teaches a small policy network (one shared net, all information sets)
  AlphaPoker v0 + search the network as the prior for PUCT information-set search, as AlphaGo uses its policy net
"""

import json
import os
import sys
import time

import numpy as np
import pyspiel
from open_spiel.python import policy as osp
from open_spiel.python import rl_environment, rl_tools
from open_spiel.python.algorithms import (
    cfr,
    exploitability,
    fictitious_play,
    ismcts,
    mcts,
    outcome_sampling_mccfr,
    tabular_qlearner,
)

GAME = pyspiel.load_game("leduc_poker")
NA = GAME.num_distinct_actions()


def expl(pol):
    return round(1000 * exploitability.exploitability(GAME, pol), 1)


def infosets():
    """Every decision information set with one representative state."""
    out = {}

    def walk(s):
        if s.is_terminal():
            return
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                walk(s.child(a))
            return
        out.setdefault(s.information_state_string(), s.clone())
        for a in s.legal_actions():
            walk(s.child(a))

    walk(GAME.new_initial_state())
    return out


def table(fn):
    """A full strategy table from fn(state) -> {action: probability}."""
    tp = osp.TabularPolicy(GAME)
    for key, s in infosets().items():
        probs = fn(s)
        arr = tp.policy_for_key(key)
        arr[:] = 0
        for a, p in probs.items():
            arr[a] = p
        tot = arr.sum()
        legal = s.legal_actions()
        if tot <= 0:
            arr[legal] = 1 / len(legal)
        else:
            arr /= tot
    return tp


def uniform(s):
    return {a: 1 / len(s.legal_actions()) for a in s.legal_actions()}


def pimc(worlds, sims, seed=0):
    rng = np.random.RandomState(seed)
    bot = mcts.MCTSBot(GAME, 2.0, sims, mcts.RandomRolloutEvaluator(1, rng), random_state=rng)

    def fn(s):
        counts = np.zeros(NA)
        for _ in range(worlds):
            w = s.resample_from_infostate(s.current_player(), pyspiel.UniformProbabilitySampler(0.0, 1.0))
            root = bot.mcts_search(w)
            best = max(root.children, key=lambda c: c.explore_count)
            counts[best.action] += 1  # each world votes for its best move, as PIMC does
        return {a: counts[a] for a in s.legal_actions()}

    return fn


def is_mcts(sims, prior=None, seed=0):
    rng = np.random.RandomState(seed)
    ev = mcts.RandomRolloutEvaluator(2, rng)
    if prior is not None:
        ev = PriorEvaluator(ev, prior)
    bot = ismcts.ISMCTSBot(
        GAME,
        ev,
        2.0,
        sims,
        random_state=rng,
        final_policy_type=ismcts.ISMCTSFinalPolicyType.NORMALIZED_VISITED_COUNT,
        child_selection_policy=ismcts.ChildSelectionPolicy.PUCT
        if prior is not None
        else ismcts.ChildSelectionPolicy.UCT,
    )

    def fn(s):
        return dict(bot.run_search(s))

    return fn


class PriorEvaluator(mcts.Evaluator):
    def __init__(self, base, prior):
        self.base, self.prior_fn = base, prior

    def evaluate(self, state):
        return self.base.evaluate(state)

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        p = self.prior_fn(state)
        return [(a, p.get(a, 0.0)) for a in state.legal_actions()]


def qlearning(episodes, seed=0):
    env = rl_environment.Environment(GAME, seed=seed)
    agents = [
        tabular_qlearner.QLearner(p, NA, step_size=0.05, epsilon_schedule=rl_tools.LinearSchedule(0.3, 0.02, episodes))
        for p in range(2)
    ]
    for _ in range(episodes):
        ts = env.reset()
        while not ts.last():
            p = ts.observations["current_player"]
            ts = env.step([agents[p].step(ts).action])
        for ag in agents:
            ag.step(ts)

    def fn(s):
        q = agents[s.current_player()]._q_values[str(s.information_state_tensor(s.current_player()))]
        legal = s.legal_actions()
        best = max(legal, key=lambda a: q[a])
        return {best: 1.0}  # a Q-learner plays its greedy action

    return fn


def network(teacher, hidden=(64,), seed=0):
    """AlphaPoker v0's policy network: one MLP over the information-state tensor, trained on the teacher's strategy."""
    from sklearn.neural_network import MLPRegressor

    sets = infosets()
    X = np.array([s.information_state_tensor() for s in sets.values()])
    Y = np.array([[teacher.action_probabilities(s).get(a, 0.0) for a in range(NA)] for s in sets.values()])
    net = MLPRegressor(hidden, max_iter=3000, random_state=seed).fit(X, Y)

    def fn(s):
        p = net.predict([s.information_state_tensor()])[0]
        legal = s.legal_actions()
        v = np.clip([p[a] for a in legal], 1e-6, None)
        return dict(zip(legal, v / v.sum()))

    return fn, sum(w.size for w in net.coefs_) + sum(b.size for b in net.intercepts_)


def main(outdir, quick=False):
    os.makedirs(outdir, exist_ok=True)
    k = 0.1 if quick else 1.0
    rep = {
        "game": "leduc_poker",
        "units": "milli-chips per hand (ante 1 chip); 0 = unbeatable",
        "infosets": len(infosets()),
        "methods": {},
        "curves": {},
    }

    def record(name, fn_or_pol, **extra):
        t0 = time.time()
        pol = table(fn_or_pol) if callable(fn_or_pol) else fn_or_pol
        rep["methods"][name] = {
            "exploitability": expl(pol),
            "seconds": round(time.time() - t0 + extra.pop("train_s", 0), 1),
            **extra,
        }
        print(name, rep["methods"][name], flush=True)
        json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
        return pol

    record("random", uniform)
    record(
        "AlphaGo search (PIMC)",
        pimc(int(20 * k) or 4, int(200 * k) or 20),
        note=f"{int(20 * k) or 4} sampled worlds x {int(200 * k) or 20} simulations",
    )
    record("AlphaGo search (IS-MCTS)", is_mcts(int(4000 * k)), note=f"{int(4000 * k)} simulations per decision")
    for eps in [int(x * k) for x in (10_000, 100_000, 1_000_000)]:
        t0 = time.time()
        fn = qlearning(eps)
        rep["curves"].setdefault("self-play RL (Q-learning)", []).append(
            {"episodes": eps, "exploitability": expl(table(fn)), "seconds": round(time.time() - t0, 1)}
        )
    rep["methods"]["self-play RL (Q-learning)"] = rep["curves"]["self-play RL (Q-learning)"][-1]
    print("qlearning", rep["curves"]["self-play RL (Q-learning)"], flush=True)
    xfp = fictitious_play.XFPSolver(GAME)
    for it in range(1, int(300 * k) + 1):
        xfp.iteration()
        if it in (10, 30, 100, 300) or it == int(300 * k):
            rep["curves"].setdefault("fictitious self-play", []).append(
                {"iterations": it, "exploitability": expl(xfp.average_policy())}
            )
    rep["methods"]["fictitious self-play"] = rep["curves"]["fictitious self-play"][-1]
    cfrp = cfr.CFRPlusSolver(GAME)
    for it in range(1, int(1000 * k) + 1):
        cfrp.evaluate_and_update_policy()
        if it in (10, 30, 100, 300, 1000) or it == int(1000 * k):
            rep["curves"].setdefault("CFR+", []).append(
                {"iterations": it, "exploitability": expl(cfrp.average_policy())}
            )
    rep["methods"]["CFR+"] = rep["curves"]["CFR+"][-1]
    print("cfr+", rep["curves"]["CFR+"], flush=True)
    mc = outcome_sampling_mccfr.OutcomeSamplingSolver(GAME)
    t0 = time.time()
    for it in range(1, int(1_000_000 * k) + 1):
        mc.iteration()
        if it in (1_000, 10_000, 100_000, 1_000_000) or it == int(1_000_000 * k):
            rep["curves"].setdefault("MCCFR (sampled self-play)", []).append(
                {"games": it, "exploitability": expl(mc.average_policy()), "seconds": round(time.time() - t0, 1)}
            )
    rep["methods"]["MCCFR (sampled self-play)"] = rep["curves"]["MCCFR (sampled self-play)"][-1]
    print("mccfr", rep["curves"]["MCCFR (sampled self-play)"], flush=True)
    teacher = mc.average_policy()
    for hidden in ((16,), (64,), (128, 64)):
        net, params = network(teacher, hidden)
        record(
            f"AlphaPoker v0 (network {hidden})",
            net,
            parameters=int(params),
            teacher_exploitability=rep["methods"]["MCCFR (sampled self-play)"]["exploitability"],
        )
    net, params = network(teacher, (64,))
    record(
        "AlphaPoker v0 + AlphaGo search (PUCT, net prior)",
        is_mcts(int(4000 * k), prior=net),
        note=f"{int(4000 * k)} simulations per decision",
    )
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], "--quick" in sys.argv)
