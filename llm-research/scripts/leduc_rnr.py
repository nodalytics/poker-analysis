"""Profit against weak players versus being unbeatable, measured exactly on Leduc poker (game theory and physics).

Usage: .venv-alphapoker/bin/python scripts/leduc_rnr.py OUTDIR [ITERATIONS]   (open_spiel environment, see alphapoker.py)

Game theory: the restricted Nash response (Johanson, Zinkevich & Bowling 2007). In a modified game the opponent plays
a known weak strategy with probability p and plays freely (as a best responder would) otherwise; solving it with CFR+
gives our strategy for that p. p = 0 is the Nash equilibrium (unbeatable, exploits nothing); p = 1 is the pure best
response (exploits most, can itself be beaten badly). In between lies the frontier of how much profit can be taken
for how much exposure. Opponents: random, calling station (always check or call), maniac (raise whenever possible),
nit (check, fold to any bet).

Physics (statistical mechanics): the quantal response equilibrium, where every player plays a Boltzmann distribution
over action values at temperature alpha, found with entropy-regularised mirror descent (MMD, OpenSpiel). Higher
temperature means more random play: how much does it cost in exploitability, and does it win more from weak players?

Both measured in milli-chips per hand, per seat and averaged over the two seats: profit against each weak opponent
(above the game's value for that seat), and exploitability (what a best-responding opponent wins above game value).
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyspiel
from open_spiel.python import policy as osp
from open_spiel.python.algorithms import best_response, expected_game_score

GAME = pyspiel.load_game("leduc_poker")
FOLD, CALL, RAISE = 0, 1, 2


def weak(name):
    def probs(state):
        legal = state.legal_actions()
        if name == "random":
            return {a: 1 / len(legal) for a in legal}
        if name == "station":
            return {CALL: 1.0}
        if name == "maniac":
            return {RAISE: 1.0} if RAISE in legal else {CALL: 1.0}
        if name == "nit":
            # facing a bet (fold is legal only then): fold; otherwise check
            return {FOLD: 1.0} if FOLD in legal else {CALL: 1.0}
        raise ValueError(name)

    return probs


def tabular(fn):
    tp = osp.TabularPolicy(GAME)
    for key, idx in tp.state_lookup.items():
        tp.action_probability_array[idx] = 0

    # fill by walking the tree (one representative state per information set)
    def walk(s):
        if s.is_terminal():
            return
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                walk(s.child(a))
            return
        arr = tp.policy_for_key(s.information_state_string())
        if arr.sum() == 0:
            for a, p in fn(s).items():
                arr[a] = p
        for a in s.legal_actions():
            walk(s.child(a))

    walk(GAME.new_initial_state())
    return tp


class RNR:
    """CFR+ for our seat against an opponent who is the fixed weak strategy with probability p, free otherwise."""

    def __init__(self, seat, fixed, p):
        self.seat, self.fixed, self.p = seat, fixed, p
        self.reg, self.avg, self.oreg = {}, {}, {}

    @staticmethod
    def match(r, legal):
        pos = np.maximum(r, 0)
        s = pos.sum()
        return pos / s if s > 0 else np.ones(len(legal)) / len(legal)

    def walk(self, s, ru, ro, branch, t):
        if s.is_terminal():
            return s.returns()[self.seat]
        if s.is_chance_node():
            return sum(pr * self.walk(s.child(a), ru, ro * pr, branch, t) for a, pr in s.chance_outcomes())
        legal = s.legal_actions()
        key = s.information_state_string()
        if s.current_player() == self.seat:
            r = self.reg.setdefault(key, np.zeros(len(legal)))
            sig = self.match(r, legal)
            v = np.array([self.walk(s.child(a), ru * sig[k], ro, branch, t) for k, a in enumerate(legal)])
            node = float(sig @ v)
            r += ro * (v - node)
            np.maximum(r, 0, out=r)
            self.avg.setdefault(key, np.zeros(len(legal)))
            self.avg[key] += t * ru * sig
            return node
        if branch == "fixed":
            f = self.fixed(s)
            return sum(p * self.walk(s.child(a), ru, ro * p, branch, t) for a, p in f.items() if p > 0)
        r = self.oreg.setdefault(key, np.zeros(len(legal)))
        sig = self.match(r, legal)
        v = np.array([self.walk(s.child(a), ru, ro * sig[k], branch, t) for k, a in enumerate(legal)])
        node = float(sig @ v)
        r += ru * (node - v)  # the free opponent minimises our value
        np.maximum(r, 0, out=r)
        return node

    def run(self, iterations):
        root = GAME.new_initial_state()
        for t in range(1, iterations + 1):
            if self.p > 0:
                self.walk(root, 1.0, self.p, "fixed", t)
            if self.p < 1:
                self.walk(root, 1.0, 1 - self.p, "free", t)

    def policy(self):
        def fn(s):
            legal = s.legal_actions()
            a = self.avg.get(s.information_state_string())
            if a is None or a.sum() == 0:
                return {x: 1 / len(legal) for x in legal}
            return dict(zip(legal, a / a.sum()))

        return fn


def evaluate(seat, our_fn, game_value):
    """Exploitability of our seat and profit against each weak opponent, in milli-chips per hand."""
    ours = tabular(our_fn)
    br = best_response.BestResponsePolicy(GAME, 1 - seat, ours)
    expl = br.value(GAME.new_initial_state()) - (-game_value[seat])
    prof = {}
    for o in ("random", "station", "maniac", "nit"):
        opp = tabular(weak(o))
        pols = [ours, opp] if seat == 0 else [opp, ours]
        v = expected_game_score.policy_value(GAME.new_initial_state(), pols)[seat]
        prof[o] = 1000 * (v - game_value[seat])
    return 1000 * expl, prof


def rnr_job(args):
    seat, opp, p, iters, gv = args
    sol = RNR(seat, weak(opp), p)
    sol.run(iters)
    expl, prof = evaluate(seat, sol.policy(), gv)
    return {"seat": seat, "opponent": opp, "p": p, "exploitability": round(expl, 1), "profit": round(prof[opp], 1)}


def mmd_job(args):
    alpha, iters, gv = args
    from open_spiel.python.algorithms import mmd_dilated

    m = mmd_dilated.MMDDilatedEnt(GAME, alpha)
    for _ in range(iters):
        m.update_sequences()
    pol = m.get_avg_policies() if alpha == 0 else m.get_policies()
    out = {"alpha": alpha, "exploitability": 0.0, "profit": {}}
    for seat in (0, 1):

        def fn(s, pol=pol):
            return {a: p for a, p in pol.action_probabilities(s).items()}

        e, prof = evaluate(seat, fn, gv)
        out["exploitability"] += e / 2
        for o, v in prof.items():
            out["profit"][o] = round(out["profit"].get(o, 0) + v / 2, 1)
    out["exploitability"] = round(out["exploitability"], 1)
    return out


def main(outdir, iters=600):
    from open_spiel.python.algorithms import cfr

    os.makedirs(outdir, exist_ok=True)
    nash = cfr.CFRPlusSolver(GAME)
    for _ in range(1000):
        nash.evaluate_and_update_policy()
    gv0 = expected_game_score.policy_value(GAME.new_initial_state(), [nash.average_policy()] * 2)[0]
    gv = [gv0, -gv0]
    ps_ = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    jobs = [(seat, o, p, iters, gv) for o in ("random", "station", "maniac", "nit") for p in ps_ for seat in (0, 1)]
    with ProcessPoolExecutor(40) as pool:
        rows = list(pool.map(rnr_job, jobs))
        mm = list(pool.map(mmd_job, [(a, 2000, gv) for a in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)]))
    frontier = {}
    for o in ("random", "station", "maniac", "nit"):
        frontier[o] = []
        for p in ps_:
            rs = [r for r in rows if r["opponent"] == o and r["p"] == p]
            frontier[o].append(
                {
                    "p": p,
                    "profit": round(sum(r["profit"] for r in rs) / 2, 1),
                    "exploitability": round(sum(r["exploitability"] for r in rs) / 2, 1),
                }
            )
    rep = {
        "game_value_seat0_mchips": round(1000 * gv0, 1),
        "units": "milli-chips per hand, averaged over seats",
        "rnr_frontier": frontier,
        "qre_mmd": mm,
    }
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], *(int(x) for x in sys.argv[2:3]))
