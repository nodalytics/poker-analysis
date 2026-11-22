"""Ensembles of strategies on Leduc poker, measured exactly where possible.

Usage: .venv-alphapoker/bin/python scripts/leduc_ensembles.py OUTDIR [SESSIONS] [HANDS]   (open_spiel environment)

Experts: the CFR equilibrium (Nash) and the robust-exploitation strategy (restricted Nash response, p = 0.25) built
against each weak player (random, station, maniac, nit); leduc_rnr.py.

1. Static ensembles, scored exactly (milli-chips per hand above game value, averaged over seats): profit against each
   weak player and against two players nobody was built for ('drunk', half maniac half random; 'loose', 70% station
   30% random), and exploitability.
   - mixture: pick one expert at random for the whole hand (a mixed strategy; exploitability is at most the average
     of the experts');
   - vote: average the experts' action probabilities at every decision (no such guarantee);
   - joint solve: one robust-exploitation solve against all four weak players at once (each with probability p/4,
     free best responder otherwise), p = 0.25 and 0.5;
   - bagged AlphaPoker: five policy networks (different seeds) averaged, against one network.
2. Online ensembles against an unknown opponent: sessions of HANDS hands, seat alternating, sampled play.
   - each expert alone;
   - Exp3 (adversarial bandit): picks an expert per hand from the chips each has won when picked;
   - Bayes ID: the posterior over which weak player (or 'someone else', modelled as Nash) the opponent is, from every
     opponent action's likelihood; plays that player's robust strategy once the posterior passes 0.95, Nash otherwise;
   - Bayes ID mixed: play each expert with its posterior probability.
   Opponents: the four weak players, drunk, loose, Nash, and a trap (plays as a maniac for the first 30% of the
   session, then the best response to the anti-maniac robust strategy).
"""

import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leduc_rnr as L  # noqa: E402

GAME = L.GAME
WEAK = ("random", "station", "maniac", "nit")


def mix_fn(fns, weights):
    """A behavioural mix of opponent-type functions at each decision (only for describing an opponent's play)."""

    def fn(s):
        out = {}
        for f, w in zip(fns, weights):
            for a, p in f(s).items():
                out[a] = out.get(a, 0.0) + w * p
        return out

    return fn


UNSEEN = {
    "drunk": mix_fn([L.weak("maniac"), L.weak("random")], [0.5, 0.5]),
    "loose": mix_fn([L.weak("station"), L.weak("random")], [0.7, 0.3]),
}


def as_dict(fn, seat=None):
    """Freeze a strategy into info-set key -> {action: prob} for every info set of either seat (or only `seat`)."""
    out = {}

    def walk(s):
        if s.is_terminal():
            return
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                walk(s.child(a))
            return
        k = s.information_state_string()
        if k not in out and seat in (None, s.current_player()):
            out[k] = dict(fn(s))
        for a in s.legal_actions():
            walk(s.child(a))

    walk(GAME.new_initial_state())
    return out


def lookup(d):
    def fn(s):
        return d[s.information_state_string()]

    return fn


def mixture(dicts, weights, seat):
    """Reach-weighted average for our seat: exactly 'pick one expert per hand' as a single behaviour strategy."""
    num, den = {}, {}

    def walk(s, reach):
        if s.is_terminal():
            return
        if s.is_chance_node():
            for a, _ in s.chance_outcomes():
                walk(s.child(a), reach)
            return
        legal = s.legal_actions()
        if s.current_player() != seat:
            for a in legal:
                walk(s.child(a), reach)
            return
        k = s.information_state_string()
        if k not in num:
            num[k] = np.zeros(GAME.num_distinct_actions())
            den[k] = 0.0
            for j, d in enumerate(dicts):
                for a, p in d[k].items():
                    num[k][a] += weights[j] * reach[j] * p
                den[k] += weights[j] * reach[j]
        for a in legal:
            walk(s.child(a), [reach[j] * dicts[j][k].get(a, 0.0) for j in range(len(dicts))])

    walk(GAME.new_initial_state(), [1.0] * len(dicts))

    def fn(s):
        k, legal = s.information_state_string(), s.legal_actions()
        if den.get(k, 0) <= 0:
            return {a: 1 / len(legal) for a in legal}
        return {a: num[k][a] / den[k] for a in legal}

    return fn


class JointRNR(L.RNR):
    """Robust exploitation against several weak players at once: each with probability p / n, free otherwise."""

    def __init__(self, seat, fixed_list, p):
        super().__init__(seat, None, p)
        self.fixed_list = fixed_list

    def run(self, iterations):
        root = GAME.new_initial_state()
        n = len(self.fixed_list)
        for t in range(1, iterations + 1):
            for f in self.fixed_list:
                self.fixed = f
                self.walk(root, 1.0, self.p / n, "fixed", t)
            if self.p < 1:
                self.walk(root, 1.0, 1 - self.p, "free", t)


def score(pols, gv):
    """pols: seat -> fn. Exact exploitability and profit against every weak and unseen player."""
    from open_spiel.python.algorithms import expected_game_score

    opps = {o: L.weak(o) for o in WEAK} | UNSEEN
    e, prof = 0.0, {o: 0.0 for o in opps}
    for seat in (0, 1):
        ours = L.tabular(pols[seat])
        from open_spiel.python.algorithms import best_response

        br = best_response.BestResponsePolicy(GAME, 1 - seat, ours)
        e += 1000 * (br.value(GAME.new_initial_state()) + gv[seat]) / 2
        for o, f in opps.items():
            op = L.tabular(f)
            v = expected_game_score.policy_value(GAME.new_initial_state(), [ours, op] if seat == 0 else [op, ours])[
                seat
            ]
            prof[o] += 1000 * (v - gv[seat]) / 2
    return {"exploitability": round(e, 1), "profit": {o: round(v, 1) for o, v in prof.items()}}


def experts(iters=600):
    out = {}
    for name, opp, p in [("nash", "random", 0.0)] + [(f"rnr_{o}", o, 0.25) for o in WEAK]:
        out[name] = {}
        for seat in (0, 1):
            sol = L.RNR(seat, L.weak(opp), p)
            sol.run(iters)
            out[name][seat] = as_dict(sol.policy())
    return out


def static_job(args):
    kind, gv = args
    ex = experts()
    names = list(ex)
    if kind == "experts":
        return {n: score({st: lookup(ex[n][st]) for st in (0, 1)}, gv) for n in names}
    if kind.startswith("mixture") or kind.startswith("vote"):
        use = [n for n in names if n != "nash"] if kind.endswith("rnr4") else names
        w = [1 / len(use)] * len(use)
        if kind.startswith("mixture"):
            pols = {st: mixture([ex[n][st] for n in use], w, st) for st in (0, 1)}
        else:
            pols = {st: mix_fn([lookup(ex[n][st]) for n in use], w) for st in (0, 1)}
        return {kind: score(pols, gv)}
    if kind.startswith("joint"):
        p = float(kind.split("_")[1])
        pols = {}
        for st in (0, 1):
            sol = JointRNR(st, [L.weak(o) for o in WEAK], p)
            sol.run(600)
            pols[st] = sol.policy()
        return {kind: score(pols, gv)}
    if kind.startswith("bag"):
        import leduc_rl_rnr as R

        base = kind.split("_", 1)[1]  # nash or rnr_maniac
        single = {st: R.alphapoker_net(lookup(ex[base][st]), seed=0) for st in (0, 1)}
        nets = {st: [R.alphapoker_net(lookup(ex[base][st]), seed=k) for k in range(5)] for st in (0, 1)}
        bag = {st: mix_fn(nets[st], [0.2] * 5) for st in (0, 1)}
        return {
            f"alphapoker_single_{base}": score(single, gv),
            f"alphapoker_bag5_{base}": score(bag, gv),
            f"cfr_{base}": score({st: lookup(ex[base][st]) for st in (0, 1)}, gv),
        }
    raise ValueError(kind)


# ---------- online ----------


def sample(pr, rng):
    r, acc = rng.random(), 0.0
    for a, p in pr.items():
        acc += p
        if r < acc:
            return a
    return a


class Exp3:
    def __init__(self, n, rng, gamma=0.1, scale=13.0):
        self.w, self.n, self.rng, self.g, self.scale = [0.0] * n, n, rng, gamma, scale

    def probs(self):
        m = max(self.w)
        e = [math.exp(x - m) for x in self.w]
        z = sum(e)
        return [(1 - self.g) * x / z + self.g / self.n for x in e]

    def pick(self):
        self.p = self.probs()
        self.k = self.rng.choices(range(self.n), weights=self.p)[0]
        return self.k

    def update(self, reward):
        r = (max(-self.scale, min(self.scale, reward)) / self.scale + 1) / 2  # into [0, 1]
        self.w[self.k] += self.g * (r / self.p[self.k]) / self.n


class BayesID:
    """Posterior over opponent types from the likelihood of each observed opponent action."""

    def __init__(self, types, eps=0.02):
        self.types, self.eps = types, eps
        self.logp = np.zeros(len(types))

    def observe(self, s, a):
        for j, f in enumerate(self.types):
            pr = f(s)
            n = len(s.legal_actions())
            self.logp[j] += math.log((1 - self.eps) * pr.get(a, 0.0) + self.eps / n)

    def post(self):
        x = np.exp(self.logp - self.logp.max())
        return x / x.sum()


def opponent_policy(name, frozen):
    if name in WEAK:
        return L.weak(name)
    if name in UNSEEN:
        return UNSEEN[name]
    if name == "nash":
        return None  # seat-dependent, handled by caller
    raise ValueError(name)


def online_job(args):
    agent, opp, sessions, hands, seed, gv, frozen = args
    rng = random.Random(seed)
    names = list(frozen["experts"])  # nash, rnr_random, rnr_station, rnr_maniac, rnr_nit
    nash = frozen["experts"]["nash"]
    type_fns = [L.weak(o) for o in WEAK]
    totals = []
    for _ in range(sessions):
        bandit = Exp3(len(names), rng)
        # Bayes ID: 'someone else' is modelled by the Nash strategy of whichever seat the opponent holds
        ids = BayesID(type_fns + [None])
        won = 0.0
        for h in range(hands):
            seat = h % 2
            if agent in names:
                mine = frozen["experts"][agent][seat]
            elif agent == "exp3":
                mine = frozen["experts"][names[bandit.pick()]][seat]
            else:
                post = ids.post()
                if agent == "bayes_id":
                    j = int(post.argmax())
                    pick = names[1 + j] if j < len(WEAK) and post[j] > 0.95 else "nash"
                    mine = frozen["experts"][pick][seat]
                else:  # bayes_mixed: expert drawn with its posterior; 'someone else' -> nash
                    j = rng.choices(range(len(post)), weights=post)[0]
                    mine = frozen["experts"][names[1 + j] if j < len(WEAK) else "nash"][seat]
            if opp == "nash":
                theirs = lookup(nash[1 - seat])
            elif opp == "trap":
                theirs = L.weak("maniac") if h < 0.3 * hands else lookup(frozen["trap_br"][1 - seat])
            else:
                theirs = opponent_policy(opp, frozen)
            s = GAME.new_initial_state()
            while not s.is_terminal():
                if s.is_chance_node():
                    outs = s.chance_outcomes()
                    s.apply_action(rng.choices([a for a, _ in outs], weights=[p for _, p in outs])[0])
                    continue
                if s.current_player() == seat:
                    a = sample(mine[s.information_state_string()], rng)
                else:
                    a = sample(theirs(s), rng)
                    if agent.startswith("bayes"):
                        # 'someone else' likelihood: the Nash strategy for the opponent's seat
                        ids.types[-1] = lookup(nash[1 - seat])
                        ids.observe(s, a)
                s.apply_action(a)
            r = s.returns()[seat] - gv[seat]
            won += r
            if agent == "exp3":
                bandit.update(r)
        totals.append(1000 * won / hands)
    m = float(np.mean(totals))
    se = float(np.std(totals, ddof=1) / math.sqrt(len(totals)))
    return {"agent": agent, "opponent": opp, "mchips_per_hand": round(m, 1), "se": round(se, 1)}


def main(outdir, sessions=200, hands=1000):
    from open_spiel.python.algorithms import best_response, cfr, expected_game_score

    os.makedirs(outdir, exist_ok=True)
    nash = cfr.CFRPlusSolver(GAME)
    for _ in range(1000):
        nash.evaluate_and_update_policy()
    gv0 = expected_game_score.policy_value(GAME.new_initial_state(), [nash.average_policy()] * 2)[0]
    gv = [gv0, -gv0]
    kinds = [
        "experts",
        "mixture_all5",
        "mixture_rnr4",
        "vote_all5",
        "vote_rnr4",
        "joint_0.25",
        "joint_0.5",
        "bag_nash",
        "bag_rnr_maniac",
    ]
    with ProcessPoolExecutor(len(kinds)) as pool:
        static = {}
        for part in pool.map(static_job, [(k, gv) for k in kinds]):
            static.update(part)
    ex = experts()
    trap_br = {}
    for seat in (0, 1):  # the trap's second phase: best response, in seat `seat`, to the anti-maniac strategy
        target = L.tabular(lookup(ex["rnr_maniac"][1 - seat]))
        br = best_response.BestResponsePolicy(GAME, seat, target)
        trap_br[seat] = as_dict(lambda s, br=br: br.action_probabilities(s), seat)
    frozen = {"experts": ex, "trap_br": trap_br}
    agents = list(ex) + ["exp3", "bayes_id", "bayes_mixed"]
    opps = list(WEAK) + list(UNSEEN) + ["nash", "trap"]
    jobs = [
        (a, o, sessions, hands, 1000 + k, gv, frozen) for k, (a, o) in enumerate((a, o) for a in agents for o in opps)
    ]
    with ProcessPoolExecutor(min(48, len(jobs))) as pool:
        online = list(pool.map(online_job, jobs))
    table = {a: {r["opponent"]: [r["mchips_per_hand"], r["se"]] for r in online if r["agent"] == a} for a in agents}
    for a in agents:
        vals = [table[a][o][0] for o in opps]
        table[a]["mean"] = round(float(np.mean(vals)), 1)
        table[a]["worst"] = round(float(min(vals)), 1)
    rep = {
        "units": "milli-chips per hand above game value, averaged over seats",
        "static_exact": static,
        "online": {"sessions": sessions, "hands_per_session": hands, "table": table},
    }
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], *(int(x) for x in a[1:3]))
