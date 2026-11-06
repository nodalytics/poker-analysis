"""Depth-limited flop search (AlphaPoker / DeepStack style): CFR+ over the flop that stops where the turn is dealt
and asks the value network, instead of solving every turn and river.

Usage: uv run python scripts/depth_limited.py SOLVER_LIBRARY_DIR VALUE_NET_JOBLIB OUTDIR [N_FLOPS] [ITERATIONS]

Game: the solver library's `spr4.5_wide` spot (pot 10, stacks 45, both players on the WIDE range), flop bets of 66%
of the pot or all-in, raises all-in only, all-in when a bet would exceed 67% of the stack: the tree TexasSolver
solved. At the end of flop betting the hand continues to the turn: the value there is the value network's prediction
for each hand, given both players' current ranges, averaged over every turn card. Folds and flop all-ins are exact.
Ground truth: TexasSolver's full solve of the same spot from the library; scored at the first player's opening
decision, the second player's decision after a check, and facing the 66% bet, as ml_flop.py is. The flops are
ml_flop.py's held-out flops, so the flop model is compared on boards it never saw either.
"""

import gzip
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_flop  # noqa: E402
import poker_agents as pa  # noqa: E402

RANKS = "23456789TJQKA"
CATS = ["High Card", "Pair", "Two Pair", "Trips", "Straight", "Flush", "Full House", "Quads", "Straight Flush"]
POT, STACK, BET = 10.0, 45.0, 0.66
SPOT = "spr4.5_wide"


def wide_combos(board):
    out = []
    used = set(board)
    for c in pa.WIDE.split(","):
        r1, r2 = c[0], c[1]
        for s1 in "cdhs":
            for s2 in "cdhs":
                a, b = r1 + s1, r2 + s2
                if r1 == r2 and s1 >= s2:
                    continue
                if a in used or b in used:
                    continue
                out.append((a, b))
    return out


class Precomputed:
    """Hand-vs-hand win rates for every turn card (averaged over rivers), static features, blocker mask."""

    def __init__(self, board, hands):
        import eval7

        self.board, self.hands, n = board, hands, len(hands)
        cards = [(eval7.Card(a), eval7.Card(b)) for a, b in hands]
        self.B = np.array([[float(not ({a, b} & {c, d})) for c, d in hands] for a, b in hands], dtype=np.float32)
        deck = [r + s for r in RANKS for s in "cdhs" if r + s not in board]
        self.turns = deck
        self.W, self.D, self.valid, self.F = [], [], [], []
        bcards = [eval7.Card(c) for c in board]
        for t in deck:
            tc = eval7.Card(t)
            ok = np.array([t not in h for h in hands])
            num = np.zeros((n, n), dtype=np.float32)
            den = np.zeros((n, n), dtype=np.float32)
            for rv in deck:
                if rv == t:
                    continue
                rc = eval7.Card(rv)
                s = np.array(
                    [
                        eval7.evaluate([a, b, *bcards, tc, rc]) if (t not in h and rv not in h) else -1
                        for (a, b), h in zip(cards, hands)
                    ]
                )
                v = s >= 0
                pair_ok = np.outer(v, v)
                num += pair_ok * ((s[:, None] > s[None, :]) + 0.5 * (s[:, None] == s[None, :]))
                den += pair_ok
            self.W.append(num / np.maximum(den, 1))
            self.valid.append(ok)
            self.F.append(self.static_features(board + [t], hands))
        self.W = np.stack(self.W) * self.B[None]  # turns x n x n, blockers removed
        self.valid = np.stack(self.valid)
        self.F = np.stack(self.F)
        # flop all-in showdown: average over turns of the turn win rates (each turn equally likely)
        vv = self.valid.astype(np.float32)
        self.E_flop = (self.W * vv[:, :, None] * vv[:, None, :]).sum(0) / np.maximum(
            (vv[:, :, None] * vv[:, None, :]).sum(0), 1
        )

    @staticmethod
    def static_features(board, hands):
        import eval7

        br = sorted((RANKS.index(c[0]) for c in board), reverse=True)
        bs = [c[1] for c in board]
        bf = [
            *(x / 12 for x in br),
            max(bs.count(s) for s in "cdhs") / 4,
            float(len(set(br)) < 4),
            (br[0] - br[3]) / 12,
        ]
        out = []
        for a, b in hands:
            if a in board or b in board:
                out.append([0.0] * 22)
                continue
            cat = eval7.handtype(eval7.evaluate([eval7.Card(a), eval7.Card(b)] + [eval7.Card(c) for c in board]))
            hr = sorted((RANKS.index(a[0]), RANKS.index(b[0])), reverse=True)
            suits = bs + [a[1], b[1]]
            allr = sorted(set(br) | set(hr))
            run = best = 1
            for x, y in zip(allr, allr[1:]):
                run = run + 1 if y == x + 1 else 1
                best = max(best, run)
            out.append(
                [
                    *bf,
                    hr[0] / 12,
                    hr[1] / 12,
                    float(a[1] == b[1]),
                    float(hr[0] == hr[1]),
                    *(float(cat == c) for c in CATS),
                    float(max(suits.count(s) for s in "cdhs") == 4),
                    float(best >= 4),
                ]
            )
        return np.array(out, dtype=np.float32)


def leaf_values(pc, net, pot, stack, r_oop, r_ip):
    """Counterfactual values at the start of the turn for both players (chips, weighted by opponent reach)."""
    T, n = pc.valid.shape
    out = {}
    eqs = {}
    for p, mine, opp in (("oop", r_oop, r_ip), ("ip", r_ip, r_oop)):
        wo = pc.valid * opp[None]  # turns x n
        num = np.einsum("tij,tj->ti", pc.W, wo)
        den = np.einsum("ij,tj->ti", pc.B, wo)
        eqs[p] = np.where(den > 0, num / np.maximum(den, 1e-9), 0.5)
    rows, keep = [], []
    for p, q, mine in (("oop", "ip", r_oop), ("ip", "oop", r_ip)):
        theirs = r_ip if p == "oop" else r_oop
        for t in range(T):
            w = mine * pc.valid[t]
            wsum = w.sum() or 1.0
            e = eqs[p][t]
            order = np.argsort(e)
            pct = np.empty(n)
            pct[order] = np.cumsum(w[order]) / wsum
            wq = theirs * pc.valid[t]
            range_eq = float((e * w).sum() / wsum)
            opp_eq = float((eqs[q][t] * wq).sum() / (wq.sum() or 1.0))
            dyn = np.column_stack(
                [
                    e,
                    pct,
                    np.full(n, range_eq),
                    np.full(n, opp_eq),
                    np.full(n, w.sum() / 1326),
                    np.full(n, wq.sum() / 1326),
                    np.full(n, stack / pot),
                    np.full(n, float(p == "ip")),
                ]
            )
            rows.append(np.hstack([pc.F[t], dyn]))
            keep.append((p, t))
    pred = net.predict(np.vstack(rows)).reshape(len(keep), n) * pot  # chips relative to equal contributions
    for p, opp in (("oop", r_ip), ("ip", r_oop)):
        vals = np.zeros(n)
        cnt = np.zeros(n)
        for k, (pp, t) in enumerate(keep):
            if pp != p:
                continue
            v = pc.valid[t]
            vals += np.where(v, pred[k], 0.0)
            cnt += v
        avg = vals / np.maximum(cnt, 1)
        out[p] = avg * (pc.B @ opp)  # counterfactual: weighted by the opponent's (unblocked) reach
    return out


class Node:
    def __init__(self, kind, player=None, pot=0.0, contrib=(0.0, 0.0), children=None, name=""):
        self.kind, self.player, self.pot, self.contrib, self.children, self.name = (
            kind,
            player,
            pot,
            contrib,
            children or [],
            name,
        )


def build():
    """The flop tree TexasSolver solved: players 0 = OOP, 1 = IP; contributions on top of 5 each from pre-flop."""

    def after_bet(bettor, c, label):
        other = 1 - bettor
        call_c = (c[bettor], c[bettor])
        kids = [
            ("FOLD", Node("fold", player=other, contrib=c)),
            ("CALL", Node("showdown" if c[bettor] >= STACK + 5 else "leaf", contrib=call_c)),
        ]
        if c[bettor] < STACK + 5:
            ai = (STACK + 5, STACK + 5)
            shove = list(c)
            shove[other] = STACK + 5
            kids.append(
                (
                    "ALLIN",
                    Node(
                        "decision",
                        player=bettor,
                        contrib=tuple(shove),
                        children=[
                            ("FOLD", Node("fold", player=bettor, contrib=tuple(shove))),
                            ("CALL", Node("showdown", contrib=ai)),
                        ],
                        name=label + "-allin",
                    ),
                )
            )
        return Node("decision", player=other, contrib=c, children=kids, name=label)

    def bet_amount(c):
        pot = c[0] + c[1]
        b = BET * pot
        return STACK if b >= 0.67 * STACK else b

    base = (5.0, 5.0)
    b = bet_amount(base)
    oop_bet = after_bet(0, (5 + b, 5.0), "ip_facing_bet")
    oop_allin = after_bet(0, (STACK + 5, 5.0), "ip_facing_allin")
    ip_bet = after_bet(1, (5.0, 5 + b), "oop_facing_bet")
    ip_allin = after_bet(1, (5.0, STACK + 5), "oop_facing_allin")
    ip_after_check = Node(
        "decision",
        player=1,
        contrib=base,
        children=[("CHECK", Node("leaf", contrib=base)), ("BET", ip_bet), ("ALLIN", ip_allin)],
        name="ip_after_check",
    )
    return Node(
        "decision",
        player=0,
        contrib=base,
        children=[("CHECK", ip_after_check), ("BET", oop_bet), ("ALLIN", oop_allin)],
        name="oop_open",
    )


class Solver:
    def __init__(self, pc, net, root):
        self.pc, self.net, self.root = pc, net, root
        n = len(pc.hands)
        self.reg, self.avg = {}, {}
        self.n = n
        self.leaf_cache = {}

    def strat(self, node):
        r = self.reg.setdefault(id(node), np.zeros((self.n, len(node.children))))
        pos = np.maximum(r, 0)
        s = pos.sum(1, keepdims=True)
        return np.where(s > 0, pos / np.maximum(s, 1e-12), 1.0 / len(node.children))

    def reaches(self, node, r):
        """Both players' reach vectors at every leaf, for this iteration's leaf evaluation."""
        if node.kind == "leaf":
            yield node, r
        if node.kind != "decision":
            return
        s = self.strat(node)
        for a, (_, child) in enumerate(node.children):
            rr = list(r)
            rr[node.player] = r[node.player] * s[:, a]
            yield from self.reaches(child, rr)

    def value(self, node, p, r, t):
        opp = 1 - p
        B = self.pc.B
        if node.kind == "fold":
            folder = node.player
            u = node.contrib[opp] if folder == opp else -node.contrib[p]
            return u * (B @ r[opp])
        if node.kind == "showdown":
            P = node.contrib[0] + node.contrib[1]
            return P * ((self.pc.E_flop @ r[opp]) - 0.5 * (B @ r[opp]))
        if node.kind == "leaf":
            return self.leaf_cache[id(node)]["oop" if p == 0 else "ip"]
        s = self.strat(node)
        if node.player == p:
            vals = np.stack([self.value(c, p, r, t) for _, c in node.children], 1)
            nv = (vals * s).sum(1)
            reg = self.reg[id(node)]
            reg += vals - nv[:, None]
            np.maximum(reg, 0, out=reg)  # CFR+
            self.avg.setdefault(id(node), np.zeros_like(s))
            self.avg[id(node)] += t * r[p][:, None] * s  # linear averaging
            return nv
        total = np.zeros(self.n)
        for a, (_, c) in enumerate(node.children):
            rr = list(r)
            rr[opp] = r[opp] * s[:, a]
            total += self.value(c, p, rr, t)
        return total

    def run(self, iterations, r0):
        for t in range(1, iterations + 1):
            for p in (0, 1):
                self.leaf_cache = {}
                for leaf, r in self.reaches(self.root, r0):
                    pot = leaf.contrib[0] + leaf.contrib[1]
                    self.leaf_cache[id(leaf)] = leaf_values(
                        self.pc, self.net, pot, STACK + 5 - leaf.contrib[0], r[0], r[1]
                    )
                self.value(self.root, p, r0, t)

    def average(self, node):
        a = self.avg.get(id(node))
        if a is None:
            return self.strat(node)
        s = a.sum(1, keepdims=True)
        return np.where(s > 0, a / np.maximum(s, 1e-12), 1.0 / len(node.children))


def slot_of(label):
    return {"CHECK": 0, "CALL": 0, "BET": 1, "ALLIN": 2, "FOLD": 3}[label]


def truth_nodes(tree):
    out = {"oop_open": tree}
    for name, child in (tree.get("childrens") or {}).items():
        if child.get("node_type") != "action_node":
            continue
        if name.startswith("CHECK"):
            out["ip_after_check"] = child
        elif float(name.split()[1]) < 40:
            out["ip_facing_bet"] = child
    return out


def one(args):
    libdir, netpath, flop_s, iterations = args
    from joblib import load

    net = load(netpath)
    board = [flop_s[i : i + 2] for i in (0, 2, 4)]
    hands = wide_combos(board)
    t0 = time.time()
    pc = Precomputed(board, hands)
    pre_s = time.time() - t0
    root = build()
    sol = Solver(pc, net, root)
    t0 = time.time()
    sol.run(iterations, [np.ones(len(hands)), np.ones(len(hands))])
    solve_s = time.time() - t0
    ours = {"oop_open": root, "ip_after_check": root.children[0][1], "ip_facing_bet": root.children[1][1]}
    tree = json.load(gzip.open(os.path.join(libdir, "solves", f"{flop_s}_{SPOT}.json.gz"), "rt"))
    truth = truth_nodes(tree)
    flop_mlp = load(os.path.join(os.path.dirname(netpath.rstrip("/")), "..", "mlflop", "mlp.joblib"))
    res = {"flop": flop_s, "precompute_s": round(pre_s, 1), "solve_s": round(solve_s, 1), "nodes": {}}
    key = {frozenset((a, b)): k for k, (a, b) in enumerate(hands)}
    for name, node in ours.items():
        tn = truth.get(name)
        if tn is None:
            continue
        acts = tn["actions"]
        strat_t = tn["strategy"]["strategy"]
        avg = sol.average(node)
        labels = [lab for lab, _ in node.children]
        agree = agree_mlp = cnt = 0
        mae = mae_mlp = 0.0
        for h, probs in strat_t.items():
            k = key.get(frozenset((h[:2], h[2:])))
            if k is None:
                continue
            y = np.zeros(4)
            for a, p in zip(acts, probs):
                y[ml_flop.slot(a)] += p
            mine = np.zeros(4)
            for lab, p in zip(labels, avg[k]):
                mine[slot_of(lab)] += p
            fm = ml_flop.normalise(flop_mlp.predict([ml_flop.features(board, h, name, SPOT)]))[0]
            agree += mine.argmax() == y.argmax()
            agree_mlp += fm.argmax() == y.argmax()
            mae += np.abs(mine - y).mean()
            mae_mlp += np.abs(fm - y).mean()
            cnt += 1
        res["nodes"][name] = {
            "hands": cnt,
            "agree_depth_limited": agree / cnt,
            "agree_flop_model": agree_mlp / cnt,
            "mae_depth_limited": mae / cnt,
            "mae_flop_model": mae_mlp / cnt,
        }
    return res


def main(libdir, netpath, outdir, n_flops=20, iterations=200, workers=20):
    from sklearn.model_selection import GroupShuffleSplit

    os.makedirs(outdir, exist_ok=True)
    # ml_flop.py's held-out flops: reproduce its split from its own loader
    X, Y, Gm, N = ml_flop.load(libdir)
    _, te_m = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, Y, Gm))
    held = sorted(set(Gm[te_m]))[:n_flops]
    with ProcessPoolExecutor(min(workers, len(held))) as pool:
        rows = list(pool.map(one, [(libdir, netpath, f, iterations) for f in held]))
    summ = {"flops": len(rows), "iterations": iterations, "spot": SPOT, "nodes": {}}
    for name in ("oop_open", "ip_after_check", "ip_facing_bet"):
        rs = [r["nodes"][name] for r in rows if name in r["nodes"]]
        tot = sum(x["hands"] for x in rs)
        summ["nodes"][name] = {
            k: round(sum(x[k] * x["hands"] for x in rs) / tot, 4)
            for k in ("agree_depth_limited", "agree_flop_model", "mae_depth_limited", "mae_flop_model")
        }
    allr = [x for r in rows for x in r["nodes"].values()]
    tot = sum(x["hands"] for x in allr)
    summ["overall"] = {
        k: round(sum(x[k] * x["hands"] for x in allr) / tot, 4)
        for k in ("agree_depth_limited", "agree_flop_model", "mae_depth_limited", "mae_flop_model")
    }
    summ["median_precompute_s"] = float(np.median([r["precompute_s"] for r in rows]))
    summ["median_solve_s"] = float(np.median([r["solve_s"] for r in rows]))
    lib = {json.loads(line)["id"]: json.loads(line) for line in open(os.path.join(libdir, "index.jsonl"))}
    summ["median_full_solve_s"] = float(
        np.median([lib[f"{r['flop']}_{SPOT}"]["seconds"] for r in rows if f"{r['flop']}_{SPOT}" in lib])
    )
    json.dump({"summary": summ, "flops": rows}, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], *(int(x) for x in a[3:5]))
