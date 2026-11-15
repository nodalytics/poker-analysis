"""Local best response (LBR; Lisy & Bowling 2017) for the 9-max tournament networks: a lower bound on how exploitable a
network is, stronger than a short exploiter training run.

One seat is LBR, the other eight are the network. LBR plays the network's own strategy, except after the flop in a pot
it contests alone against one network opponent. There it keeps the opponent's range over all 1,326 holdings, each
weighted by the probability the network would have taken every action it took this hand with that holding (the
network reads the table and its own bucket, so one batch over the holdings' buckets gives every weight). It then
picks the action with the highest value assuming the hand is checked down afterwards, as in the paper:
    fold    0
    check   wp x pot
    call    wp x (pot + to call) - to call
    bet b   fp x pot + (1 - fp) x (wp' x (pot + b + c) - b)
with wp its equity against the range over sampled run-outs, fp the share of the range the network folds to the bet
(its response, read from the network after the bet), wp' the equity against the hands that continue and c the
opponent's call. The values are in chips; the match is scored in ICM equity, as the network trains.

Result: LBR in each seat in turn against eight copies of the network, minus the network itself in the same seats on
the same deals (paired by deal): LBR's gain, bb/100 with a 95% interval. A gain near 0 means LBR finds no hole of
this kind; a large one is a leak, and the report says on which street LBR's bets made the network fold too often.

Usage (lab, GPU venv, from llm-research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_lbr.py blueprint/nm25 [--hands 2000] [--runouts 48] \\
      [--out results/ninemax/lbr_nm25.json]
"""

import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ninemax_engine import Dealer, Menu, Tables, strength
from ninemax_eval import load
from ninemax_search import result, take, with_pay
from ninemax_selfplay import context, features, masked

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NBOARD = {0: 0, 1: 3, 2: 4, 3: 5}
HOLDINGS = np.array(list(itertools.combinations(range(52), 2)), np.int32)  # 1,326


def bucket_rows(dealer, rows):
    """Buckets [n, 4] and showdown ranks [n] for rows of one player's cards, [n, 7] (two hole cards, five board
    cards): the library's bp_bucket_cards with one player (Dealer.buckets_for assumes the table's nine)."""
    import ctypes

    n = rows.shape[0]
    c = np.ascontiguousarray(rows, dtype=np.int32)
    bk = np.zeros((n, 1, 4), np.int32)
    rk = np.zeros((n, 1), np.int32)
    dealer.lib.bp_bucket_cards.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 3
    dealer.lib.bp_bucket_cards(n, 1, c.ctypes.data, bk.ctypes.data, rk.ctypes.data)
    return bk[:, 0], rk[:, 0]


class LBR:
    def __init__(self, net, dealer, runouts, rng):
        self.net, self.dealer, self.K, self.rng = net, dealer, runouts, rng
        self.stats = {s: {"bets": 0} for s in (1, 2, 3)}

    def _rows(self, hole, board, valid_h):
        """Each valid holding's buckets for the streets dealt so far, [H, 4], and LBR's equity against it over
        sampled run-outs, [H] (ties half). The hand indexer refuses a repeated card, so every row it sees is made of
        distinct cards: a holding's bucket row is completed with filler cards outside it, and on each run-out only the
        holdings that share no card with it are compared."""
        known = set(hole) | set(board)
        rest = np.array([c for c in range(52) if c not in known], np.int32)
        need = 5 - len(board)
        hs = HOLDINGS[valid_h]
        H = len(hs)
        # buckets: the streets dealt so far depend only on the board, so any distinct filler serves
        rows = np.zeros((H, 7), np.int32)
        rows[:, :2] = hs
        rows[:, 2 : 2 + len(board)] = board
        for k in range(H):
            fill = [c for c in rest if c != hs[k, 0] and c != hs[k, 1]][:need]
            rows[k, 2 + len(board) :] = fill
        bk, _ = bucket_rows(self.dealer, rows)
        wins = np.zeros(H)
        count = np.zeros(H)
        for _ in range(self.K):
            run = self.rng.choice(rest, need, replace=False) if need else np.zeros(0, np.int32)
            ok = ~np.isin(hs, run).any(1)  # a run-out that uses a holding's card does not apply to it
            idx = np.nonzero(ok)[0]
            full = np.concatenate([np.array(board, np.int32), run])
            r = np.zeros((len(idx) + 1, 7), np.int32)
            r[: len(idx), :2] = hs[idx]
            r[len(idx), :2] = hole
            r[:, 2:] = full
            _, rk = bucket_rows(self.dealer, r)
            mine = rk[-1]
            wins[idx] += (rk[:-1] > mine) + 0.5 * (rk[:-1] == mine)  # a lower rank wins
            count[idx] += 1
        return torch.tensor(bk, dtype=torch.long, device=self.dealer.dev), wins / np.maximum(count, 1)

    def _policy(self, feat, legal, street, bk):
        """The network's policy at one decision for every holding: feat [nf], legal [A], bk [H, 4] -> [H, A]."""
        H = bk.shape[0]
        st = torch.full((H,), street, dtype=torch.long, device=bk.device)
        s = strength(self.dealer, bk[:, None, :], st).squeeze(1)
        lg, _ = self.net(feat[None].expand(H, -1), bk, st, s)
        return F.softmax(masked(lg, legal[None].expand(H, -1)), 1)

    def decide(self, T, t, me, opp, hole, board, history):
        """LBR's slot at table t. history: the opponent's decisions this hand, (feat, legal, street, action)."""
        dead = set(hole) | set(board)
        valid = np.array([not (a in dead or b in dead) for a, b in HOLDINGS])
        valid_h = np.nonzero(valid)[0]
        bk, eq = self._rows(list(hole), list(board), valid_h)
        # the range: what the network would have done with each holding at each of its decisions (a holding's bucket
        # for an earlier street depends only on the board then, so the current rows serve every street)
        w = torch.ones(len(valid_h), device=bk.device, dtype=torch.float64)
        for feat, legal, street, act in history:
            pr = self._policy(feat, legal, street, bk)
            w *= pr[:, act].double().clamp(min=1e-12)
        w = (w / w.sum()).cpu().numpy() if float(w.sum()) > 0 else np.full(len(valid_h), 1 / len(valid_h))
        wp = float((w * eq).sum())
        g = torch.tensor([t], device=T.stack.device)
        i = torch.tensor([me], device=T.stack.device)
        _, legal = features(T, g, i)
        L = legal[0].bool().tolist()
        _, targets = T.legal(g, i)
        pot = float(T.contrib[t].sum() + T.dead[t])
        to_call = float(T.bet[t] - T.contrib[t, me])
        best, best_v = 1, -1e18
        for a, ok in enumerate(L):
            if not ok:
                continue
            if a == 0:
                v = 0.0
            elif a == 1:
                v = wp * (pot + to_call) - to_call
            else:
                U = take(T, g)
                U.step(
                    torch.zeros(1, dtype=torch.long, device=U.stack.device), torch.tensor([a], device=U.stack.device)
                )
                if bool(U.done[0]) or int(U.actor[0]) != opp:
                    continue
                f2, l2 = features(U, torch.zeros(1, dtype=torch.long, device=U.stack.device), U.actor[:1])
                pr = self._policy(f2[0], l2[0], int(U.street[0]), bk).cpu().numpy()
                fold = pr[:, 0]
                fp = float((w * fold).sum())
                cont = w * (1 - fold)
                wpc = float((cont * eq).sum() / max(cont.sum(), 1e-12))
                b = float(U.contrib[0, me] - T.contrib[t, me])
                c = float(torch.minimum(U.bet[0], U.avail[0, opp]) - U.contrib[0, opp])
                v = fp * pot + (1 - fp) * (wpc * (pot + b + c) - b)
            if v > best_v:
                best, best_v = a, v
        if best >= 2:
            st = int(T.street[t])
            self.stats[st]["bets"] += 1
        return best


def match(net, dealer, menu, n, dev, lbr):
    """LBR in each seat against 8 copies of the network, and the network itself in the same seats on the same deals:
    the per-deal results (bb/100) of both, and LBR's decisions."""
    P = dealer.P
    bk, rk, cd = dealer.deal(n, cards=True)
    stacks, pay = context(n, P, dev)
    out = {"lbr": torch.zeros(n, device=dev), "net": torch.zeros(n, device=dev)}
    decisions = 0
    for who in ("net", "lbr"):
        for h in range(P):
            T = with_pay(Tables(n, P, menu, dev, stacks=stacks), pay)
            hist = [[] for _ in range(n)]
            while not bool(T.done.all()):
                g = (~T.done).nonzero().squeeze(1)
                i = T.actor[g]
                feat, legal = features(T, g, i)
                s = strength(dealer, bk[g], T.street[g]).gather(1, i[:, None]).squeeze(1)
                with torch.no_grad():
                    lg, _ = net(feat, bk[g, i], T.street[g], s)
                a = torch.multinomial(F.softmax(masked(lg, legal), 1), 1).squeeze(1)
                if who == "lbr":
                    heads_up = (~T.folded[g]).sum(1) == 2
                    mine = (i == h) & (T.street[g] >= 1) & heads_up
                    for k in mine.nonzero().squeeze(1).tolist():
                        t = int(g[k])
                        live = (~T.folded[t]).nonzero().squeeze(1).tolist()
                        opp = [x for x in live if x != h][0]
                        nb = NBOARD[int(T.street[t])]
                        hole = [int(cd[t, 2 * h]), int(cd[t, 2 * h + 1])]
                        board = [int(x) for x in cd[t, 2 * P : 2 * P + nb]]
                        hist_opp = [(f, lgl, st, act) for (seat, f, lgl, st, act) in hist[t] if seat == opp]
                        with torch.no_grad():
                            a[k] = lbr.decide(T, t, h, opp, hole, board, hist_opp)
                        decisions += 1
                    for k in range(len(g)):
                        hist[int(g[k])].append((int(i[k]), feat[k], legal[k], int(T.street[g[k]]), int(a[k])))
                T.step(g, a)
            out[who] += result(T, rk)[:, h]
    for k in out:
        out[k] = out[k] / P * 100
    return out, decisions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--hands", type=int, default=2000)
    ap.add_argument("--runouts", type=int, default=48)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out")
    ap.add_argument("--lib", default=os.path.join(ROOT, "blueprint", "libbp7.so"))
    ap.add_argument("--abs", default=os.path.join(ROOT, "blueprint", "out"))
    a = ap.parse_args()
    dev = "cuda"
    dealer = Dealer(a.lib, a.abs, 9, 0, dev, seed=a.seed)
    net, hands = load(a.run, dealer.B, dev)
    d = os.path.dirname(a.run) if a.run.endswith(".pt") else a.run
    menu = Menu(json.load(open(os.path.join(d, "config.json"))).get("menu", "medium"), dev)
    lbr = LBR(net, dealer, a.runouts, np.random.default_rng(a.seed))
    t0 = time.time()
    with torch.no_grad():
        out, decisions = match(net, dealer, menu, a.hands, dev, lbr)
    gain = (out["lbr"] - out["net"]).cpu().numpy()
    m, se = float(gain.mean()), float(gain.std() / math.sqrt(len(gain)))
    res = {"run": a.run, "hands_trained": int(hands), "hands": a.hands, "runouts": a.runouts,
           "lbr_gain_bb100": round(m, 2), "ci95": [round(m - 1.96 * se, 2), round(m + 1.96 * se, 2)],
           "lbr_decisions": decisions, "lbr_bets_by_street": {str(k): v["bets"] for k, v in lbr.stats.items()},
           "net_self_bb100": round(float(out["net"].mean()), 2), "seconds": round(time.time() - t0)}  # fmt: skip
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
