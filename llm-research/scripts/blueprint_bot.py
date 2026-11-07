"""Play the heads-up blueprint (blueprint/mccfr.cpp) in a real hand.

The hand so far is read from PokerKit's operation log and walked down the abstract betting tree:
  fold / check / call      the matching child
  a bet or raise           the child with the nearest size, measured as a fraction of the pot after calling; between
                           two abstract sizes A < x < B it maps to A with probability (B-x)(1+A) / ((B-A)(1+x)), the
                           randomized pseudo-harmonic mapping of Ganzfried & Sandholm used by Libratus and Pluribus
Our own decision: the average strategy before the flop and the current regret-matching strategy after it (as Pluribus;
BLUEPRINT_POSTFLOP=average to change), at that node for our card bucket (libbp.so: the trainer's abstraction),
actions under 5% dropped ("purification", harder to exploit), sampled. An abstract raise is played at the same
fraction of the real pot. Returns None (caller falls back to the solver bot) for multi-way pots, or when the hand
leaves the tree.

Env: BLUEPRINT_RUN (tree.tsv + avg.f32; default blueprint/run2), BLUEPRINT_ABS (bucket tables; blueprint/out),
BLUEPRINT_LIB (libbp.so; blueprint/libbp.so). Units: the harness's chips are scaled so its big blind = 100.
"""

import ctypes
import json
import os
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.environ.get("BLUEPRINT_RUN", os.path.join(ROOT, "blueprint", "run2"))
ABS = os.environ.get("BLUEPRINT_ABS", os.path.join(ROOT, "blueprint", "out"))
LIB = os.environ.get("BLUEPRINT_LIB", os.path.join(ROOT, "blueprint", "libbp.so"))
PURIFY = 0.05
_RNG = random.Random(97)
_BP = {}


def card_int(c):
    return "23456789TJQKA".index(c[0]) * 4 + "cdhs".index(c[1])


class Blueprint:
    def __init__(self, run=RUN, absdir=ABS, lib=LIB):
        rows = [ln.rstrip("\n").split("\t") for ln in open(os.path.join(run, "tree.tsv"))][1:]
        self.nodes = {}
        for r in rows:
            self.nodes[int(r[0])] = {
                "kind": int(r[1]),
                "player": int(r[2]),
                "street": int(r[3]),
                "c": (int(r[4]), int(r[5])),
                "offset": int(r[6]),
                "labels": r[7].split(",") if r[7] else [],
                "children": [int(x) for x in r[8].split(",")] if r[8] else [],
            }
        self.avg = np.memmap(os.path.join(run, "avg.f32"), dtype=np.float32, mode="r")
        # after the flop, play the current (regret-matching) strategy as Pluribus did: rarely reached spots such as
        # facing an all-in keep a near-uniform average for a long time, and LBR won +560 bb/100 calling them down
        rp = os.path.join(run, "regret.f32")
        self.regret = np.memmap(rp, dtype=np.float32, mode="r") if os.path.exists(rp) else None
        self.postflop = os.environ.get("BLUEPRINT_POSTFLOP", "average")  # LBR: average +153, current +286
        self._mapped = {}  # betting history -> child: an off-tree bet keeps one reading for the whole hand
        self.lib = ctypes.CDLL(lib)
        if self.lib.bp_init(absdir.encode()) != 0:
            raise RuntimeError(f"blueprint buckets not found in {absdir}")

    def bucket(self, cards):
        arr = (ctypes.c_int * len(cards))(*[card_int(c) for c in cards])
        return self.lib.bp_bucket(arr, len(cards))

    # ---- walking the abstract tree ----
    @staticmethod
    def frac(to, their, mine, pot):
        """Raise size as a fraction of the pot after calling: (to - their bet) / (pot + to_call)."""
        call = their - mine
        return (to - their) / max(pot + call, 1)

    def child_by_label(self, node, *labels):
        n = self.nodes[node]
        for lab in labels:
            if lab in n["labels"]:
                return n["children"][n["labels"].index(lab)]
        return None

    def map_raise(self, node, x):
        """Child for a real raise of size fraction x (pseudo-harmonic randomized mapping)."""
        n = self.nodes[node]
        p = n["player"]
        cands = []
        for lab, ch in zip(n["labels"], n["children"]):
            if lab in ("fold", "check", "call"):
                continue
            c = self.nodes[ch]["c"]
            mine, their = n["c"][p], n["c"][1 - p]
            to = c[p]
            cands.append((self.frac(to, their, mine, n["c"][0] + n["c"][1]), ch))
        if not cands:
            return None
        cands.sort()
        if x <= cands[0][0]:
            return cands[0][1]
        if x >= cands[-1][0]:
            return cands[-1][1]
        for (a, ca), (b, cb) in zip(cands, cands[1:]):
            if a <= x <= b:
                pa = ((b - x) * (1 + a)) / max((b - a) * (1 + x), 1e-9)
                return ca if _RNG.random() < pa else cb
        return cands[-1][1]

    def walk(self, state, scale):
        """(abstract node, real street contributions info) after the hand's actions, or None off the tree."""
        node = 0
        n_seats = len(state.statuses)
        if n_seats != 2:
            return None
        bets = [0, 0]  # real chips this street, by blueprint player (0 = button / small blind)
        pot = 0
        sig = []  # the betting so far, for the mapping cache
        for op in state.operations:
            name = type(op).__name__
            if name in ("Folding", "CheckingOrCalling", "CompletionBettingOrRaisingTo", "BoardDealing"):
                sig.append((name[0], getattr(op, "player_index", -1), getattr(op, "amount", 0)))
            if name == "BlindOrStraddlePosting":
                bets[1 - op.player_index] += op.amount * scale  # PokerKit heads-up: seat 0 big blind, 1 button
            elif name == "BetCollection":
                pot += sum(bets)
                bets = [0, 0]
            elif name in ("Folding", "CheckingOrCalling", "CompletionBettingOrRaisingTo"):
                if self.nodes[node]["kind"] != 0:
                    return None
                p = 1 - op.player_index
                if self.nodes[node]["player"] != p:
                    return None
                if name == "Folding":
                    node = self.child_by_label(node, "fold")
                elif name == "CheckingOrCalling":
                    bets[p] = bets[1 - p]
                    node = self.child_by_label(node, "check", "call")
                else:
                    to = op.amount * scale
                    x = self.frac(to, bets[1 - p], bets[p], pot + sum(bets))
                    bets[p] = to
                    # keyed by the action history so far (not id(state): Python reuses ids of discarded copies, and
                    # LBR's hypothetical bets then lent their reading to real ones)
                    key = (tuple(sig), round(x, 6))
                    if key not in self._mapped:
                        if len(self._mapped) > 200000:
                            self._mapped.clear()
                        self._mapped[key] = self.map_raise(node, x)
                    node = self._mapped[key]
                if node is None:
                    return None
        return node, bets, pot

    # ---- our decision ----
    def distribution(self, state, i, scale, hole=None, walked=None):
        """[(kind, raise_to_in_harness_units or None, prob)] for player i, or None. `walked`: a walk() result to reuse
        (LBR asks for every holding at the same spot)."""
        w = walked or self.walk(state, scale)
        if w is None:
            return None
        node, bets, pot = w
        n = self.nodes[node]
        p = 1 - i
        if n["kind"] != 0 or n["player"] != p:
            return None
        board = [repr(c) for c in state.get_board_cards(0)] if state.board_count else []
        hole = hole or [repr(c) for c in state.hole_cards[i]]
        b = self.bucket(hole + board)
        if b < 0:
            return None
        k = len(n["labels"])
        lo_i, hi_i = n["offset"] + b * k, n["offset"] + (b + 1) * k
        s = np.array(self.avg[lo_i:hi_i], dtype=float)
        s = s / s.sum() if s.sum() > 0 else np.full(k, 1 / k)
        if n["street"] > 0 and self.postflop == "current" and self.regret is not None:
            r = np.clip(np.array(self.regret[lo_i:hi_i], dtype=float), 0, None)
            if r.sum() > 0:
                s = r / r.sum()
        s[s < PURIFY] = 0.0
        s = s / s.sum() if s.sum() > 0 else np.full(k, 1 / k)
        can_raise = state.can_complete_bet_or_raise_to()
        lo = state.min_completion_betting_or_raising_to_amount if can_raise else None
        hi = state.max_completion_betting_or_raising_to_amount if can_raise else None
        out = []
        for lab, ch, pr in zip(n["labels"], n["children"], s):
            if pr <= 0:
                continue
            if lab == "fold":
                out.append(("fold", None, pr))
            elif lab in ("check", "call") or not can_raise:
                out.append(("call", None, pr))
            elif lab == "allin":
                out.append(("raise", hi, pr))
            else:
                c = self.nodes[ch]["c"]
                f = self.frac(c[p], n["c"][1 - p], n["c"][p], n["c"][0] + n["c"][1])
                real_to = bets[1 - p] + f * (pot + sum(bets) + bets[1 - p] - bets[p])
                out.append(("raise", int(max(lo, min(hi, round(real_to / scale)))), pr))
        return out


def get():
    if "bp" not in _BP:
        _BP["bp"] = Blueprint()
    return _BP["bp"]


def reply(state, i):
    """JSON reply like the other bots, or None (the caller falls back)."""
    big = max(getattr(state, "blinds_or_straddles", (1, 2)))
    dist = get().distribution(state, i, 100 / big)
    if not dist:
        return None
    kind, to, _ = _RNG.choices(dist, weights=[p for _, _, p in dist])[0]
    if kind == "fold" and not state.checking_or_calling_amount:
        kind = "call"
    if kind == "raise":
        return json.dumps({"action": "raise", "amount": int(to)})
    return json.dumps(
        {"action": "fold" if kind == "fold" else ("call" if state.checking_or_calling_amount else "check")}
    )
