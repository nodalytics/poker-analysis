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
import itertools
import json
import os
import random

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# training runs (run2 + out + libbp.so) when present, else the advisor's data pack (blueprint/pack, with bp.dll on
# Windows): tree, 8-bit strategy, flop/turn buckets and river cut points
_PACK = os.path.join(ROOT, "blueprint", "pack")
_TRAINED = os.path.exists(os.path.join(ROOT, "blueprint", "run2", "avg.f32"))
RUN = os.environ.get("BLUEPRINT_RUN", os.path.join(ROOT, "blueprint", "run2") if _TRAINED else _PACK)
ABS = os.environ.get("BLUEPRINT_ABS", os.path.join(ROOT, "blueprint", "out") if _TRAINED else _PACK)
LIB = os.environ.get(
    "BLUEPRINT_LIB",
    os.path.join(ROOT, "blueprint", "libbp.so")
    if _TRAINED
    else os.path.join(_PACK, "bp.dll" if os.name == "nt" else "libbp.so"),
)
PURIFY = float(os.environ.get("BLUEPRINT_PURIFY", 0.05))  # drop actions under this probability
_RNG = random.Random(97)
_DECK = [r + s for r in "23456789TJQKA" for s in "cdhs"]
_COMBO_INDEX = {a + b: j for j, (a, b) in enumerate((a, b) for k, a in enumerate(_DECK) for b in _DECK[k + 1 :])}
_BP = {}


def card_int(c):
    return "23456789TJQKA".index(c[0]) * 4 + "cdhs".index(c[1])


class Blueprint:
    def __init__(self, run=RUN, absdir=ABS, lib=LIB):
        self.run, self.absdir = run, absdir  # for the safe re-solver, which loads the same table
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
        # avg.f32 from training, or avg.u8 in the advisor's data pack (blueprint/advisor_pack.py): every reader
        # normalises per (node, bucket), so 8-bit probabilities play the same strategy within rounding
        f32 = os.path.join(run, "avg.f32")
        self.avg = (
            np.memmap(f32, dtype=np.float32, mode="r")
            if os.path.exists(f32)
            else np.memmap(os.path.join(run, "avg.u8"), dtype=np.uint8, mode="r")
        )
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

    def walk(self, state, scale, full=False):
        """(abstract node, real street bets, real pot before this street) after the hand's actions, or None off the
        tree. full=True adds the history [(node, child index, player, board cards dealt by then)] and this street's
        actions as [(kind, units added)] for the re-solver."""
        node = 0
        history, street_acts, board_n = [], [], 0
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
            elif name == "BoardDealing":
                board_n += len(op.cards)
                street_acts = []
            elif name in ("Folding", "CheckingOrCalling", "CompletionBettingOrRaisingTo"):
                if self.nodes[node]["kind"] != 0:
                    return None
                p = 1 - op.player_index
                if self.nodes[node]["player"] != p:
                    return None
                before = node
                if name == "Folding":
                    street_acts.append(("fold", 0))
                    node = self.child_by_label(node, "fold")
                elif name == "CheckingOrCalling":
                    street_acts.append(("call" if op.amount else "check", op.amount))
                    bets[p] = bets[1 - p]
                    node = self.child_by_label(node, "check", "call")
                else:
                    to = op.amount * scale
                    street_acts.append(("raise", int(round((to - bets[p]) / scale))))
                    x = self.frac(to, bets[1 - p], bets[p], pot + sum(bets))
                    bets[p] = to
                    # keyed by the action history so far (not id(state): Python reuses ids of discarded copies, and
                    # LBR's hypothetical bets then lent their reading to real ones)
                    key = (tuple(sig), round(x, 6))
                    # least recently used first out: every reading this hand has used was just touched, so a full
                    # cache never drops one mid-hand (clearing it all could re-draw an earlier bet's reading)
                    node = self._mapped.pop(key) if key in self._mapped else self.map_raise(node, x)
                    self._mapped[key] = node
                    if len(self._mapped) > 200000:
                        for old in list(itertools.islice(self._mapped, 100000)):
                            del self._mapped[old]
                if node is None:
                    return None
                history.append((before, self.nodes[before]["children"].index(node), p, board_n))
        if full:
            return node, bets, pot, history, street_acts
        return node, bets, pot

    # ---- re-solving (phase A) ----
    def reach(self, history, player, board, blocked):
        """Weight of every holding for `player`: the product of the blueprint's probabilities of that player's actions
        so far, each read at the holding's bucket on the board dealt by then."""
        deck = [r + s for r in "23456789TJQKA" for s in "cdhs"]
        combos = [a + b for a, b in itertools.combinations(deck, 2) if a not in blocked and b not in blocked]
        w = np.ones(len(combos))
        for node, child, p, nb in history:
            if p != player:
                continue
            n = self.nodes[node]
            k = len(n["labels"])
            bks = np.array([self.bucket([c[:2], c[2:]] + board[:nb]) for c in combos])
            ok = bks >= 0
            idx = n["offset"] + np.where(ok, bks, 0)[:, None] * k + np.arange(k)[None, :]
            s = np.asarray(self.avg[idx.ravel()], dtype=float).reshape(-1, k)
            tot = s.sum(1)
            pr = np.where(tot > 0, s[:, child] / np.maximum(tot, 1e-12), 1.0 / k)
            w *= np.where(ok, pr, 0.0)
        return dict(zip(combos, w))

    @staticmethod
    def solver_range(w):
        top = max(w.values()) if w else 0
        if top <= 0:
            return None
        return ",".join(f"{c}:{x / top:.3f}" for c, x in w.items() if x / top >= 0.001)

    def resolve(self, state, i, scale, extra_opponent_range=None, p_exploit=0.0):
        """TexasSolver's strategy for player i's hand on the turn or river, solved from the start of the street with
        both players' blueprint ranges; {"BET 12": 0.3, ...} or None. extra_opponent_range / p_exploit: phase B, a
        read of the opponent blended into their range."""
        import poker_agents as pa

        if not os.path.exists(pa.SOLVER):
            return None
        board = [repr(c) for c in state.get_board_cards(0)] if state.board_count else []
        if len(board) < 4 or len(state.statuses) != 2:
            return None
        w = self.walk(state, scale, full=True)
        if w is None:
            return None
        node, bets, pot, history, street_acts = w
        hole = [repr(c) for c in state.hole_cards[i]]
        me = 1 - i  # our blueprint player
        # ranges at the start of this street: this street's actions are part of the solved tree
        before = [h for h in history if h[3] < len(board)]
        r_me = self.reach(before, me, board, set(board))
        r_op = self.reach(before, 1 - me, board, set(board) | set(hole))
        if extra_opponent_range and p_exploit > 0:
            tot_a = sum(r_op.values()) or 1.0
            tot_b = sum(extra_opponent_range.get(c, 0.0) for c in r_op) or 1.0
            r_op = {
                c: (1 - p_exploit) * x / tot_a + p_exploit * extra_opponent_range.get(c, 0.0) / tot_b
                for c, x in r_op.items()
            }
        deck = [r + s for r in "23456789TJQKA" for s in "cdhs"]
        mine = "".join(sorted(hole, key=deck.index))  # the range keys list cards in deck order
        r_me[mine] = max(r_me.get(mine, 0.0), max(r_me.values()) * 0.02)  # our hand must be in our range
        s_me, s_op = self.solver_range(r_me), self.solver_range(r_op)
        if not s_me or not s_op:
            return None
        oop_is_me = i == 0  # PokerKit heads-up: seat 0 (big blind) acts first after the flop
        ranges = (s_me, s_op) if oop_is_me else (s_op, s_me)
        street_pot = int(round(pot / scale))
        eff = min(state.stacks[j] + state.bets[j] for j in (0, 1))  # behind at the start of the street
        key = ("bp-resolve", tuple((h[0], h[1]) for h in before), tuple(board), i, mine)
        tree, _ = pa._solve(key, street_pot, eff, board, 0, None, ranges)
        nd = pa._walk(tree, street_acts) if tree else None
        strat = (nd or {}).get("strategy", {}).get("strategy", {}) if nd else {}
        probs = strat.get(hole[0] + hole[1]) or strat.get(hole[1] + hole[0])
        if not probs:
            return None
        return dict(zip(nd["actions"], probs))

    # ---- safe re-solving on the river (resolve.cpp: exact cards, the re-solve gadget) ----
    _RS = None

    def safe_strategies(self, state, i, scale):
        """Safe re-solve of this river spot for player i: {combo: probabilities over the node's actions} for every
        holding, or None. Solved at the current node with both players' blueprint ranges; the gadget keeps every
        opponent hand at or below its value against the blueprint (blueprint/resolve.cpp). Cached per spot."""
        board = [repr(c) for c in state.get_board_cards(0)] if state.board_count else []
        if len(board) != 5 or len(state.statuses) != 2:
            return None
        w = self.walk(state, scale, full=True)
        if w is None:
            return None
        node, _, _, history, _ = w
        if self.nodes[node]["kind"] != 0 or self.nodes[node]["player"] != 1 - i:
            return None
        # solved from the start of the river with the ranges there, so the opponent's bets on this street (bluffs
        # included) are inside the solve; we read our strategy at the current node. Re-solving from after the
        # opponent's bet trusts the blueprint's betting range, and LBR's bluffs exploited that (+176 bb/100)
        before = [h for h in history if h[3] < len(board)]
        on_river = [h for h in history if h[3] == len(board)]
        root = on_river[0][0] if on_river else node
        key = ("safe", tuple((h[0], h[1]) for h in history), tuple(board))
        if key in self._mapped:
            return self._mapped[key]
        if Blueprint._RS is None:
            lib = ctypes.CDLL(os.environ.get("BLUEPRINT_RESOLVE_LIB", os.path.join(ROOT, "blueprint", "libresolve.so")))
            lib.rs_solve.argtypes = (
                [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            )
            lib.rs_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
            if lib.rs_init(self.absdir.encode(), self.run.encode(), PURIFY, 0.02) != 0:
                raise RuntimeError("libresolve could not load the blueprint")
            Blueprint._RS = lib
        deck = [r + s for r in "23456789TJQKA" for s in "cdhs"]
        combos = [a + b for k, a in enumerate(deck) for b in deck[k + 1 :]]
        reach = [np.zeros(1326, np.float32), np.zeros(1326, np.float32)]
        for p in (0, 1):
            for c, x in self.reach(before, p, board, set(board)).items():
                reach[p][_COMBO_INDEX[c]] = x
        cards = np.array([card_int(c) for c in board], np.int32)
        out = np.zeros(1326 * 6, np.float32)
        iters = int(os.environ.get("BLUEPRINT_RESOLVE_ITERS", 500))
        rc = Blueprint._RS.rs_solve(
            root, node, cards.ctypes.data, reach[0].ctypes.data, reach[1].ctypes.data, iters, 1, out.ctypes.data
        )
        k = len(self.nodes[node]["labels"])
        res = None if rc else {c: out[j * 6 : j * 6 + k].astype(float) for j, c in enumerate(combos)}
        self._mapped[key] = res
        return res

    def resolve_all(self, state, i, scale):
        """One re-solve of this spot for every holding in player i's blueprint range (LBR needs the whole range):
        {combo: {action name: prob}} or None. Same solve as resolve(), without forcing a particular hand in."""
        import poker_agents as pa

        if not os.path.exists(pa.SOLVER):
            return None
        board = [repr(c) for c in state.get_board_cards(0)] if state.board_count else []
        if len(board) < 4 or len(state.statuses) != 2:
            return None
        w = self.walk(state, scale, full=True)
        if w is None:
            return None
        node, bets, pot, history, street_acts = w
        me = 1 - i
        before = [h for h in history if h[3] < len(board)]
        r_me = self.reach(before, me, board, set(board))
        r_op = self.reach(before, 1 - me, board, set(board))
        s_me, s_op = self.solver_range(r_me), self.solver_range(r_op)
        if not s_me or not s_op:
            return None
        ranges = (s_me, s_op) if i == 0 else (s_op, s_me)
        eff = min(state.stacks[j] + state.bets[j] for j in (0, 1))
        key = ("bp-resolve-all", tuple((h[0], h[1]) for h in before), tuple(board), i)
        tree, _ = pa._solve(key, int(round(pot / scale)), eff, board, 0, None, ranges)
        nd = pa._walk(tree, street_acts) if tree else None
        strat = (nd or {}).get("strategy", {}).get("strategy", {}) if nd else {}
        if not strat:
            return None
        out = {}
        for c, probs in strat.items():
            key2 = c if c in r_me else c[2:] + c[:2]
            out[key2] = dict(zip(nd["actions"], probs))
        return out

    # ---- minimum defence frequency ----
    MDF_FROM = float(os.environ.get("BLUEPRINT_MDF_FROM", 0.9))  # bet / pot at which the floor applies; 0 disables

    def own_range_strength(self, state, i, scale, board):
        """(our range from the blueprint's probabilities of our actions so far, strength per holding), cached per spot."""
        w = self.walk(state, scale, full=True)
        if w is None:
            return None
        history = w[3]
        key = ("own", tuple((h[0], h[1]) for h in history), tuple(board), i)
        if key not in self._mapped:
            import ranges

            r = self.reach(history, 1 - i, board, set(board))
            st = ranges.strengths(board)
            self._mapped[key] = (r, {c: st.get(ranges.norm(c), (0.5, 0, 0))[0] for c in r})
        return self._mapped[key]

    def defence_floor(self, state, i, scale, n, s, hole, board):
        """Facing a bet of MDF_FROM x the pot or more after the flop: a hand in the top P/(P+B) of our own range
        never folds (its fold probability moves to call). Otherwise a bettor's bluffs show a profit: LBR won with
        pot and all-in bets while the blueprint folded 61% / 71% / 79% on flop / turn / river (2026-11-08)."""
        if not self.MDF_FROM or not board or "fold" not in n["labels"]:
            return s
        to_call = state.checking_or_calling_amount
        pot_before = state.total_pot_amount - to_call
        if to_call < self.MDF_FROM * max(pot_before, 1):
            return s
        rs = self.own_range_strength(state, i, scale, board)
        if not rs:
            return s
        r, strength = rs
        import ranges

        mine = strength.get(next((c for c in r if {c[:2], c[2:]} == set(hole)), ""), None)
        if mine is None:
            mine = ranges.strengths(board).get(ranges.norm("".join(hole)), (0.5, 0, 0))[0]
        tot = sum(r.values()) or 1.0
        pct = sum(x for c, x in r.items() if strength[c] < mine) / tot
        mdf = pot_before / max(pot_before + to_call, 1)
        if pct < 1 - mdf:
            return s
        s = s.copy()
        fi, ci = n["labels"].index("fold"), n["labels"].index("call") if "call" in n["labels"] else None
        if ci is not None and s[fi] > 0:
            s[ci] += s[fi]
            s[fi] = 0.0
        return s

    # ---- our decision ----
    def distribution(self, state, i, scale, hole=None, walked=None, strategy=None):
        """[(kind, raise_to_in_harness_units or None, prob)] for player i, or None. `walked`: a walk() result to reuse
        (LBR asks for every holding at the same spot). `strategy`: probabilities over the node's actions to play
        instead of the table's (a safe re-solve); purified, with the defence floor unless BLUEPRINT_SAFE_FLOOR=0."""
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
        if strategy is not None and np.sum(strategy) > 0:
            s = np.array(strategy, dtype=float)[:k]
            s[s < PURIFY] = 0.0
            s = s / s.sum() if s.sum() > 0 else np.full(k, 1 / k)
            if os.environ.get("BLUEPRINT_SAFE_FLOOR", "1") == "1":  # the defence floor on top of the re-solve
                s = self.defence_floor(state, i, scale, n, s, hole, board)
        else:
            lo_i, hi_i = n["offset"] + b * k, n["offset"] + (b + 1) * k
            s = np.array(self.avg[lo_i:hi_i], dtype=float)
            s = s / s.sum() if s.sum() > 0 else np.full(k, 1 / k)
            if n["street"] > 0 and self.postflop == "current" and self.regret is not None:
                r = np.clip(np.array(self.regret[lo_i:hi_i], dtype=float), 0, None)
                if r.sum() > 0:
                    s = r / r.sum()
            s[s < PURIFY] = 0.0
            s = s / s.sum() if s.sum() > 0 else np.full(k, 1 / k)
            s = self.defence_floor(state, i, scale, n, s, hole, board)
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


def reply(state, i, resolve=None):
    """JSON reply like the other bots, or None (the caller falls back). resolve: True re-solves turns and rivers with
    TexasSolver (unsafe), "safe" re-solves rivers with our safe re-solver; None reads BLUEPRINT_RESOLVE (1 or safe)."""
    big = max(getattr(state, "blinds_or_straddles", (1, 2)))
    mode = os.environ.get("BLUEPRINT_RESOLVE") if resolve is None else resolve
    safe_mode = mode == "safe"  # our own safe river re-solve; True or "1": TexasSolver (unsafe)
    resolve = mode in (True, "1")
    if resolve:
        try:
            strat = get().resolve(state, i, 100 / big)
        except Exception:  # a failed solve falls back to the blueprint
            strat = None
        if strat:
            pick = _RNG.choices(list(strat), weights=list(strat.values()))[0]
            kind = pick.split()[0]
            to_call = state.checking_or_calling_amount
            if kind == "FOLD":
                return json.dumps({"action": "fold" if to_call else "check"})
            if kind in ("CHECK", "CALL") or not state.can_complete_bet_or_raise_to():
                return json.dumps({"action": "call" if to_call else "check"})
            lo = state.min_completion_betting_or_raising_to_amount
            hi = state.max_completion_betting_or_raising_to_amount
            to = max(lo, min(hi, state.bets[i] + int(float(pick.split()[1]))))
            return json.dumps({"action": "raise", "amount": int(to)})
    strategy = None
    if safe_mode:  # our own safe river re-solve (resolve.cpp)
        try:
            all_s = get().safe_strategies(state, i, 100 / big)
        except Exception:  # a failed solve falls back to the blueprint
            all_s = None
        if all_s:
            deck = [r + s for r in "23456789TJQKA" for s in "cdhs"]
            hole = sorted((repr(c) for c in state.hole_cards[i]), key=deck.index)
            strategy = all_s.get(hole[0] + hole[1])
    dist = get().distribution(state, i, 100 / big, strategy=strategy)
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
