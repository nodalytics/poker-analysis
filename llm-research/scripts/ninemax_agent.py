"""Play a 9-max network (ninemax_selfplay.py) as a seat in poker_selfplay / bot_sim: "nine#<run>" loads the moving
average network of blueprint/<run>/checkpoint.pt (or NINEMAX_DIR/<run>).

The adapter rebuilds the engine's view of the hand (ninemax_engine.Tables, one table) from the live PokerKit state by
replaying its operations: chips in per player, folds, all-ins, who has acted since the last full raise, the bet to
match, the last raise size, raises this street and pre-flop, the last aggressor and the street. Chips are rescaled so
that the big blind is 100, as in training. The cards become the training buckets through the same library
(bp_bucket: pre-flop class, then flop, turn and river buckets) and the same 0..1 strength.

The network's action is sampled from its policy over the legal slots and turned into a PokerKit action: fold,
check/call, a menu raise (the engine's raise-to total, converted to this street's raise-to and clamped to the legal
range), or all-in. Off-menu bets by opponents need no mapping: the inputs are continuous (pot, price, chips in).

Returns None, so the caller falls back to another bot, when the game is not the one the network was trained for:
a table other than 9 players, or stacks other than 100 big blinds. Run bot_sim with SIM_STACKS=100,100.
Needs PyTorch: run in .venv-gpu (CPU is enough; one thread per worker).
"""

import json
import os
import random

import torch
from pokerkit import BlindOrStraddlePosting, BoardDealing, CheckingOrCalling, CompletionBettingOrRaisingTo, Folding

from ninemax_engine import BB, STACK, Dealer, Menu, Tables
from ninemax_selfplay import FEAT, NineNet, features, masked, set_mode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.environ.get("NINEMAX_LIB", os.path.join(ROOT, "blueprint", "libbp5.so"))
ABS = os.environ.get("NINEMAX_ABS", os.path.join(ROOT, "blueprint", "out"))
RUNS = os.environ.get("NINEMAX_DIR", os.path.join(ROOT, "blueprint"))
PLAYERS = 9
_CACHE = {}
_RNG = random.Random(9009)


def _card(c):
    return "23456789TJQKA".index(str(c.rank)[0]) * 4 + "cdhs".index(str(c.suit)[0])


class Seat:
    def __init__(self, run):
        torch.set_num_threads(1)
        self.B = int(open(os.path.join(ABS, "meta.txt")).read().split()[1])
        self.dealer = Dealer(LIB, ABS, PLAYERS, self.B, "cpu")
        d = run if os.path.isabs(run) else os.path.join(RUNS, run)
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.mode = (
            cfg.get("stacks", "100"),
            cfg.get("tourney", False),
            cfg.get("depths"),
            not cfg.get("no_open_limp", False),
            cfg.get("ante", 0.0),
            cfg.get("short", 0.0),
            cfg.get("seats"),
            cfg.get("bounty", 0.0),
        )
        self.menu = Menu(cfg.get("menu", "medium"), "cpu")
        self.net = NineNet(cfg["nfeat"], cfg["actions"], cfg.get("buckets", self.B), arch=cfg.get("arch", "mlp"))
        st = torch.load(os.path.join(d, "checkpoint.pt"), map_location="cpu", weights_only=True)
        self.net.load_state_dict(st["avg"])
        self.net.eval()

    def buckets(self, hole, board):
        """[4] buckets (pre-flop class, flop, turn, river; 0 for streets not dealt) and the current street."""
        import ctypes

        out = []
        for n in (0, 3, 4, 5):
            if len(board) < n:
                out.append(0)
                continue
            cs = hole + board[:n]
            arr = (ctypes.c_int * len(cs))(*cs)
            out.append(max(self.dealer.lib.bp_bucket(arr, len(cs)), 0))
        street = {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]
        return out, street

    def table(self, state, i):
        """The engine's view of PokerKit's hand (one table), or None if it is not a 9-handed 100bb game."""
        P = state.player_count
        k = BB / max(state.blinds_or_straddles)
        set_mode(*self.mode)
        starts = [round(s * k) for s in state.starting_stacks]
        if P != PLAYERS or FEAT["tourney"]:  # a tournament network needs the payouts, which a cash table has not
            return None
        if any(s != STACK for s in starts) if FEAT["stacks"] is None else min(starts) < 2 * BB:
            return None  # a 100bb network at other depths, or a stack under 2bb
        contrib, base = [0] * P, [0] * P  # chips in this hand, and before this street (engine units)
        folded, acted = [False] * P, [False] * P
        bet, last_raise, raises, pre_raises, aggressor, street, n_board = 0, BB, 0, 0, -1, 0, 0
        for op in state.operations:
            if isinstance(op, BlindOrStraddlePosting):
                contrib[op.player_index] += round(op.amount * k)
                bet = max(bet, contrib[op.player_index])
            elif isinstance(op, BoardDealing):
                n_board += len(op.cards)
                s = {3: 1, 4: 2, 5: 3}.get(n_board, street)
                if s != street:
                    street, raises, last_raise, base = s, 0, BB, list(contrib)
                    acted = [False] * P
            elif isinstance(op, Folding):
                folded[op.player_index] = acted[op.player_index] = True
            elif isinstance(op, CheckingOrCalling):
                contrib[op.player_index] += round(op.amount * k)
                acted[op.player_index] = True
            elif isinstance(op, CompletionBettingOrRaisingTo):
                p = op.player_index
                total = base[p] + round(op.amount * k)
                inc = total - bet
                if inc >= last_raise:  # a full raise reopens the betting
                    raises += 1
                    pre_raises += street == 0
                    last_raise = inc
                    acted = [False] * P
                aggressor = p if inc > 0 else aggressor
                contrib[p], bet = total, max(bet, total)
                acted[p] = True
        T = Tables(1, P, self.menu, "cpu", stacks=torch.tensor([starts]))
        T.contrib[0] = torch.tensor(contrib)
        T.folded[0] = torch.tensor(folded)
        T.allin[0] = torch.tensor([c >= s and not f for c, f, s in zip(contrib, folded, starts)])
        T.acted[0] = torch.tensor(acted)
        for name, v in (("bet", bet), ("last_raise", last_raise), ("raises", raises), ("pre_raises", pre_raises),
                        ("aggressor", aggressor), ("street", street), ("to_act", (i - 1) % P), ("actor", i)):  # fmt: skip
            getattr(T, name)[0] = v
        T.done[0] = False
        return T, base, k

    def act(self, state, i):
        view = self.table(state, i)
        if view is None:
            return None
        T, base, k = view
        hole = [_card(c) for c in state.hole_cards[i]]
        board = [_card(c) for c in state.get_board_cards(0)] if state.board_count else []
        bk, street = self.buckets(hole, board)
        if street != int(T.street[0]):
            return None  # the board and the operations disagree: not a hand this adapter understands
        g, ii = torch.zeros(1, dtype=torch.long), torch.tensor([i])
        feat, legal = features(T, g, ii)
        _, t = T.legal(g, ii)
        can_raise = state.can_complete_bet_or_raise_to()
        legal[0, 0] &= state.can_fold()
        legal[0, 2:] &= can_raise
        s = self.dealer.pre_strength[bk[0]] if street == 0 else torch.tensor(bk[street] / (self.B - 1))
        with torch.no_grad():
            lg, _ = self.net(feat, torch.tensor([bk]), torch.tensor([street]), s.reshape(1).float())
        probs = torch.softmax(masked(lg, legal), 1)[0].tolist()
        a = _RNG.choices(range(len(probs)), weights=probs)[0]
        S = self.menu.S
        if a == 0:
            return json.dumps({"action": "fold"})
        if a == 1 or not can_raise:
            return json.dumps({"action": "call" if state.checking_or_calling_amount else "check"})
        lo, hi = state.min_completion_betting_or_raising_to_amount, state.max_completion_betting_or_raising_to_amount
        if a == S + 2:
            amt = hi
        else:  # the engine's raise-to total for the hand -> PokerKit's raise-to for this street
            amt = max(lo, min(hi, round((float(t[0, a - 2]) - base[i]) / k)))
        return json.dumps({"action": "raise", "amount": amt})


def reply(state, i, ctx, name):
    run = name.split("#", 1)[1]
    if run not in _CACHE:
        _CACHE[run] = Seat(run)
    return _CACHE[run].act(state, i)
