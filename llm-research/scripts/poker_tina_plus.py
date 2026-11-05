"""Tina+: the tight-aggressive simulated human with her two weaknesses fixed.

1. Stack awareness: at 20 big blinds or less she plays the push/fold equilibrium (results/preflop): shove or fold
   when the pot is unopened, and call an all-in only with the chart's call range for her seat against that shover.
2. Harder to read: preflop raise sizes vary (2.2-3.5 big blinds), she sometimes flat-calls in position and slow-plays
   monsters, and after the flop every bet is 66% of the pot whatever she holds, with bluffs taken mainly from draws at
   about one bluff per two value bets (the balance for that size), so her sizing no longer gives her hand away.
The original Tina is unchanged: she is a fixed anchor in the running ladders.
"""

import json
import os
import random

from pokerkit import StandardHighHand

import poker_selfplay as ps

NAME = "tinaplus"
KIND = "tight-aggressive, stack-aware and balanced (Tina+), simulated human"
_RNG = random.Random(4040)
_PRE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "preflop")
_CHARTS = {}


def _charts(name):
    if name not in _CHARTS:
        try:
            _CHARTS[name] = json.load(open(os.path.join(_PRE, name)))["charts"]
        except OSError:
            _CHARTS[name] = []
    return _CHARTS[name]


def hand_class(hole):
    order = "23456789TJQKA"
    (r1, s1), (r2, s2) = [(repr(c)[0], repr(c)[1]) for c in hole]
    hi, lo = sorted((r1, r2), key=order.index, reverse=True)
    return hi + lo if hi == lo else hi + lo + ("s" if s1 == s2 else "o")


def position(i, n):
    """PokerKit seats: 0 small blind, 1 big blind, last the button; the multi-way charts name the rest pos0, pos1..."""
    return "small blind" if i == 0 else "big blind" if i == 1 else "button" if i == n - 1 else f"pos{i - 2}"


def push_fold(state, i, bb):
    """'shove', 'call', 'fold' or None (not a push/fold spot) from the equilibrium charts."""
    n = len(state.statuses)
    eff = (state.stacks[i] + state.bets[i]) / bb
    if eff > 20:
        return None
    hc = hand_class(state.hole_cards[i])
    top = max(state.bets)
    raisers = [j for j in range(n) if state.bets[j] == top and j != i]
    if top <= bb:  # unopened or limped
        if position(i, n) == "big blind":
            return None
        if n == 2:
            c = min(_charts("push_fold_nash.json"), key=lambda x: abs(x["stack_bb"] - eff), default=None)
            rng = c["push"] if c else []
        else:
            cs = [c for c in _charts("push_fold_multiway_ante.json") if c["players"] == min(n, 6)]
            c = min(cs, key=lambda x: abs(x["stack_bb"] - eff), default=None)
            rng = c["shove"].get(position(i, min(n, 6)), {}).get("range", []) if c else []
        return "shove" if hc in rng else "fold"
    shover = raisers[0] if raisers else None  # facing a raise: treat it as the all-in it nearly always is here
    if shover is None:
        return None
    if n == 2:
        c = min(_charts("push_fold_nash.json"), key=lambda x: abs(x["stack_bb"] - eff), default=None)
        rng = c["call"] if c else []
    else:
        cs = [c for c in _charts("push_fold_multiway_ante.json") if c["players"] == min(n, 6)]
        c = min(cs, key=lambda x: abs(x["stack_bb"] - eff), default=None)
        key = f"{position(i, min(n, 6))} vs {position(shover, min(n, 6))}"
        rng = c["call"].get(key, {}).get("range", []) if c else []
    return "call" if hc in rng else "fold"


def _draws(hole, board):
    cards = [repr(c) for c in hole] + board
    flush = max(sum(c[1] == s for c in cards) for s in "cdhs") == 4
    ranks = sorted({"23456789TJQKA".index(c[0]) for c in cards})
    run = best = 1
    for a, b in zip(ranks, ranks[1:]):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return flush or best >= 4


def reply(state, i):
    rng = _RNG
    bb = max(getattr(state, "blinds_or_straddles", (ps.SB, ps.BB)))
    to_call, pot = state.checking_or_calling_amount, state.total_pot_amount
    can_raise = state.can_complete_bet_or_raise_to()
    lo = state.min_completion_betting_or_raising_to_amount if can_raise else 0
    hi = state.max_completion_betting_or_raising_to_amount if can_raise else 0
    call = json.dumps({"action": "call" if to_call else "check"})
    fold = json.dumps({"action": "fold" if to_call else "check"})

    def raise_to(x):
        return json.dumps({"action": "raise", "amount": int(max(lo, min(hi, x)))}) if can_raise else call

    board = [repr(c) for c in state.get_board_cards(0)] if state.board_count else []
    if not board:
        pf = push_fold(state, i, bb)
        if pf == "shove":
            return raise_to(hi)
        if pf == "call":
            return call
        if pf == "fold":
            return fold
        c = ps._chen(list(state.hole_cards[i]))
        n = len(state.statuses)
        in_position = i in (n - 1, n - 2)
        if c >= 12 and rng.random() < 0.15:  # slow-play a monster now and then
            return call
        if c >= 8:
            return raise_to(state.bets[i] + to_call + int(bb * rng.uniform(2.2, 3.5)))
        if c >= 6 and in_position and to_call <= 3 * bb and rng.random() < 0.3:
            return call
        return fold
    label = StandardHighHand.from_game("".join(repr(c) for c in state.hole_cards[i]), "".join(board)).entry.label
    strength = list(type(label)).index(label)
    bet = state.bets[i] + to_call + int(0.66 * (pot + to_call))  # one size for value and bluffs alike
    draw = _draws(state.hole_cards[i], board)
    if strength >= 2:
        return call if rng.random() < 0.15 else raise_to(bet)  # trap 15%
    if strength == 1:
        if not to_call:
            return raise_to(bet) if rng.random() < 0.5 else call
        return call if to_call <= 0.66 * pot else fold
    if draw:  # bluff with equity: about one bluff per two value bets at this size
        if not to_call:
            return raise_to(bet) if rng.random() < 0.5 else call
        return call if to_call <= pot / 3 else fold
    if not to_call:
        return raise_to(bet) if rng.random() < 0.05 else call
    return fold
