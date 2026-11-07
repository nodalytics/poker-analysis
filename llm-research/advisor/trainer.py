"""Trainer: play no-limit hold'em against our bots, with the engine's advice on request and a verdict after each move.

Game logic only (TrainerGame); the window is in app.py. Opponents (poker_selfplay / poker_agents):
  max      maniac: raises and bluffs a lot          carl     calling station: calls too much
  nora     nit: plays few hands, folds to bets      bob      recreational: loose, passive
  tina     tight-aggressive regular                 monte    Monte Carlo equity bot
  solverbot  the solver stack (beat Slumbot)        adaptbot  solver stack bent toward what it reads in you
Blinds 1/2 units (1 big blind = 2 units). Stacks reset every hand (cash game with reloads); 'short' deals 5-20 big
blinds each hand, 'deep' 100. Results are tracked in big blinds.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advisor import engine  # noqa: E402

import poker_selfplay as ps  # noqa: E402  (scripts/ is on the path via engine)

pa = engine.pa
BOTS = {
    "max": "Max (maniac)",
    "carl": "Carl (calling station)",
    "nora": "Nora (nit)",
    "bob": "Bob (recreational)",
    "tina": "Tina (tight-aggressive)",
    "monte": "Monte (equity bot)",
    "solverbot": "Solver bot",
    "adaptbot": "Adaptive bot",
}
HERO = "You"


class TrainerGame:
    def __init__(self, opponents, depth="deep", seed=None):
        self.opponents = list(opponents)
        self.order = [HERO] + self.opponents
        self.depth = depth
        self.rng = random.Random(seed)
        self.hand_no = 0
        self.net = {p: 0.0 for p in self.order}
        self.decisions = {"agrees": 0, "questionable": 0, "mistake": 0}
        blank = {"hands": 0, "vpip": 0, "post_raises": 0, "post_calls": 0, "faced": 0, "folded": 0}
        self.profile = {p: dict(blank) for p in self.order}
        self.history = []  # text log of the current hand
        self.table = None

    # ---- hand flow ----
    def new_hand(self):
        self.hand_no += 1
        k = (self.hand_no - 1) % len(self.order)
        seats = self.order[k:] + self.order[:k]  # PokerKit order: small blind, big blind, ..., button
        if len(seats) == 2:
            seats = [seats[1], seats[0]]  # heads-up: big blind first, then the button
        self.seats = seats
        self.start = [
            2 * (self.rng.randint(5, 20) if self.depth == "short" else 100) for _ in seats
        ]  # units; 2 per big blind
        self.table = engine.Table(self.start, (1, 2), [None] * len(seats), seed=self.rng.randrange(10**9))
        self.folded = set()
        self.voluntary = set()
        self.history = [f"Hand {self.hand_no}: " + ", ".join(f"{p} {s / 2:g}bb" for p, s in zip(seats, self.start))]
        self.history.append(f"Your cards: {self.cards(seats.index(HERO))}")
        self._advance()

    def cards(self, i):
        return " ".join(repr(c) for c in self.table.state.hole_cards[i])

    def board(self):
        s = self.table.state
        return [repr(c) for c in s.get_board_cards(0)] if s.board_count else []

    def _advance(self):
        """Deal boards as needed until someone must act or the hand is over."""
        while self.table.needs_board():
            self.table.deal_street()
            self.history.append(f"*** {engine.STREETS[self.table.street].upper()} *** {' '.join(self.board())}")

    def over(self):
        return not self.table.state.status

    def to_act(self):
        """The name of the player to act, or None when the hand is over."""
        if self.over():
            return None
        i = self.table.actor()
        return None if i is None else self.seats[i]

    def ctx(self):
        c = self.table.ctx()
        c["seats"], c["profile"], c["recent"] = self.seats, self.profile, self.profile
        return c

    # ---- moves ----
    def legal(self):
        s = self.table.state
        can_raise = s.can_complete_bet_or_raise_to()
        return {
            "to_call": s.checking_or_calling_amount,
            "can_fold": s.can_fold() and s.checking_or_calling_amount > 0,
            "can_raise": can_raise,
            "min": s.min_completion_betting_or_raising_to_amount if can_raise else None,
            "max": s.max_completion_betting_or_raising_to_amount if can_raise else None,
            "pot": s.total_pot_amount,
            "bet": s.bets[s.actor_index],
        }

    def advice(self):
        i = self.table.actor()
        return engine.advise(self.table.state, i, self.ctx())

    def hero_act(self, kind, to=None, coach=True):
        """kind: fold / call (check-call) / raise (to = raise-to). Returns the engine's verdict when coach is on."""
        verdict = None
        if coach:
            adv = self.advice()
            from advisor.review import grade

            p = adv["mix"].get(kind, 0.0)
            g = grade(p)
            if adv["source"] == "Monte Carlo equity" and g == "mistake":
                g = "questionable"
            self.decisions[g] += 1
            verdict = {"grade": g, "pct": p, "advice": adv}
        self._apply(HERO, kind, to)
        return verdict

    def bot_act(self):
        name = self.to_act()
        i = self.table.actor()
        s = self.table.state
        if name in ps.PERSONAS:
            text = ps.bot_reply(name, s, i)
        elif name == "solverbot":
            text = pa.solver_play(s, i, self.ctx()) or pa.monte_reply(s, i)
        elif name == "adaptbot":
            text = pa.adapt_play(s, i, self.ctx())
        else:
            text = pa.monte_reply(s, i)
        act, amt = ps.parse(text)
        if act in ("all-in", "all in"):
            act, amt = "raise", s.max_completion_betting_or_raising_to_amount
        kind = {"check": "call", "bet": "raise"}.get(act, act)
        if kind not in ("fold", "call", "raise"):
            kind = "call"
        self._apply(name, kind, amt)

    def _apply(self, name, kind, to):
        s = self.table.state
        i = self.table.actor()
        to_call = s.checking_or_calling_amount
        if kind == "fold" and not to_call:
            kind = "call"  # never fold when checking is free
        if kind == "raise" and not s.can_complete_bet_or_raise_to():
            kind = "call"
        street = engine.STREETS[self.table.street]
        self.table.act(kind, to)
        if kind == "fold":
            self.folded.add(name)
            txt = "folds"
        elif kind == "raise":
            bet_first = not to_call and not any(k == "raise" for k, _ in self.table.street_actions[:-1])
            txt = f"{'bets' if bet_first else 'raises to'} {s.bets[i] / 2:g}bb" if s.bets[i] else "raises"
            if street == "preflop":
                self.voluntary.add(name)
        else:
            txt = f"calls {to_call / 2:g}bb" if to_call else "checks"
            if street == "preflop" and to_call:
                self.voluntary.add(name)
        if street != "preflop":
            pr = self.profile[name]
            pr["post_raises"] += kind == "raise"
            pr["post_calls"] += kind == "call" and bool(to_call)
            if to_call:
                pr["faced"] += 1
                pr["folded"] += kind == "fold"
        self.history.append(f"{name} {txt}")
        self._advance()
        if self.over():
            self._finish()

    def _finish(self):
        s = self.table.state
        for p in self.seats:
            self.profile[p]["hands"] += 1
            self.profile[p]["vpip"] += p in self.voluntary
        live = [p for p in self.seats if p not in self.folded]
        if len(live) > 1:
            self.history.append("Showdown: " + "; ".join(f"{p} {self.cards(self.seats.index(p))}" for p in live))
        for k, p in enumerate(self.seats):
            d = (s.stacks[k] - self.start[k]) / 2
            self.net[p] += d
            if d > 0:
                self.history.append(f"{p} wins {d:g}bb")

    def showdown_names(self):
        return [p for p in self.seats if p not in self.folded] if self.over() else []
