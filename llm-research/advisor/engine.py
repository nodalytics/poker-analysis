"""The advice engine shared by session review, the trainer and spot study.

advise(state, i, ctx) answers "what does the solver stack play here?" as a mix over fold / call (check-call) /
raise, with the source and a reason, using the same layers as the solver bot (scripts/poker_agents.solver_play):
push/fold chart, short-stack defence, deep pre-flop strategy, TexasSolver on heads-up turns and rivers (only when a
solver binary is configured; with hand reading from the betting), the flop model on heads-up flops, and Monte Carlo
equity elsewhere.

Table(...) builds a PokerKit state from a hand's description and replays its actions while keeping the context the
advisors need (this street's actions, the pot at its start, the hand log for hand reading).

Never used at a live real-money table: it reads finished hands, a practice table, or a spot typed in for study.
"""

import itertools
import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))

from pokerkit import Automation, Mode, NoLimitTexasHoldem  # noqa: E402

import poker_agents as pa  # noqa: E402
import ranges  # noqa: E402

AUTO = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.CARD_BURNING,
    Automation.RUNOUT_COUNT_SELECTION,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)
DECK = [r + s for r in "23456789TJQKA" for s in "cdhs"]
STREETS = ["preflop", "flop", "turn", "river"]
_IDS = itertools.count(1)


class Table:
    """One hand, in PokerKit seat order (0 small blind, 1 big blind, ..., last the button; heads-up 0 is the big
    blind and 1 the button, which posts the small blind). stacks and blinds are integers in one unit (cents, or
    chips). holes: per seat a 4-character string or None (unknown cards are dealt at random)."""

    def __init__(self, stacks, blinds, holes, seed=0, pay=True, exclude=()):
        """pay=False stops before the pot is awarded (a replay's stand-in cards must not decide who wins), so
        start stack minus stack is what each player put in."""
        self.n = len(stacks)
        auto = AUTO if pay else tuple(a for a in AUTO if a not in (Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING))
        self.state = NoLimitTexasHoldem.create_state(
            auto, True, 0, tuple(blinds), max(blinds), tuple(stacks), self.n, mode=Mode.CASH_GAME
        )
        rng = random.Random(seed)
        known = {h[k : k + 2] for h in holes if h for k in (0, 2)} | set(exclude)  # exclude: the board to come
        free = [c for c in DECK if c not in known]
        rng.shuffle(free)
        for h in holes:
            if h:
                self.state.deal_hole(h)
            else:
                self.state.deal_hole(free.pop() + free.pop())
        self.deck = free  # the cards nobody holds, for dealing the board in a live game (deal_street)
        self.street = 0
        self.street_actions = []
        self.street_pot = self.state.total_pot_amount
        self.log = []
        self.hand_id = next(_IDS) * 1000 + random.randrange(1000)

    def deal(self, cards):
        """Deal board cards ('Qs7h2c' or a list); the next street starts."""
        cs = cards if isinstance(cards, str) else "".join(cards)
        self.state.deal_board(cs)
        self.street = STREETS.index({3: "flop", 4: "turn", 5: "river"}[len(list(self.state.get_board_cards(0)))])
        self.street_actions, self.street_pot = [], self.state.total_pot_amount

    def deal_street(self):
        """Deal the next street from the deck (live play: trainer)."""
        n = 3 if not list(self.state.get_board_cards(0)) else 1
        self.deal("".join(self.deck.pop() for _ in range(n)))

    def needs_board(self):
        return bool(self.state.status) and self.state.actor_index is None and self.state.can_deal_board()

    def actor(self):
        return self.state.actor_index

    def ctx(self):
        return {
            "hand": self.hand_id,
            "street_actions": list(self.street_actions),
            "street_pot": self.street_pot,
            "seats": [f"p{k}" for k in range(self.n)],
            "profile": None,
            "recent": None,
            "log": list(self.log),
        }

    def act(self, kind, to=None):
        """kind: fold, check, call, raise (to = raise-to total on this street). Applies to the player to act."""
        s, i = self.state, self.state.actor_index
        raises = sum(1 for k, _ in self.street_actions if k == "raise")
        to_call, before = s.checking_or_calling_amount, s.bets[i]
        code = 0 if kind == "fold" else 2 if kind == "raise" else 1
        self.log.append(ranges.entry(s, i, STREETS[self.street], raises, code))
        if kind == "fold" and s.can_fold():
            s.fold()
            self.street_actions.append(("fold", 0))
        elif kind == "raise" and s.can_complete_bet_or_raise_to():
            lo, hi = s.min_completion_betting_or_raising_to_amount, s.max_completion_betting_or_raising_to_amount
            s.complete_bet_or_raise_to(max(lo, min(hi, int(to if to is not None else hi))))
            self.street_actions.append(("raise", s.bets[i] - before))
        else:
            s.check_or_call()
            self.street_actions.append(("call" if to_call else "check", to_call))


def _mix(fold=0.0, call=0.0, raise_=0.0):
    t = fold + call + raise_ or 1.0
    return {"fold": fold / t, "call": call / t, "raise": raise_ / t}


def advise(state, i, ctx, use_solver=True):
    """{'source', 'mix': {fold, call, raise}, 'detail': {...}, 'why'} for player i to act."""
    to_call = state.checking_or_calling_amount
    bb = max(getattr(state, "blinds_or_straddles", (1, 2)))
    pf = pa.push_fold_advice(state, i)
    if pf:
        shove = pf["shove"]
        return {
            "source": "push/fold chart",
            "mix": _mix(raise_=1.0) if shove else (_mix(fold=1.0) if to_call else _mix(call=1.0)),
            "detail": {"all-in": 1.0} if shove else {},
            "why": f"{pf['hand']} at {pf['stack_bb']}bb: {'shove' if shove else 'fold'} ({pf['chart']})",
        }
    sd = pa.short_defence(state, i)
    if sd:
        a = sd["action"]
        return {
            "source": "short-stack defence",
            "mix": _mix(**{"raise_" if a == "raise" else a: 1.0}),
            "detail": {},
            "why": sd["why"],
        }
    deep = pa.preflop_deep_decision(state, i, ctx)
    if deep:
        a = {"check": "call"}.get(deep["action"], deep["action"])
        amt = deep.get("amount")
        return {
            "source": "pre-flop strategy",
            "mix": _mix(**{"raise_" if a == "raise" else a: 1.0}),
            "detail": {f"raise to {amt / bb:.1f}bb": 1.0} if amt else {},
            "why": deep["why"],
        }
    ob = pa.overbet_defence(state, i, ctx)
    if ob:
        return {
            "source": "overbet defence",
            "mix": _mix(**{ob["action"]: 1.0}),
            "detail": {},
            "why": ob["why"],
        }
    if use_solver and os.path.exists(pa.SOLVER):
        adv = pa.solver_strategy(state, i, ctx)
        if adv.get("available"):
            f = c = r = 0.0
            detail = {}
            for name, p in adv["strategy"].items():
                k = name.split()[0]
                if k == "FOLD":
                    f += p
                elif k in ("CHECK", "CALL"):
                    c += p
                else:
                    r += p
                    detail[f"{k.lower()} {float(name.split()[1]) / bb:.1f}bb"] = p
            return {
                "source": "TexasSolver",
                "mix": _mix(f, c, r),
                "detail": detail,
                "why": f"solved live; ranges: {adv['assumed_ranges']}",
            }
    sized = pa.sized_flop_strategy(state, i, ctx)
    if sized.get("available"):
        s = sized["strategy"]
        raise_keys = [k for k in s if k not in ("check/call", "fold")]
        faced = f", facing {sized['bet_faced']:.0%} pot" if sized["bet_faced"] else ""
        return {
            "source": "flop model (bet sizes)",
            "mix": _mix(s.get("fold", 0), s.get("check/call", 0), sum(s[k] for k in raise_keys)),
            "detail": {k: s[k] for k in raise_keys if s[k] > 0.005},
            "why": f"predicts the solver's flop play ({sized['node'].replace('_', ' ')}{faced})",
        }
    flop = pa.flop_model_strategy(state, i, ctx)
    if flop.get("available"):
        s = flop["strategy"]
        return {
            "source": "flop model",
            "mix": _mix(
                s.get("fold", 0), s.get("check/call", 0), s.get("bet or raise about 2/3 pot", 0) + s.get("all-in", 0)
            ),
            "detail": {k: v for k, v in s.items() if k in ("bet or raise about 2/3 pot", "all-in")},
            "why": f"predicts the solver's flop play ({flop['node'].replace('_', ' ')})",
        }
    hole = pa._cards(state.hole_cards[i])
    board = pa._cards(state.get_board_cards(0)) if state.board_count else []
    others = [j for j, s in enumerate(state.statuses) if s and j != i]
    live = len(others)
    eq, against = None, f"{live} random hand(s)"
    if live == 1 and ctx.get("log") and any(e["player"] == others[0] for e in ctx["log"]) and ranges.available():
        eq = equity_vs_range(state, i, others[0], ctx["log"])
        against = "the range their betting implies (hand reading)"
    if eq is None:
        eq = pa.equity_vs_random(hole, board, max(1, live))
    pot = state.total_pot_amount
    price = to_call / (pot + to_call) if to_call else 0.0
    fair = 1 / (live + 1)
    if state.can_complete_bet_or_raise_to() and eq > min(0.85, fair * 1.8):
        m = _mix(raise_=1.0)
    elif not to_call or eq > price:
        m = _mix(call=1.0)
    else:
        m = _mix(fold=1.0)
    return {
        "source": "Monte Carlo equity",
        "mix": m,
        "detail": {"equity": eq, "price": price},
        "why": f"equity {eq:.0%} against {against}; the price is {price:.0%} (rough: ignores later betting)",
    }


def equity_vs_range(state, i, j, log, sims=1500, seed=0):
    """Hero's equity against player j's hand-reading posterior (ranges.py), by weighted Monte Carlo."""
    import eval7

    rng = random.Random(seed)
    hole = [repr(c) for c in state.hole_cards[i]]
    board = [repr(c) for c in state.get_board_cards(0)] if state.board_count else []
    w = ranges.weights(log, j, set(hole) | set(board))
    combos = [c for c, x in w.items() if x > 1e-4]
    if not combos:
        return None
    deck = [c for c in DECK if c not in set(hole) | set(board)]
    card = {c: eval7.Card(c) for c in DECK}
    me = [card[c] for c in hole]
    bc = [card[c] for c in board]
    draws = rng.choices(combos, weights=[w[c] for c in combos], k=sims)  # one weighted draw for all sims
    need = 5 - len(board)
    win = 0.0
    for v in draws:
        rest = []
        while len(rest) < need:  # rejection sampling: far faster than rebuilding the deck each time
            c = deck[rng.randrange(len(deck))]
            if c != v[:2] and c != v[2:] and c not in rest:
                rest.append(c)
        b = bc + [card[c] for c in rest]
        a, o = eval7.evaluate(me + b), eval7.evaluate([card[v[:2]], card[v[2:]]] + b)
        win += 1.0 if a > o else 0.5 if a == o else 0.0
    return win / sims
