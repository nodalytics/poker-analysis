"""Local best response (LBR; Lisý & Bowling 2017) against our solver bot: a lower bound on how beatable it is, and
where it leaks.

Usage: uv run python scripts/lbr.py OUTDIR HANDS [WORKERS] [STACK_BB]
Heads-up no-limit, blinds 1/2 units, STACK_BB big blinds (default 100), positions alternate.

LBR keeps the bot's range: a weight for each of the 1,326 holdings, multiplied after every bot action by the
probability the bot takes that action with that holding. At each of its own decisions after the flop LBR picks the
action with the highest value, assuming the hand is then checked down:
    fold    0
    call    wp x (pot + to_call) - to_call
    bet b   fp x pot + (1 - fp) x (wp' x (pot + 2b... ) - b)   with fp the share of the range that folds to the bet
            and wp' the equity against the hands that continue (bot responses from its own policy)
Pre-flop LBR only calls, as in the paper. Its expected winnings are a lower bound on the bot's exploitability.

The bot's policy for every holding, mirroring solver_play:
  pre-flop   push/fold chart, short-stack defence, deep pre-flop strategy (per hand class)
  flop       the flop model where it applies (all holdings in one batch)
  turn/river TexasSolver, solved on the bot's tracked range against LBR's full range (one solve per spot)
  elsewhere  a strength rule standing in for the Monte Carlo fallback (bet with the top 20% of holdings on the
             board, call when the holding's rank beats the price)
The bot's real cards are played from the same policy, so the range LBR tracks is the bot's actual one.
Reports LBR's winnings (bb/100, card luck removed as in aivat_lite.py), and per street how often the bot folds to
LBR's bets.
"""

import copy
import json
import math
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advisor import engine  # noqa: E402

pa, ranges = engine.pa, engine.ranges
BB = 2
FOLD, CALL = "fold", "call"


def card_objs(h):
    from pokerkit import Card

    return list(Card.parse(h))


class Hand:
    def __init__(self, stack_bb, lbr_seat, rng):
        self.rng = rng
        deck = engine.DECK[:]
        rng.shuffle(deck)
        self.holes = [deck.pop() + deck.pop(), deck.pop() + deck.pop()]
        self.deck = deck
        self.t = engine.Table([stack_bb * BB] * 2, (1, 2), self.holes, seed=rng.randrange(10**9))
        self.t.deck = deck
        self.lbr, self.bot = lbr_seat, 1 - lbr_seat
        self.reach = {h: 1.0 for h in ranges.ALL if not ({h[:2], h[2:]} & set(self.holes[self.lbr]))}
        self.events = []

    # ---- the bot's policy for every holding ----
    def with_hole(self, state, i, combo):
        s = copy.deepcopy(state)
        s.hole_cards[i] = card_objs(combo)
        return s

    def policy(self, state, ctx):
        """combo -> list of (kind, raise_to, prob) for the bot at this state."""
        board = [repr(c) for c in state.get_board_cards(0)] if state.board_count else []
        combos = [h for h, w in self.reach.items() if w > 1e-9 and not ({h[:2], h[2:]} & set(board))]
        if not board:
            return self.preflop(state, ctx, combos)
        ob = self.overbet(state, ctx, combos, board)
        if ob is not None:
            return ob
        if len(board) == 3:
            p = self.flop(state, ctx, combos, board)
            if p is not None:
                return p
        elif os.path.exists(pa.SOLVER):
            p = self.solve(state, ctx, combos, board)
            if p is not None:
                return p
        return self.strength_rule(state, combos, board)

    def preflop(self, state, ctx, combos):
        by_class = {}
        out = {}
        for h in combos:
            c = ranges.cls(h)
            if c not in by_class:
                s = self.with_hole(state, self.bot, h)
                pf = pa.push_fold_advice(s, self.bot)
                if pf:
                    hi = s.max_completion_betting_or_raising_to_amount if s.can_complete_bet_or_raise_to() else None
                    d = [("raise", hi, 1.0)] if pf["shove"] and hi else [(FOLD, None, 1.0)]
                else:
                    sd = pa.short_defence(s, self.bot)
                    dd = sd or pa.preflop_deep_decision(s, self.bot, ctx)
                    if dd:
                        a = {"check": CALL}.get(dd["action"], dd["action"])
                        d = [(a, dd.get("amount"), 1.0)]
                    else:
                        d = [(CALL, None, 1.0)]
                by_class[c] = d
            out[h] = by_class[c]
        return out

    def overbet(self, state, ctx, combos, board):
        """Mirror of poker_agents.overbet_defence for every holding: equity against LBR's range as hand reading reads
        it (computed once, blocked cards removed per holding) against the price."""
        to_call = state.checking_or_calling_amount
        pot_before = state.total_pot_amount - to_call
        if not to_call or to_call < pa.OVERBET_RATIO * max(pot_before, 1) or not ctx.get("log"):
            return None
        w = ranges.weights(ctx["log"], self.lbr, set(board))
        price = to_call / (state.total_pot_amount + to_call)
        margin = 0.0 if state.stacks[self.lbr] == 0 or state.stacks[self.bot] <= to_call else 0.05
        mdf = pot_before / max(pot_before + to_call, 1)
        st = ranges.strengths(board)
        tot = sum(self.reach[h] for h in combos) or 1.0
        strength = {h: st.get(ranges.norm(h), (0.5, 0, 0))[0] for h in combos}
        out = {}
        for h in combos:
            # mirrors overbet_defence: top of range by strength (MDF) and equity beats the price
            pct = sum(self.reach[g] for g in combos if strength[g] < strength[h]) / tot
            if pct < 1 - mdf:
                out[h] = [(FOLD, None, 1.0)]
                continue
            if to_call <= 1.5 * max(pot_before, 1):  # mirrors overbet_defence: MDF alone up to 1.5x pot
                out[h] = [(CALL, None, 1.0)]
                continue
            eq = ranges.equity_vs_weights(h, board, w, 150, self.rng)
            ok = eq is not None and eq > price + margin
            out[h] = [(CALL, None, 1.0)] if ok else [(FOLD, None, 1.0)]
        return out

    def flop_sized(self, state, ctx, combos, board):
        """Mirror of poker_agents.sized_flop_strategy for every holding (one batch)."""
        probe = pa.sized_flop_strategy(state, self.bot, ctx)  # node, bet size and model availability
        if not probe.get("available"):
            return None
        import ml_flop_sized

        live = [0, 1]
        behind = min(state.stacks[j] for j in live) + min(state.bets[j] for j in live)
        spot = "spr9.5_wide" if behind / max(1, ctx["street_pot"]) > 7 else "spr4.5_wide"
        X = [ml_flop_sized.feats(board, h, probe["node"], spot, probe["bet_faced"]) for h in combos]
        P = np.clip(pa._FLOP_SIZED["model"].predict(X), 0, None)
        to_call = state.checking_or_calling_amount
        out = {}
        for h, p in zip(combos, P):
            p = list(p)
            if not to_call:
                p[0], p[7], p[5] = p[0] + p[7], 0.0, 0.0
            else:
                p[1:5] = [0.0] * 4
            s = sum(p) or 1.0
            d = []
            for name, v in zip(pa.SIZED_SLOTS, p):
                if v <= 0:
                    continue
                reply = json.loads(pa.sized_action(state, self.bot, name))
                kind = {"check": CALL, "call": CALL, "fold": FOLD}.get(reply["action"], "raise")
                d.append((kind, reply.get("amount"), v / s))
            out[h] = d
        return out

    def flop(self, state, ctx, combos, board):
        sized = self.flop_sized(state, ctx, combos, board)
        if sized is not None:
            return sized
        live = [0, 1]
        acts = [k for k, _ in ctx["street_actions"] if k != "fold"]
        oop = self.bot == min(live)
        node = (
            "oop_open"
            if oop and not acts
            else "ip_after_check"
            if not oop and acts == ["check"]
            else "ip_facing_bet"
            if not oop and acts == ["raise"]
            else None
        )
        if node is None:
            return None
        if "model" not in pa._FLOP:
            pa.flop_model_strategy(state, self.bot, ctx)  # loads the model
        model = pa._FLOP.get("model")
        if model is None:
            return None
        import ml_flop

        behind = min(state.stacks) + min(state.bets)
        spot = "spr9.5_wide" if behind / max(1, ctx["street_pot"]) > 7 else "spr4.5_wide"
        X = [ml_flop.features(board, h, node, spot) for h in combos]
        P = np.clip(model.predict(X), 0, None)
        to_call, pot = state.checking_or_calling_amount, state.total_pot_amount
        can_raise = state.can_complete_bet_or_raise_to()
        lo = state.min_completion_betting_or_raising_to_amount if can_raise else None
        hi = state.max_completion_betting_or_raising_to_amount if can_raise else None
        size = max(lo, min(hi, state.bets[self.bot] + to_call + int(0.66 * (pot + to_call)))) if can_raise else None
        out = {}
        for h, p in zip(combos, P):
            cc, bet, allin, fold = (float(x) for x in p)
            if not to_call:
                cc, fold = cc + fold, 0.0
            if not can_raise:
                cc, bet, allin = cc + bet + allin, 0.0, 0.0
            t = cc + bet + allin + fold or 1.0
            out[h] = [(CALL, None, cc / t), ("raise", size, bet / t), ("raise", hi, allin / t), (FOLD, None, fold / t)]
        return out

    def solve(self, state, ctx, combos, board):
        live = [0, 1]
        eff = min(state.stacks[j] + state.bets[j] for j in live)
        bot_range = ranges.solver_range({h: self.reach[h] for h in combos})
        full = ranges.solver_range({h: 1.0 for h in ranges.ALL if not ({h[:2], h[2:]} & set(board))})
        oop = min(live)
        rng_pair = (bot_range, full) if self.bot == oop else (full, bot_range)
        key = ("lbr", id(self), len(board), ctx["street_pot"], tuple(ctx["street_actions"][:0]))
        tree, _ = pa._solve(key, ctx["street_pot"], eff, board, oop, None, rng_pair)
        node = pa._walk(tree, ctx["street_actions"]) if tree else None
        if not node or "strategy" not in node:
            return None
        strat = (node.get("strategy") or {}).get("strategy") or {}
        if not strat:
            return None
        names = node["actions"]
        to_call = state.checking_or_calling_amount
        can_raise = state.can_complete_bet_or_raise_to()
        lo = state.min_completion_betting_or_raising_to_amount if can_raise else None
        hi = state.max_completion_betting_or_raising_to_amount if can_raise else None
        out = {}
        for h in combos:
            probs = strat.get(h) or strat.get(h[2:] + h[:2])
            if probs is None:
                continue
            d = []
            for name, p in zip(names, probs):
                k = name.split()[0]
                if k == "FOLD":
                    d.append((FOLD if to_call else CALL, None, p))
                elif k in ("CHECK", "CALL") or not can_raise:
                    d.append((CALL, None, p))
                else:
                    d.append(("raise", max(lo, min(hi, state.bets[self.bot] + int(float(name.split()[1])))), p))
            out[h] = d
        return out or None

    def strength_rule(self, state, combos, board):
        st = ranges.strengths(board)
        to_call, pot = state.checking_or_calling_amount, state.total_pot_amount
        price = to_call / (pot + to_call) if to_call else 0.0
        can_raise = state.can_complete_bet_or_raise_to()
        size = None
        if can_raise:
            lo, hi = (
                state.min_completion_betting_or_raising_to_amount,
                state.max_completion_betting_or_raising_to_amount,
            )
            size = max(lo, min(hi, to_call + state.bets[self.bot] + int(0.75 * (pot + to_call))))
        out = {}
        for h in combos:
            s = st.get(ranges.norm(h), (0.5, 0, 0))[0]
            if can_raise and s > 0.8:
                out[h] = [("raise", size, 1.0)]
            elif not to_call or s > price:
                out[h] = [(CALL, None, 1.0)]
            else:
                out[h] = [(FOLD, None, 1.0)]
        return out

    # ---- LBR's choice ----
    def equity(self, combos_w, board, sims=600):
        import eval7

        hero = self.holes[self.lbr]
        combos = [h for h, w in combos_w.items() if w > 1e-9]
        if not combos:
            return 1.0
        weights = [combos_w[h] for h in combos]
        card = {c: eval7.Card(c) for c in engine.DECK}
        me = [card[hero[:2]], card[hero[2:]]]
        used = set(board) | {hero[:2], hero[2:]}
        deck = [c for c in engine.DECK if c not in used]
        win = 0.0
        for v in self.rng.choices(combos, weights=weights, k=sims):
            rest = []
            while len(rest) < 5 - len(board):
                c = deck[self.rng.randrange(len(deck))]
                if c not in (v[:2], v[2:]) and c not in rest:
                    rest.append(c)
            b = [card[c] for c in board + rest]
            a, o = eval7.evaluate(me + b), eval7.evaluate([card[v[:2]], card[v[2:]]] + b)
            win += 1.0 if a > o else 0.5 if a == o else 0.0
        return win / sims

    def lbr_choice(self, state, ctx):
        board = [repr(c) for c in state.get_board_cards(0)] if state.board_count else []
        to_call, pot = state.checking_or_calling_amount, state.total_pot_amount
        if not board:
            return CALL, None, {}
        live = {h: w for h, w in self.reach.items() if w > 1e-9 and not ({h[:2], h[2:]} & set(board))}
        tot = sum(live.values())
        wp = self.equity(live, board)
        best = (0.0 if to_call else -1e-9, FOLD if to_call else CALL, None)
        ev_call = wp * (pot + to_call) - to_call if to_call else wp * pot
        if ev_call >= best[0]:
            best = (ev_call, CALL, None)
        info = {"wp": wp, "ev_call": ev_call}
        if state.can_complete_bet_or_raise_to():
            lo, hi = (
                state.min_completion_betting_or_raising_to_amount,
                state.max_completion_betting_or_raising_to_amount,
            )
            sizes = {max(lo, min(hi, state.bets[self.lbr] + to_call + (pot + to_call))), hi}
            for to in sorted(sizes):
                s2 = copy.deepcopy(state)
                t2 = copy.copy(self.t)
                t2.state = s2
                t2.street_actions = list(self.t.street_actions)
                t2.log = list(self.t.log)
                t2.act("raise", to)
                if not s2.status or s2.actor_index != self.bot:
                    continue
                resp = self.policy(s2, t2.ctx())
                fold_w = sum(live[h] * sum(p for k, _, p in resp.get(h, []) if k == FOLD) for h in live)
                fp = fold_w / tot if tot else 0.0
                cont = {h: live[h] * (1 - sum(p for k, _, p in resp.get(h, []) if k == FOLD)) for h in live}
                wp2 = self.equity(cont, board, 400)
                risk = to - state.bets[self.lbr]
                ev = fp * pot + (1 - fp) * (wp2 * (pot + 2 * risk - to_call) - risk)
                info[f"ev_raise_{to}"] = ev
                if ev > best[0]:
                    best = (ev, "raise", to)
        return best[1], best[2], info

    # ---- play ----
    def play(self):
        t = self.t
        while t.state.status:
            if t.needs_board():
                t.deal_street()
                continue
            i = t.actor()
            if i is None:
                break
            ctx = t.ctx()
            street = engine.STREETS[t.street]
            if i == self.bot:
                pol = self.policy(t.state, ctx)
                mine = (
                    pol.get(ranges.norm(self.holes[self.bot])) or pol.get(self.holes[self.bot]) or [(CALL, None, 1.0)]
                )
                k = self.rng.choices(range(len(mine)), weights=[max(p, 0) + 1e-12 for _, _, p in mine])[0]
                kind, to, _ = mine[k]
                faced = bool(ctx["street_actions"]) and ctx["street_actions"][-1][0] == "raise"
                # reach update: probability of the same action kind (and size bucket) for every holding
                for h in list(self.reach):
                    d = pol.get(h)
                    if d is None:  # not a possible holding here (blocked by the board, or already out of range)
                        self.reach[h] = 0.0
                        continue
                    self.reach[h] *= sum(p for kk, tt, p in d if kk == kind and (kind != "raise" or tt == to))
                if faced and street != "preflop":
                    self.events.append(("bot_faced_bet", street, kind))
                t.act(kind, to)
            else:
                kind, to, _ = self.lbr_choice(t.state, ctx)
                if kind == "raise":
                    s0 = t.state
                    pot_before = s0.total_pot_amount + s0.checking_or_calling_amount
                    hi = s0.max_completion_betting_or_raising_to_amount
                    size = (
                        "all-in"
                        if to >= hi
                        else f"{(to - s0.bets[i] - s0.checking_or_calling_amount) / max(pot_before, 1):.1f}x pot"
                    )
                    self.events.append(("lbr_bet", street, size))
                t.act(kind, to)


def play_hand(args):
    seed, stack_bb = args
    rng = random.Random(seed)
    h = Hand(stack_bb, seed % 2, rng)
    start = list(h.t.state.stacks)
    try:
        h.play()
    except Exception as e:  # a failed hand is reported, not fatal
        return {"error": repr(e)[:200]}
    # start is after the blinds were posted: LBR's blind (2 units in the big blind, seat 0; 1 on the button) is lost too
    won = (h.t.state.stacks[h.lbr] - start[h.lbr] - (2 if h.lbr == 0 else 1)) / BB
    return {"won_bb": won, "lbr_seat": h.lbr, "events": h.events}


def main(outdir, hands, workers=48, stack_bb=100):
    os.makedirs(outdir, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        res = list(pool.map(play_hand, [(s, stack_bb) for s in range(hands)], chunksize=4))
    ok = [r for r in res if "won_bb" in r]
    x = np.array([r["won_bb"] for r in ok])
    se = x.std(ddof=1) / math.sqrt(len(x))
    folds = {}
    for st in ("flop", "turn", "river"):
        ev = [e for r in ok for e in r["events"] if e[0] == "bot_faced_bet" and e[1] == st]
        if ev:
            folds[st] = {"faced": len(ev), "fold_rate": round(sum(e[2] == FOLD for e in ev) / len(ev), 3)}
    # where LBR's money comes from: hands grouped by its first bet (street and size) and the bot's answer
    lines = {}
    for r in ok:
        first = next((e for e in r["events"] if e[0] == "lbr_bet"), None)
        if first:
            ans = next((e[2] for e in r["events"] if e[0] == "bot_faced_bet" and e[1] == first[1]), "-")
            key = f"{first[1]} {first[2]}, bot {ans}"
        else:
            key = "LBR never bets"
        n, w = lines.get(key, (0, 0.0))
        lines[key] = (n + 1, w + r["won_bb"])
    by_line = {
        k: {"hands": n, "lbr_bb_per_100_hands_overall": round(100 * w / len(ok), 1)}
        for k, (n, w) in sorted(lines.items(), key=lambda kv: -kv[1][1])
    }
    rep = {
        "hands": len(ok),
        "where_lbr_wins": by_line,
        "errors": len(res) - len(ok),
        "stack_bb": stack_bb,
        "lbr_wins_bb100": round(100 * x.mean(), 1),
        "ci95": [round(100 * (x.mean() - 1.96 * se), 1), round(100 * (x.mean() + 1.96 * se), 1)],
        "by_lbr_position": {
            p: round(100 * float(np.mean([r["won_bb"] for r in ok if r["lbr_seat"] == k])), 1)
            for k, p in ((0, "big blind"), (1, "button"))
        },
        "bot_fold_rate_facing_bets": folds,
        "sample_errors": [r["error"] for r in res if "error" in r][:5],
        "note": "positive = the bot is beatable by at least this much (lower bound on exploitability)",
    }
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]), *(int(x) for x in a[2:4]))
