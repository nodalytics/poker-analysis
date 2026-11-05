"""Multi-table tournaments: rising blinds and antes, eliminations, table balancing, a final table, payouts.

Usage: uv run python scripts/tournament_env.py OUTDIR TOURNAMENTS ENTRANT [ENTRANT ...]
ENTRANT: MODEL#plain, MODEL#advised (Monte Carlo equity and pot odds every decision, plus the push/fold
equilibrium's answer for its exact seat and table size when it is short: 20 big blinds or less, unopened pot),
or a non-learning player: bob, tina, max, carl, nora (personalities) or monte; repeat one as name#2, name#3.

Each tournament: up to 6-max tables, 1,500 chips each, blinds and antes up a level every 8 hands at each table,
one hand per table per round, players removed when they bust (ties broken by chips at the start of the hand),
tables rebalanced when they differ by two or more, and a single final table at 6 or fewer. Pays the top three
50/30/20% of a 100-per-player pool. Scores: mean prize, share of tournaments in the money, mean finishing place.
"""

import json
import os
import random
import sys
import time

from pokerkit import Mode, NoLimitTexasHoldem

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arena  # noqa: E402
import poker_agents as pa  # noqa: E402
import poker_selfplay as ps  # noqa: E402

LEVELS = [
    (10, 20, 0),
    (15, 30, 0),
    (25, 50, 5),
    (50, 100, 10),
    (75, 150, 15),
    (100, 200, 25),
    (150, 300, 40),
    (200, 400, 50),
    (300, 600, 75),
    (500, 1000, 100),
    (700, 1400, 150),
    (1000, 2000, 200),
    (1500, 3000, 300),
]
HANDS_PER_LEVEL, START, MAX_SEATS = 8, 1500, 6
NASH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "preflop")


def base(e):
    return e.split("#")[0]


def chart_advice(state, i, n, bb):
    """Push/fold equilibrium answer for this seat, if the hero is short and nobody has raised yet."""
    stack_bb = (state.stacks[i] + state.bets[i]) / bb
    if stack_bb > 20 or max(state.bets) > bb:
        return ""
    hole = [repr(c) for c in state.hole_cards[i]]
    order = "23456789TJQKA"
    hi, lo = sorted((hole[0][0], hole[1][0]), key=order.index, reverse=True)
    hc = hi + lo if hi == lo else hi + lo + ("s" if hole[0][1] == hole[1][1] else "o")
    try:
        if n == 2:
            c = min(
                json.load(open(os.path.join(NASH, "push_fold_nash.json")))["charts"],
                key=lambda x: abs(x["stack_bb"] - stack_bb),
            )
            rng = c["push"] if i == 0 else c["call"]
        else:
            charts = [
                c
                for c in json.load(open(os.path.join(NASH, "push_fold_multiway_ante.json")))["charts"]
                if c["players"] == n
            ]
            c = min(charts, key=lambda x: abs(x["stack_bb"] - stack_bb))
            pos = "small blind" if i == 0 else "big blind" if i == 1 else "button" if i == n - 1 else f"pos{i - 2}"
            if pos == "big blind":
                return ""
            rng = c["shove"][pos]["range"]
    except (OSError, KeyError, ValueError):
        return ""
    return (
        f"Push/fold equilibrium ({n} players, ~{c['stack_bb']}bb): {hc} is "
        f"{'a SHOVE (all-in)' if hc in rng else 'a FOLD'} here.\n"
    )


def act(entrant, state, i, n, bb, ctx, history):
    b = base(entrant)
    if b == "tinaplus":
        import poker_tina_plus

        return poker_tina_plus.reply(state, i)
    if b in ps.PERSONAS:
        return ps.bot_reply(b, state, i)
    if b == "monte":
        return pa.monte_reply(state, i)
    board = ps.cards(list(state.get_board_cards(0))) if state.board_count else "-"
    to_call = state.checking_or_calling_amount
    can_raise = state.can_complete_bet_or_raise_to()
    lo = state.min_completion_betting_or_raising_to_amount if can_raise else None
    hi = state.max_completion_betting_or_raising_to_amount if can_raise else None
    legal = (
        (["fold"] if state.can_fold() else [])
        + (["call"] if to_call else ["check"])
        + ([f"raise (to between {lo} and {hi})"] if can_raise else [])
    )
    prompt = (
        f"Multi-table No-Limit Hold'em tournament (play chips). Blinds {bb // 2}/{bb}. Street: {ps.street(state)}. "
        f"Your cards: {ps.cards(state.hole_cards[i])}. Board: {board}. Pot: {state.total_pot_amount}. "
        f"Your stack: {state.stacks[i]} ({state.stacks[i] / bb:.1f} big blinds). To call: {to_call}. "
        f"Players at your table: {n}. Only the top 3 of the field are paid.\n"
        f"Action so far: {'; '.join(history[-10:]) or 'none'}\nLegal actions: {', '.join(legal)}.\n"
    )
    if entrant.endswith("#advised"):
        prompt += chart_advice(state, i, n, bb) + pa.solver_brief(state, i, ctx) + "\n"
    prompt += 'Reply with ONLY JSON: {"action": "fold"|"check"|"call"|"raise", "amount": <raise-to total>}'
    text, _ = ps.ask(entrant, prompt)  # ask() drops the '#flow' tag to find the model
    return text


def play_hand(table, chips, level, rng, log, ctx_hand):
    sb, bb, ante = level
    seats = table[ctx_hand % len(table) :] + table[: ctx_hand % len(table)]  # move the button
    n = len(seats)
    state = NoLimitTexasHoldem.create_state(
        ps.AUTO, True, ante, (sb, bb), bb, tuple(chips[e] for e in seats), n, mode=Mode.TOURNAMENT
    )
    start = {e: chips[e] for e in seats}
    ctx, street_now, history = {"hand": ctx_hand, "street_actions": [], "street_pot": 0}, None, []
    while state.status:
        i = state.actor_index
        if i is None:
            break
        st = ps.street(state)
        if st != street_now:
            street_now, ctx["street_actions"], ctx["street_pot"] = st, [], state.total_pot_amount
        e = seats[i]
        text = act(e, state, i, n, bb, ctx, history)
        a, amt = ps.parse(text)
        to_call = state.checking_or_calling_amount
        before = state.bets[i]
        if a in ("all-in", "all in"):
            a, amt = ("raise", None) if state.can_complete_bet_or_raise_to() else ("call", None)
        if a == "fold" and state.can_fold():
            state.fold()
            done = "folds"
        elif a in ("check", "call") and state.can_check_or_call():
            state.check_or_call()
            done = "calls" if to_call else "checks"
        elif a in ("raise", "bet") and state.can_complete_bet_or_raise_to():
            lo = state.min_completion_betting_or_raising_to_amount
            hi = state.max_completion_betting_or_raising_to_amount
            amt = hi if amt is None else max(lo, min(hi, amt))
            state.complete_bet_or_raise_to(amt)
            done = f"raises to {amt}"
        elif state.can_check_or_call() and not to_call:
            state.check_or_call()
            done = "checks"
        else:
            state.fold()
            done = "folds"
        history.append(f"{arena.cs.short(e)} {done}")
        kind = done.split()[0].rstrip("s")
        ctx["street_actions"].append((kind, to_call if kind == "call" else max(0, state.bets[i] - before)))
    for k, e in enumerate(seats):
        chips[e] = state.stacks[k]
    return start


def balance(tables):
    tables = [t for t in tables if t]
    total = sum(len(t) for t in tables)
    if total <= MAX_SEATS and len(tables) > 1:
        return [[e for t in tables for e in t]]
    while True:
        big, small = max(tables, key=len), min(tables, key=len)
        if len(big) - len(small) < 2:
            return tables
        small.append(big.pop())


def tournament(entrants, rng, log, tno):
    field = list(entrants)
    rng.shuffle(field)
    k = -(-len(field) // MAX_SEATS)
    tables = [field[j::k] for j in range(k)]
    chips = {e: START for e in field}
    finish, hands_played, rnd = {}, [0] * 64, 0
    while sum(1 for e in chips if chips[e] > 0) > 1:
        rnd += 1
        for ti, table in enumerate(tables):
            if len(table) < 2:
                continue
            level = LEVELS[min(len(LEVELS) - 1, hands_played[ti] // HANDS_PER_LEVEL)]
            start = play_hand(table, chips, level, rng, log, hands_played[ti])
            hands_played[ti] += 1
            out = [e for e in table if chips[e] <= 0]
            for e in sorted(out, key=lambda e: start[e]):  # the smaller stack at the start of the hand goes first
                finish[e] = sum(1 for x in chips if chips[x] > 0) + 1 + sum(1 for x in out if start[x] > start[e])
                table.remove(e)
        tables = balance(tables)
        if rnd > 4000:
            break
    winner = max(chips, key=chips.get)
    finish[winner] = 1
    pool = 100 * len(entrants)
    prize = {
        e: (0.5 * pool if f == 1 else 0.3 * pool if f == 2 else 0.2 * pool if f == 3 else 0.0)
        for e, f in finish.items()
    }
    log(
        f"tournament {tno}: "
        + ", ".join(f"{f}. {arena.cs.short(e)}" for e, f in sorted(finish.items(), key=lambda x: x[1])[:6])
        + f" ({rnd} rounds)"
    )
    return finish, prize


def main(outdir, n, entrants):
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(9)
    logf = open(os.path.join(outdir, "tournaments.log"), "a", buffering=1)

    def log(s):
        logf.write(s + "\n")

    rows = []
    for t in range(1, n + 1):
        t0 = time.time()
        finish, prize = tournament(entrants, rng, log, t)
        rows.append({"tournament": t, "finish": finish, "prize": prize, "seconds": round(time.time() - t0)})
        json.dump(rows, open(os.path.join(outdir, "results.json"), "w"), indent=1)
    summary = {}
    for e in entrants:
        fs = [r["finish"][e] for r in rows if e in r["finish"]]
        ps_ = [r["prize"][e] for r in rows if e in r["prize"]]
        summary[e] = {
            "tournaments": len(fs),
            "mean_prize": round(sum(ps_) / len(ps_), 1),
            "in_the_money": round(sum(f <= 3 for f in fs) / len(fs), 3),
            "mean_finish": round(sum(fs) / len(fs), 2),
            "wins": sum(f == 1 for f in fs),
        }
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    for e, s in sorted(summary.items(), key=lambda x: -x[1]["mean_prize"]):
        log(
            f"  {arena.cs.short(e):28} prize {s['mean_prize']:7.1f}  ITM {s['in_the_money']:.0%}  "
            f"finish {s['mean_finish']:5.2f}  wins {s['wins']}"
        )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
