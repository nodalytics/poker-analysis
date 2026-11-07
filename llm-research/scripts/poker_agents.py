"""Algorithmic and tool-using poker players for poker_selfplay.py.

- monte: Monte Carlo equity with eval7 (the poker-analysis toolkit's evaluator) against a random hand for every
  live opponent, compared with the pot odds.
- sol: TexasSolver (from the poker-analysis toolkit) when the hand is heads-up on the turn or river, sampling
  from the solved strategy for its exact hand; Monte Carlo everywhere else and whenever a solve fails. Opponent
  ranges are a fixed wide assumption, since nobody at the table shows theirs.
- model#toolkit: a language model with the poker-analysis toolkit as native tool calls (equity against a range,
  hand evaluation, push/fold chart, and solver_advice: TexasSolver's strategy for its hand, heads-up turn/river). It needs a tool-capable model such as qwen2.5vl-tools:32b.

Paths: POKER_ANALYSIS_ROOT (default: the repo above llm-research) for src/ and tools/TexasSolver.
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.request

import eval7

ROOT = os.environ.get(
    "POKER_ANALYSIS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
# TEXASSOLVER_BIN: e.g. the optimised build (~/solver-fast/build/console_solver), same answers, 2-3x faster
SOLVER = os.environ.get("TEXASSOLVER_BIN", os.path.join(ROOT, "tools", "TexasSolver", "build", "console_solver"))
# Hand-ranking tables TexasSolver needs. Without them it still runs, ranks every hand equal and returns nonsense
# (on the lab, ~/llm-research has no tools/ beside it: river solves shoved ace-high into a flush until 2026-11-07).
RESOURCES = os.environ.get("TEXASSOLVER_RESOURCES") or next(
    (
        d
        for d in (
            os.path.join(ROOT, "tools", "TexasSolver", "resources"),
            os.path.expanduser("~/poker-analysis/tools/TexasSolver/resources"),
        )
        if os.path.exists(os.path.join(d, "compairer", "card5_dic_sorted.txt"))
    ),
    os.path.join(ROOT, "tools", "TexasSolver", "resources"),
)
SOLVER_THREADS, SOLVER_TIMEOUT = int(os.environ.get("SOLVER_THREADS", 8)), 90
SOLVER_ITERS = int(os.environ.get("SOLVER_RIVER_ITERS", 200))  # river solves are cheap; 60 left bluff shoves
WIDE = (
    "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AK,AQ,AJ,AT,A9,A8,A7,A6,A5,A4,A3,A2,KQ,KJ,KT,K9,K8,QJ,QT,Q9,"
    "JT,J9,T9,T8,98,97,87,76,65,54"
)
ALGOS = {
    "monte": "Monte Carlo equity (eval7, poker-analysis toolkit)",
    "sol": "TexasSolver heads-up turn/river, else Monte Carlo",
}
_RNG = random.Random(4242)
_DECK = eval7.Deck().cards
_SOLVES = {}
stats = {
    "solver_used": 0,
    "solver_failed": 0,
    "solver_seconds": 0.0,
    "toolkit_calls": 0,
    "flop_model_used": 0,
    "push_fold_used": 0,
    "preflop_deep_used": 0,
    "ranges_narrowed": 0,
}
# The flop model (ml_flop.py) trained on the solver library; the MLP is 325KB and within 0.006 MAE of the 2.6GB
# extra-trees model on held-out flops, so it is the one loaded per decision.
FLOP_MODEL = os.environ.get(
    "FLOP_MODEL", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mlflop", "mlp.joblib")
)
_FLOP = {}


def _cards(cs):
    return [eval7.Card(repr(c)) for c in cs]


def equity_vs_random(hole, board, opponents, sims=2000):
    known = set(hole + board)
    rest = [c for c in _DECK if c not in known]
    win = 0.0
    for _ in range(sims):
        draw = _RNG.sample(rest, 2 * opponents + 5 - len(board))
        b = board + draw[2 * opponents :]
        me = eval7.evaluate(hole + b)
        best = max(eval7.evaluate(draw[2 * k : 2 * k + 2] + b) for k in range(opponents))
        win += 1.0 if me > best else 0.5 if me == best else 0.0
    return win / sims


def _reply(action, amount=None):
    return json.dumps({"action": action, **({"amount": int(amount)} if amount is not None else {})})


def monte_reply(state, i):
    hole = _cards(state.hole_cards[i])
    board = _cards(state.get_board_cards(0)) if state.board_count else []
    live = sum(1 for j, s in enumerate(state.statuses) if s and j != i)
    eq = equity_vs_random(hole, board, max(1, live))
    to_call, pot = state.checking_or_calling_amount, state.total_pot_amount
    can_raise = state.can_complete_bet_or_raise_to()
    fair = 1 / (live + 1)
    if can_raise and eq > min(0.85, fair * 1.8):
        lo, hi = state.min_completion_betting_or_raising_to_amount, state.max_completion_betting_or_raising_to_amount
        return _reply("raise", max(lo, min(hi, to_call + int(pot * 0.75))))
    if not to_call:
        return _reply("check")
    return _reply("call") if eq > to_call / (pot + to_call) else _reply("fold")


def hand_class(hole):
    """TexasSolver range notation for a hand: 'AK' (suited and offsuit together, as WIDE uses) or '77'."""
    order = "23456789TJQKA"
    a, b = sorted((repr(c)[0] for c in hole), key=order.index, reverse=True)
    return a + b


def _solve(key, pot, eff, board, oop, extra=None, ranges=None):
    if key in _SOLVES:
        return _SOLVES[key]
    tree = None
    if os.path.exists(SOLVER) and not os.path.exists(os.path.join(RESOURCES, "compairer", "card5_dic_sorted.txt")):
        raise RuntimeError(f"TexasSolver hand-ranking tables missing at {RESOURCES}; set TEXASSOLVER_RESOURCES")
    if os.path.exists(SOLVER):
        street = "turn" if len(board) == 4 else "river"
        rng = WIDE if not extra or extra in WIDE.split(",") else f"{WIDE},{extra}"
        with tempfile.TemporaryDirectory() as tmp:
            lines = [
                f"set_pot {pot}",
                f"set_effective_stack {eff}",
                f"set_board {','.join(board)}",
                f"set_range_oop {ranges[0] if ranges else rng}",
                f"set_range_ip {ranges[1] if ranges else rng}",
            ]
            for p in ("oop", "ip"):
                lines += [
                    f"set_bet_sizes {p},{street},bet,66",
                    f"set_bet_sizes {p},{street},raise,100",
                ]
                # an explicit all-in only when the stack is at most 4 pots: deeper, a 60-iteration solve leaves bluff
                # shoves of 50 pots (Slumbot at 200bb called them); bets still become all-in past the threshold
                if eff <= 4 * pot:
                    lines.append(f"set_bet_sizes {p},{street},allin")
            lines += [
                "set_allin_threshold 0.67",
                "build_tree",
                f"set_thread_num {SOLVER_THREADS}",
                "set_accuracy 1.0",
                f"set_max_iteration {SOLVER_ITERS if street == 'river' else 60}",
                "set_print_interval 20",
                "set_use_isomorphism 1",
                "start_solve",
                "set_dump_rounds 1",
                f"dump_result {tmp}/out.json",
            ]
            open(f"{tmp}/in.txt", "w").write("\n".join(lines) + "\n")
            t0 = time.time()
            try:
                subprocess.run(
                    ["nice", "-n", "10", SOLVER, "--input_file", f"{tmp}/in.txt", "--resource_dir", RESOURCES],
                    capture_output=True,
                    timeout=SOLVER_TIMEOUT,
                    check=True,
                )
                tree = json.load(open(f"{tmp}/out.json"))
            except (subprocess.SubprocessError, OSError, ValueError):
                tree = None
            stats["solver_seconds"] += time.time() - t0
    _SOLVES[key] = (tree, oop)
    return _SOLVES[key]


def _walk(node, actions):
    """Follow this street's actions (kind, chips added) down the solved tree."""
    for kind, amount in actions:
        if not node or node.get("node_type") != "action_node":
            return None
        names = node["actions"]
        if kind in ("check", "call", "fold"):
            want = [a for a in names if a.split()[0] == kind.upper()]
        else:
            sized = [a for a in names if a.split()[0] in ("BET", "RAISE")]
            want = sorted(sized, key=lambda a: abs(float(a.split()[1]) - amount))[:1]
        if not want:
            return None
        node = node.get("childrens", {}).get(want[0])
    return node


def sol_reply(state, i, ctx):
    board_cards = list(state.get_board_cards(0)) if state.board_count else []
    live = [j for j, s in enumerate(state.statuses) if s]
    if len(live) == 2 and len(board_cards) in (4, 5) and min(state.stacks[j] for j in live) > 0:
        board = [repr(c) for c in board_cards]
        eff = min(state.stacks[j] + state.bets[j] for j in live)
        cls = hand_class(state.hole_cards[i])
        key = (ctx["hand"], len(board), i)
        tree, _ = _solve(key, ctx["street_pot"], eff, board, min(live), cls)
        node = _walk(tree, ctx["street_actions"]) if tree else None
        hole = [repr(c) for c in state.hole_cards[i]]
        strat = (node or {}).get("strategy", {}).get("strategy", {}) if node else {}
        probs = strat.get(hole[0] + hole[1]) or strat.get(hole[1] + hole[0])
        if probs:
            stats["solver_used"] += 1
            pick = _RNG.choices(node["actions"], weights=probs)[0]
            kind = pick.split()[0]
            if kind in ("CHECK", "CALL", "FOLD"):
                return _reply(kind.lower())
            if not state.can_complete_bet_or_raise_to():  # the solver bets but no raise is legal (opponent all-in)
                return _reply("call" if state.checking_or_calling_amount else "check")
            lo = state.min_completion_betting_or_raising_to_amount
            hi = state.max_completion_betting_or_raising_to_amount
            return _reply("raise", max(lo, min(hi, state.bets[i] + int(float(pick.split()[1])))))
        stats["solver_failed"] += 1
    return monte_reply(state, i)


def narrowed_ranges(state, i, ctx, board, live):
    """(oop, ip) TexasSolver ranges read from this hand's betting (ranges.py), or None without a hand log or model.
    Heads-up only: the out-of-position player is the lowest live seat (PokerKit acting order after the flop)."""
    log = ctx.get("log")
    if not log or len(live) != 2 or os.environ.get("NARROW_RANGES", "1") == "0":
        return None
    import ranges

    if not ranges.available():
        return None
    hole = [repr(c) for c in state.hole_cards[i]]
    out = []
    for j in sorted(live):
        blocked = set(board) | (set() if j == i else set(hole))
        w = ranges.weights(log, j, blocked)
        out.append(ranges.solver_range(w, include=ranges.cls("".join(hole)) if j == i else None))
    stats["ranges_narrowed"] += 1
    return tuple(out)


def solver_strategy(state, i, ctx):
    """TexasSolver's action frequencies for player i's exact hand in this spot, or a reason there are none."""
    board_cards = list(state.get_board_cards(0)) if state.board_count else []
    live = [j for j, s in enumerate(state.statuses) if s]
    if len(live) != 2:
        return {"available": False, "reason": f"{len(live)} players in the hand; the solver handles heads-up only"}
    if len(board_cards) not in (4, 5):
        return {
            "available": False,
            "reason": "the solver is used on the turn and river only (a flop solve is too slow)",
        }
    board = [repr(c) for c in board_cards]
    eff = min(state.stacks[j] + state.bets[j] for j in live)
    cls = hand_class(state.hole_cards[i])
    rngs = narrowed_ranges(state, i, ctx, board, live)
    tree, _ = _solve((ctx["hand"], len(board), i), ctx["street_pot"], eff, board, min(live), cls, rngs)
    node = _walk(tree, ctx["street_actions"]) if tree else None
    hole = [repr(c) for c in state.hole_cards[i]]
    strat = (node or {}).get("strategy", {}).get("strategy", {}) if node else {}
    probs = strat.get(hole[0] + hole[1]) or strat.get(hole[1] + hole[0])
    if not probs:
        return {
            "available": False,
            "reason": "no solution for this exact line (solve failed or the betting left the tree)",
        }
    return {
        "available": True,
        "assumed_ranges": "narrowed by each player's betting (ranges.py)"
        if rngs
        else "wide (plus your own hand class), the same for both players",
        "strategy": {a: round(p, 3) for a, p in zip(node["actions"], probs)},
        "note": "BET/RAISE amounts are chips added this street; frequencies are the solver's mixed strategy",
    }


def flop_model_strategy(state, i, ctx):
    """The flop model's prediction of TexasSolver's strategy, heads-up on the flop at the three points it was trained
    on: first to act, in position after a check, and in position facing a bet. Or a reason there is none."""
    board_cards = list(state.get_board_cards(0)) if state.board_count else []
    live = [j for j, s in enumerate(state.statuses) if s]
    if len(board_cards) != 3 or len(live) != 2:
        return {"available": False, "reason": "the flop model covers heads-up flops only"}
    acts = [k for k, _ in ctx["street_actions"] if k != "fold"]
    oop = i == min(live)
    if oop and not acts:
        node = "oop_open"
    elif not oop and acts == ["check"]:
        node = "ip_after_check"
    elif not oop and acts == ["raise"]:
        node = "ip_facing_bet"
    else:
        return {"available": False, "reason": "the flop model covers the first decision of each player only"}
    if "model" not in _FLOP:
        try:
            from joblib import load

            _FLOP["model"] = load(FLOP_MODEL)
        except (OSError, ImportError) as e:
            _FLOP["model"] = None
            _FLOP["error"] = str(e)
    if _FLOP["model"] is None:
        return {"available": False, "reason": f"no flop model ({_FLOP.get('error', 'not loaded')})"}
    import ml_flop

    behind = min(state.stacks[j] for j in live) + min(state.bets[j] for j in live)
    spot = "spr9.5_wide" if behind / max(1, ctx["street_pot"]) > 7 else "spr4.5_wide"
    hole = "".join(repr(c) for c in state.hole_cards[i])
    x = ml_flop.features([repr(c) for c in board_cards], hole, node, spot)
    p = [max(0.0, float(v)) for v in _FLOP["model"].predict([x])[0]]
    if not state.checking_or_calling_amount:
        p[0], p[3] = p[0] + p[3], 0.0  # never fold when checking is free
    t = sum(p) or 1.0
    names = ["check/call", "bet or raise about 2/3 pot", "all-in", "fold"]
    return {"available": True, "node": node, "strategy": {n: round(v / t, 3) for n, v in zip(names, p)}}


PREFLOP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "preflop")
_CHARTS = {}


def _chart(name):
    if name not in _CHARTS:
        try:
            _CHARTS[name] = json.load(open(os.path.join(PREFLOP, name)))["charts"]
        except OSError:
            _CHARTS[name] = []
    return _CHARTS[name]


def hand_class169(hole):
    """One of the 169 preflop classes the push/fold charts use: 'AKs', 'AKo' or '77'."""
    order = "23456789TJQKA"
    (r1, s1), (r2, s2) = [(repr(c)[0], repr(c)[1]) for c in hole]
    hi, lo = sorted((r1, r2), key=order.index, reverse=True)
    return hi + lo if hi == lo else hi + lo + ("s" if s1 == s2 else "o")


def push_fold_advice(state, i):
    """The push/fold equilibrium's answer when the pot is unopened (blinds only, no limpers) and the hero has 20 big
    blinds or less. With k players still behind, a shove meets the same callers as the first seat of a (k+1)-handed
    table, so that chart is used (capped at 6 handed, slightly loose for longer tables; at most one caller assumed).
    Or None when it does not apply."""
    n = len(state.statuses)
    bb = max(getattr(state, "blinds_or_straddles", (1, 2)))
    big = 0 if n == 2 else 1  # heads-up PokerKit seats the big blind at 0 and the button (small blind) at 1
    if list(state.get_board_cards(0)) or i == big or max(state.bets) > bb:
        return None
    if any(state.bets[j] for j in range(2, n)):  # a limper: not unopened
        return None
    order = [1, 0] if n == 2 else list(range(2, n)) + [0, 1]  # pre-flop acting order
    behind = [j for j in order[order.index(i) + 1 :] if state.statuses[j]]
    if not behind:
        return None
    eff = min(state.stacks[i] + state.bets[i], max(state.stacks[j] + state.bets[j] for j in behind)) / bb
    if eff > 20:
        return None
    hc = hand_class169(state.hole_cards[i])
    m = len(behind) + 1
    if m == 2:
        c = min(_chart("push_fold_nash.json"), key=lambda x: abs(x["stack_bb"] - eff), default=None)
        rng, table = (c["push"] if c else []), "heads-up"
    else:
        size = min(m, 6)
        cs = [c for c in _chart("push_fold_multiway.json") if c["players"] == size]
        c = min(cs, key=lambda x: abs(x["stack_bb"] - eff), default=None)
        pos = "button" if size == 3 else "pos0"
        rng, table = (c["shove"].get(pos, {}).get("range", []) if c else []), f"{size}-handed, first to act"
    if not c:
        return None
    return {"hand": hc, "shove": hc in rng, "stack_bb": round(eff, 1), "chart": f"{table}, {c['stack_bb']}bb"}


def short_defence(state, i):
    """Heads-up, 20 big blinds or less, facing a raise before the flop: the push/fold equilibrium's calling range.
    Facing an all-in (or a raise of 40%+ of our stack): call with the chart's calling range, else fold. Facing a
    smaller raise: re-shove the calling range (it is stronger than the raiser's opening range), call when the
    equity against a wide (80%) opening range, x0.85 (short stacks realise most of their equity: little play is left after the flop), covers the price, else fold.
    Returns {"action", "amount"?, "why"} or None."""
    n = len(state.statuses)
    bb = max(getattr(state, "blinds_or_straddles", (1, 2)))
    live = [j for j in range(n) if state.statuses[j]]
    to_call = state.checking_or_calling_amount
    if list(state.get_board_cards(0)) or len(live) != 2 or not to_call or max(state.bets) <= bb:
        return None
    j = next(x for x in live if x != i)
    eff = min(state.stacks[i] + state.bets[i], state.stacks[j] + state.bets[j]) / bb
    if eff > 20:
        return None
    c = min(_chart("push_fold_nash.json"), key=lambda x: abs(x["stack_bb"] - eff), default=None)
    if not c:
        return None
    hc = hand_class169(state.hole_cards[i])
    strong = hc in c["call"]
    can_raise = state.can_complete_bet_or_raise_to()
    if state.stacks[j] == 0 or to_call >= 0.4 * state.stacks[i] or not can_raise:
        return {"action": "call" if strong else "fold", "why": f"{hc} vs all-in, {c['stack_bb']}bb calling chart"}
    if strong:
        return {"action": "raise", "amount": state.max_completion_betting_or_raising_to_amount, "why": "re-shove"}
    import preflop_deep

    e = preflop_deep.equity(hc, preflop_deep.top(0.8))
    need = to_call / (state.total_pot_amount + to_call)
    return {"action": "call" if e * 0.85 >= need else "fold", "why": f"equity {e:.0%} vs a wide open, price {need:.0%}"}


def preflop_deep_decision(state, i, ctx):
    """The deep-stack pre-flop strategy (preflop_deep.py) for this spot, or None."""
    import preflop_deep

    if push_fold_advice(state, i):
        return None
    return preflop_deep.decide(state, i, ctx, hand_class169(state.hole_cards[i]))


def solver_play(state, i, ctx):
    """The solver acts itself wherever it has an answer, so the model cannot overrule it: the push/fold chart
    (unopened pot, 20 big blinds or less), TexasSolver (heads-up turn and river) and the flop model (heads-up, first
    decision each), each sampled from its mixed strategy. Returns a JSON reply, or None where no solver covers the
    spot and the model decides."""
    can_raise = state.can_complete_bet_or_raise_to()
    lo = state.min_completion_betting_or_raising_to_amount if can_raise else 0
    hi = state.max_completion_betting_or_raising_to_amount if can_raise else 0
    to_call, pot = state.checking_or_calling_amount, state.total_pot_amount
    passive = _reply("call" if to_call else "check")
    pf = push_fold_advice(state, i)
    if pf:
        stats["push_fold_used"] += 1
        return _reply("raise", hi) if pf["shove"] and can_raise else (_reply("fold") if to_call else passive)
    sd = short_defence(state, i)
    if sd:
        stats["push_fold_used"] += 1
        return _reply(sd["action"], sd.get("amount"))
    deep = preflop_deep_decision(state, i, ctx)
    if deep:
        stats["preflop_deep_used"] += 1
        return _reply(deep["action"], deep.get("amount"))
    adv = solver_strategy(state, i, ctx)
    if adv.get("available"):
        stats["solver_used"] += 1
        pick = _RNG.choices(list(adv["strategy"]), weights=list(adv["strategy"].values()))[0]
        kind = pick.split()[0]
        if kind == "FOLD":
            return _reply("fold") if to_call else passive
        if kind in ("CHECK", "CALL") or not can_raise:
            return passive
        return _reply("raise", max(lo, min(hi, state.bets[i] + int(float(pick.split()[1])))))
    flop = flop_model_strategy(state, i, ctx)
    if flop.get("available"):
        stats["flop_model_used"] += 1
        pick = _RNG.choices(list(flop["strategy"]), weights=list(flop["strategy"].values()))[0]
        if pick == "fold":
            return _reply("fold") if to_call else passive
        if pick == "check/call" or not can_raise:
            return passive
        if pick == "all-in":
            return _reply("raise", hi)
        return _reply("raise", max(lo, min(hi, state.bets[i] + to_call + int(0.66 * (pot + to_call)))))
    return None


def opponent_read(ctx, i, recent=False):
    """Pooled tendencies of the live opponents: VPIP, post-flop aggression, fold to a bet. recent=True reads the
    exponentially forgotten counts (memory of about 7 hands), which follow an opponent who changes style."""
    seats, prof = ctx.get("seats"), ctx.get("recent" if recent else "profile")
    if not seats or not prof:
        return None
    need = 3 if recent else 8
    rows = [prof[m] for j, m in enumerate(seats) if j != i and prof[m]["hands"] >= need]
    if not rows:
        return None
    h = sum(r["hands"] for r in rows)
    acts = sum(r["post_raises"] + r["post_calls"] for r in rows)
    faced = sum(r["faced"] for r in rows)
    if recent:
        return {
            "vpip": sum(r["vpip"] for r in rows) / h,
            "aggression": sum(r["post_raises"] for r in rows) / acts if acts > 0.5 else 0.5,
            "fold_to_bet": sum(r["folded"] for r in rows) / faced if faced >= 1.5 else 0.5,
        }
    return {
        "vpip": sum(r["vpip"] for r in rows) / h,
        "aggression": sum(r["post_raises"] for r in rows) / acts if acts else 0.5,
        "fold_to_bet": sum(r["folded"] for r in rows) / faced if faced >= 5 else 0.5,
    }


def adapt_play(state, i, ctx, recent=False):
    """The solver strategy, bent toward the opponents it has seen: the classic way poker profit is made.
    Against calling stations (fold to a bet under 35%, VPIP over 40%): no bluffs, bet strong hands for value. Against
    players who fold too much (over 60%): bet when checked to. Against maniacs (post-flop aggression over 60%): call
    down medium hands instead of folding. Unchanged when the opponents look balanced or unknown."""
    base = solver_play(state, i, ctx) or monte_reply(state, i)
    board = list(state.get_board_cards(0))
    read = opponent_read(ctx, i, recent)
    if not board or read is None:
        return base
    stats["adapt_reads"] = stats.get("adapt_reads", 0) + 1
    live = sum(1 for j, s in enumerate(state.statuses) if s and j != i)
    eq = equity_vs_random(_cards(state.hole_cards[i]), _cards(board), max(1, live), 400)
    to_call, pot = state.checking_or_calling_amount, state.total_pot_amount
    can_raise = state.can_complete_bet_or_raise_to()
    act = json.loads(base).get("action")

    def bet(frac):
        lo = state.min_completion_betting_or_raising_to_amount
        hi = state.max_completion_betting_or_raising_to_amount
        return _reply("raise", max(lo, min(hi, state.bets[i] + to_call + int(frac * (pot + to_call)))))

    station = read["fold_to_bet"] < 0.35 and read["vpip"] > 0.4
    folder = read["fold_to_bet"] > 0.6
    maniac = read["aggression"] > 0.6
    if station and act == "raise" and eq < 0.5:
        stats["adapt_changes"] = stats.get("adapt_changes", 0) + 1
        return _reply("call" if to_call else "check")  # no bluffs into a calling station
    if station and act in ("check", "call") and eq > 0.65 and can_raise:
        stats["adapt_changes"] = stats.get("adapt_changes", 0) + 1
        return bet(0.75)  # thin value against someone who pays off
    if folder and not to_call and act == "check" and can_raise:
        stats["adapt_changes"] = stats.get("adapt_changes", 0) + 1
        return bet(0.5)  # steal from someone who folds too often
    if maniac and act == "fold" and eq > 0.45:
        stats["adapt_changes"] = stats.get("adapt_changes", 0) + 1
        return _reply("call")  # catch the maniac's bluffs
    return base


def solver_brief(state, i, ctx):
    """What a solver-first player is told before it decides: the solver's strategy when it can solve the spot,
    otherwise Monte Carlo equity against the live opponents and the pot odds."""
    adv = solver_strategy(state, i, ctx)
    if adv.get("available"):
        stats["solver_used"] += 1
        mix = ", ".join(f"{a.split('.')[0]} {p:.0%}" for a, p in adv["strategy"].items())
        return f"Solver (TexasSolver, game-theory strategy for your exact hand here): {mix}."
    flop = flop_model_strategy(state, i, ctx)
    if flop.get("available"):
        stats["flop_model_used"] += 1
        mix = ", ".join(f"{a} {p:.0%}" for a, p in flop["strategy"].items() if p >= 0.005)
        return (
            "Solver flop model (trained on 400 TexasSolver flop solves, predicts its strategy for your hand here; "
            f"about 81% agreement on unseen flops): {mix}."
        )
    deep = preflop_deep_decision(state, i, ctx)
    if deep:
        stats["preflop_deep_used"] += 1
        amount = f" to {deep['amount']}" if deep["action"] == "raise" else ""
        return (
            f"Pre-flop strategy (deep stacks, from our equity tables): {deep['action'].upper()}{amount}. {deep['why']}."
        )
    pf = push_fold_advice(state, i)
    if pf:
        stats["push_fold_used"] += 1
        return (
            f"Push/fold equilibrium ({pf['chart']} chart, you have {pf['stack_bb']} big blinds, folded to you): "
            f"{pf['hand']} is {'a SHOVE: go all-in' if pf['shove'] else 'a FOLD'}."
        )
    hole = _cards(state.hole_cards[i])
    board = _cards(state.get_board_cards(0)) if state.board_count else []
    # range only the opponents with chips in the pot: preflop, players still to act are not random hands yet
    postflop = bool(board)
    live = sum(1 for j, s in enumerate(state.statuses) if s and j != i and (postflop or state.bets[j]))
    eq = equity_vs_random(hole, board, max(1, live))
    to_call, pot = state.checking_or_calling_amount, state.total_pot_amount
    odds = to_call / (pot + to_call) if to_call else 0.0
    return (
        f"Simulation (Monte Carlo, 2000 run-outs vs {live} random hand(s)): your equity {eq:.0%}; "
        f"calling needs {odds:.0%}. ({adv.get('reason', 'no solve')}.)"
    )


# --- the poker-analysis toolkit as native tools for a language model ---------------------------------------
def _toolkit():
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from src.core.cards import parse_hand
    from src.core.evaluator import evaluate_best
    from src.equity.calculator import hand_vs_range_equity
    from src.mtt.tournament import push_fold_range

    def equity(hand, villain_range, board=""):
        h = [hand.replace(" ", "")[k : k + 2] for k in (0, 2)]
        b = [board.replace(" ", "")[k : k + 2] for k in range(0, len(board.replace(" ", "")), 2)]
        return {"equity_pct": hand_vs_range_equity(h, villain_range, b or None, num_simulations=8000)}

    def evaluate(cards):
        best = evaluate_best(parse_hand(cards.replace(" ", "")))
        return {"hand": best.name}

    def chart(big_blinds, position):
        return {"push_range": push_fold_range(float(big_blinds), position)}

    return {"equity_hand_vs_range": equity, "evaluate_hand": evaluate, "push_fold_chart": chart}


TOOLKIT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "solver_advice",
            "description": "Ask TexasSolver (poker-analysis toolkit) for the game-theory strategy for your exact hand in "
            "the current spot. No arguments: the table fills in the pot, stacks, board and betting. Works heads-up on "
            "the turn and river only; elsewhere it says why not. Takes about 8 seconds.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "equity_hand_vs_range",
            "description": "Your equity in percent against a villain range (PokerStove notation, e.g. 'TT+, AJs+, KQs'), "
            "from the poker-analysis toolkit. hand like 'AhKd'; board '' preflop or 3-5 cards like 'Qh9h3d'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hand": {"type": "string"},
                    "villain_range": {"type": "string"},
                    "board": {"type": "string"},
                },
                "required": ["hand", "villain_range"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_hand",
            "description": "Best five-card hand from 5-7 cards, e.g. 'AhKhQh9h2h'.",
            "parameters": {"type": "object", "properties": {"cards": {"type": "string"}}, "required": ["cards"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "push_fold_chart",
            "description": "Short-stack open-shove range by big blinds and position (utg, mp, co, btn, sb). A hand-typed "
            "table, not a Nash calculation.",
            "parameters": {
                "type": "object",
                "properties": {"big_blinds": {"type": "number"}, "position": {"type": "string"}},
                "required": ["big_blinds", "position"],
            },
        },
    },
]


def toolkit_reply(model, prompt, max_calls=3, state=None, i=None, ctx=None):
    fns = _toolkit()
    fns["solver_advice"] = lambda: solver_strategy(state, i, ctx) if state is not None else {"available": False}
    messages = [
        {
            "role": "user",
            "content": prompt + "\nYou may call the poker-analysis tools first, then give "
            "your action as the JSON above.",
        }
    ]
    for _ in range(max_calls + 1):
        body = {
            "model": model.partition("#")[0],
            "stream": False,
            "messages": messages,
            "tools": TOOLKIT_TOOLS,
            "options": {"temperature": 0.5, "num_predict": 200},
        }
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            msg = json.load(r)["message"]
        calls = msg.get("tool_calls") or []
        messages.append(msg)
        if not calls:
            return msg.get("content", "")
        for c in calls:
            fn = c["function"]
            stats["toolkit_calls"] += 1
            try:
                out = fns[fn["name"]](**(fn.get("arguments") or {}))
            except Exception as e:  # a model's bad arguments are its mistake, reported back to it
                out = {"error": str(e)[:200]}
            messages.append({"role": "tool", "content": json.dumps(out), "tool_name": fn["name"]})
    return messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
