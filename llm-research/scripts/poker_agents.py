"""Algorithmic and tool-using poker players for poker_selfplay.py.

- monte: Monte Carlo equity with eval7 (the poker-analysis toolkit's evaluator) against a random hand for every
  live opponent, compared with the pot odds.
- sol: TexasSolver (from the poker-analysis toolkit) when the hand is heads-up on the turn or river, sampling
  from the solved strategy for its exact hand; Monte Carlo everywhere else and whenever a solve fails. Opponent
  ranges are a fixed wide assumption, since nobody at the table shows theirs.
- model#toolkit: a language model with the poker-analysis toolkit as native tool calls (equity against a range,
  hand evaluation, push/fold chart). It needs a tool-capable model such as qwen2.5vl-tools:32b.

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
SOLVER = os.path.join(ROOT, "tools", "TexasSolver", "build", "console_solver")
RESOURCES = os.path.join(ROOT, "tools", "TexasSolver", "resources")
SOLVER_THREADS, SOLVER_TIMEOUT = 8, 90
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
stats = {"solver_used": 0, "solver_failed": 0, "solver_seconds": 0.0, "toolkit_calls": 0}


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


def _solve(key, pot, eff, board, oop):
    if key in _SOLVES:
        return _SOLVES[key]
    tree = None
    if os.path.exists(SOLVER):
        street = "turn" if len(board) == 4 else "river"
        with tempfile.TemporaryDirectory() as tmp:
            lines = [
                f"set_pot {pot}",
                f"set_effective_stack {eff}",
                f"set_board {','.join(board)}",
                f"set_range_oop {WIDE}",
                f"set_range_ip {WIDE}",
            ]
            for p in ("oop", "ip"):
                lines += [
                    f"set_bet_sizes {p},{street},bet,66",
                    f"set_bet_sizes {p},{street},raise,100",
                    f"set_bet_sizes {p},{street},allin",
                ]
            lines += [
                "set_allin_threshold 0.67",
                "build_tree",
                f"set_thread_num {SOLVER_THREADS}",
                "set_accuracy 1.0",
                "set_max_iteration 60",
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
        key = (ctx["hand"], len(board))
        tree, _ = _solve(key, ctx["street_pot"], eff, board, min(live))
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
            lo = state.min_completion_betting_or_raising_to_amount
            hi = state.max_completion_betting_or_raising_to_amount
            return _reply("raise", max(lo, min(hi, state.bets[i] + int(float(pick.split()[1])))))
        stats["solver_failed"] += 1
    return monte_reply(state, i)


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


def toolkit_reply(model, prompt, max_calls=3):
    fns = _toolkit()
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
