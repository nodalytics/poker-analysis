"""Spot study: describe a situation in a short line and get the engine's answer, each opponent's likely range and
your equity against it.

A spot: number of players, stacks (big blinds), your position and cards, and the action so far as tokens in
acting order:
    f       fold                 x   check            c   call
    r2.5    bet or raise TO 2.5 big blinds (this street's total)       a   all-in
    [Qs7h2c]  the board cards of the next street
Example (6-max, you on the button with AhKd):  "f f r2.5 c [Qs7h2c] x"   -> the small blind folded... and so on.
The analysis is for the player to act next (normally you).
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advisor import engine  # noqa: E402

ranges = engine.ranges
BB = 2  # units per big blind (blinds 1/2)


def positions(n):
    """Seat names in PokerKit order (0 small blind ... last the button; heads-up 0 big blind, 1 button)."""
    if n == 2:
        return ["BB", "BTN"]
    names = {3: ["BTN"], 4: ["CO", "BTN"], 5: ["HJ", "CO", "BTN"], 6: ["UTG", "HJ", "CO", "BTN"]}
    if n in names:
        return ["SB", "BB"] + names[n]
    return ["SB", "BB"] + [f"UTG+{k}" if k else "UTG" for k in range(n - 5)] + ["HJ", "CO", "BTN"]


TOKEN = re.compile(r"\[([2-9TJQKA][cdhs](?:\s*[2-9TJQKA][cdhs])*)\]|(f|x|c|a|r\d+(?:\.\d+)?)", re.I)


def parse_line(line):
    out = []
    pos = 0
    for m in TOKEN.finditer(line):
        if line[pos : m.start()].strip():
            raise ValueError(f"cannot read '{line[pos : m.start()].strip()}'")
        pos = m.end()
        if m.group(1):
            out.append(("board", m.group(1).replace(" ", "")))
        else:
            t = m.group(2).lower()
            out.append(("act", t))
    if line[pos:].strip():
        raise ValueError(f"cannot read '{line[pos:].strip()}'")
    return out


def build(n, stacks_bb, hero_pos, hero_cards, line, villain_cards=None):
    """Replay the spot. stacks_bb: one number for everyone or a list in PokerKit order. Returns (table, hero index)."""
    names = positions(n)
    if hero_pos not in names:
        raise ValueError(f"position must be one of {', '.join(names)}")
    hi = names.index(hero_pos)
    stacks = [int(round(float(s) * BB)) for s in (stacks_bb if isinstance(stacks_bb, list) else [stacks_bb] * n)]
    holes = [None] * n
    holes[hi] = hero_cards.replace(" ", "")
    board_cards = [m.replace(" ", "") for m in re.findall(r"\[([^\]]*)\]", line)]
    exclude = [b[k : k + 2] for b in board_cards for k in range(0, len(b), 2)]
    t = engine.Table(stacks, (1, 2), holes, seed=7, pay=False, exclude=exclude)
    for kind, tok in parse_line(line):
        if kind == "board":
            if not t.needs_board():
                raise ValueError("board given before the betting round finished")
            t.deal(tok)
            continue
        if t.actor() is None:
            raise ValueError("an action after the hand or street ended (a board is missing?)")
        s = t.state
        if tok == "f":
            t.act("fold")
        elif tok in ("x", "c"):
            t.act("call")
        elif tok == "a":
            t.act("raise", s.max_completion_betting_or_raising_to_amount)
        else:
            t.act("raise", int(round(float(tok[1:]) * BB)))
    return t, hi


def top_classes(w, k=16):
    """The holdings most consistent with the betting: each class's average weight relative to the likeliest."""
    acc = {}
    for h, x in w.items():
        c = ranges.cls(h)
        s, m = acc.get(c, (0.0, 0))
        acc[c] = (s + x, m + 1)
    mean = {c: s / m for c, (s, m) in acc.items()}
    top = max(mean.values()) or 1.0
    return sorted(((c, v / top) for c, v in mean.items()), key=lambda x: -x[1])[:k]


def analyse(n, stacks_bb, hero_pos, hero_cards, line, use_solver=False):
    t, hi = build(n, stacks_bb, hero_pos, hero_cards, line)
    s = t.state
    names = positions(n)
    out = {"to_act": None, "advice": None, "ranges": {}, "equity": None, "pot_bb": s.total_pot_amount / BB}
    i = t.actor()
    if t.needs_board():
        out["note"] = "the betting round is over: add the next board cards, e.g. [Qs7h2c]"
        return out
    if i is None:
        out["note"] = "the hand is over"
        return out
    out["to_act"] = names[i]
    if i != hi:
        out["note"] = f"{names[i]} is to act, not you: add their action to the line"
        return out
    out["to_call_bb"] = s.checking_or_calling_amount / BB
    out["advice"] = engine.advise(s, i, t.ctx(), use_solver=use_solver)
    live = [j for j, x in enumerate(s.statuses) if x and j != hi]
    board = [repr(c) for c in s.get_board_cards(0)] if s.board_count else []
    if ranges.available():
        for j in live:
            if any(e["player"] == j for e in t.log):
                w = ranges.weights(t.log, j, set(board) | {hero_cards[:2], hero_cards[2:4]})
                out["ranges"][names[j]] = top_classes(w)
        if len(live) == 1 and any(e["player"] == live[0] for e in t.log):
            out["equity"] = engine.equity_vs_range(s, hi, live[0], t.log)
    return out


def describe(res):
    """Plain-text answer for the window."""
    if res.get("note"):
        return res["note"]
    a = res["advice"]
    m = a["mix"]
    lines = [
        f"{res['to_act']} to act; pot {res['pot_bb']:g}bb"
        + (f", {res['to_call_bb']:g}bb to call." if res["to_call_bb"] else "."),
        f"Engine ({a['source']}): fold {m['fold']:.0%}, check/call {m['call']:.0%}, bet/raise {m['raise']:.0%}.",
    ]
    det = {k: v for k, v in a.get("detail", {}).items() if isinstance(v, float) and k not in ("equity", "price")}
    if det:
        lines.append("Sizes: " + ", ".join(f"{k} {v:.0%}" for k, v in det.items()))
    lines.append(f"Why: {a['why']}")
    if res["equity"] is not None:
        lines.append(f"Your equity against their likely range: {res['equity']:.0%}")
    for who, cls in res["ranges"].items():
        lines.append(f"{who}'s likeliest holdings (relative): " + ", ".join(f"{c} {p:.0%}" for c, p in cls))
    return "\n".join(lines)
