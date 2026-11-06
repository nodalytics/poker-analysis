"""Deep-stack pre-flop strategy (more than 20 big blinds): open, 3-bet, call, 4-bet and all-in decisions by position.

Usage: uv run python scripts/preflop_deep.py   (prints the ranges it implies and writes results/preflop/deep_ranges.json)

Built from our own 169x169 equity table (results/preflop/equity169.json, 20,000 run-outs a pair), not typed-in
charts. Only these inputs are assumptions, each stated here:
  OPEN_SHARE   share of all 1,326 starting hands opened from each position, close to published 100bb solver
               opening frequencies for a full ring (early 13% ... button 42%); hands ranked by equity vs a random hand.
  REALISE      share of raw equity a caller turns into chips: 0.8 in position, 0.65 out of position (the usual
               rule of thumb; out of position realises less).
  VALUE_3BET   3-bet when equity against the opener's range is at least 0.58; VALUE_4BET 0.56 against a 3-bet range.
  R3_SHARE     a 3-bettor's range: top 6% of hands (9% from the blinds against a late open); R4_SHARE a 4-bettor's: 2.5%.
Facing an open: 3-bet for value, else call when equity x realisation covers the pot odds, else fold. No bluff
3-bets, no squeezes, no limping; card removal ignored. Sizes: open 2.5bb (+1 a limper; 3bb from the small blind),
3-bet 3x in position or 4x out (+1 open a caller), 4-bet 2.3x, then all-in.
Facing a 4-bet or more, continuing is all-in, priced on the whole effective stack against the final pot.
Calling is tighter than a solver's: the big blind defends about 24% of hands against a button open where solvers
defend about 60%, because the realisation factor is a single number.
"""

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN_SHARE = {"EP": 0.13, "LJ": 0.17, "HJ": 0.21, "CO": 0.27, "BTN": 0.42, "SB": 0.36}
REALISE = {True: 0.8, False: 0.65}
OPEN_SHARE_HU = 0.8  # heads-up button (small blind) opening range, close to published 100bb solver ranges
VALUE_3BET, VALUE_4BET = 0.58, 0.56
R3_SHARE, R3_SHARE_BLINDS, R4_SHARE = 0.06, 0.09, 0.025
_EQ = {}


def _table():
    if not _EQ:
        d = json.load(open(os.path.join(HERE, "results", "preflop", "equity169.json")))
        _EQ["classes"], _EQ["eq"] = d["classes"], d["equity"]
        _EQ["w"] = {c: 6 if len(c) == 2 else 4 if c[2] == "s" else 12 for c in d["classes"]}
        _EQ["vs_random"] = {c: equity(c, d["classes"]) for c in d["classes"]}
        _EQ["by_strength"] = sorted(d["classes"], key=lambda c: -_EQ["vs_random"][c])
    return _EQ


def equity(hand, rng):
    """Equity of a hand class against a range of classes, weighted by combinations (card removal ignored)."""
    t = _table()  # safe while the table is being built: classes, eq and w are set before this is first called
    eq, w = t["eq"], t["w"]
    tot = sum(w[c] for c in rng)
    return sum(w[c] * eq[hand][c] for c in rng) / tot if tot else 0.5


def top(share):
    """The strongest hand classes (by equity against a random hand) covering `share` of all 1,326 combinations."""
    t = _table()
    out, n = [], 0
    for c in t["by_strength"]:
        if n >= share * 1326:
            break
        out.append(c)
        n += t["w"][c]
    return out


def position(i, n):
    """PokerKit seats: 0 small blind, 1 big blind, n-1 button; named by how many players act after them. Heads-up
    PokerKit reverses the blinds: seat 1 is the button and posts the small blind, seat 0 the big blind."""
    if n == 2:
        return "SB" if i == 1 else "BB"
    if i == 0:
        return "SB"
    if i == 1:
        return "BB"
    behind = n - 1 - i
    return ["BTN", "CO", "HJ", "LJ"][behind] if behind < 4 else "EP"


def decide(state, i, ctx, hand):
    """{"action": fold|check|call|raise, "amount": raise-to, "why": text} for a deep-stack pre-flop spot, or None
    when the spot is not pre-flop or a stack is 20 big blinds or less (the push/fold chart covers those)."""
    n = len(state.statuses)
    bb = max(getattr(state, "blinds_or_straddles", (1, 2)))
    if list(state.get_board_cards(0)):
        return None
    live = [j for j in range(n) if state.statuses[j]]
    eff = min(state.stacks[i] + state.bets[i], max(state.stacks[j] + state.bets[j] for j in live if j != i)) / bb
    if eff <= 20:
        return None
    to_call, pot, top_bet = state.checking_or_calling_amount, state.total_pot_amount, max(state.bets)
    can_raise = state.can_complete_bet_or_raise_to()
    lo = state.min_completion_betting_or_raising_to_amount if can_raise else 0
    hi = state.max_completion_betting_or_raising_to_amount if can_raise else 0
    raises = sum(1 for k, _ in ctx.get("street_actions", []) if k == "raise")
    pos = position(i, n)

    def act(a, amount=None, why=""):
        if a == "raise" and not can_raise:
            a = "call" if to_call else "check"
        if a == "fold" and not to_call:
            a = "check"
        out = {"action": a, "why": why}
        if a == "raise":
            out["amount"] = int(max(lo, min(hi, amount)))
        return out

    def continue_or_fold(e, ip, why):
        need = to_call / (pot + to_call) if to_call else 0.0
        if e * REALISE[ip] >= need:
            return act("call", why=f"{why}; equity {e:.0%} x {REALISE[ip]} realisation covers the {need:.0%} price")
        return act("fold", why=f"{why}; equity {e:.0%} does not cover the {need:.0%} price")

    if raises == 0:
        limpers = sum(1 for j in range(2, n) if state.bets[j] == bb and j != i)
        if pos == "BB":
            if limpers and hand in top(0.15):
                return act("raise", (4 + limpers) * bb, f"{hand} is in the top 15%: raise the limpers")
            return act("check", why="big blind, no raise: check")
        share = OPEN_SHARE_HU if n == 2 else OPEN_SHARE[pos]
        if hand in top(share):
            size = (2.5 if n == 2 else 3 if pos == "SB" else 2.5 + limpers) * bb
            return act("raise", size, f"{hand} is in the {pos} opening range (top {share:.0%})")
        return act("fold", why=f"{hand} is outside the {pos} opening range (top {share:.0%})")

    raiser = max(range(n), key=lambda j: (state.bets[j], -j))
    ip = (i > raiser) and pos not in ("SB", "BB") or (raiser in (0, 1) and i > 1)
    if n == 2:
        ip = pos == "SB"  # heads-up the button posts the small blind and acts last after the flop
    if raises == 1:
        opener_pos = position(raiser, n)
        rng = top(OPEN_SHARE_HU if n == 2 else OPEN_SHARE.get(opener_pos, 0.42))
        e = equity(hand, rng)
        callers = sum(1 for j in range(n) if j not in (i, raiser) and state.bets[j] == top_bet)
        if e >= VALUE_3BET:
            size = (3 if ip else 4) * top_bet + callers * top_bet
            return act("raise", size, f"{hand} has {e:.0%} equity against a {opener_pos} opening range: 3-bet")
        return continue_or_fold(e, ip, f"facing a {opener_pos} open")
    if raises == 2:
        blinds_vs_late = raiser in (0, 1)
        rng = top(R3_SHARE_BLINDS if blinds_vs_late else R3_SHARE)
        e = equity(hand, rng)
        if e >= VALUE_4BET:
            return act("raise", 2.3 * top_bet, f"{hand} has {e:.0%} equity against a 3-bet range: 4-bet")
        if state.bets[i] > bb:  # we opened and were 3-bet: we may call
            return continue_or_fold(e, ip, "facing a 3-bet")
        return act("fold", why=f"cold facing a 3-bet: {hand} has only {e:.0%} against it")
    # facing a 4-bet or more, continuing means all-in: price the whole effective stack against the final pot
    e = equity(hand, top(R4_SHARE))
    stake = eff * bb
    dead = pot - state.bets[i] - top_bet
    need = (stake - state.bets[i]) / (dead + 2 * stake)
    if e >= need:
        return act("raise", hi, f"{hand} has {e:.0%} against a 4-bet range, all-in needs {need:.0%}: all-in")
    return act("fold", why=f"facing a 4-bet: {hand} has {e:.0%}, all-in needs {need:.0%}")


if __name__ == "__main__":
    out = {"open": {p: top(s) for p, s in OPEN_SHARE.items()}}
    for p, s in OPEN_SHARE.items():
        print(f"{p:3} opens {s:.0%}: {len(out['open'][p])} classes, weakest {', '.join(out['open'][p][-5:])}")
    for opener in ("EP", "CO", "BTN"):
        rng = top(OPEN_SHARE[opener])
        three = [c for c in _table()["classes"] if equity(c, rng) >= VALUE_3BET]
        out[f"3bet_vs_{opener}"] = three
        print(f"3-bet vs {opener} open: {', '.join(three)}")
    for need, ip, label in ((1.5 / 5.5, False, "BB vs BTN 2.5bb open"), (2.5 / 6.5, True, "BTN vs CO 2.5bb open")):
        rng = top(OPEN_SHARE["BTN" if "BTN open" in label else "CO"])
        calls = [c for c in _table()["classes"] if VALUE_3BET > equity(c, rng) and equity(c, rng) * REALISE[ip] >= need]
        cw = sum(_table()["w"][c] for c in calls) / 1326
        out[f"call_{label}"] = calls
        print(f"call, {label}: {cw:.0%} of hands, e.g. {', '.join(calls[:12])}")
    json.dump(out, open(os.path.join(HERE, "results", "preflop", "deep_ranges.json"), "w"), indent=1)
