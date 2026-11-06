"""Local models play No-Limit Hold'em against each other, with play chips, on a PokerKit engine.

Usage: python3 poker_selfplay.py OUTDIR HANDS MODEL [MODEL ...]
Writes OUTDIR/table.log (readable, follow it live with `tail -f`) and OUTDIR/hands.jsonl (every decision).

PokerKit enforces the rules, side pots and showdowns; models only choose actions. Each model sees its own
hole cards, the board, pot, stacks and the hand's action so far. An unreadable or illegal reply is counted
and replaced by check (if allowed) or fold. Blinds 1/2, 200-chip stacks (100 big blinds), no rebuys.
"""

import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
import warnings

from pokerkit import Automation, Mode, NoLimitTexasHoldem, StandardHighHand

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poker_agents as pa  # noqa: E402

warnings.filterwarnings("ignore", message="There is no reason for this player to fold")

OLLAMA = "http://127.0.0.1:11434"
SB, BB, START = 1, 2, 200
AUTO = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.CARD_BURNING,
    Automation.HOLE_DEALING,
    Automation.BOARD_DEALING,
    Automation.RUNOUT_COUNT_SELECTION,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)


def short(model):
    """Display name; keeps an entrant tag such as '#learn'."""
    base, _, tag = model.partition("#")
    name = base.split("/")[-1].split(":")[0].replace("-GGUF", "")
    return f"{name}#{tag}" if tag else name


def notes_block(notes, who):
    """The entrant's own skill and memory notes, if any, as a prompt prefix. Entrant ids are 'model#tag'."""
    text = (notes or {}).get(who, "").strip()
    return f"Your own notes from earlier games (you wrote these; use them if they help):\n{text}\n\n" if text else ""


def cards(cs):
    return " ".join(repr(c) for c in cs) or "-"


def street(state):
    n = len(list(state.get_board_cards(0))) if state.board_count else 0
    return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(n, f"{n} cards")


def ask(model, prompt):
    body = {
        "model": model.partition("#")[0],
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.7, "num_predict": 120},
    }
    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            text = json.load(r)["message"]["content"]
    except urllib.error.HTTPError:  # e.g. a 2k-context model overflowing: an invalid move, not a crash
        text = ""
    return text, round(time.time() - t0, 1)


# --- Simulated humans: rule-based personalities, not language models ---------------------------------------
# Each plays a classic human type with some randomness. None of them learns, so they are fixed anchors on any
# ladder. Parameters: raise/call/limp are Chen-score thresholds preflop; cap is the most (in big blinds) they will
# call preflop; bluff is the chance to bet with nothing; sticky the chance to call a big bet with one pair;
# size the bet as a fraction of the pot.
PERSONAS = {
    "bob": dict(kind="recreational", raise_=10, call=7, limp=5, cap=3, bluff=0.08, sticky=0.2, size=0.5),
    "tina": dict(kind="tight-aggressive", raise_=8, call=99, limp=99, cap=0, bluff=0.12, sticky=0.1, size=0.75),
    "max": dict(kind="maniac", raise_=4, call=2, limp=0, cap=10, bluff=0.35, sticky=0.4, size=1.0),
    "carl": dict(kind="calling station", raise_=13, call=3, limp=2, cap=8, bluff=0.02, sticky=0.8, size=0.33),
    "nora": dict(kind="nit", raise_=12, call=11, limp=99, cap=0, bluff=0.0, sticky=0.05, size=0.5),
}
BOB = "bob"
MAX_HUMANS_PER_TABLE = 2  # capped for now: Bob plus one rotating personality
_BOT_RNG = random.Random(2026)
_HIGH = {"A": 10, "K": 8, "Q": 7, "J": 6}
_VALUE = {r: i for i, r in enumerate("23456789TJQKA", 2)}


def is_bot(entrant):
    """Rule-based or algorithmic players: no language model behind them."""
    return entrant in PERSONAS or entrant in pa.ALGOS or entrant.split("#")[0] in ("tinaplus", "solverbot")


def kind_of(entrant):
    if entrant.split("#")[0] == "tinaplus":
        return "tight-aggressive, stack-aware and balanced (Tina+), simulated human"
    if entrant.split("#")[0] == "solverbot":
        return "the solver stack alone: charts, flop model and solves, Monte Carlo where none applies"
    if entrant in PERSONAS:
        return PERSONAS[entrant]["kind"] + ", simulated human"
    if entrant in pa.ALGOS:
        return pa.ALGOS[entrant]
    if entrant.endswith("#solverfirst"):
        return "language model given solver or Monte Carlo advice every decision"
    if entrant.endswith("#solverplays"):
        return "solver acts wherever it has an answer; language model decides the rest, with advice"
    return "poker-analysis toolkit via tool calls" if entrant.endswith("#toolkit") else "language model"


def bots_for_round(rnd):
    """Bob every round plus one rotating personality (capped at MAX_HUMANS_PER_TABLE), then Monte and Sol."""
    others = [b for b in PERSONAS if b != BOB]
    k = MAX_HUMANS_PER_TABLE - 1
    start = (rnd - 1) * k % len(others)
    return [BOB] + [others[(start + j) % len(others)] for j in range(k)] + list(pa.ALGOS)


def _chen(hole):
    (r1, s1), (r2, s2) = [(repr(c)[0], repr(c)[1]) for c in hole]
    v1, v2 = _VALUE[r1], _VALUE[r2]
    hi = max(v1, v2)
    score = _HIGH.get("23456789TJQKA"[hi - 2], hi / 2)
    if v1 == v2:
        return max(5.0, score * 2)
    score += 2 if s1 == s2 else 0
    gap = abs(v1 - v2) - 1
    score -= {0: 0, 1: 1, 2: 2, 3: 4}.get(gap, 5)
    if gap <= 1 and hi < 12:
        score += 1
    return score


def bot_reply(name, state, i):
    p, rng = PERSONAS[name], _BOT_RNG
    hole = list(state.hole_cards[i])
    board = list(state.get_board_cards(0)) if state.board_count else []
    to_call = state.checking_or_calling_amount
    pot = state.total_pot_amount
    can_raise = state.can_complete_bet_or_raise_to()
    lo = state.min_completion_betting_or_raising_to_amount if can_raise else 0
    hi = state.max_completion_betting_or_raising_to_amount if can_raise else 0

    def raise_to(amount):
        return json.dumps({"action": "raise", "amount": int(max(lo, min(hi, amount)))}) if can_raise else call

    call = json.dumps({"action": "call" if to_call else "check"})
    fold = json.dumps({"action": "fold" if to_call else "check"})
    bet = int(max(BB, pot * p["size"]))
    if not board:
        c = _chen(hole)
        if c >= p["raise_"]:
            return raise_to(to_call + 3 * BB)
        if c >= p["call"] and to_call <= p["cap"] * BB:
            return call
        if c >= p["limp"] and to_call <= BB:
            return call
        return call if to_call <= 2 * BB and rng.random() < p["bluff"] / 2 else fold
    label = StandardHighHand.from_game("".join(map(repr, hole)), "".join(map(repr, board))).entry.label
    strength = list(type(label)).index(label)  # 0 high card, 1 pair, 2 two pair, 3+ trips and better
    if strength >= 2:
        return raise_to(to_call + bet)
    if strength == 1:
        if not to_call:
            return raise_to(bet) if rng.random() < 0.3 + p["bluff"] else call
        return call if to_call <= pot // 2 or rng.random() < p["sticky"] else fold
    if not to_call:
        return raise_to(bet) if rng.random() < p["bluff"] else call
    return call if rng.random() < p["sticky"] / 4 else fold


def parse(text):
    m = re.search(r"\{[^{}]*\}", text)
    if m:
        try:
            obj = json.loads(m.group())
            act = str(obj.get("action", "")).lower().strip()
            amt = obj.get("amount")
            return act, int(amt) if isinstance(amt, (int, float)) or str(amt).isdigit() else None
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    low = text.lower()
    for act in ("fold", "check", "call", "raise", "bet", "all-in", "all in"):
        if act in low:
            n = re.search(rf"{act}\D{{0,12}}(\d+)", low)
            return act, int(n.group(1)) if n else None
    return None, None


_SHORT_RNG = random.Random(77)


def main(outdir, hands, models, notes=None, short_stack=None):
    """short_stack=(lo, hi): every hand deals each player a fresh lo-hi big blinds; chips then track net winnings."""
    os.makedirs(outdir, exist_ok=True)
    log = open(os.path.join(outdir, "table.log"), "a", buffering=1)
    jl = open(os.path.join(outdir, "hands.jsonl"), "a", buffering=1)
    chips = {m: START for m in models}
    stats = {m: {"decisions": 0, "invalid": 0, "vpip_hands": 0, "hands": 0, "seconds": 0.0} for m in models}

    def say(s=""):
        log.write(s + "\n")

    say(f"=== Self-play NLHE, blinds {SB}/{BB}, {START} chips each, {hands} hands max ===")
    say("Seats: " + ", ".join(short(m) for m in models))
    order = list(models)
    for h in range(1, hands + 1):
        alive = list(order) if short_stack else [m for m in order if chips[m] > 0]
        if len(alive) < 2:
            break
        # PokerKit seats: index 0 small blind, 1 big blind, last is the button. Rotate one seat per hand.
        seat = alive[(h - 1) % len(alive) :] + alive[: (h - 1) % len(alive)]
        rs = _SHORT_RNG
        dealt = (
            {m: BB * rs.randint(short_stack[0], short_stack[1]) for m in seat}
            if short_stack
            else {m: chips[m] for m in seat}
        )
        state = NoLimitTexasHoldem.create_state(
            AUTO, True, 0, (SB, BB), BB, tuple(dealt[m] for m in seat), len(seat), mode=Mode.CASH_GAME
        )
        say(
            f"\n--- Hand {h} --- "
            + "  ".join(
                f"{short(m)} {dealt[m]} (net {chips[m] - START:+})" if short_stack else f"{short(m)} {chips[m]}"
                for m in seat
            )
        )
        for i, m in enumerate(seat):
            stats[m]["hands"] += 1
        history, voluntarily = [], set()
        ctx, street_now = {"hand": h, "street_actions": [], "street_pot": 0}, None
        while state.status:
            i = state.actor_index
            if i is None:
                break
            m = seat[i]
            hole, st = cards(state.hole_cards[i]), street(state)
            if st != street_now:
                street_now, ctx["street_actions"], ctx["street_pot"] = st, [], state.total_pot_amount
            bet_before = state.bets[i]
            board = cards(list(state.get_board_cards(0))) if state.board_count else "-"
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
                notes_block(notes, m)
                + "You are playing No-Limit Texas Hold'em for play chips against other AI players.\n"
                f"Street: {street(state)}. Your cards: {cards(state.hole_cards[i])}. Board: {board}.\n"
                f"Pot: {state.total_pot_amount}. Your stack: {state.stacks[i]}. To call: {to_call}.\n"
                f"Players: "
                + ", ".join(
                    f"{short(p)}{' (you)' if j == i else ''} stack {state.stacks[j]}"
                    f"{' folded' if not state.statuses[j] else ''}"
                    for j, p in enumerate(seat)
                )
                + "\n"
                f"Action so far this hand: {'; '.join(history) or 'none'}\n"
                f"Legal actions: {', '.join(legal)}.\n"
                'Reply with ONLY JSON: {"action": "fold"|"check"|"call"|"raise", "amount": <raise-to total, raise only>}'
            )
            t_start = time.time()
            if m.split("#")[0] == "tinaplus":
                import poker_tina_plus

                text = poker_tina_plus.reply(state, i)
            elif m.split("#")[0] == "solverbot":
                text = pa.solver_play(state, i, ctx) or pa.monte_reply(state, i)
            elif m in PERSONAS:
                text = bot_reply(m, state, i)
            elif m in pa.ALGOS:
                text = pa.monte_reply(state, i) if m == "monte" else pa.sol_reply(state, i, ctx)
            elif m.endswith("#solverplays"):
                text = pa.solver_play(state, i, ctx)
                if text is None:
                    text, _ = ask(
                        m, prompt.replace("Legal actions:", pa.solver_brief(state, i, ctx) + "\nLegal actions:", 1)
                    )
            elif m.endswith("#solverfirst"):
                brief = pa.solver_brief(state, i, ctx)
                text, _ = ask(m, prompt.replace("Legal actions:", brief + "\nLegal actions:", 1))
            elif m.endswith("#toolkit"):
                text = pa.toolkit_reply(m, prompt, state=state, i=i, ctx=ctx)
            else:
                text = None
            if text is None:
                text, secs = ask(m, prompt)
            else:
                secs = round(time.time() - t_start, 2)
            stats[m]["decisions"] += 1
            stats[m]["seconds"] += secs
            act, amt = parse(text)
            if act in ("all-in", "all in"):
                act, amt = ("raise", hi) if can_raise else ("call", None)
            elif act == "bet":
                act = "raise"
            done, note = None, ""
            if act == "fold" and state.can_fold():
                state.fold()
                done = "folds"
            elif act in ("check", "call") and state.can_check_or_call():
                state.check_or_call()
                done = f"calls {to_call}" if to_call else "checks"
            elif act == "raise" and can_raise:
                amt = hi if amt is None else max(lo, min(hi, amt))
                state.complete_bet_or_raise_to(amt)
                done = f"raises to {amt}" + (" (all-in)" if amt == hi else "")
            if done is None:
                stats[m]["invalid"] += 1
                note = f" [invalid reply: {' '.join(text.split())[:60]!r}]"
                if state.can_check_or_call() and not to_call:
                    state.check_or_call()
                    done = "checks"
                else:
                    state.fold()
                    done = "folds"
            if st == "preflop" and done.startswith(("calls", "raises")):
                voluntarily.add(m)
            history.append(f"{short(m)} {done}")
            kind = done.split()[0].rstrip("s")
            added = to_call if kind == "call" else (state.bets[i] - bet_before if kind == "raise" else 0)
            ctx["street_actions"].append((kind, added))
            say(f"  [{st:7}] {short(m):22} {hole}  {done}{note}  ({secs}s)")
            jl.write(
                json.dumps(
                    {
                        "hand": h,
                        "model": m,
                        "seat": i,
                        "seats": [short(x) for x in seat],
                        "stacks": list(state.stacks),
                        "bets": list(state.bets),
                        "street": st,
                        "hole": hole,
                        "board": board,
                        "pot": state.total_pot_amount,
                        "to_call": to_call,
                        "legal": legal,
                        "reply": text,
                        "action": done,
                        "invalid": bool(note),
                        "seconds": secs,
                    }
                )
                + "\n"
            )
        for m in voluntarily:
            stats[m]["vpip_hands"] += 1
        board = cards(list(state.get_board_cards(0))) if state.board_count else "-"
        for i, m in enumerate(seat):
            chips[m] += state.stacks[i] - dealt[m]  # net change this hand
        result = ", ".join(f"{short(m)} {'+' if p > 0 else ''}{p}" for m, p in zip(seat, state.payoffs) if p)
        jl.write(
            json.dumps(
                {
                    "hand": h,
                    "end": True,
                    "board": board,
                    "seats": [short(m) for m in seat],
                    "holes": [cards(state.hole_cards[i]) for i in range(len(seat))],
                    "in_at_end": [bool(x) for x in state.statuses],
                    "payoffs": list(state.payoffs),
                    "stacks": list(state.stacks),
                }
            )
            + "\n"
        )
        say(
            f"  board {board} | "
            + "  ".join(f"{short(m)} {cards(state.hole_cards[i])}" for i, m in enumerate(seat) if state.statuses[i])
            + f" | {result}"
        )
    say("\n=== Final chips ===")
    for m in sorted(models, key=lambda m: -chips[m]):
        s = stats[m]
        say(
            f"  {short(m):22} {chips[m]:>5}  ({chips[m] - START:+})  hands {s['hands']}, played {s['vpip_hands']}, "
            f"decisions {s['decisions']}, invalid {s['invalid']}, avg {s['seconds'] / max(1, s['decisions']):.1f}s"
        )
    json.dump({"chips": chips, "stats": stats}, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    return chips, stats


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
