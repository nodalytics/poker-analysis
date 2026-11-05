"""Local models play No-Limit Hold'em against each other, with play chips, on a PokerKit engine.

Usage: python3 poker_selfplay.py OUTDIR HANDS MODEL [MODEL ...]
Writes OUTDIR/table.log (readable, follow it live with `tail -f`) and OUTDIR/hands.jsonl (every decision).

PokerKit enforces the rules, side pots and showdowns; models only choose actions. Each model sees its own
hole cards, the board, pot, stacks and the hand's action so far. An unreadable or illegal reply is counted
and replaced by check (if allowed) or fold. Blinds 1/2, 200-chip stacks (100 big blinds), no rebuys.
"""

import json
import os
import re
import sys
import time
import urllib.request
import warnings

from pokerkit import Automation, Mode, NoLimitTexasHoldem

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
    return model.split("/")[-1].split(":")[0].replace("-GGUF", "")


def cards(cs):
    return " ".join(repr(c) for c in cs) or "-"


def street(state):
    n = len(list(state.get_board_cards(0))) if state.board_count else 0
    return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(n, f"{n} cards")


def ask(model, prompt):
    body = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.7, "num_predict": 120},
    }
    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.load(r)["message"]["content"]
    return text, round(time.time() - t0, 1)


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


def main(outdir, hands, models):
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
        alive = [m for m in order if chips[m] > 0]
        if len(alive) < 2:
            break
        # PokerKit seats: index 0 small blind, 1 big blind, last is the button. Rotate one seat per hand.
        seat = alive[(h - 1) % len(alive) :] + alive[: (h - 1) % len(alive)]
        state = NoLimitTexasHoldem.create_state(
            AUTO, True, 0, (SB, BB), BB, tuple(chips[m] for m in seat), len(seat), mode=Mode.CASH_GAME
        )
        say(f"\n--- Hand {h} --- " + "  ".join(f"{short(m)} {chips[m]}" for m in seat))
        for i, m in enumerate(seat):
            stats[m]["hands"] += 1
        history, voluntarily = [], set()
        while state.status:
            i = state.actor_index
            if i is None:
                break
            m = seat[i]
            hole, st = cards(state.hole_cards[i]), street(state)
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
                f"You are playing No-Limit Texas Hold'em for play chips against other AI players.\n"
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
            text, secs = ask(m, prompt)
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
            chips[m] = state.stacks[i]
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


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
