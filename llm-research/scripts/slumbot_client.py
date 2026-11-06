"""Our bots against Slumbot, a strong public heads-up no-limit bot (slumbot.com): the external benchmark.

Usage: uv run python scripts/slumbot_client.py OUTDIR BOT HANDS [WORKERS]
BOT: solverbot (charts + flop model + TexasSolver), adaptbot (solverbot bent toward the opponent's read tendencies),
monte (Monte Carlo equity only), or any poker_selfplay persona.

Slumbot's game: blinds 50/100, 20,000 chips each (200 big blinds), reset every hand; we alternate big blind
(client_pos 0) and small blind (1). Each decision replays Slumbot's action string into a PokerKit state in our units
(1 unit = 50 chips, so the 1/2 blinds and 400-unit stacks match), with our real hole cards, the board as revealed,
and random cards for Slumbot (no bot reads them), then asks the bot. Bets are rounded to the nearest unit (half a big
blind) for the replay only; replies go back in chips, clamped to Slumbot's legal range. The adaptive bot reads
Slumbot's tendencies from every finished hand of its worker's session.

Writes OUTDIR/hands.jsonl (one line per hand: action string, cards, winnings) and OUTDIR/summary.json: big blinds
per 100 hands with a bootstrap 95% interval. Workers each play their own session; keep them few out of courtesy.
"""

import itertools
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pokerkit import Automation, Mode, NoLimitTexasHoldem  # noqa: E402

import poker_selfplay as ps  # noqa: E402

HOST = "https://slumbot.com/api/"
SCALE = 50  # Slumbot chips per unit of ours
BIG, STACK = 100, 20000
AUTO = tuple(a for a in ps.AUTO if a not in (Automation.HOLE_DEALING, Automation.BOARD_DEALING))
DECK = [r + s for r in "23456789TJQKA" for s in "cdhs"]
_HAND_IDS = itertools.count(1)


def post(path, body, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(
                HOST + path, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except Exception:  # network hiccup: back off and retry
            time.sleep(2**k)
    raise RuntimeError(f"slumbot {path} failed")


def tokens(action):
    """Slumbot action string -> list of street lists of ('k'|'c'|'f'|'b', chips street total or None)."""
    streets, k = [[]], 0
    while k < len(action):
        ch = action[k]
        if ch == "/":
            streets.append([])
            k += 1
        elif ch == "b":
            j = k + 1
            while j < len(action) and action[j].isdigit():
                j += 1
            streets[-1].append(("b", int(action[k + 1 : j])))
            k = j
        else:
            streets[-1].append((ch, None))
            k += 1
    return streets


def legal_range(streets):
    """Slumbot's min and max raise-to (street total, chips) and whether a bet is faced, for the player to act."""
    contrib_prev = [0, 0]  # chips in on earlier streets, by Slumbot position (0 BB, 1 SB)
    for si, st in enumerate(streets):
        if si == 0:
            street = [BIG, BIG // 2]
            actor, last, raise_size = 1, BIG, BIG
        else:
            street = [0, 0]
            actor, last, raise_size = 0, 0, BIG
        for a, amt in st:
            if a == "b":
                raise_size = max(raise_size, amt - last)
                last = amt
                street[actor] = amt
            elif a == "c":
                street[actor] = last
            actor = 1 - actor
        if si < len(streets) - 1:
            for p in (0, 1):
                contrib_prev[p] += street[p]
    me = actor
    max_to = STACK - contrib_prev[me]
    return min(last + raise_size, max_to), max_to, street[me] < last, last


def replay(resp, rng):
    """Build our PokerKit state at the current decision. Returns (state, ctx street info)."""
    pos = resp["client_pos"]
    state = NoLimitTexasHoldem.create_state(
        AUTO, True, 0, (ps.SB, ps.BB), ps.BB, (STACK // SCALE, STACK // SCALE), 2, mode=Mode.CASH_GAME
    )
    hole = "".join(resp["hole_cards"])
    board = resp.get("board", [])
    used = set(resp["hole_cards"]) | set(board)
    theirs = "".join(rng.sample([c for c in DECK if c not in used], 2))
    for seat in (0, 1):
        state.deal_hole(hole if seat == pos else theirs)
    streets = tokens(resp["action"])
    board_iter = iter(board)
    street_actions, street_pot = [], state.total_pot_amount
    for si, st in enumerate(streets):
        if si > 0:
            n = 3 if si == 1 else 1
            if state.can_deal_board():
                state.deal_board("".join(next(board_iter) for _ in range(n)))
            street_actions, street_pot = [], state.total_pot_amount
        for a, amt in st:
            i = state.actor_index
            before = state.bets[i]
            to_call = state.checking_or_calling_amount
            if a == "f":
                state.fold()
                street_actions.append(("fold", 0))
            elif a in ("k", "c"):
                state.check_or_call()
                street_actions.append(("call", to_call))
            else:
                if state.can_complete_bet_or_raise_to():
                    lo = state.min_completion_betting_or_raising_to_amount
                    hi = state.max_completion_betting_or_raising_to_amount
                    state.complete_bet_or_raise_to(max(lo, min(hi, round(amt / SCALE))))
                    street_actions.append(("raise", state.bets[i] - before))
                else:
                    state.check_or_call()
                    street_actions.append(("call", to_call))
    # a street ended exactly at the last action: deal the next board Slumbot already showed
    while state.status and state.actor_index is None and state.can_deal_board():
        k = len(list(state.get_board_cards(0)))
        n = 3 if k == 0 else 1
        state.deal_board("".join(next(board_iter) for _ in range(n)))
        street_actions, street_pot = [], state.total_pot_amount
    return state, streets, street_actions, street_pot


def update_profile(prof, action, slum_pos):
    """Slumbot's tendencies from a finished hand's action string, in poker_selfplay's profile format."""
    streets = tokens(action)
    prof["hands"] += 1
    for si, st in enumerate(streets):
        actor = 1 if si == 0 else 0
        facing = si == 0  # the big blind is a bet preflop
        for a, _ in st:
            if actor == slum_pos:
                if si == 0 and a in ("b", "c") and not (a == "c" and not facing):
                    prof["_vp"] = True
                if si > 0:
                    prof["post_raises"] += a == "b"
                    prof["post_calls"] += a == "c"
                    if facing:
                        prof["faced"] += 1
                        prof["folded"] += a == "f"
            facing = a == "b"
            actor = 1 - actor
    if prof.pop("_vp", False):
        prof["vpip"] += 1


def decide(bot, state, i, ctx):
    name = bot.split("#")[0]
    if name == "solverbot":
        return ps.pa.solver_play(state, i, ctx) or ps.pa.monte_reply(state, i)
    if name == "adaptbot":
        return ps.pa.adapt_play(state, i, ctx)
    if name == "monte":
        return ps.pa.monte_reply(state, i)
    return ps.bot_reply(name, state, i)


def session(args):
    outdir, bot, hands, wid = args
    rng = random.Random(wid)
    blank = {"hands": 0, "vpip": 0, "post_raises": 0, "post_calls": 0, "faced": 0, "folded": 0}
    profile = {"us": dict(blank), "slumbot": dict(blank)}
    token, out = None, open(os.path.join(outdir, f"hands_{wid}.jsonl"), "a", buffering=1)
    errors = 0
    for _ in range(hands):
        body = {"token": token} if token else {}
        resp = post("new_hand", body)
        token = resp.get("token", token)
        hid = wid * 10_000_000 + next(_HAND_IDS)
        while "winnings" not in resp:
            if "error_msg" in resp:
                raise RuntimeError(resp["error_msg"])
            state, streets, sa, spot = replay(resp, rng)
            pos = resp["client_pos"]
            seats = ["us", "slumbot"] if pos == 0 else ["slumbot", "us"]
            ctx = {
                "hand": hid,
                "street_actions": sa,
                "street_pot": spot,
                "seats": seats,
                "profile": profile,
                "recent": profile,
            }
            act, amt = ps.parse(decide(bot, state, pos, ctx))
            lo, hi, facing, last = legal_range(streets)
            if act in ("all-in", "all in"):
                act, amt = "raise", None
            if act in ("raise", "bet") and lo <= hi and hi > last:
                chips = hi if amt is None else max(lo, min(hi, amt * SCALE))
                incr = f"b{chips}"
            elif act == "fold" and facing:
                incr = "f"
            else:
                incr = "c" if facing else "k"
            resp = post("act", {"token": token, "incr": incr})
            token = resp.get("token", token)
            if "error_msg" in resp:  # our move was illegal for Slumbot: fall back to check or call once
                errors += 1
                resp = post("act", {"token": token, "incr": "c" if facing else "k"})
                token = resp.get("token", token)
        slum_pos = 1 - resp["client_pos"]
        update_profile(profile["slumbot"], resp["action"], slum_pos)
        out.write(
            json.dumps(
                {
                    "bot": bot,
                    "pos": resp["client_pos"],
                    "hole": resp["hole_cards"],
                    "slumbot_hole": resp.get("bot_hole_cards"),
                    "board": resp.get("board"),
                    "action": resp["action"],
                    "winnings_chips": resp["winnings"],
                }
            )
            + "\n"
        )
    return errors


def main(outdir, bot, hands, workers=2):
    os.makedirs(outdir, exist_ok=True)
    per = -(-hands // workers)
    with ProcessPoolExecutor(workers) as pool:
        errors = sum(pool.map(session, [(outdir, bot, per, w) for w in range(workers)]))
    rows = []
    for w in range(workers):
        rows += [json.loads(x) for x in open(os.path.join(outdir, f"hands_{w}.jsonl"))]
    x = [r["winnings_chips"] / BIG for r in rows]
    rng = random.Random(1)
    boots = sorted(sum(rng.choice(x) for _ in x) / len(x) * 100 for _ in range(1000))
    summ = {
        "bot": bot,
        "hands": len(x),
        "bb100": round(sum(x) / len(x) * 100, 1),
        "ci95": [round(boots[25], 1), round(boots[974], 1)],
        "fallbacks_after_illegal_reply": errors,
    }
    json.dump(summ, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]), *(int(x) for x in a[3:4]))
