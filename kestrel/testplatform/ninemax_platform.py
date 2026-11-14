"""A small 9-max tournament platform for testing Kestrel's structured connector (connectors/testplatform.yaml).

PokerKit deals and keeps the rules, so the table logic is independent of Kestrel's own. One seat is the client; the
others are rule bots that limp, min-raise and bet odd sizes, so every size the connector meets must be translated.
Stacks vary, the button moves, blinds rise and a big-blind ante is posted; busted players finish in places with
final-table payouts. Binds 127.0.0.1 only.

  POST /join  {"name": "kestrel"}                         -> {"token": ...}       a new tournament
  GET  /state ?token=                                     -> the client's view (below)
  POST /act   {"token", "action": fold|check|call|raise|allin, "amount": street total for raise}

The view: you, hand {id, players [{name, start_stack}], button, sb, bb, ante, to_act, hole, board, actions [{player,
street, type, to}]}, payouts (prize shares of the places still to be paid, first place first), finished, place, prize,
chips. "to" is the player's street total after a bet, raise or all-in.

Usage: uv run python ../kestrel/testplatform/ninemax_platform.py [--port 8793] [--players 9] [--seed 1]
"""

import argparse
import json
import random
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from pokerkit import Automation, Mode, NoLimitTexasHoldem

AUTO = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.CARD_BURNING,
    Automation.HOLE_DEALING,
    Automation.BOARD_DEALING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)
PAYOUTS = [0.30, 0.20, 0.145, 0.105, 0.08, 0.065, 0.05, 0.035, 0.02]
RANKS = "23456789TJQKA"
STREETS = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
BOTS = ["limper", "station", "odd", "tag", "minraiser", "odd", "limper", "tag"]


def strength(hole, board):
    """A rough 0..1 hand strength for the bots: pairs, high cards, and a made pair with the board."""
    r = sorted((RANKS.index(repr(c)[0]) for c in hole), reverse=True)
    s = (r[0] + r[1]) / 24 * 0.6 + (0.35 if r[0] == r[1] else 0) + (0.05 if repr(hole[0])[1] == repr(hole[1])[1] else 0)
    b = [RANKS.index(repr(c)[0]) for c in board]
    if any(x in b for x in r):
        s += 0.3 + 0.02 * max(x for x in r if x in b)
    return min(s, 1.0)


class Tournament:
    def __init__(self, name, players, rng, ante=True, outbox=None):
        self.use_ante, self.outbox, self.board_seen = ante, outbox, 0
        self.rng = rng
        self.names = [name] + [f"{BOTS[k % len(BOTS)]}{k}" for k in range(1, players)]
        self.kind = {n: n.rstrip("0123456789") for n in self.names[1:]}
        self.me = name
        self.rng.shuffle(self.names)
        self.chips = {n: 20 * int(20 * rng.uniform(0.5, 2.5)) for n in self.names}
        self.ring = list(self.names)  # seat order, busted players removed
        self.button = rng.randrange(len(self.ring))
        self.hands = 0
        self.places = {}
        self.state = None
        self.new_hand()

    def bb(self):
        return 20 * round(1.5 ** (self.hands // 10))  # blinds rise every 10 hands

    def new_hand(self):
        while True:
            for n in [n for n in self.ring if self.chips[n] <= 0]:
                self.places[n] = len(self.ring)
                self.ring.remove(n)
            if self.me not in self.ring or len(self.ring) == 1:
                if len(self.ring) == 1:
                    self.places[self.ring[0]] = 1
                self.state = None
                return
            n = len(self.ring)
            self.button %= n
            if n == 2:
                order = [
                    self.ring[(self.button + 1) % n],
                    self.ring[self.button],
                ]  # PokerKit: BB first, then the button
            else:
                order = [self.ring[(self.button + 1 + k) % n] for k in range(n)]
            self.hands += 1
            bb = self.bb()
            self.order, self.sb_, self.bb_, self.ante = order, bb // 2, bb, bb if self.use_ante else 0
            self.start = {p: self.chips[p] for p in order}
            self.actions = []
            self.state = NoLimitTexasHoldem.create_state(
                AUTO, False, {1: self.ante}, (bb // 2, bb), bb, [self.chips[p] for p in order], n,
                mode=Mode.TOURNAMENT,
            )  # fmt: skip
            self.board_seen = 0
            if self.me in order:
                self.out({"type": "hand_start", "hand_id": f"h-{self.hands}", "seat": self.seat(self.me),
                          "dealer_seat": self.seat(order[-1]), "blinds": {"small_blind": bb // 2, "big_blind": bb}})  # fmt: skip
                self.out(
                    {"type": "hole_cards", "cards": [repr(c) for c in self.state.hole_cards[order.index(self.me)]]}
                )
            self.run_bots()
            if self.state is not None:
                return

    def finish_hand(self):
        for p, s in zip(self.order, self.state.stacks):
            self.chips[p] = s
        self.out({"type": "hand_result", "final_stacks": {str(self.seat(p)): self.chips[p] for p in self.ring}})
        self.button = (self.ring.index(self.order[-1]) + 1) % len(self.ring)
        self.state = None

    def seat(self, name):
        return self.names.index(name)

    def out(self, msg):
        if self.outbox is not None:
            self.outbox.append(msg)

    def board_events(self):
        board = [repr(c) for cs in self.state.board_cards for c in cs] if self.state else []
        if len(board) > self.board_seen:
            self.out({"type": "community_cards", "cards": board[self.board_seen :], "street": STREETS[len(board)]})
            self.board_seen = len(board)

    def apply(self, i, action, amount=0):
        """Play a move; the street total is worked out first, as PokerKit collects the bets when a street closes."""
        st = self.state
        street, top, mine = st.street_index, max(st.bets), st.bets[i]
        if action == "fold" and st.can_fold():
            st.fold()
            kind, to = "fold", mine
        elif action in ("raise", "allin") and st.can_complete_bet_or_raise_to():
            lo = st.min_completion_betting_or_raising_to_amount
            hi = st.max_completion_betting_or_raising_to_amount
            to = hi if action == "allin" else max(lo, min(hi, int(round(float(amount)))))
            st.complete_bet_or_raise_to(to)
            kind = "allin" if to == hi else "raise" if top else "bet"
        else:
            to = min(top, mine + st.stacks[i])
            st.check_or_call()
            kind = "call" if top > mine else "check"
        self.actions.append({"player": self.order[i], "street": street, "type": kind, "to": to, "add": to - mine})
        self.out({"type": "player_action", "seat": self.seat(self.order[i]), "name": self.order[i],
                  "action": {"allin": "all_in", "bet": "raise"}.get(kind, kind),
                  "amount": None if kind in ("fold", "check") else to - mine if kind == "call" else to,
                  "street": STREETS[[0, 3, 4, 5][street]], "contribution_delta": to - mine,
                  "stack": st.stacks[i]})  # fmt: skip
        self.board_events()

    def bot(self, i):
        st, name = self.state, self.order[i]
        kind = self.kind[name]
        hole = st.hole_cards[i]
        board = [c for cs in st.board_cards for c in cs]
        s = strength(hole, board)
        to_call = max(st.bets) - st.bets[i]
        pot = st.total_pot_amount
        r = self.rng.random()
        can_raise = st.can_complete_bet_or_raise_to()
        if kind == "station":
            return ("call", 0) if to_call else ("check", 0)
        if kind == "limper":
            if st.street_index == 0 and to_call <= self.bb_ and s > 0.25:
                return "call", 0
            if s > 0.75 and can_raise:
                return "raise", max(st.bets) + pot * 0.63
            return ("call", 0) if to_call and s > 0.45 else ("check", 0) if not to_call else ("fold", 0)
        if kind == "minraiser":
            if s > 0.55 and can_raise:
                return "raise", max(st.bets) * 2 if max(st.bets) else self.bb_
            return ("call", 0) if to_call and s > 0.4 else ("check", 0) if not to_call else ("fold", 0)
        if kind == "odd":
            if s > 0.5 and can_raise and r < 0.7:
                return "raise", max(st.bets) + pot * self.rng.choice([0.27, 0.41, 0.83, 1.37, 2.9])
            if r < 0.1 and can_raise:
                return "allin", 0
            return ("call", 0) if to_call and s > 0.35 else ("check", 0) if not to_call else ("fold", 0)
        # tag
        if s > 0.65 and can_raise:
            return "raise", max(st.bets) * 3 if st.street_index == 0 else max(st.bets) + pot * 0.55
        if (
            st.street_index == 0
            and not any(a["street"] == 0 and a["type"] != "fold" for a in self.actions)
            and s > 0.45
        ):
            return ("raise", 2.3 * self.bb_) if can_raise else ("call", 0)
        return ("call", 0) if to_call and s > 0.5 else ("check", 0) if not to_call else ("fold", 0)

    def run_bots(self):
        while self.state is not None:
            st = self.state
            if not st.status:
                self.finish_hand()
                self.new_hand()
                continue
            i = st.actor_index
            if i is None:
                self.finish_hand()
                self.new_hand()
                continue
            if self.order[i] == self.me:
                return
            self.apply(i, *self.bot(i))

    def view(self):
        alive = len(self.ring)
        if self.me in self.places:
            p = self.places[self.me]
            return {"you": self.me, "finished": True, "place": p, "prize": PAYOUTS[p - 1] if p <= len(PAYOUTS) else 0,
                    "chips": self.chips.get(self.me, 0), "hands": self.hands}  # fmt: skip
        st = self.state
        i = self.order.index(self.me)
        return {
            "you": self.me,
            "finished": False,
            "payouts": PAYOUTS[:alive],
            "chips": self.chips[self.me],
            "hand": {
                "id": self.hands,
                "players": [{"name": p, "start_stack": self.start[p]} for p in self.ring if p in self.start],
                "button": self.order[-1],
                "sb": self.sb_,
                "bb": self.bb_,
                "ante": self.ante,
                "to_act": self.order[st.actor_index] if st.actor_index is not None else None,
                "hole": [repr(c) for c in st.hole_cards[i]],
                "board": [repr(c) for cs in st.board_cards for c in cs],
                "actions": self.actions,
            },
        }


class Platform:
    def __init__(self, players, seed):
        self.players, self.rng, self.lock, self.t = players, random.Random(seed), threading.Lock(), {}

    def handle(self, method, path, q, body):
        with self.lock:
            if method == "POST" and path == "/join":
                tok = secrets.token_hex(8)
                self.t[tok] = Tournament(str(body.get("name", "kestrel")), self.players, self.rng)
                return {"token": tok}
            t = self.t.get(q.get("token", [body.get("token", "")])[0] if method == "GET" else body.get("token", ""))
            if t is None:
                return {"error": "unknown token"}
            if method == "GET" and path == "/state":
                return t.view()
            if method == "POST" and path == "/act":
                st = t.state
                if st is None or st.actor_index is None or t.order[st.actor_index] != t.me:
                    return {"error": "not your turn"}
                a = body.get("action", "")
                if a not in ("fold", "check", "call", "raise", "allin"):
                    return {"error": "unknown action " + a}
                if a == "check" and max(st.bets) > st.bets[st.actor_index]:
                    return {"error": "cannot check facing a bet"}
                t.apply(st.actor_index, a, body.get("amount") or 0)
                t.run_bots()
                return {"ok": True}
            return {"error": "no route"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8793)
    ap.add_argument("--players", type=int, default=9)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    plat = Platform(a.players, a.seed)

    class H(BaseHTTPRequestHandler):
        def reply(self, method):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            out = json.dumps(plat.handle(method, u.path, parse_qs(u.query), body)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            self.reply("GET")

        def do_POST(self):
            self.reply("POST")

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
