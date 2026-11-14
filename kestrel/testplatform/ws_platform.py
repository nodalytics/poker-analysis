"""The test platform over a WebSocket, speaking Open Poker's bot protocol (docs.openpoker.ai, revision 2026-09-02).

For testing Kestrel's WebSocket connectors (connectors/testplatform-ws.yaml) before an Open Poker account exists. The
table is the same PokerKit tournament as ninemax_platform.py, without an ante (Open Poker has none). Messages:

  client: join_lobby, action {hand_id, action, amount?, client_action_id, turn_token}
  server: connected, lobby_joined, table_joined, hand_start, hole_cards, player_action (with contribution_delta),
          community_cards, your_turn {valid_actions, turn_token, players, pot}, action_ack, action_rejected {code},
          hand_result {final_stacks}, busted, error {code}

The key is checked as Open Poker checks it: `Authorization: Bearer <key>`, else error auth_failed and close 4001.
Binds 127.0.0.1 only.

Usage: uv run --with websockets==14.1 python kestrel/testplatform/ws_platform.py [--port 8798] [--key test]
"""

import argparse
import asyncio
import json
import random
import secrets
import sys
from pathlib import Path

from websockets.asyncio.server import serve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ninemax_platform import Tournament  # noqa: E402


class Session:
    def __init__(self, ws, rng, players):
        self.ws, self.rng, self.players = ws, rng, players
        self.t, self.outbox, self.token = None, [], None

    async def flush(self):
        """Send what the table did, then our turn if it is ours."""
        while self.outbox:
            await self.ws.send(json.dumps(self.outbox.pop(0)))
        t = self.t
        if t is None:
            return
        if t.state is None:
            if t.me in t.places:
                await self.ws.send(json.dumps({"type": "busted", "options": ["rebuy", "leave"]}))
                self.t = None
            return
        st = t.state
        if st.actor_index is None or t.order[st.actor_index] != t.me or self.token:
            return
        i = st.actor_index
        to_call = max(st.bets) - st.bets[i]
        valid = [{"action": "fold"}]
        valid.append({"action": "call", "amount": min(to_call, st.stacks[i])} if to_call else {"action": "check"})
        if st.can_complete_bet_or_raise_to():
            lo, hi = st.min_completion_betting_or_raising_to_amount, st.max_completion_betting_or_raising_to_amount
            valid.append({"action": "raise", "min": lo, "max": hi})
            valid.append({"action": "all_in", "amount": hi})
        self.token = secrets.token_hex(8)
        await self.ws.send(json.dumps({
            "type": "your_turn", "hand_id": f"h-{t.hands}", "valid_actions": valid, "pot": st.total_pot_amount,
            "community_cards": [repr(c) for cs in st.board_cards for c in cs],
            "players": [{"seat": t.seat(p), "name": p, "stack": s} for p, s in zip(t.order, st.stacks)],
            "turn_token": self.token,
        }))  # fmt: skip

    async def on(self, m):
        kind = m.get("type")
        if kind == "join_lobby":
            if self.t is not None:
                return await self.ws.send(json.dumps({"type": "error", "code": "already_seated", "message": "seated"}))
            await self.ws.send(json.dumps({"type": "lobby_joined", "position": 1, "estimated_wait": "~0s"}))
            self.outbox.clear()
            self.t = Tournament("kestrel", self.players, self.rng, ante=False, outbox=self.outbox)
            # table_joined goes before the first hand's events
            self.outbox.insert(0, {"type": "table_joined", "table_id": "t-1", "seat": self.t.seat("kestrel"),
                                   "players": [{"seat": self.t.seat(p), "name": p, "stack": self.t.start.get(p, 0)}
                                               for p in self.t.names]})  # fmt: skip
            return await self.flush()
        if kind != "action":
            return await self.ws.send(json.dumps({"type": "error", "code": "unknown_message", "message": kind}))
        t = self.t

        async def reject(code):
            await self.ws.send(
                json.dumps({"type": "action_rejected", "code": code, "reason": code, "details": {"code": code}})
            )

        if t is None or t.state is None:
            return await reject("not_at_table")
        if m.get("hand_id") != f"h-{t.hands}":
            return await reject("stale_hand_action")
        if not self.token or m.get("turn_token") != self.token:
            return await reject("stale_turn_token")
        st, i = t.state, t.state.actor_index
        a = m.get("action")
        amount = m.get("amount")
        if a == "raise":
            ok = (
                isinstance(amount, int)
                and st.can_complete_bet_or_raise_to()
                and (
                    st.min_completion_betting_or_raising_to_amount
                    <= amount
                    <= st.max_completion_betting_or_raising_to_amount
                )
            )
            if not ok:
                return await reject("invalid_action")
        elif a == "check" and max(st.bets) > st.bets[i]:
            return await reject("invalid_action")
        elif a not in ("fold", "check", "call", "raise", "all_in"):
            return await reject("invalid_action")
        self.token = None
        await self.ws.send(
            json.dumps({"type": "action_ack", "client_action_id": m.get("client_action_id"), "status": "accepted"})
        )
        t.apply(i, {"all_in": "allin"}.get(a, a), amount or 0)
        t.run_bots()
        await self.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8798)
    ap.add_argument("--players", type=int, default=6)
    ap.add_argument("--key", default="test")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    async def handler(ws):
        if ws.request.headers.get("Authorization") != f"Bearer {a.key}":
            await ws.send(json.dumps({"type": "error", "code": "auth_failed", "message": "invalid key"}))
            return await ws.close(4001, "auth_failed")
        await ws.send(json.dumps({"type": "connected", "agent_id": "test", "name": "kestrel"}))
        s = Session(ws, rng, a.players)
        async for raw in ws:
            try:
                m = json.loads(raw)
            except ValueError:
                await ws.send(json.dumps({"type": "error", "code": "invalid_message", "message": "not JSON"}))
                continue
            await s.on(m)

    async with serve(handler, "127.0.0.1", a.port):
        await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    asyncio.run(main())
