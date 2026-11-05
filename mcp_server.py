#!/usr/bin/env python3
"""MCP server for the poker analysis tools, so a model calculates instead of recalling.

Run (stdio, for a local MCP client):   python3 mcp_server.py
Run (HTTP, for a remote client):        python3 mcp_server.py --http --host 127.0.0.1 --port 8770

Needs: mcp==2.3.0, eval7==0.1.11 (see requirements-mcp.txt). The HTTP transport has no
authentication, so keep it on 127.0.0.1 and reach it through an SSH tunnel.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

from src.analysis.analyzer import TournamentAnalyzer  # noqa: E402
from src.core.cards import parse_hand  # noqa: E402
from src.core.evaluator import evaluate_best  # noqa: E402
from src.equity.calculator import hand_vs_hand_equity, hand_vs_range_equity  # noqa: E402
from src.icm.calculator import bubble_factor as _bubble_factor  # noqa: E402
from src.icm.calculator import icm_equity  # noqa: E402
from src.mtt.tournament import BlindLevel, categorize_stack, push_fold_range  # noqa: E402

MAX_ICM_PLAYERS = 10  # the ICM recursion is factorial in the number of paid places
MAX_SIMULATIONS = 500_000
POSITIONS = ("utg", "mp", "co", "btn", "sb")

mcp = MCPServer(
    name="poker-analysis",
    instructions=(
        "Exact poker tournament maths. Use these tools for any equity, ICM or bubble-factor number "
        "instead of recalling one. Equity is Monte Carlo (preflop and flop) or exact (turn and river). "
        "Cards are rank+suit like 'Ah', 'Td', '2c'; hands may be written 'AhKd' or 'Ah Kd'."
    ),
)


def _cards(text, n=None, what="cards"):
    text = (text or "").replace(",", " ").replace(" ", "")
    if len(text) % 2:
        raise ToolError(f"{what}: '{text}' is not a list of two-character cards like 'AhKd'")
    cards = [text[i : i + 2] for i in range(0, len(text), 2)]
    for c in cards:
        try:
            parse_hand(c)
        except ValueError:
            raise ToolError(f"{what}: '{c}' is not a card (ranks 23456789TJQKA, suits cdhs)") from None
    if n is not None and len(cards) != n:
        raise ToolError(f"{what}: expected {n} cards, got {len(cards)}")
    return [c[0].upper() + c[1].lower() for c in cards]


def _no_duplicates(*groups):
    seen = [c for g in groups for c in g]
    dup = sorted({c for c in seen if seen.count(c) > 1})
    if dup:
        raise ToolError(f"the same card appears twice: {', '.join(dup)}")


def _board(board):
    cards = _cards(board, what="board") if board else []
    if len(cards) not in (0, 3, 4, 5):
        raise ToolError(f"board: expected 0, 3, 4 or 5 cards, got {len(cards)}")
    return cards


def _stacks_payouts(stacks, payouts):
    if not 2 <= len(stacks) <= MAX_ICM_PLAYERS:
        raise ToolError(f"stacks: need 2 to {MAX_ICM_PLAYERS} players, got {len(stacks)}")
    if any(s < 0 for s in stacks) or sum(stacks) == 0:
        raise ToolError("stacks: must be non-negative with at least one chip in play")
    if not payouts or any(p < 0 for p in payouts):
        raise ToolError("payouts: need at least one non-negative payout")
    if list(payouts) != sorted(payouts, reverse=True):
        raise ToolError("payouts: list them from 1st place down")


@mcp.tool()
def equity_hand_vs_hand(hand1: str, hand2: str, board: str = "", simulations: int = 50_000) -> dict:
    """Equity of two specific hands against each other, in percent.

    hand1, hand2: two cards each, e.g. 'AhAd' and 'KhKd'. board: '' preflop, or 3-5 cards.
    Preflop and flop use Monte Carlo with `simulations` runs (results move by a few tenths
    between calls); turn and river are exact.
    """
    h1, h2, b = _cards(hand1, 2, "hand1"), _cards(hand2, 2, "hand2"), _board(board)
    _no_duplicates(h1, h2, b)
    sims = max(1_000, min(simulations, MAX_SIMULATIONS))
    r = hand_vs_hand_equity(h1, h2, b or None, num_simulations=sims)
    return {
        "hand1": "".join(h1),
        "hand2": "".join(h2),
        "board": "".join(b) or "preflop",
        "hand1_equity_pct": r["hand1_equity"],
        "hand2_equity_pct": r["hand2_equity"],
        "method": "exact" if len(b) >= 4 else f"monte carlo, {sims} runs",
    }


@mcp.tool()
def equity_hand_vs_range(hand: str, villain_range: str, board: str = "", simulations: int = 50_000) -> dict:
    """Equity of one hand against a range, in percent.

    villain_range uses PokerStove notation: 'TT+, AJs+, AQo+, KQs'. Also reports how many
    combinations the range holds and what share of all 1,326 starting hands that is.
    """
    h, b = _cards(hand, 2, "hand"), _board(board)
    _no_duplicates(h, b)
    import eval7

    try:
        combos = len(eval7.HandRange(villain_range).hands)
    except Exception:
        raise ToolError(f"villain_range: '{villain_range}' is not a PokerStove range like 'TT+, AJs+'") from None
    sims = max(1_000, min(simulations, MAX_SIMULATIONS))
    eq = hand_vs_range_equity(h, villain_range, b or None, num_simulations=sims)
    return {
        "hand": "".join(h),
        "villain_range": villain_range,
        "board": "".join(b) or "preflop",
        "equity_pct": eq,
        "range_combos": combos,
        "range_share_pct": round(combos / 1326 * 100, 2),
        "method": "exact" if len(b) >= 4 else f"monte carlo, {sims} runs",
    }


@mcp.tool()
def evaluate_hand(cards: str) -> dict:
    """Best five-card hand from 5 to 7 cards, e.g. 'AhKhQh9h2h3d7c' gives Flush."""
    c = _cards(cards, what="cards")
    if not 5 <= len(c) <= 7:
        raise ToolError(f"cards: expected 5 to 7 cards, got {len(c)}")
    _no_duplicates(c)
    best = evaluate_best(parse_hand("".join(c)))
    return {"cards": "".join(c), "hand": best.name, "best_five": " ".join(str(x) for x in best.cards)}


@mcp.tool()
def icm(stacks: list[int], payouts: list[float], names: list[str] | None = None) -> dict:
    """Dollar value of each stack under the Independent Chip Model (Malmuth-Harville).

    stacks in seat order; payouts from 1st place down. Up to 10 players.
    """
    _stacks_payouts(stacks, payouts)
    eq = icm_equity(stacks, payouts)
    names = names or [f"Player {i + 1}" for i in range(len(stacks))]
    total = sum(stacks)
    return {
        "players": [
            {"name": n, "chips": s, "chip_pct": round(s / total * 100, 2), "icm_value": v}
            for n, s, v in zip(names, stacks, eq)
        ],
        "prize_pool": sum(payouts),
    }


@mcp.tool()
def bubble_factor(stacks: list[int], payouts: list[float], hero: int, villain: int) -> dict:
    """Bubble factor for hero calling villain's all-in (indexes into `stacks`, from 0).

    Bubble factor = ICM loss if hero busts or loses the overlap / ICM gain if hero wins it.
    Required equity to call ignoring dead money = BF / (1 + BF). Blinds and antes already in
    the pot lower the real requirement; this figure leaves them out.
    """
    _stacks_payouts(stacks, payouts)
    for k, v in (("hero", hero), ("villain", villain)):
        if not 0 <= v < len(stacks):
            raise ToolError(f"{k}: index {v} is outside 0..{len(stacks) - 1}")
    if hero == villain:
        raise ToolError("hero and villain must be different players")
    bf = _bubble_factor(stacks, payouts, hero, villain)
    return {"bubble_factor": bf, "required_equity_pct_ignoring_dead_money": round(bf / (1 + bf) * 100, 2)}


@mcp.tool()
def spot_decision(
    hero_hand: str,
    villain_range: str,
    stacks: list[int],
    payouts: list[float],
    hero: int,
    villain: int,
    small_blind: int,
    big_blind: int,
    ante: int = 0,
    board: str = "",
) -> dict:
    """Call-or-fold for hero facing villain's all-in: card equity against the range versus ICM-required equity.

    Ignores the blinds and antes already in the pot, so it leans towards folding.
    """
    _stacks_payouts(stacks, payouts)
    if hero == villain or not (0 <= hero < len(stacks) and 0 <= villain < len(stacks)):
        raise ToolError("hero and villain must be two different valid indexes into stacks")
    h, b = _cards(hero_hand, 2, "hero_hand"), _board(board)
    _no_duplicates(h, b)
    eq = hand_vs_range_equity(h, villain_range, b or None)
    bf = _bubble_factor(stacks, payouts, hero, villain)
    req = bf / (1 + bf) * 100
    level = BlindLevel(level=1, small_blind=small_blind, big_blind=big_blind, ante=ante)
    bb = level.effective_bb(stacks[hero])
    return {
        "card_equity_pct": eq,
        "bubble_factor": bf,
        "required_equity_pct": round(req, 2),
        "decision": "CALL" if eq > req else "FOLD",
        "equity_surplus_pct": round(eq - req, 2),
        "hero_bb": round(bb, 1),
        "hero_zone": categorize_stack(bb)["zone"],
    }


@mcp.tool()
def push_fold_chart(big_blinds: float, position: str) -> dict:
    """Suggested open-shove range for a short stack.

    WARNING: a hand-typed table (5 positions x 15/12/10/8/6 big blinds), not a Nash or ICM
    calculation; it ignores antes, table size and payouts. Treat it as a starting point and
    check a specific hand with equity_hand_vs_range and spot_decision.
    """
    pos = position.lower()
    if pos not in POSITIONS:
        raise ToolError(f"position: one of {', '.join(POSITIONS)}")
    return {
        "big_blinds": big_blinds,
        "position": pos,
        "push_range": push_fold_range(big_blinds, pos),
        "zone": categorize_stack(big_blinds)["zone"],
        "source": "hand-typed table in src/mtt/tournament.py",
    }


@mcp.tool()
def final_table(
    stacks: list[int],
    payouts: list[float],
    small_blind: int,
    big_blind: int,
    ante: int = 0,
    names: list[str] | None = None,
) -> dict:
    """Final-table report: chips, ICM value, big blinds, M-ratio and zone per player, biggest stack first.

    M-ratio counts one ante per player at this table (the analyzer's own m_ratio assumes 9).
    """
    _stacks_payouts(stacks, payouts)
    names = names or [f"Player {i + 1}" for i in range(len(stacks))]
    if len(names) != len(stacks):
        raise ToolError("names: one per stack")
    level = BlindLevel(level=1, small_blind=small_blind, big_blind=big_blind, ante=ante)
    report = TournamentAnalyzer().final_table_report(
        stacks=stacks, payouts=payouts, player_names=names, blind_level=level
    )
    n = len(stacks)
    for p in report["players"]:
        p["zone"] = p["zone"]["zone"]
        p["m_ratio"] = round(level.m_ratio(p["chips"], num_players=n), 1)
    return {"players": report["players"], "prize_pool": report["total_prizes"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    if a.http:
        mcp.run("streamable-http", host=a.host, port=a.port)
    else:
        mcp.run("stdio")
