"""End-to-end tests for mcp_server.py through a real MCP client over stdio.

Run: python -m pytest -q tests/test_mcp_server.py   (needs requirements-mcp.txt installed)
These use assert, unlike test_milestone1.py, so pytest really fails them.
"""

import asyncio
import json
import os
import sys

import pytest

pytest.importorskip("mcp")
from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = StdioServerParameters(command=sys.executable, args=[os.path.join(ROOT, "mcp_server.py")])


def call_all(calls):
    """Run several tool calls in one server session; return (tool names, results)."""

    async def go():
        async with stdio_client(SERVER) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = [t.name for t in (await s.list_tools()).tools]
                out = []
                for name, args in calls:
                    res = await s.call_tool(name, args)
                    text = "".join(c.text for c in res.content if getattr(c, "type", "") == "text")
                    out.append((res.is_error, json.loads(text) if not res.is_error else text))
                return names, out

    return asyncio.run(go())


def test_tools_listed_and_answers_are_right():
    names, res = call_all(
        [
            ("equity_hand_vs_hand", {"hand1": "AhAd", "hand2": "Kh Kd", "simulations": 200000}),
            ("equity_hand_vs_hand", {"hand1": "AhKh", "hand2": "QdQc", "board": "Qh9h3d2s"}),
            ("equity_hand_vs_range", {"hand": "AhAd", "villain_range": "TT+, AJs+, AQo+, KQs", "simulations": 20000}),
            ("evaluate_hand", {"cards": "AhKhQh9h2h3d7c"}),
            ("icm", {"stacks": [50000, 30000, 15000, 5000], "payouts": [10000, 6000, 3500, 1500]}),
            (
                "bubble_factor",
                {"stacks": [50000, 30000, 15000, 5000], "payouts": [10000, 6000, 3500, 1500], "hero": 1, "villain": 0},
            ),
            (
                "final_table",
                {
                    "stacks": [5000, 50000, 15000, 30000],
                    "payouts": [10000, 6000, 3500, 1500],
                    "small_blind": 300,
                    "big_blind": 600,
                    "ante": 75,
                    "names": ["D", "A", "C", "B"],
                },
            ),
            ("push_fold_chart", {"big_blinds": 8, "position": "BTN"}),
        ]
    )
    assert set(names) >= {
        "equity_hand_vs_hand",
        "equity_hand_vs_range",
        "evaluate_hand",
        "icm",
        "bubble_factor",
        "spot_decision",
        "push_fold_chart",
        "final_table",
    }
    assert not any(err for err, _ in res), res

    aa_kk = res[0][1]
    assert 81.0 <= aa_kk["hand1_equity_pct"] <= 83.5  # published value is about 82.4% with ties split
    assert res[1][1]["method"] == "exact"
    rng = res[2][1]
    assert rng["range_combos"] == 70 and rng["range_share_pct"] == 5.28  # not the demo's "~7%"
    assert res[3][1]["hand"] == "Flush"

    icm = res[4][1]
    assert abs(sum(p["icm_value"] for p in icm["players"]) - 21000) < 0.05
    assert icm["players"][0]["icm_value"] == pytest.approx(7521.06, abs=0.01)
    assert res[5][1]["bubble_factor"] == pytest.approx(3.35, abs=0.01)

    # Names must stay with their own stacks after sorting (the GUI's final-table tab gets this wrong).
    by_name = {p["name"]: p["chips"] for p in res[6][1]["players"]}
    assert by_name == {"A": 50000, "B": 30000, "C": 15000, "D": 5000}
    assert [p["name"] for p in res[6][1]["players"]] == ["A", "B", "C", "D"]

    assert "hand-typed" in res[7][1]["source"]


def test_bad_input_is_an_error_not_a_guess():
    _, res = call_all(
        [
            ("equity_hand_vs_hand", {"hand1": "AhAh", "hand2": "KhKd"}),  # duplicate card
            ("equity_hand_vs_hand", {"hand1": "AhXd", "hand2": "KhKd"}),  # bad rank
            ("equity_hand_vs_hand", {"hand1": "AhAd", "hand2": "KhKd", "board": "2c3c"}),  # 2-card board
            ("icm", {"stacks": [1] * 11, "payouts": [1]}),  # too many players
            ("icm", {"stacks": [100, 50], "payouts": [10, 30]}),  # payouts not descending
            ("push_fold_chart", {"big_blinds": 8, "position": "dealer"}),
        ]
    )
    assert all(err for err, _ in res), res
