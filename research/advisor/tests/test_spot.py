"""Spot study: the line notation, positions, and answers on known spots."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from advisor import spot  # noqa: E402

spot.engine.pa.SOLVER = "/nonexistent"


def test_positions():
    assert spot.positions(2) == ["BB", "BTN"]
    assert spot.positions(6) == ["SB", "BB", "UTG", "HJ", "CO", "BTN"]


def test_parse_line():
    assert spot.parse_line("f r2.5 c [Qs7h2c] x a") == [
        ("act", "f"),
        ("act", "r2.5"),
        ("act", "c"),
        ("board", "Qs7h2c"),
        ("act", "x"),
        ("act", "a"),
    ]
    with pytest.raises(ValueError):
        spot.parse_line("f zz")


def test_short_stack_shove_chart():
    r = spot.analyse(2, 15, "BTN", "Kh7c", "")
    assert r["advice"]["source"] == "push/fold chart" and r["advice"]["mix"]["raise"] == 1.0


def test_not_your_turn_and_board_needed():
    assert "not you" in spot.analyse(6, 100, "BTN", "AhKd", "f f")["note"]
    assert "board" in spot.analyse(2, 100, "BB", "AhKd", "r2.5 c")["note"]


def test_postflop_reads_range():
    r = spot.analyse(6, 100, "BB", "AhKd", "f f r2.5 f f c [Qs7h2c] x r3")
    assert r["to_act"] == "BB" and r["equity"] is not None and "CO" in r["ranges"]
