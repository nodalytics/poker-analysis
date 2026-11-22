"""Trainer: whole hands against every bot, the hero following the engine's advice; chips are conserved."""

import os
import random
import sys

os.environ["TEXASSOLVER_BIN"] = "/nonexistent"  # no live solves in tests (the solver bot falls back)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from advisor import trainer  # noqa: E402

trainer.pa.SOLVER = "/nonexistent"  # another test may have imported poker_agents first; never solve in tests


def play(opponents, depth, hands, seed):
    g = trainer.TrainerGame(opponents, depth, seed=seed)
    rng = random.Random(seed)
    for _ in range(hands):
        g.new_hand()
        steps = 0
        while not g.over():
            steps += 1
            assert steps < 200
            if g.to_act() == trainer.HERO:
                mix = g.advice()["mix"]
                kind = rng.choices(list(mix), weights=[v + 1e-9 for v in mix.values()])[0]
                lg = g.legal()
                g.hero_act(kind, lg["min"] if kind == "raise" else None)
            else:
                g.bot_act()
        assert sum(g.table.state.stacks) == sum(g.start)  # chips conserved within the hand
    assert abs(sum(g.net.values())) < 1e-9  # zero-sum (no rake)
    return g


def test_heads_up_every_bot():
    for bot in trainer.BOTS:
        play([bot], "deep", 2, seed=sum(map(ord, bot)))


def test_six_max_short():
    g = play(["max", "carl", "nora", "tina", "solverbot"], "short", 5, seed=7)
    assert sum(g.decisions.values()) > 0
