"""Play an MLP action model as a seat in poker_selfplay: "clone#<name>" loads results/human_clones/<name>.npz
(human_clone.py), or CLONE_DIR/<name>.npz. NumPy only, so it runs in the main venv without PyTorch.

The inputs are opp_mlp_prep.features, computed from the live PokerKit state exactly as they were from the logs. The
model's action class is sampled and turned into a legal action: 0 fold (check when free), 1 check/call, 2-5 a raise of
the class's mid-point fraction of the pot after calling (opp_mlp_prep.RAISE_FRACTION), 6 all-in.
"""

import json
import os
import random

import numpy as np

from opp_mlp_prep import RAISE_FRACTION, features, split_cards, strength

try:
    from scipy.special import erf
except ImportError:  # the tanh form of GELU differs by under 1e-3
    erf = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS = {}
_RNG = random.Random(4242)


def _gelu(x):
    if erf is not None:
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def load(name):
    if name not in _MODELS:
        d = os.environ.get("CLONE_DIR", os.path.join(ROOT, "results", "human_clones"))
        z = np.load(os.path.join(d, f"{name}.npz"))
        n = len([k for k in z.files if k.startswith("w")])
        _MODELS[name] = [(z[f"w{k}"], z[f"b{k}"]) for k in range(n)]
    return _MODELS[name]


def probs(name, f):
    h = f
    layers = load(name)
    for k, (w, b) in enumerate(layers):
        h = h @ w.T + b
        if k < len(layers) - 1:
            h = _gelu(h)
    e = np.exp(h - h.max())
    return e / e.sum()


def reply(state, i, ctx, entrant):
    name = entrant.split("#", 1)[1]
    hole = split_cards(" ".join(repr(c) for c in state.hole_cards[i]))
    board = split_cards(" ".join(repr(c) for c in state.get_board_cards(0))) if state.board_count else []
    st = {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]
    to_call = state.checking_or_calling_amount
    pot = state.total_pot_amount
    raises = sum(1 for k, _ in ctx.get("street_actions", []) if k == "raise")
    n_in, nseat = sum(1 for s in state.statuses if s), len(state.statuses)
    hs = strength(hole, board, _RNG)
    f = features(st, hs, to_call, pot, state.stacks[i], raises, n_in, i, nseat)
    f[14] = 0  # clones were trained on 6-max hands (human_clone.py)
    p = probs(name, f)
    can_raise = state.can_complete_bet_or_raise_to()
    if not can_raise:
        p[2:] = 0
    if not to_call:
        p[1] += p[0]  # never fold when checking is free
        p[0] = 0
    cls = _RNG.choices(range(7), weights=p.tolist())[0]
    if cls == 0:
        return json.dumps({"action": "fold"})
    if cls == 1:
        return json.dumps({"action": "call" if to_call else "check"})
    lo = state.min_completion_betting_or_raising_to_amount
    hi = state.max_completion_betting_or_raising_to_amount
    if cls == 6:
        return json.dumps({"action": "raise", "amount": hi})
    to = max(state.bets) + RAISE_FRACTION[cls] * (pot + to_call)
    return json.dumps({"action": "raise", "amount": int(max(lo, min(hi, round(to))))})
