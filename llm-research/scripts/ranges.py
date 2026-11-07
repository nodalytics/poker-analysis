"""Ranges narrowed by the betting, for the solver: Bayesian hand reading with the action model of range_inference.py.

Every one of the 1,326 holdings starts equally likely (minus cards we can see) and is reweighted after each of the
player's actions by P(that action | holding's strength, street, price, raises so far, seat), the model trained on
Pluribus's hands (scored there: the true hand lands in the top 10% half the time after 2+ actions). The result is
turned into TexasSolver's weighted class notation ('AA:1,AKs:0.43,...').

The hand log is ctx["log"]: one entry per action, appended before the action is applied, with
{player, street, board, pot, to_call, raises, seat_feature, action (0 fold, 1 check/call, 2 bet/raise)}.
Seat feature: the player's 6-max seat over 5 (Pluribus convention, 0 small blind .. 1 button); heads-up the button
maps to 1.0 and the big blind to 0.2. Players feature: 6 (the model only saw 6-handed play).
Model: $RANGE_MODEL or range_inference/action_model.joblib (from range_inference.py; not in git, built on the lab).
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get("RANGE_MODEL", os.path.join(ROOT, "range_inference", "action_model.joblib"))
RANKS = "23456789TJQKA"
DECK = [r + s for r in RANKS for s in "cdhs"]
ALL = [a + b for a, b in itertools.combinations(DECK, 2)]
STREETS = ["preflop", "flop", "turn", "river"]
_MODELS = {}
_STRENGTH = {}


def model(path=None):
    """The action model at `path` (default $RANGE_MODEL or the Pluribus-trained one), or False if missing."""
    path = path or os.environ.get("RANGE_MODEL", MODEL_PATH)
    if path not in _MODELS:
        from joblib import load

        _MODELS[path] = load(path) if os.path.exists(path) else False
    return _MODELS[path]


def norm(h):
    a, b = h[:2], h[2:]
    return a + b if DECK.index(a) < DECK.index(b) else b + a


def available():
    return bool(model())


def seat_feature(i, n):
    """PokerKit seat -> the model's seat feature."""
    if n == 2:
        return 1.0 if i == 1 else 0.2
    return i / max(n - 1, 1)


def entry(state, i, street, raises, action):
    board = [repr(c) for c in state.get_board_cards(0)] if state.board_count else []
    return {
        "player": i,
        "street": street,
        "board": board,
        "pot": state.total_pot_amount,
        "to_call": state.checking_or_calling_amount,
        "raises": raises,
        "seat_feature": seat_feature(i, len(state.statuses)),
        "action": action,
    }


def strengths(board):
    key = tuple(board)
    if key not in _STRENGTH:
        import range_inference

        _STRENGTH[key] = range_inference.strengths(list(board))
    return _STRENGTH[key]


def weights(log, player, blocked, model_path=None):
    """Posterior weight per holding (dict combo -> weight, max 1) for `player`, from their actions in `log`."""
    cand = [h for h in ALL if h[:2] not in blocked and h[2:] not in blocked]
    logp = np.zeros(len(cand))
    m = model(model_path)
    for e in log:
        if e["player"] != player:
            continue
        st = strengths(e["board"])
        S = np.array([st.get(h, (0.0, 0.0, 0.0)) for h in cand])
        pot = max(e["pot"], 1)
        ctx = [*(float(e["street"] == x) for x in STREETS), e["to_call"] / pot, float(e["to_call"] > 0)]
        ctx += [e["raises"], e["seat_feature"], 6]
        X = np.hstack([S, np.tile(ctx, (len(cand), 1))])
        P = m.predict_proba(X)[:, e["action"]]
        logp += np.log(np.clip(P, 1e-4, 1))
    w = np.exp(logp - logp.max())
    return dict(zip(cand, w))


def cls(h):
    a, b = h[:2], h[2:]
    hi, lo = sorted((a, b), key=lambda c: RANKS.index(c[0]), reverse=True)
    if hi[0] == lo[0]:
        return hi[0] * 2
    return hi[0] + lo[0] + ("s" if hi[1] == lo[1] else "o")


def solver_range(w, floor=0.02, include=None):
    """Weighted TexasSolver range string from combo weights (class weight = mean over its live combos)."""
    acc = {}
    for h, x in w.items():
        c = cls(h)
        s, n = acc.get(c, (0.0, 0))
        acc[c] = (s + x, n + 1)
    cw = {c: s / n for c, (s, n) in acc.items()}
    top = max(cw.values()) or 1.0
    cw = {c: v / top for c, v in cw.items()}
    if include:
        cw[include] = max(cw.get(include, 0.0), 0.1)
    return ",".join(f"{c}:{v:.3f}" for c, v in sorted(cw.items(), key=lambda x: -x[1]) if v >= floor)
