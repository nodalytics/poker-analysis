"""Clones of human professionals, for the blueprint and the MLPs to play against.

A recorded hand cannot be replayed against a new strategy (nobody knows how the human would have answered a different
bet), so the humans are cloned instead: one MLP per professional, plus one for all of them pooled, trained on their
decisions in the Pluribus hands (Brown & Sandholm 2019; the PHH dataset, MIT), where every player's cards are
known. Features and network are the per-player study's (opp_mlp_prep.features, opp_mlp_train.fit); mlp_agent.py
plays a clone as a seat ("clone#MrBlue", "clone#pros").

    extract (main venv):  python scripts/human_clone.py extract PLURIBUS_DIR results/human_clones/decisions.npz
    train   (GPU venv):   python scripts/human_clone.py train results/human_clones/decisions.npz results/human_clones

Caveat: the pros played 6-max at 100bb; at a heads-up 200bb table a clone plays as it would in a two-way pot.
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opp_mlp_prep import hand_examples  # noqa: E402

MIN_HANDS = 2000  # professionals with fewer hands go into the pooled clone only


def rows_of(path):
    """One PHH hand replayed in PokerKit, as poker_selfplay-style decision rows (state before the action, pot and
    stacks after it)."""
    from pokerkit import HandHistory

    hh = HandHistory.load(open(path, "rb"))
    names = list(hh.players)
    rows, pre = [], None
    for state, action in hh.state_actions:
        if pre is not None and action and action.startswith("p") and action.split()[1] in ("f", "cc", "cbr"):
            seat = int(action.split()[0][1:]) - 1
            kind = action.split()[1]
            act = (
                "folds"
                if kind == "f"
                else ("calls" if pre["to_call"] else "checks")
                if kind == "cc"
                else f"raises to {action.split()[2]}"
            )
            rows.append(
                {
                    "model": names[seat],
                    "seat": seat,
                    "seats": names,
                    "street": pre["street"],
                    "hole": pre["holes"][seat],
                    "board": pre["board"],
                    "to_call": pre["to_call"],
                    "pot": state.total_pot_amount,
                    "stacks": list(state.stacks),
                    "action": act,
                    "start_pot": rows[0]["start_pot"] if rows else pre["pot"],
                }
            )
        board = [repr(c) for c in state.get_board_cards(0)] if state.board_count else []
        pre = {
            "street": {0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(len(board), "river"),
            "holes": [" ".join(repr(c) for c in h) for h in state.hole_cards],
            "board": " ".join(board) or "-",
            "to_call": state.checking_or_calling_amount if state.actor_index is not None else 0,
            "pot": state.total_pot_amount,
        }
    return rows


def extract(src, out):
    from multiprocessing import Pool

    files = sorted(glob.glob(os.path.join(src, "**", "*.phh"), recursive=True))
    with Pool(int(os.environ.get("WORKERS", 16))) as pool:
        hands = pool.map(rows_of, files, chunksize=50)
        per = pool.starmap(hand_examples, [(rows, {r["model"] for r in rows}, None) for rows in hands if rows])
    names = sorted({m for ex in per for m, *_ in ex})
    M, F, Y, H = [], [], [], []
    for h, ex in enumerate(per):
        for m, f, c, _ in ex:
            M.append(names.index(m))
            F.append(f)
            Y.append(c)
            H.append(h)
    np.savez_compressed(out, player=np.array(M), feats=np.array(F), y=np.array(Y), hand=np.array(H), names=names)
    counts = np.bincount(M, minlength=len(names))
    print(f"{len(files)} hands, {len(Y)} decisions:", dict(zip(names, counts.tolist())))


def train(data, outdir):
    import torch

    from opp_mlp_train import fit, nll

    d = np.load(data)
    X, y, P, H, names = d["feats"], d["y"].astype(np.int64), d["player"], d["hand"], list(d["names"])
    X[:, 14] = 0  # the heads-up flag: the pros never played heads-up; clones always see 6-max-style inputs
    rng = np.random.default_rng(0)
    hands = np.unique(H)
    rng.shuffle(hands)
    role = np.zeros(H.max() + 1, np.int8)
    role[hands[: len(hands) // 5]] = 2
    role[hands[len(hands) // 5 : len(hands) // 5 + len(hands) // 10]] = 1
    R = role[H]
    os.makedirs(outdir, exist_ok=True)
    report = {}
    pros = np.array([n != "Pluribus" for n in names])[P]
    groups = {"pros": pros}
    for k, n in enumerate(names):
        if (P == k).sum() >= MIN_HANDS:
            groups[n] = P == k
    for g, mask in groups.items():
        tr, va, te = (np.where(mask & (R == r))[0] for r in (0, 1, 2))
        model = fit(X, y, tr, va)
        v, acc = nll(model, X, y, te)
        base = np.bincount(y[tr], minlength=7) + 1
        base = base / base.sum()
        report[g] = {
            "decisions": int(mask.sum()),
            "test_log_loss": float(v.mean()),
            "test_accuracy": acc,
            "prior_log_loss": float(-np.log(base[y[te]]).mean()),
        }
        layers = [m for m in model if isinstance(m, torch.nn.Linear)]
        np.savez(
            os.path.join(outdir, f"{g}.npz"),
            **{f"w{k}": lin.weight.detach().cpu().numpy() for k, lin in enumerate(layers)},
            **{f"b{k}": lin.bias.detach().cpu().numpy() for k, lin in enumerate(layers)},
        )
        print(g, report[g], flush=True)
    json.dump(report, open(os.path.join(outdir, "clones.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3])
    else:
        train(sys.argv[2], sys.argv[3])
