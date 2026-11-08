"""Does the small MLP read every kind of player? Next-action log-loss per player, from opp_mlp_prep.py.

For each player, scored on the same held-out 20% of their hands (lower is better):
    freq      frequencies by street, facing a bet and raise count, fitted on the player's own training hands
    pooled    one MLP trained on every OTHER player: the generic model, never seen this player
    own       an MLP trained on this player alone
    own_N     the same on only N of the player's decisions, for N = 300, 1,000, 3,000, 10,000: how fast a player is
              learned, which sets how many hands phase B needs before its read beats the generic one
Intervals: paired bootstrap over hands of own minus pooled.

Usage: .venv-gpu/bin/python scripts/opp_mlp_train.py results/llm_ideas/players.npz OUT.json
"""

import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from opp_mlp_prep import PLAYERS  # noqa: E402

NCLS = 7
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def freq_probs(ftr, ytr, fte):
    """Frequencies by street, facing a bet and raise count (feature columns 0-3, 10 and 9), lightly smoothed."""

    def key(f):
        return np.argmax(f[:, :4], 1) * 100 + f[:, 10].astype(int) * 10 + np.round(f[:, 9] * 4).astype(int)

    ktr, kte = key(ftr), key(fte)
    glob_p = (np.bincount(ytr, minlength=NCLS) + 1) / (len(ytr) + NCLS)
    out = np.tile(glob_p, (len(fte), 1))
    for k in np.unique(kte):
        c = np.bincount(ytr[ktr == k], minlength=NCLS) + 5 * glob_p
        out[kte == k] = c / c.sum()
    return out


def mlp(nf, hidden=128):
    return nn.Sequential(
        nn.Linear(nf, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, NCLS)
    ).to(DEV)


def fit(X, y, tr, va, epochs=60, bs=512, seed=0):
    torch.manual_seed(seed)
    model = mlp(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    Xt, yt = torch.tensor(X, device=DEV), torch.tensor(y, device=DEV)
    tr_t, va_t = torch.tensor(tr, device=DEV), torch.tensor(va, device=DEV)
    best, state, bad = 1e9, None, 0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        perm = tr_t[torch.tensor(rng.permutation(len(tr)), device=DEV)]
        for i in range(0, len(perm), bs):
            b = perm[i : i + bs]
            loss = F.cross_entropy(model(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = F.cross_entropy(model(Xt[va_t]), yt[va_t]).item()
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(state)
    return model


def nll(model, X, y, idx):
    with torch.no_grad():
        lp = F.log_softmax(model(torch.tensor(X[idx], device=DEV)), -1).cpu().numpy()
    return -lp[np.arange(len(idx)), y[idx]], float((lp.argmax(1) == y[idx]).mean())


def boot(d, groups, rng, n=2000):
    uh, inv = np.unique(groups, return_inverse=True)
    s, c = np.bincount(inv, weights=d), np.bincount(inv)
    b = []
    for _ in range(n):
        k = rng.integers(0, len(uh), len(uh))
        b.append(s[k].sum() / c[k].sum())
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


def main():
    d, out = np.load(sys.argv[1]), sys.argv[2]
    X, y, P, H = d["feats"], d["y"].astype(np.int64), d["player"], d["hand"]
    rng = np.random.default_rng(0)
    hands = np.unique(H)
    rng.shuffle(hands)
    n = len(hands)
    role = np.zeros(H.max() + 1, np.int8)  # 0 train, 1 val, 2 test, by hand for everyone
    role[hands[: n // 5]] = 2
    role[hands[n // 5 : n // 5 + n // 10]] = 1
    R = role[H]
    res = {}
    for pi, name in enumerate(PLAYERS):
        me = P == pi
        if me.sum() < 2000:
            continue
        tr, va, te = (np.where(me & (R == k))[0] for k in (0, 1, 2))
        o_tr, o_va = np.where(~me & (R == 0))[0], np.where(~me & (R == 1))[0]
        pf = freq_probs(X[tr], y[tr], X[te])
        r = {"decisions": int(me.sum()), "test": len(te), "classes": np.bincount(y[me], minlength=NCLS).tolist()}
        base = -np.log(pf[np.arange(len(te)), y[te]])
        r["freq"] = float(base.mean())
        pooled, r["pooled_acc"] = nll(fit(X, y, o_tr, o_va), X, y, te)
        own, r["own_acc"] = nll(fit(X, y, tr, va), X, y, te)
        r["pooled"], r["own"] = float(pooled.mean()), float(own.mean())
        r["own_minus_pooled"] = float((own - pooled).mean())
        r["own_minus_pooled_ci"] = boot(own - pooled, H[te], rng)
        for k in (300, 1000, 3000, 10000):
            if k < len(tr):
                sub = rng.choice(tr, k, replace=False)
                v, _ = nll(fit(X, y, sub, va), X, y, te)
                r[f"own_{k}"] = float(v.mean())
        res[name] = r
        print(
            f"{name[:40]:40s} n={me.sum():6d} freq {r['freq']:.3f} pooled {r['pooled']:.3f} own {r['own']:.3f} "
            f"acc {r['own_acc']:.3f}  own-pooled {r['own_minus_pooled']:+.3f} {r['own_minus_pooled_ci']}  "
            + " ".join(f"{k}:{r[k]:.3f}" for k in r if k.startswith("own_") and k[4:].isdigit()),
            flush=True,
        )
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
