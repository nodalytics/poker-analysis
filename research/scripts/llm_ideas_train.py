"""Do the model styles behind LLMs read an opponent better? Next-action prediction on Slumbot's decisions.

Data: llm_ideas_prep.py. The same held-out hands score every model, and the score is log-loss (lower is better) on
Slumbot's action class. Models, from simplest to most LLM-like:
    freq        frequencies by street, facing a bet or not, and raise count (the generic model we use now)
    logreg      logistic regression on the hand-made features (street, hand strength, price, pot, stack, raises)
    mlp         two hidden layers on the same features
    lstm        a recurrent network over the token sequence, plus the features (the pre-transformer sequence model)
    tf_raw      a transformer over the tokens alone, end to end, as an LLM learns: cards and actions, no features
    tf          a transformer over the tokens plus the features
    tf_moe      the same with a mixture-of-experts feed-forward layer (4 experts, top-1 routing)
    tf_s/tf_l   the transformer at a quarter and at 4x the width, for a small scaling check
Intervals: a paired bootstrap over hands of each model's log-loss minus the MLP's.

Usage: .venv-gpu/bin/python scripts/llm_ideas_train.py results/llm_ideas/slumbot_actions.npz OUT.json
"""

import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB, NCLS = 72, 7
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


class FeatNet(nn.Module):
    def __init__(self, nf, hidden):
        super().__init__()
        self.net = (
            nn.Linear(nf, NCLS)
            if hidden == 0
            else nn.Sequential(
                nn.Linear(nf, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, NCLS)
            )
        )

    def forward(self, tok, feat):
        return self.net(feat)


class LSTMNet(nn.Module):
    def __init__(self, nf, d=128):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d, padding_idx=0)
        self.rnn = nn.LSTM(d, d, batch_first=True)
        self.head = nn.Sequential(nn.Linear(d + nf, d), nn.GELU(), nn.Linear(d, NCLS))

    def forward(self, tok, feat):
        lens = (tok != 0).sum(1)
        h, _ = self.rnn(self.emb(tok))
        last = h[torch.arange(len(tok), device=tok.device), lens - 1]
        return self.head(torch.cat([last, feat], 1))


class MoE(nn.Module):
    def __init__(self, d, experts=4):
        super().__init__()
        self.gate = nn.Linear(d, experts)
        self.ff = nn.ModuleList(
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(experts)
        )

    def forward(self, x):
        g = F.softmax(self.gate(x), -1)
        top = g.argmax(-1)
        out = torch.zeros_like(x)
        for e, ff in enumerate(self.ff):
            m = top == e
            if m.any():
                out[m] = ff(x[m]) * g[..., e][m].unsqueeze(-1)
        return out


class Block(nn.Module):
    def __init__(self, d, heads, moe):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = MoE(d) if moe else nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, pad):
        h = self.n1(x)
        x = x + self.att(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        return x + self.ff(self.n2(x))


class TFNet(nn.Module):
    """Pre-norm transformer encoder; reads the hidden state at the final (query) token, like an LLM's next token."""

    def __init__(self, nf, d=64, layers=3, heads=4, use_feat=True, moe=False, maxlen=48):
        super().__init__()
        self.use_feat = use_feat
        self.emb = nn.Embedding(VOCAB, d, padding_idx=0)
        self.pos = nn.Embedding(maxlen, d)
        self.blocks = nn.ModuleList(Block(d, heads, moe) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d + (nf if use_feat else 0), d), nn.GELU(), nn.Linear(d, NCLS))

    def forward(self, tok, feat):
        pad = tok == 0
        x = self.emb(tok) + self.pos(torch.arange(tok.shape[1], device=tok.device))
        for b in self.blocks:
            x = b(x, pad)
        last = self.norm(x)[torch.arange(len(tok), device=tok.device), (~pad).sum(1) - 1]
        return self.head(torch.cat([last, feat], 1) if self.use_feat else last)


def freq_probs(ftr, ytr, fte):
    def key(f):
        return np.argmax(f[:, :4], 1) * 100 + f[:, 9].astype(int) * 10 + np.round(f[:, 8] * 4).astype(int)

    ktr, kte = key(ftr), key(fte)
    glob_p = (np.bincount(ytr, minlength=NCLS) + 1) / (len(ytr) + NCLS)
    out = np.tile(glob_p, (len(fte), 1))
    for k in np.unique(kte):
        m = ktr == k
        c = np.bincount(ytr[m], minlength=NCLS) + 5 * glob_p
        out[kte == k] = c / c.sum()
    return out


def train(model, tok, feat, y, tr, va, epochs=40, lr=2e-3, bs=512):
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    T, Fe, Y = (torch.tensor(a, device=DEV) for a in (tok, feat, y))
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.tensor(np.random.permutation(tr), device=DEV)
        for i in range(0, len(perm), bs):
            b = perm[i : i + bs]
            loss = F.cross_entropy(model(T[b], Fe[b]), Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        vl = F.cross_entropy(predict_logits(model, T, Fe, va), Y[torch.tensor(va, device=DEV)]).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_logits(model, T, Fe, idx, bs=4096):
    model.eval()
    idx = torch.tensor(idx, device=DEV)
    return torch.cat([model(T[idx[i : i + bs]], Fe[idx[i : i + bs]]) for i in range(0, len(idx), bs)])


def main():
    data, out = np.load(sys.argv[1]), sys.argv[2]
    tok, feat, y, hand = data["tokens"].astype(np.int64), data["feats"], data["y"].astype(np.int64), data["hand"]
    nf = feat.shape[1]
    rng = np.random.default_rng(0)
    hands = np.unique(hand)
    rng.shuffle(hands)
    n = len(hands)
    test_h, val_h = set(hands[: n // 5]), set(hands[n // 5 : n // 5 + n // 10])
    te = np.array([i for i in range(len(y)) if hand[i] in test_h])
    va = np.array([i for i in range(len(y)) if hand[i] in val_h])
    tr = np.array([i for i in range(len(y)) if hand[i] not in test_h and hand[i] not in val_h])
    print(f"train {len(tr)} val {len(va)} test {len(te)} examples on {DEV}", flush=True)
    T, Fe = torch.tensor(tok, device=DEV), torch.tensor(feat, device=DEV)
    nll = {}
    info = {}
    p = freq_probs(feat[tr], y[tr], feat[te])
    nll["freq"] = -np.log(p[np.arange(len(te)), y[te]])
    info["freq"] = {"params": 0}
    builders = {
        "logreg": lambda: FeatNet(nf, 0),
        "mlp": lambda: FeatNet(nf, 128),
        "lstm": lambda: LSTMNet(nf),
        "tf_raw": lambda: TFNet(nf, use_feat=False),
        "tf": lambda: TFNet(nf),
        "tf_moe": lambda: TFNet(nf, moe=True),
        "tf_s": lambda: TFNet(nf, d=32, heads=2),
        "tf_l": lambda: TFNet(nf, d=128, layers=4, heads=8),
    }
    for name, make in builders.items():
        t0 = time.time()
        model = train(make(), tok, feat, y, tr, va)
        lp = F.log_softmax(predict_logits(model, T, Fe, te), -1).cpu().numpy()
        nll[name] = -lp[np.arange(len(te)), y[te]]
        acc = float((lp.argmax(1) == y[te]).mean())
        info[name] = {"params": sum(q.numel() for q in model.parameters()), "acc": acc, "seconds": time.time() - t0}
        print(f"{name}: log-loss {nll[name].mean():.4f} acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
    # paired bootstrap over test hands against the MLP
    th = hand[te]
    uh, inv = np.unique(th, return_inverse=True)
    res = {}
    for name, v in nll.items():
        d = v - nll["mlp"]
        per_h = np.bincount(inv, weights=d, minlength=len(uh))
        cnt = np.bincount(inv, minlength=len(uh))
        boots = []
        for _ in range(2000):
            s = rng.integers(0, len(uh), len(uh))
            boots.append(per_h[s].sum() / cnt[s].sum())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res[name] = {
            "log_loss": float(v.mean()),
            "vs_mlp": float(d.mean()),
            "vs_mlp_ci": [float(lo), float(hi)],
            "perplexity": float(math.exp(v.mean())),
            **info[name],
        }
    res["_meta"] = {"train": len(tr), "val": len(va), "test": len(te), "classes": np.bincount(y, minlength=7).tolist()}
    json.dump(res, open(out, "w"), indent=1)
    for k, r in res.items():
        if not k.startswith("_"):
            print(
                f"{k:8s} {r['log_loss']:.4f}  vs mlp {r['vs_mlp']:+.4f} [{r['vs_mlp_ci'][0]:+.4f}, {r['vs_mlp_ci'][1]:+.4f}]"
            )


if __name__ == "__main__":
    main()
