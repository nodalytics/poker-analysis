"""A poker foundation model, stage 1: a decoder-only transformer pretrained on hands as tokens (poker_tokens.py,
fm_data.py), by next-token prediction, and scored on held-out hands as an opponent model and a card reader.

Model: GPT-style, pre-norm, causal attention, learned positions, the output tied to the input embedding; the default
(d 512, 8 layers, 8 heads, context 512) has about 25 M parameters, Kronos-small's size. Training windows start at a
hand's BOS and run on into the next hands (a window holds several); sources are mixed by --mix.

Held out: the first --val-tokens of each source's first shard (never trained on). Scored there, per source:
  loss      every token, bits
  action    the action tokens (FOLD, CHECK, CALL_b, BET_b, RAISE_b, ALLIN): what a player does next, in bits, against
            a count model of the same tokens given the player's position and the street (from the training shards)
  cards     the cards after REVEAL (what every player held, given the whole hand): bits a card, against the uniform
            guess over the cards not yet seen

Usage (lab, GPU, from research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/fm_train.py DATA OUT [--tokens 2e9] [--mix self:0.4,human:0.5,pluribus:0.1]
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from poker_tokens import ID, VOCAB

ACTION = np.array([t in ("FOLD", "CHECK", "ALLIN") or t.split("_")[0] in ("CALL", "BET", "RAISE") for t in VOCAB])
CARD = np.array([t.startswith("C_") for t in VOCAB])
POS = np.array([t.startswith("POS_") for t in VOCAB])
STREET = {ID["PRE"]: 0, ID["FLOP"]: 1, ID["TURN"]: 2, ID["RIVER"]: 3}


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h = heads

    def forward(self, x):
        n, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(n, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(n, t, d)
        x = x + self.proj(y)
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d=512, layers=8, heads=8, ctx=512):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.ctx = ctx

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        for b in self.blocks:
            h = b(h)
        return self.ln(h) @ self.tok.weight.T  # tied output


class MambaBlock(nn.Module):
    """A pre-norm residual Mamba layer (mamba_ssm: the selective state-space scan, linear in the sequence)."""

    def __init__(self, d):
        super().__init__()
        from mamba_ssm import Mamba

        self.ln, self.m = nn.LayerNorm(d), Mamba(d_model=d, d_state=16, d_conv=4, expand=2)

    def forward(self, x):
        return x + self.m(self.ln(x))


class MambaLM(nn.Module):
    """arch mamba: Mamba layers only (order comes from the recurrence: no position embedding); arch hybrid: every
    attn_every-th layer a causal attention block instead (exact recall within the context, as Jamba mixes them)."""

    def __init__(self, vocab, d=512, layers=16, heads=8, attn_every=0):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(
            Block(d, heads) if attn_every and (k + 1) % attn_every == 0 else MambaBlock(d) for k in range(layers)
        )
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        h = self.tok(x)
        for b in self.blocks:
            h = b(h)
        return self.ln(h) @ self.tok.weight.T


FOCUS_BINS = ((0, 0), (1, 4), (5, 19), (20, 10**9))  # the focus player's earlier hands in the context


class Data:
    """Every source's shards (memory-mapped) and the BOS positions to start windows at; the held-out part apart."""

    def __init__(self, root, mix, ctx, val_tokens, seed=1):
        self.ctx, self.rng = ctx, np.random.default_rng(seed)
        self.src, self.val = {}, {}
        for name, w in mix.items():
            shards = sorted(glob.glob(os.path.join(root, f"{name}*_[0-9][0-9][0-9].npy")))
            if not shards:
                continue
            arrs = [np.load(f, mmap_mode="r") for f in shards]
            held = min(val_tokens, len(arrs[0]) // 5)
            self.val[name] = np.asarray(arrs[0][:held])
            starts = []
            for k, a in enumerate(arrs):
                b = np.nonzero(np.asarray(a) == ID["BOS"])[0]
                if k == 0:
                    b = b[b >= held]
                starts.append(b[b + ctx + 1 <= len(a)])
            self.src[name] = (w, arrs, starts, np.array([len(s) for s in starts], float))
        tot = sum(v[0] for v in self.src.values())
        self.names = list(self.src)
        self.p = np.array([self.src[n][0] / tot for n in self.names])

    def batch(self, B, dev):
        x = np.empty((B, self.ctx + 1), np.int64)
        for i, s in enumerate(self.rng.choice(len(self.names), B, p=self.p)):
            _, arrs, starts, cnt = self.src[self.names[s]]
            k = self.rng.choice(len(arrs), p=cnt / cnt.sum())
            b = starts[k][self.rng.integers(len(starts[k]))]
            x[i] = arrs[k][b : b + self.ctx + 1]
        x = torch.from_numpy(x).to(dev, non_blocking=True)
        return x[:, :-1], x[:, 1:]


def count_model(data, n=20_000_000):
    """P(action token | the acting position token, street) from up to n training tokens of each source, add-one."""
    tab = np.ones((len(VOCAB), 4, len(VOCAB)))
    for _, arrs, starts, _ in data.src.values():
        a = np.asarray(arrs[-1][:n])
        st, prev = 0, -1
        for k in range(len(a)):
            t = a[k]
            if t in STREET:
                st = STREET[t]
            elif ACTION[t] and prev >= 0 and POS[prev]:
                tab[prev, st, t] += 1
            prev = t
    tab /= tab[:, :, ACTION].sum(2, keepdims=True)
    return tab


@torch.no_grad()
def score(model, seq, tab, dev, ctx):
    """Bits a token, an action token (and the count model's), and a REVEAL card (and the uniform guess's)."""
    out = {"tokens": 0, "bits": 0.0, "actions": 0, "action_bits": 0.0, "count_bits": 0.0, "cards": 0,
           "card_bits": 0.0, "uniform_bits": 0.0}  # fmt: skip
    # windows start at hands (BOS) or, in player sequences (SEP between players), at sequence starts
    if bool((seq == ID["SEP"]).any()):
        bos = np.concatenate([[0], np.nonzero(seq == ID["SEP"])[0] + 1])
        bos = bos[bos < len(seq)]
    else:
        bos = np.nonzero(seq == ID["BOS"])[0]
    focus = {b: [0, 0.0] for b in FOCUS_BINS}
    k = 0
    while k < len(bos):  # windows of whole hands, ctx + 1 tokens at most (a longer hand is skipped)
        b0, e = bos[k], bos[k]
        while k < len(bos):
            nxt = bos[k + 1] if k + 1 < len(bos) else len(seq)
            if nxt - b0 > ctx + 1:
                break
            e, k = nxt, k + 1
        if e - b0 < 2:
            k += 1
            continue
        w = seq[b0:e]
        x = torch.from_numpy(w[:-1].astype(np.int64))[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x).float()
        lp = torch.log_softmax(lg[0], -1).cpu().numpy()
        tgt = w[1:]
        nll = -lp[np.arange(len(tgt)), tgt] / math.log(2)
        out["tokens"] += len(tgt)
        out["bits"] += float(nll.sum())
        st, prev, reveal, seen = 0, -1, False, set()
        hand, fpos, last_pos = -1, -1, -1  # the hand's index in its player sequence, the focus seat's position token
        for j, t in enumerate(w):
            if t == ID["SEP"]:
                hand = -1
            if t == ID["BOS"]:
                st, reveal, seen = 0, False, set()
                hand, fpos = hand + 1, -1
            if POS[t] and st == 0 and fpos == -1:
                last_pos = t
            if t == ID["FOCUS"]:
                fpos = last_pos
            if t in STREET:
                st = STREET[t]
            if t == ID["REVEAL"]:
                reveal = True
            if t == ID["END"]:
                reveal = False
            if j >= 1:
                if ACTION[t] and POS[prev] and prev == fpos:  # the focus player acts
                    for b in FOCUS_BINS:
                        if b[0] <= hand <= b[1]:
                            focus[b][0] += 1
                            focus[b][1] += float(nll[j - 1])
                if ACTION[t] and POS[prev]:
                    out["actions"] += 1
                    out["action_bits"] += float(nll[j - 1])
                    out["count_bits"] += -math.log2(tab[prev, st, t])
                if reveal and CARD[t]:
                    out["cards"] += 1
                    out["card_bits"] += float(nll[j - 1])
                    out["uniform_bits"] += math.log2(52 - len(seen))
            if CARD[t]:
                seen.add(int(t))
            prev = t
    r = {"bits_a_token": out["bits"] / max(out["tokens"], 1)}
    if out["actions"]:
        r["action_bits"] = out["action_bits"] / out["actions"]
        r["count_model_action_bits"] = out["count_bits"] / out["actions"]
    if out["cards"]:
        r["reveal_card_bits"] = out["card_bits"] / out["cards"]
        r["uniform_card_bits"] = out["uniform_bits"] / out["cards"]
    r["actions"], r["cards"] = out["actions"], out["cards"]
    for b, (n, bits) in focus.items():
        if n:
            r[f"focus_bits_{b[0]}_{b[1] if b[1] < 10**9 else 'more'}"] = bits / n
            r[f"focus_n_{b[0]}_{b[1] if b[1] < 10**9 else 'more'}"] = n
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--tokens", type=float, default=2e9)
    ap.add_argument("--mix", default="self:0.4,human:0.5,pluribus:0.1")
    ap.add_argument("--arch", choices=("gpt", "mamba", "hybrid"), default="gpt")
    ap.add_argument("--attn-every", type=int, default=4, help="hybrid: an attention layer every this many layers")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-tokens", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=2000, help="steps")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(1)
    mix = {k: float(v) for k, v in (x.split(":") for x in a.mix.split(","))}
    data = Data(a.data, mix, a.ctx, a.val_tokens)
    tab = count_model(data)
    if a.arch == "gpt":
        model = GPT(len(VOCAB), a.d, a.layers, a.heads, a.ctx).to(dev)
    else:
        model = MambaLM(len(VOCAB), a.d, a.layers, a.heads, a.attn_every if a.arch == "hybrid" else 0).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps = int(a.tokens // (a.batch * a.ctx))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.02, anneal_strategy="cos")
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train.log"), "a", buffering=1)
    log.write(f"{nparam / 1e6:.1f} M parameters, {steps} steps of {a.batch} x {a.ctx} tokens, sources {data.names}\n")
    t0 = time.time()
    res = {}
    for step in range(1, steps + 1):
        x, y = data.batch(a.batch, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).float().view(-1, len(VOCAB)), y.reshape(-1), ignore_index=ID["PAD"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0 or step == steps:
            model.eval()
            res = {n: score(model, v, tab, dev, a.ctx) for n, v in data.val.items()}
            model.train()
            seen = step * a.batch * a.ctx
            log.write(f"step {step} {seen / 1e9:.2f} B tokens {time.time() - t0:.0f}s train {loss.item() / math.log(2):.3f} "
                      f"bits | " + " | ".join(f"{n}: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()
                                                                 if isinstance(v, float)) for n, r in res.items()) + "\n")  # fmt: skip
            torch.save({"model": model.state_dict(), "args": vars(a), "step": step}, os.path.join(a.out, "model.pt"))
    json.dump({"parameters": nparam, "tokens": steps * a.batch * a.ctx, "sources": data.names, "mix": mix,
               "held_out": res, "seconds": round(time.time() - t0)}, open(os.path.join(a.out, "result.json"), "w"),
              indent=1)  # fmt: skip


if __name__ == "__main__":
    main()
