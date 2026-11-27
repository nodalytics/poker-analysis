"""A size-head copy of nm25 (bet sizing, plan B): the start of a --size-head run.

SizeHeadNet scores a raise by its size, so it has no per-size rows to copy from nm25. It starts from nm25's trunk
(embeddings, residual blocks, value; the input weights of the public features, and of the fold, call and all-in legal
flags) and nm25's fold, call and all-in rows, then learns nm25's play by distillation: nm25 plays itself in its own game
on its own menu (mtt), and the copy, offered exactly nm25's legal moves and sizes, is trained to give the same
probabilities (KL from nm25's policy) and value. Its size head is then a smooth function that agrees with nm25 at
nm25's three sizes; between and beyond them it interpolates, and the --size-head run's random sizes teach it the rest.

Writes OUT/checkpoint.pt (net, magnet and avg the copy) and OUT/config.json (nm25's, with size_head, nbase, the
training menu size8, play menu grid16 and rule menu mtt), and reports the copy against nm25 at an mtt table (offered
nm25's sizes) and on the grid16 menu.

Usage (lab, GPU, from research):
  CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_sizehead_init.py \
      blueprint/nm25 OUT [--steps 3000] [--hands 4096]
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from ninemax_engine import Menu
from ninemax_eval import load, setup
from ninemax_selfplay import FEAT, SizeHeadNet, board_of, context, duplicate, masked, n_features, play


def to_sizehead(feat, A, S):
    """A slot-menu feature row (features(): public, then A legal flags, then S size fractions / 10) in the size-head
    layout (features(..., sizehead=True)): public, fold/call/all-in legal, any raise, smallest and largest legal
    raise, the S fractions, the S legal flags."""
    nb = feat.shape[1] - A - S
    base, legal, frac = feat[:, :nb], feat[:, nb : nb + A] > 0.5, feat[:, nb + A :]
    ls = legal[:, 2 : 2 + S]
    big = torch.full_like(frac, 9.0)
    anyr = ls.any(1)
    lo = torch.where(anyr, torch.where(ls, frac, big).min(1).values, torch.zeros_like(frac[:, 0]))
    hi = torch.where(anyr, torch.where(ls, frac, -big).max(1).values, torch.zeros_like(frac[:, 0]))
    out = [
        base,
        legal[:, [0, 1, A - 1]].float(),
        anyr.float()[:, None],
        lo[:, None],
        hi[:, None],
        frac * ls,
        ls.float(),
    ]
    return torch.cat(out, 1), legal


def init_from(student, teacher, A, S):
    """Copy the trunk, and the fold, call and all-in rows, of a slot-menu network into a SizeHeadNet."""
    t, s = teacher.state_dict(), student.state_dict()
    for k in s:
        if k.startswith(("emb.", "blocks.", "norm.", "v.")):
            s[k] = t[k].clone()
    emb_tail = 4 * student.emb[0].embedding_dim + 1
    tw = t["inp.weight"]
    nb0 = tw.shape[1] - emb_tail - A - S
    sw = torch.zeros_like(s["inp.weight"])
    sw[:, :nb0] = tw[:, :nb0]
    sw[:, nb0 : nb0 + 3] = tw[:, [nb0, nb0 + 1, nb0 + A - 1]]  # fold, call, all-in legal
    sw[:, -emb_tail:] = tw[:, -emb_tail:]
    s["inp.weight"], s["inp.bias"] = sw, t["inp.bias"].clone()
    s["pi.weight"] = t["pi.weight"][[0, 1, 2, A - 1]].clone()  # raise: the smallest size's row, as a start
    s["pi.bias"] = t["pi.bias"][[0, 1, 2, A - 1]].clone()
    student.load_state_dict(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teacher")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--hands", type=int, default=4096, help="deals of self-play per step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-hands", type=int, default=20000)
    ap.add_argument("--players", type=int, default=9)
    ap.add_argument("--menu", default="mtt")
    ap.add_argument("--abs", default="blueprint/out")
    ap.add_argument("--lib", default="blueprint/libbp7.so")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(1)
    dealer, menu, B = setup(a, dev)
    teacher, hands = load(a.teacher, B, dev)
    cfg = json.load(open(os.path.join(a.teacher, "config.json")))
    assert cfg.get("menu") == "mtt" and menu.name == "mtt", "the teacher plays mtt"
    P, A, S = dealer.P, menu.A, menu.S
    nbase = n_features(P, menu, dev, sizehead=True) - 2 * S
    student = SizeHeadNet(nbase, cfg.get("buckets", B)).to(dev)
    init_from(student, teacher, A, S)
    opt = torch.optim.Adam(student.parameters(), lr=a.lr)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "init.log"), "a", buffering=1)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        bk, rk, cd = dealer.deal(a.hands, cards=FEAT["texture"])
        _, rec = play(dealer, menu, [teacher] * P, bk, rk, record=True, dev=dev, ctx=context(a.hands, P, dev),
                      board=board_of(cd, P, dev) if cd is not None else None)  # fmt: skip
        feat = torch.cat([r[2] for r in rec])
        mybk = torch.cat([r[4] for r in rec])
        street = torch.cat([r[5] for r in rec])
        s = torch.cat([r[6] for r in rec])
        with torch.no_grad():
            lt, vt = teacher(feat, mybk, street, s)
        fs, legal = to_sizehead(feat, A, S)
        lt = masked(lt, legal)
        ls_, vs = student(fs, mybk, street, s)
        ls_ = masked(ls_, legal)
        pt = F.softmax(lt, 1)
        kl = (pt * (F.log_softmax(lt, 1) - F.log_softmax(ls_, 1))).sum(1).mean()
        loss = kl + 0.5 * F.mse_loss(vs, vt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == 1:
            log.write(f"step {step} kl {kl.item():.5f} value mse {F.mse_loss(vs, vt).item():.4f} "
                      f"decisions {len(feat)} {time.time() - t0:.0f}s\n")  # fmt: skip
    student.eval().requires_grad_(False)
    student.texture = bool(cfg.get("texture", False))
    for name in ("mtt", "grid16"):  # offered nm25's own sizes, then the play grid
        student.menu_name = name
        m, lo, hi = duplicate(dealer, menu, student, teacher, a.eval_hands, dev)
        log.write(f"copy on {name} vs 8 x {os.path.basename(a.teacher)}: {m:+.2f} bb/100 [{lo:+.2f}, {hi:+.2f}]\n")
    sd = student.state_dict()
    torch.save({"net": sd, "magnet": sd, "avg": sd, "hands": hands}, os.path.join(a.out, "checkpoint.pt"))
    train_menu = Menu("size8", dev)
    json.dump({**cfg, "size_head": True, "nbase": nbase, "nfeat": nbase + 2 * train_menu.S, "actions": train_menu.A,
               "menu": "size8", "play_menu": "grid16", "rule_menu": "mtt", "init_from": a.teacher},
              open(os.path.join(a.out, "config.json"), "w"), indent=1)  # fmt: skip
    log.write("saved\n")


if __name__ == "__main__":
    main()
