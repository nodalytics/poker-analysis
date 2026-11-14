"""What does the training actually push the big blind towards, facing the button's min-raise? Read off the update.

ninemax_sizecheck.py shows that, played out against the same network, the big blind should defend every hand
against the button's min-raise at 20bb (ICM), and that the networks defend about a third and give up ~1.1bb a spot.
Changing the KL pull or NeuRD's logit bound did not move it. This script takes that spot exactly as training sees it
(the network's own policy, sampled, the hand finished by the policies, the result in ICM equity) and computes, per
deal, what one training step does to the big blind's logits there:

  pg      NeuRD: the sampled action's value estimate q = (G - v) / pi(a) on the centred logits, zero where the logit
          is already beyond +-beta in the direction it would move (as ninemax_selfplay.py), divided by the trainer's
          advantage scale (the mean |G - v| over a self-play batch);
  kl      the pull towards the magnet: -kl * d KL(pi || magnet) / d logits;
  ent     the entropy bonus: +entropy * d H(pi) / d logits.

Every deal is weighted by the chance that the players before the big blind (the opponent: the network itself, or a
rule player) fold and min-raise as scripted with their cards, so the big blind faces their real ranges. Averaged
over deals (by hand class, and over all hands) each is the expected move of each action's logit, so the
sum says which way training drifts in this spot and which term wins. It also reads the critic: v against the mean
result G, and the result of each action the policy chose (on-policy; compare ninemax_sizecheck.py's forced ones).

Usage (lab, GPU venv, from llm-research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_signal.py blueprint/nm16/checkpoint.pt \\
      [--opp self|reg|tag|maniac] [--deals 400000] [--out results/ninemax/signal_nm16_self.json]
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from icm import end_stacks, icm, scenarios
from ninemax_engine import BB, Dealer, Menu, Tables, strength
from ninemax_eval import load
from ninemax_selfplay import features, masked, play
from ninemax_sizecheck import class_ids, class_label, probs_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = 9
NAMES = ("fold", "call", "raise", "allin")


def groups(A):
    """Menu actions folded into fold, call, raise (every sized raise), all-in: [A, 4]."""
    m = torch.zeros(A, 4)
    m[0, 0], m[1, 1], m[A - 1, 3] = 1, 1, 1
    m[2 : A - 1, 2] = 1
    return m


def adv_scale(net, dealer, menu, dev, n=8192):
    """The trainer's advantage scale, approximately: mean |G - v| over the decisions of a self-play batch."""
    bk, rk, _ = dealer.deal(n)
    u, rec = play(dealer, menu, [net] * P, bk, rk, record=True, dev=dev)
    g, i, f, _, mybk, street, s, _, _ = (torch.cat([r[k] for r in rec]) for k in range(9))
    with torch.no_grad():
        _, v = net(f, mybk, street, s)
    return float((u[g, i] / BB - v).abs().mean())


def spot(net, magnet, opp, dealer, menu, dev, n, stack_bb, script, hero, cfg, scale):
    """One batch of the spot. Returns per deal: hero's hand class, sampled action, result G (bb), critic v, policy
    [n, A], the pg / kl / ent logit pushes [n, A], and the deal's weight (the chance the players before the hero
    take the scripted actions with their cards, so the hero faces their real ranges)."""
    bk, rk, cd = dealer.deal(n, cards=True)
    pay = scenarios(n, P, dev, avg_bb=(stack_bb, stack_bb))[1].float()
    T = Tables(n, P, menu, dev, stacks=torch.full((n, P), int(stack_bb * BB), device=dev))
    T.pay, T.eq0 = pay, icm(T.stack.float(), pay).float()
    g = torch.arange(n, device=dev)
    z = torch.zeros(n, dtype=torch.long, device=dev)
    w = torch.ones(n, device=dev)  # the chance the players before the hero act as scripted: their real ranges
    for seat, a in [*script, (hero, None)]:
        while int(T.actor[0]) != seat:
            w = w * probs_for(opp, T, g, bk, dealer)[:, 0]
            T.step(g, z)
        if a is not None:
            w = w * probs_for(opp, T, g, bk, dealer)[:, a]
            T.step(g, torch.full((n,), a, dtype=torch.long, device=dev))
    i = T.actor[g]
    s = strength(dealer, bk, T.street).gather(1, i[:, None]).squeeze(1)
    feat, legal = features(T, g, i)
    with torch.no_grad():
        lg, v = net(feat, bk[g, i], T.street, s)
        lm, _ = magnet(feat, bk[g, i], T.street, s)
    logp = F.log_softmax(masked(lg, legal), 1)
    logm = F.log_softmax(masked(lm, legal), 1)
    p = logp.exp()
    act = torch.multinomial(p, 1).squeeze(1)
    T.step(g, act)
    while not bool(T.done.all()):
        gg = (~T.done).nonzero().squeeze(1)
        ii = T.actor[gg]
        a = torch.zeros(len(gg), dtype=torch.long, device=dev)
        mine = ii == hero
        for m, pol in ((mine, net), (~mine, opp)):
            if bool(m.any()):
                a[m] = torch.multinomial(probs_for(pol, T, gg[m], bk, dealer), 1).squeeze(1)
        T.step(gg, a)
    u = T.payoffs(rk)
    eq1 = icm(end_stacks(T.stack, u), pay).float()
    G = ((eq1 - T.eq0) * T.stack.sum(1, keepdim=True).float())[:, hero] / BB

    lf = legal.float()
    nl = lf.sum(1, keepdim=True)
    y = lg.masked_fill(~legal, 0)
    y = y - y.sum(1, keepdim=True) / nl
    y_a = y.gather(1, act[:, None]).squeeze(1)
    p_a = p.gather(1, act[:, None]).squeeze(1)
    q_a = (G - v) / p_a.clamp(min=0.05)
    beta = cfg.get("neurd_beta", 2.0)
    live = (((q_a > 0) & (y_a < beta)) | ((q_a < 0) & (y_a > -beta))).float()
    onehot = F.one_hot(act, menu.A).float()
    pg = (q_a * live)[:, None] * (onehot - lf / nl) / scale  # minus the gradient of -(q_a * y_a)
    lpz, lmz = logp.masked_fill(~legal, 0), logm.masked_fill(~legal, 0)
    kl = (p * (lpz - lmz)).sum(1, keepdim=True)
    H = -(p * lpz).sum(1, keepdim=True)
    klp = -cfg.get("kl", 0.3) * p * (lpz - lmz - kl)
    entp = -cfg.get("entropy", 0.01) * p * (lpz + H)
    return class_ids(torch.as_tensor(cd, device=dev), hero), act, G, v, p, pg, klp * lf, entp * lf, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="a checkpoint.pt (its live net and magnet)")
    ap.add_argument("--opp", default="self", help="self, or a rule player for the other seats: reg, tag, maniac")
    ap.add_argument("--deals", type=int, default=400000)
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--out")
    ap.add_argument("--lib", default=os.path.join(ROOT, "blueprint", "libbp5.so"))
    ap.add_argument("--abs", default=os.path.join(ROOT, "blueprint", "out"))
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(11)
    B = int(open(os.path.join(a.abs, "meta.txt")).read().split()[1])
    dealer = Dealer(a.lib, a.abs, P, B, dev, seed=11)
    bk0, _, _ = dealer.raw(1 << 16)
    pre_s = dealer.pre_strength[torch.tensor(bk0[:, :, 0].ravel(), device=dev)]
    dealer.pre_q = (torch.quantile(pre_s, 0.88).item(), torch.quantile(pre_s, 0.65).item())
    cfg = json.load(open(os.path.join(os.path.dirname(a.run), "config.json")))
    net, _ = load(a.run, B, dev, "net")
    magnet, _ = load(a.run, B, dev, "magnet")
    menu = Menu(cfg.get("menu", "medium"), dev)
    opp = net if a.opp == "self" else a.opp
    scale = adv_scale(net, dealer, menu, dev)
    print(f"{a.run} (live net) vs {a.opp}: kl {cfg.get('kl')}, entropy {cfg.get('entropy')}, "
          f"beta {cfg.get('neurd_beta')}, advantage scale {scale:.2f} bb", flush=True)  # fmt: skip

    S, A = 169, menu.A
    gm = groups(A).to(dev).double()
    acc = {k: torch.zeros(S, 4, device=dev, dtype=torch.float64) for k in ("p", "pg", "kl", "ent", "Gsum", "Gn")}
    cnt = torch.zeros(S, device=dev, dtype=torch.float64)
    vsum = torch.zeros(S, device=dev, dtype=torch.float64)
    gsum = torch.zeros(S, device=dev, dtype=torch.float64)
    for _ in range(a.deals // a.chunk):
        cls, act, G, v, p, pg, klp, entp, wt = spot(net, magnet, opp, dealer, menu, dev, a.chunk, 20,
                                                ((8, 2), (0, 0)), 1, cfg, scale)  # fmt: skip
        wt = wt.double()
        cnt += torch.bincount(cls, weights=wt, minlength=S)
        vsum.index_add_(0, cls, v.double() * wt)
        gsum.index_add_(0, cls, G.double() * wt)
        for k, x in (("p", p), ("pg", pg), ("kl", klp), ("ent", entp)):
            acc[k].index_add_(0, cls, (x.double() @ gm) * wt[:, None])
        oh = (F.one_hot(act, A).double() @ gm) * wt[:, None]
        acc["Gsum"].index_add_(0, cls, oh * G.double()[:, None])
        acc["Gn"].index_add_(0, cls, oh)
    c = cnt.clamp(min=1e-12)[:, None]
    w = (cnt / cnt.sum())[:, None]
    mean = {k: acc[k] / c for k in ("p", "pg", "kl", "ent")}
    tot = mean["pg"] + mean["kl"] + mean["ent"]
    Gact = acc["Gsum"] / acc["Gn"].clamp(min=1e-12)

    def row(x):
        return {n: round(float(y), 5) for n, y in zip(NAMES, x)}

    out = {"run": a.run, "opp": a.opp, "deals": a.deals, "adv_scale_bb": scale, "cfg": {k: cfg.get(k) for k in
           ("kl", "entropy", "neurd_beta", "pg")}, "overall": {}, "classes": {}}  # fmt: skip
    for k, x in (*mean.items(), ("total", tot)):
        out["overall"][k] = row((w * x).sum(0))
    out["overall"]["G_given_action_bb"] = row(acc["Gsum"].sum(0) / acc["Gn"].sum(0).clamp(min=1e-12))
    out["overall"]["critic_v_bb"] = float(vsum.sum() / cnt.sum())
    out["overall"]["mean_G_bb"] = float(gsum.sum() / cnt.sum())
    for k in range(S):
        out["classes"][class_label(k)] = {
            "weight": round(float(cnt[k]), 2), "v": round(float(vsum[k] / c[k, 0]), 3), "G": round(float(gsum[k] / c[k, 0]), 3),
            "policy": row(mean["p"][k]), "G_given_action": row(Gact[k]), "pg": row(mean["pg"][k]),
            "kl": row(mean["kl"][k]), "ent": row(mean["ent"][k]), "total": row(tot[k]),
        }  # fmt: skip
    o = out["overall"]
    print(f"  policy        {o['p']}")
    print(f"  result | act  {o['G_given_action_bb']}  (bb, on-policy)")
    print(f"  critic v {o['critic_v_bb']:+.3f} bb against mean result {o['mean_G_bb']:+.3f} bb")
    for k in ("pg", "kl", "ent", "total"):
        print(f"  push {k:5s}    {o[k]}")
    for name in ("AA", "AKo", "KTo", "J7o", "T4o", "72o"):
        r = out["classes"][name]
        print(f"  {name:4s} v {r['v']:+.2f} G {r['G']:+.2f} policy {r['policy']} | G|a {r['G_given_action']}"
              f" | total {r['total']} | pg {r['pg']}")  # fmt: skip
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
