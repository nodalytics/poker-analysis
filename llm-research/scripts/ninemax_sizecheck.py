"""Is the network's short-stack min-raise, and its tight big blind, actually worse than the textbook play? Measured
by playing it out, not by charts.

The poker-sense checks (ninemax_sense.py) say the tournament networks min-raise at 8bb where the charts shove, and
defend the big blind about half as often as the charts. Either the networks are wrong, or the charts do not fit the
opponents the networks meet. This script plays the spots out against a given opponent and reads the expected
result of each action, by hand class:

  open 8bb   9-handed, everyone 8bb, folded to UTG or the cutoff: each deal is played twice, once with a min-raise
             and once with an all-in, and finished by the policies (hero's own network after its forced action,
             the opponent for everyone else). Folding is 0. Per hand class the best of fold, min-raise and all-in.
  bb 20bb    everyone 20bb, the button min-raises, the small blind folds: the big blind folds, calls or shoves.
             Folding loses the blind (and the ante).

From the class EVs it reports the play a best response to this opponent would make (the share of hands played, the
share of those that shove, the big blind's defence share), next to what the network does, and the EV per hand the
network gives up against that best response (bb). Chips, winner-takes-all payouts (as ninemax_sense.py), the run's
own ante; with --icm, the training's tournament payouts instead (icm.scenarios: final tables, bubbles, in the money)
and results in ICM equity (bb equivalents), as the networks are trained. Every action, folding too, is played out
(under ICM a fold is worth something when others bust). A best response to one opponent is not an equilibrium; the point is whether the network's choice loses
against the players it trains with.

Usage (lab, GPU venv, from llm-research):
  PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_sizecheck.py blueprint/nm16/avg_8h.pt \\
      [--opp self|reg|tag] [--deals 400000] [--out results/ninemax/sizecheck_nm16_8h_self.json]
"""

import argparse
import json
import os

import torch

from icm import end_stacks, icm, scenarios
from ninemax_engine import BB, RULES, Dealer, Menu, Tables, strength
from ninemax_eval import load
from ninemax_pushfold_check import RANKS, policy
from ninemax_selfplay import features, rule_probs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = 9
ICM = {"on": False}


def class_ids(cd, seat):
    """Hand class 0..168 of the hero's cards [G]: row = higher rank, column = lower; suited above the diagonal."""
    c1, c2 = cd[:, 2 * seat].long(), cd[:, 2 * seat + 1].long()
    r1, r2 = c1 // 4, c2 // 4
    hi, lo = torch.maximum(r1, r2), torch.minimum(r1, r2)
    suited = (c1 % 4) == (c2 % 4)
    return torch.where(suited, hi * 13 + lo, lo * 13 + hi)  # pairs on the diagonal


def class_label(k):
    a, b = divmod(k, 13)
    if a == b:
        return RANKS[a] * 2
    return RANKS[max(a, b)] + RANKS[min(a, b)] + ("s" if a > b else "o")


def probs_for(pol, T, g, bk, dealer):
    if isinstance(pol, str):
        i = T.actor[g]
        s = strength(dealer, bk[g], T.street[g]).gather(1, i[:, None]).squeeze(1)
        _, legal = features(T, g, i)
        return rule_probs(pol, T, g, i, legal, s, dealer, bk)
    return policy(pol, T, g, bk, dealer)[0]


def play_spot(net, opp, dealer, menu, dev, deal, stack_bb, script, hero, hero_action):
    """One batch of deals: the scripted actions ((seat, slot), everyone in between folds), everyone before hero
    folds, hero takes hero_action (None: read its policy and stop), the rest is played out. Returns hero's result
    in bb [G], or hero's policy [G, A]."""
    bk, rk, cd, pay = deal
    G = bk.shape[0]
    T = Tables(G, P, menu, dev, stacks=torch.full((G, P), int(stack_bb * BB), device=dev))
    T.pay = pay
    T.eq0 = icm(T.stack.float(), T.pay).float()
    g = torch.arange(G, device=dev)
    z = torch.zeros(G, dtype=torch.long, device=dev)
    for seat, a in [*script, (hero, hero_action)]:
        while int(T.actor[0]) != seat:
            T.step(g, z)
        if a is None:
            return policy(net, T, g, bk, dealer)[0]
        T.step(g, torch.full((G,), a, dtype=torch.long, device=dev))
    while not bool(T.done.all()):
        g = (~T.done).nonzero().squeeze(1)
        i = T.actor[g]
        a = torch.zeros(len(g), dtype=torch.long, device=dev)
        mine = i == hero
        for m, pol in ((mine, net), (~mine, opp)):
            if bool(m.any()):
                a[m] = torch.multinomial(probs_for(pol, T, g[m], bk, dealer), 1).squeeze(1)
        T.step(g, a)
    u = T.payoffs(rk)
    eq1 = icm(end_stacks(T.stack, u), pay).float()  # winner-takes-all: the chip result
    return ((eq1 - T.eq0) * T.stack.sum(1, keepdim=True).float())[:, hero] / BB


def measure(net, opp, dealer, menu, dev, n, chunk, stack_bb, script, hero, actions):
    """Class EVs [169, len(actions)] (bb), class counts [169] and the network's mean policy per class [169, A]."""
    S = 169
    ev = torch.zeros(S, len(actions), device=dev, dtype=torch.float64)
    cnt = torch.zeros(S, device=dev, dtype=torch.float64)
    pol = torch.zeros(S, menu.A, device=dev, dtype=torch.float64)
    for _ in range(n // chunk):
        if ICM["on"]:
            pay = scenarios(chunk, P, dev, avg_bb=(stack_bb, stack_bb))[1].float()
        else:
            pay = torch.zeros(chunk, P, device=dev)
            pay[:, 0] = 1.0
        deal = (*dealer.deal(chunk, cards=True), pay)
        cls = class_ids(torch.as_tensor(deal[2], device=dev), hero)
        cnt += torch.bincount(cls, minlength=S).double()
        pr = play_spot(net, opp, dealer, menu, dev, deal, stack_bb, script, hero, None)
        pol.index_add_(0, cls, pr.double())
        for k, a in enumerate(actions):
            u = play_spot(net, opp, dealer, menu, dev, deal, stack_bb, script, hero, a)
            ev[:, k].index_add_(0, cls, u.double())
    c = cnt.clamp(min=1)[:, None]
    return ev / c, cnt, pol / c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--opp", default="self", help="self (the same network) or a rule player: reg, tag")
    ap.add_argument("--deals", type=int, default=400000)
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--icm", action="store_true", help="tournament payouts and ICM results, as in training")
    ap.add_argument("--out")
    ap.add_argument("--part", default="avg", help="checkpoint part: avg (the averaged policy) or net (the live one)")
    ap.add_argument("--lib", default=os.path.join(ROOT, "blueprint", "libbp5.so"))
    ap.add_argument("--abs", default=os.path.join(ROOT, "blueprint", "out"))
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(7)
    B = int(open(os.path.join(a.abs, "meta.txt")).read().split()[1])
    dealer = Dealer(a.lib, a.abs, P, B, dev, seed=7)
    bk0, _, _ = dealer.raw(1 << 16)
    pre_s = dealer.pre_strength[torch.tensor(bk0[:, :, 0].ravel(), device=dev)]
    dealer.pre_q = (torch.quantile(pre_s, 0.88).item(), torch.quantile(pre_s, 0.65).item())
    ICM["on"] = a.icm
    net, hands = load(a.run, B, dev, a.part)
    d = os.path.dirname(a.run) if a.run.endswith(".pt") else a.run
    menu = Menu(json.load(open(os.path.join(d, "config.json"))).get("menu", "medium"), dev)
    opp = net if a.opp == "self" else a.opp
    ALLIN, MIN = menu.A - 1, 2
    ante = RULES["ante"]
    out = {"run": a.run, "part": a.part, "opp": a.opp, "icm": a.icm, "ante_bb": ante, "deals": a.deals, "spots": {}}
    print(
        f"{a.run} ({a.part}) vs {a.opp}, {'ICM' if a.icm else 'chips'}, ante {ante:g}bb, {a.deals} deals per action",
        flush=True,
    )

    for name, seat in (("UTG", 2), ("CO", 7)):
        ev, cnt, pol = measure(net, opp, dealer, menu, dev, a.deals, a.chunk, 8, (), seat, (0, MIN, ALLIN))
        w = cnt / cnt.sum()
        best = ev.argmax(1)  # 0 fold, 1 min-raise, 2 all-in
        play_br = float(w[best > 0].sum())
        shove_br = float(w[best == 2].sum()) / max(play_br, 1e-9)
        p_play = 1 - pol[:, 0]
        play_net = float((w * p_play).sum())
        shove_net = float((w * pol[:, ALLIN]).sum()) / max(play_net, 1e-9)
        # the network's EV with its own mix, other open sizes counted as the min-raise (an approximation)
        net_ev = (
            pol[:, 0] * ev[:, 0] + pol[:, MIN:ALLIN].sum(1) * ev[:, 1] + pol[:, ALLIN] * ev[:, 2] + pol[:, 1] * ev[:, 1]
        )
        gap = float((w * (ev.max(1).values - net_ev)).sum())
        # on the hands the network plays: how much better is the all-in than the min-raise, weighted by play
        diff = float((w * p_play * (ev[:, 2] - ev[:, 1])).sum()) / max(play_net, 1e-9)
        res = {"best_response_play": play_br, "best_response_shove_share": shove_br, "net_play": play_net,
               "net_shove_share": shove_net, "net_gives_up_bb_per_hand": gap,
               "shove_minus_minraise_bb_on_net_played_hands": diff,
               "classes": {class_label(k): [round(float(x), 3) for x in ev[k]] for k in range(169)}}  # fmt: skip
        out["spots"][f"open 8bb {name}"] = res
        print(f"  open 8bb {name}: best response plays {play_br:.0%}, shoves {shove_br:.0%} of those | network plays "
              f"{play_net:.0%}, shoves {shove_net:.0%} | all-in minus min-raise on the hands it plays {diff:+.2f} bb"
              f" | it gives up {gap:.3f} bb per hand dealt", flush=True)  # fmt: skip

    ev, cnt, pol = measure(net, opp, dealer, menu, dev, a.deals, a.chunk, 20, ((8, MIN), (0, 0)), 1, (0, 1, ALLIN))
    w = cnt / cnt.sum()
    best = ev.argmax(1)
    defend_br = float(w[best > 0].sum())
    defend_net = float((w * (1 - pol[:, 0])).sum())
    call_ok = float(w[ev[:, 1] > ev[:, 0]].sum())
    net_ev = pol[:, 0] * ev[:, 0] + pol[:, 1] * ev[:, 1] + pol[:, 2:ALLIN].sum(1) * ev[:, 1] + pol[:, ALLIN] * ev[:, 2]
    gap = float((w * (ev.max(1).values - net_ev)).sum())
    out["spots"]["bb vs button min-raise 20bb"] = {
        "best_response_defend": defend_br, "call_beats_fold": call_ok, "net_defend": defend_net,
        "net_gives_up_bb_per_hand": gap,
        "classes": {class_label(k): [round(float(x), 3) for x in ev[k]] for k in range(169)},
    }  # fmt: skip
    print(f"  bb vs button min-raise 20bb: best response defends {defend_br:.0%} (calling beats folding with "
          f"{call_ok:.0%}) | network defends {defend_net:.0%} | it gives up {gap:.3f} bb per hand dealt", flush=True)  # fmt: skip
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
