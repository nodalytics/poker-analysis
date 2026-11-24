"""A3: does an ICM push/fold chart player beat nm25 at short-stacked ICM tables, in prize equity?

The chart player is nm25 except before the flop, in the three spots the push/fold model covers:
  first in          (everyone before it folded, no limp): shove all in or fold, from the chart's shove range;
  facing a shove    (one player shoved first in, everyone else folded): call or fold, from the chart's call range;
  shove and a call  (one player shoved first in, one called, the rest folded): call or fold, from the overcall range.
Everywhere else (a limp or a raise that is not all in before it, a re-shove, three or more players in, after the
flop) it plays nm25's policy. The charts are icm_pushfold.solve2: up to two callers, ICM (Malmuth-Harville, as the
engine's reward), the engine's big-blind ante, the table's real payouts.

Tables (fixed configurations, so every chart is solved once for the exact stacks it plays): K stack vectors per stage,
9 players, average stack log-uniform in --avg (8-12.5bb by default), each stack avg x lognormal(sd 0.7) clamped at
2bb (icm.scenarios' spread). Two stages:
  final   a 9-handed final table, places 1-9 paid: FT9 = 26, 17.5, 12.6, 9.4, 7.2, 5.6, 4.4, 3.6, 3.0 % of the pool
  bubble  9 players, 8 paid: the same ladder's top 8, ninth gets nothing (the bubble with one more player than paid)
Deal g plays configuration g mod K.

The match is duplicate and paired: for each deal and each of the 9 seats, the chart player in that seat against 8 x
nm25, and nm25 itself in that seat against the same 8 x nm25 (the control arm, zero in expectation), on the same
cards and with common random numbers: every decision draws its action by inverse CDF from one uniform per (deal,
seating, decision number), shared by both arms. Until the chart player first acts differently, the two arms play
the same hand exactly, so the difference measures the chart's decisions, not card luck or sampling. Scored in
prize equity (ICM change x the table's chips, in big blinds: bb/100-equivalent) and in chips.

Usage (lab, from the work directory with scripts/, blueprint/nm25, blueprint/out, blueprint/libbp7.so,
results/preflop/{equity169.json,equity3_169.npz}):
  PYTHONPATH=scripts python scripts/ninemax_chart_match.py solve --stage final --configs 40 --workers 6
  PYTHONPATH=scripts python scripts/ninemax_chart_match.py solve --stage bubble --configs 40 --workers 6
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts python scripts/ninemax_chart_match.py \
      match --stages final,bubble --deals 200000 --out results/pushfold_a3/match.json
"""

import argparse
import json
import math
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FT9 = [26.0, 17.5, 12.6, 9.4, 7.2, 5.6, 4.4, 3.6, 3.0]
PAYOUTS = {"final": FT9, "bubble": FT9[:8] + [0.0]}
P = 9
BBC = 100  # chips in a big blind (ninemax_engine.BB)


def action_index(seat, n=P):
    """Engine seat (0 SB, 1 BB, 2 UTG .. n-1 button) -> chart action order (pos0 .. button, SB, BB)."""
    return seat - 2 if seat >= 2 else n - 2 + seat


def make_configs(stage, K, avg, spread, seed):
    """K stack vectors [K, P] in chips by engine seat, and the payouts [P]."""
    rng = np.random.default_rng(seed + (0 if stage == "final" else 1000))
    lo, hi = avg
    a = lo * (hi / lo) ** rng.random((K, 1))
    st = a * np.exp(spread * rng.standard_normal((K, P)) - spread**2 / 2)
    st = np.maximum(np.round(st * BBC / 10) * 10, 2 * BBC).astype(np.int64)  # to 0.1bb, at least 2bb
    return st, np.array(PAYOUTS[stage])


def _solve_one(job):
    stacks_chips, pay, bb_ante, iters = job
    from icm_pushfold import load_equity3, load_matrix, solve2

    if "O" not in _CACHE:
        _CACHE["O"], _CACHE["m"] = load_equity3(), load_matrix()
    order = sorted(range(P), key=action_index)  # engine seats in action order
    sb = [stacks_chips[s] / BBC for s in order]
    t0 = time.time()
    c = solve2(sb, list(pay), iters=iters, matrix=_CACHE["m"], bb_ante=bb_ante, equity3=_CACHE["O"], arrays=True)
    a = c.pop("arrays")
    summary = {k: c[k] for k in ("stacks_bb", "icm_equity", "max_gain_bb")}
    summary["shove_share"] = [v["share"] for v in c["shove"].values()]
    summary["seconds"] = round(time.time() - t0, 1)
    return a["push"].astype(np.float32), a["call1"].astype(np.float32), a["call2"].astype(np.float32), summary


_CACHE = {}


def cmd_solve(args):
    from multiprocessing import Pool

    st, pay = make_configs(args.stage, args.configs, tuple(args.avg), args.spread, args.seed)
    out = os.path.join(args.dir, f"charts_{args.stage}.npz")
    part = os.path.join(args.dir, f"parts_{args.stage}")  # one file per configuration: an interrupted solve resumes
    os.makedirs(part, exist_ok=True)
    name = lambda i: os.path.join(part, f"{i:03d}_{args.iters}.npz")  # noqa: E731
    todo = [i for i in range(args.configs) if not os.path.exists(name(i))]
    jobs = [(st[i], pay, args.bb_ante, args.iters) for i in todo]
    print(f"{args.stage}: {args.configs - len(todo)} configurations already solved, {len(todo)} to go", flush=True)
    with Pool(args.workers) as pool:
        for i, r in zip(todo, pool.imap(_solve_one, jobs)):
            np.savez(name(i) + ".tmp.npz", push=r[0], call1=r[1], call2=r[2], summary=json.dumps(r[3]))
            os.replace(name(i) + ".tmp.npz", name(i))
            print(f"{args.stage} {i + 1}/{args.configs}: max gain {r[3]['max_gain_bb']} bb, shove shares "
                  f"{r[3]['shove_share']}, {r[3]['seconds']}s", flush=True)  # fmt: skip
    res = []
    for i in range(args.configs):
        z = np.load(name(i))
        assert np.array_equal(np.round(np.array(json.loads(str(z["summary"]))["stacks_bb"]) * BBC),
                              np.round(np.array([st[i][s] for s in sorted(range(P), key=action_index)]))), i  # fmt: skip
        res.append((z["push"], z["call1"], z["call2"], json.loads(str(z["summary"]))))
    np.savez_compressed(
        out,
        stacks=st,
        payouts=pay,
        push=np.stack([r[0] for r in res]).astype(np.float16),
        call1=np.stack([r[1] for r in res]).astype(np.float16),
        call2=np.stack([r[2] for r in res]).astype(np.float16),
    )
    meta = {"stage": args.stage, "iters": args.iters, "bb_ante": args.bb_ante, "avg": args.avg,
            "spread": args.spread, "seed": args.seed, "configs": [r[3] for r in res]}  # fmt: skip
    json.dump(meta, open(os.path.join(args.dir, f"charts_{args.stage}.json"), "w"), indent=1)
    gains = [r[3]["max_gain_bb"] for r in res]
    print(f"wrote {out}; max gain over configs: median {np.median(gains):.4f}, worst {max(gains):.4f} bb", flush=True)


# ---- the match (GPU) ----


def class_lut():
    """[52, 52] card pair -> class index in equity169.json's order."""
    from ninemax_pushfold_check import class_name

    classes = json.load(open(os.path.join(ROOT, "results", "preflop", "equity169.json")))["classes"]
    ix = {c: i for i, c in enumerate(classes)}
    lut = np.zeros((52, 52), np.int64)
    for a in range(52):
        for b in range(52):
            if a != b:
                lut[a, b] = ix[class_name(a, b)]
    return lut


def chart_override(T, g, i, probs, legal, hero, ch, cfg, cls, aidx, stats, A):
    """Replace probs for the hero's pre-flop decisions in a push/fold spot by the chart's. In place."""
    import torch

    from ninemax_engine import BB, SB

    m = (i == hero) & (T.street[g] == 0)
    if not bool(m.any()):
        return
    gm = g[m]
    rows = m.nonzero().squeeze(1)
    n = len(gm)
    ar = torch.arange(n, device=g.device)
    c = T.contrib[gm]
    post = torch.zeros(P, dtype=torch.long, device=g.device)
    post[0], post[1] = SB, BB
    vol = c > post[None]
    hero_vol = vol[:, hero]
    others = vol.clone()
    others[:, hero] = False
    nv = others.sum(1)
    big = 99
    ordv = torch.where(others, aidx[None, :].expand(n, P), torch.full_like(c, big))
    s1 = ordv.argmin(1)
    ord2 = ordv.clone()
    ord2[ar, s1] = big
    s2 = ord2.argmin(1)
    pr = T.pre_raises[gm]
    s1_allin = T.allin[gm, s1]
    k = cfg[gm]
    hc = cls[gm, hero]
    ha = int(aidx[hero])
    first = ~hero_vol & (nv == 0) & (pr == 0) & (hero != 1)
    facing = ~hero_vol & (nv == 1) & (pr == 1) & s1_allin
    over = ~hero_vol & (nv == 2) & (pr == 1) & s1_allin & (c[ar, s2] <= c[ar, s1])
    pn = probs[rows]
    new = torch.zeros_like(pn)
    legal_allin = legal[rows, A - 1]
    p_push = ch["push"][k, min(ha, P - 2), hc]  # the big blind is never first in (first is False there)
    p_call = ch["call1"][k, aidx[s1], ha, hc]
    p_over = ch["call2"][k, aidx[s1], aidx[s2], ha, hc]
    use_first = first & legal_allin
    new[use_first, 0] = 1 - p_push[use_first]
    new[use_first, A - 1] = p_push[use_first]
    new[facing, 0] = 1 - p_call[facing]
    new[facing, 1] = p_call[facing]
    new[over, 0] = 1 - p_over[over]
    new[over, 1] = p_over[over]
    use = use_first | facing | over
    for name, msk, nm_in, ch_in in (
        ("first_in", use_first, pn[:, A - 1], p_push),
        ("facing_shove", facing, pn[:, 1], p_call),
        ("shove_and_call", over, pn[:, 1], p_over),
    ):
        st = stats.setdefault(name, [0, 0.0, 0.0, 0.0])
        st[0] += int(msk.sum())
        st[1] += float(ch_in[msk].sum())  # the chart's probability of going in (shove or call)
        st[2] += float(nm_in[msk].sum())  # nm25's probability of the same action
        st[3] += float((1 - pn[msk, 0]).sum())  # nm25's probability of going in at all
    stats["first_in_allin_illegal"] = stats.get("first_in_allin_illegal", 0) + int((first & ~legal_allin).sum())
    probs[rows[use]] = new[use].to(probs.dtype)


def play_crn(dealer, menu, net, bk, rk, ctx, unif, hero, ch=None, cfg=None, cls=None, stats=None):
    """One seating: nm25 everywhere, the chart player (ch) or nm25 in seat hero. Actions by inverse CDF from unif
    [G, L]. Returns the hero's prize-equity change and chip change, in big blinds [G]."""
    import torch
    import torch.nn.functional as F

    from ninemax_engine import BB, Tables, strength
    from ninemax_selfplay import end_equity, features, masked, start_equity

    dev = bk.device
    G = bk.shape[0]
    stacks, pay = ctx
    T = Tables(G, P, menu, dev, stacks=stacks)
    T.pay, T.eq0 = pay.float(), start_equity(T.stack, pay)
    T.board = None
    steps = torch.zeros(G, dtype=torch.long, device=dev)
    L = unif.shape[1]
    aidx = torch.tensor([action_index(s) for s in range(P)], device=dev)
    A = menu.A
    while not bool(T.done.all()):
        g = (~T.done).nonzero().squeeze(1)
        i = T.actor[g]
        s = strength(dealer, bk[g], T.street[g]).gather(1, i[:, None]).squeeze(1)
        feat, legal = features(T, g, i)
        with torch.no_grad():
            lg, _ = net(feat, bk[g, i], T.street[g], s)
        probs = F.softmax(masked(lg, legal), 1)
        if ch is not None:
            chart_override(T, g, i, probs, legal, hero, ch, cfg, cls, aidx, stats, A)
        u = unif[g, steps[g].clamp(max=L - 1)]
        cum = probs.cumsum(1)
        a = (cum < u[:, None]).sum(1)
        last = torch.where(probs > 0, torch.arange(A, device=dev)[None, :], torch.zeros_like(legal, dtype=torch.long))
        a = torch.minimum(a, last.max(1).values)
        T.step(g, a)
        steps[g] += 1
    chips = T.payoffs(rk)
    eq = (end_equity(T, chips, rk) - T.eq0) * T.stack.sum(1, keepdim=True).float()
    return eq[:, hero] / BB, chips[:, hero] / BB


def interval(x):
    x = np.asarray(x, dtype=float)
    m, se = x.mean() * 100, x.std() / math.sqrt(len(x)) * 100
    return [round(float(m), 3), round(float(m - 1.96 * se), 3), round(float(m + 1.96 * se), 3)]


def cmd_match(args):
    import torch

    from ninemax_eval import load, setup
    from ninemax_selfplay import FEAT

    dev = args.device
    torch.manual_seed(args.seed)
    args.players = P
    dealer, menu, B = setup(args, dev)
    net, _ = load(args.net, B, dev)
    assert FEAT["tourney"] and not FEAT["bounty"], "nm25's game: a tournament without bounties"
    from ninemax_engine import RULES

    lut = torch.tensor(class_lut(), device=dev)
    res = {"net": args.net, "deals": args.deals, "batch": args.batch, "seed": args.seed, "rules": dict(RULES),
           "stages": {}}  # fmt: skip
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for stage in args.stages.split(","):
        z = np.load(os.path.join(args.dir, f"charts_{stage}.npz"))
        K = len(z["stacks"])
        ch = {k: torch.tensor(z[k].astype(np.float32), device=dev) for k in ("push", "call1", "call2")}
        cfg_stacks = torch.tensor(z["stacks"], device=dev)
        pay = torch.tensor(z["payouts"] / z["payouts"].sum(), device=dev, dtype=torch.float32)
        eqc, eqb, chc, chb, depth = [], [], [], [], []
        stats = {}
        t0 = time.time()
        done = 0
        gen = torch.Generator(device=dev)
        while done < args.deals:
            n = min(args.batch, args.deals - done)
            bk, rk, cd = dealer.deal(n, cards=True)
            cdt = torch.tensor(cd, device=dev, dtype=torch.long)
            cls = lut[cdt[:, 0 : 2 * P : 2], cdt[:, 1 : 2 * P : 2]]  # [n, P] class by seat
            cfg = (torch.arange(n, device=dev) + done) % K
            ctx = (cfg_stacks[cfg].clone(), pay[None].expand(n, P).clone())
            e_c = torch.zeros(n, P, device=dev)
            e_b, c_c, c_b = torch.zeros_like(e_c), torch.zeros_like(e_c), torch.zeros_like(e_c)
            for seat in range(P):
                gen.manual_seed(args.seed * 1_000_003 + done * 17 + seat)
                unif = torch.rand(n, 96, device=dev, generator=gen).clamp(1e-7, 1 - 1e-7)
                e_c[:, seat], c_c[:, seat] = play_crn(dealer, menu, net, bk, rk, ctx, unif, seat, ch, cfg, cls, stats)
                e_b[:, seat], c_b[:, seat] = play_crn(dealer, menu, net, bk, rk, ctx, unif, seat)
            eqc.append(e_c.cpu().numpy())
            eqb.append(e_b.cpu().numpy())
            chc.append(c_c.cpu().numpy())
            chb.append(c_b.cpu().numpy())
            depth.append((ctx[0] / 100).cpu().numpy())
            done += n
            d = np.concatenate(eqc).mean(1) - np.concatenate(eqb).mean(1)
            print(f"{stage} {done:,}/{args.deals:,} deals, {time.time() - t0:.0f}s: chart - nm25 {interval(d)} bb/100",
                  flush=True)  # fmt: skip
        eqc, eqb, chc, chb, depth = (np.concatenate(x) for x in (eqc, eqb, chc, chb, depth))
        r = {
            "configs": K,
            "deals": int(len(eqc)),
            "equity_difference": interval(eqc.mean(1) - eqb.mean(1)),
            "equity_chart": interval(eqc.mean(1)),
            "equity_control_nm25": interval(eqb.mean(1)),
            "chips_difference": interval(chc.mean(1) - chb.mean(1)),
            "spots": {
                k: v
                if not isinstance(v, list)
                else {
                    "count": v[0],
                    "chart_in": round(v[1] / max(v[0], 1), 3),
                    "nm25_same_action": round(v[2] / max(v[0], 1), 3),
                    "nm25_in_any": round(v[3] / max(v[0], 1), 3),
                }
                for k, v in stats.items()
            },
            "by_hero_stack": {},
            "seconds": round(time.time() - t0),
        }
        diff = eqc - eqb  # [deal, seat]
        for lo, hi in ((0, 8), (8, 15), (15, 1e9)):
            msk = (depth >= lo) & (depth < hi)
            # per (deal, seat) pairs in the band; deals share cards across seats, so the interval is approximate
            r["by_hero_stack"][f"{lo}-{hi if hi < 1e9 else 'up'}bb"] = {
                "share": round(float(msk.mean()), 3),
                "difference_per_seat_hand": interval(diff[msk]),
            }
        res["stages"][stage] = r
        print(json.dumps({stage: r}, indent=1), flush=True)
        json.dump(res, open(args.out, "w"), indent=1)
    print("done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("solve")
    s.add_argument("--stage", choices=sorted(PAYOUTS), required=True)
    s.add_argument("--configs", type=int, default=40)
    s.add_argument("--iters", type=int, default=1000)
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--bb-ante", type=float, default=1.0, help="nm25's game: a big-blind ante of 1bb")
    s.add_argument("--avg", type=float, nargs=2, default=[8.0, 12.5])
    s.add_argument("--spread", type=float, default=0.7)
    s.add_argument("--seed", type=int, default=3)
    s.add_argument("--dir", default=os.path.join(ROOT, "results", "pushfold_a3"))
    m = sub.add_parser("match")
    m.add_argument("--stages", default="final,bubble")
    m.add_argument("--deals", type=int, default=200000)
    m.add_argument("--batch", type=int, default=25000)
    m.add_argument("--net", default="blueprint/nm25")
    m.add_argument("--out", required=True)
    m.add_argument("--dir", default=os.path.join(ROOT, "results", "pushfold_a3"))
    m.add_argument("--menu", default="mtt")
    m.add_argument("--abs", default="blueprint/out")
    m.add_argument("--lib", default="blueprint/libbp7.so")
    m.add_argument("--seed", type=int, default=21)
    m.add_argument("--device", default="cuda", help="cpu: a smoke test only")
    a = ap.parse_args()
    cmd_solve(a) if a.cmd == "solve" else cmd_match(a)


if __name__ == "__main__":
    main()
