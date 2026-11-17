"""What the other players at Kestrel's tables do, from the hands in kestrel.db (each 9-max hand's "table": players,
starting stacks, every action, shown cards): behaviour that self-play never meets (tilt, a player who changes,
overvalued suited hands) and collusion (soft play between partners). Every family of tests is Holm-corrected
(multitest.py), and each detector has a control so it cannot find tournament structure instead of behaviour.

  tilt        each player's VPIP and pre-flop raise rate in the 25 hands after losing 15bb or more in a hand,
              against the 25 hands before that loss (before and after: a loose player loses more, so other
              comparisons read loose play as tilt)
  change      each player's VPIP, lifetime against recent (forgetting 0.986 a hand, as Kestrel's profile): the first
              hand where the recent rate is 3 standard errors from the lifetime one
  soft play   for each pair: A's bets after the flop with B in the pot and A's folds to B's bets, against logistic
              main effects (A's own style, everyone's treatment of B); collusion is the pair's leftover, required in
              both directions (the weaker direction scores the pair)
  suited      the share of suited hands among the hands each player shows, against everyone's (overplaying suited
              cards shows up as more of them at showdown)

Usage: uv run python scripts/table_analysis.py KESTREL_DB [--connector testplatform] [--out results/...json]
Planted behaviours to check it with: kestrel/testplatform/ninemax_platform.py --bots tag,switcher,tilter,ally,ally,...
"""

import argparse
import json
import math
import sqlite3
from collections import defaultdict

import numpy as np

from multitest import holm

LOSS_BB, WINDOW, DECAY = 15.0, 25, 0.986


def z2(x1, n1, x2, n2):
    """Two-proportion z: (x1/n1 - x2/n2) over the pooled standard error."""
    if n1 < 5 or n2 < 5:
        return 0.0
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p), 1e-9) * (1 / n1 + 1 / n2))
    return (x1 / n1 - x2 / n2) / se


def p_of(z):
    return math.erfc(abs(z) / math.sqrt(2))


def load(db, connector):
    rows = sqlite3.connect(db).execute(
        "SELECT session_id, seq, data FROM hands WHERE connector = ? ORDER BY session_id, seq", (connector,)
    )
    hands = []
    for _, _, data in rows:
        h = json.loads(data)
        if "table" in h:
            hands.append(h)
    # each player's result in each hand: the next hand's starting stack less this one's (busted: all of it), within a
    # tournament (a new tournament starts where a player's stack does not follow on: a hand id that does not increase)
    for k, h in enumerate(hands):
        nxt = hands[k + 1] if k + 1 < len(hands) else None
        same = nxt is not None and str(nxt.get("hand_id", "")).isdigit() and str(h.get("hand_id", "")).isdigit() \
            and int(nxt["hand_id"]) > int(h["hand_id"])  # fmt: skip
        res = {}
        for p, st in h["table"]["stacks_bb"].items():
            if not same:
                continue
            after = nxt["table"]["stacks_bb"].get(p)
            # stacks are in each hand's own big blinds; the next hand's bb may be bigger, so compare in chips
            res[p] = (after * nxt["table"].get("bb", 1) if after is not None else 0.0) - st * h["table"].get("bb", 1)
        h["result_bb_each"] = res
    return hands


def preflop_flags(h):
    vpip, pfr = set(), set()
    for p, street, kind in h["table"]["actions"]:
        if street == 0 and kind in ("call", "bet", "raise", "allin"):
            vpip.add(p)
        if street == 0 and kind in ("bet", "raise", "allin"):
            pfr.add(p)
    return vpip, pfr


def tilt(hands, me):
    """Before and after: each player's VPIP and pre-flop raise rate in the 25 hands after each of its own losses of
    15bb or more, against the 25 hands just before that loss. A loose player loses more, so comparing after-loss hands
    with other times reads loose play as tilt (the arrow runs the other way); before-and-after keeps the player's
    style on both sides, and only a reaction to the loss differs. One-sided (tilt is looser or more aggressive)."""
    seq = defaultdict(list)  # player -> [(vpip, pfr, lost big this hand)] in order
    for h in hands:
        vpip, pfr = preflop_flags(h)
        bb = h["table"].get("bb", 1)
        for p in h["table"]["players"]:
            if p != me:
                r = h["result_bb_each"].get(p)
                seq[p].append((p in vpip, p in pfr, r is not None and -r / bb >= LOSS_BB))
    out = []
    for p, xs in sorted(seq.items()):
        a = [0, 0, 0]  # after: hands, vpip, pfr
        b = [0, 0, 0]  # before
        for t, x in enumerate(xs):
            if not x[2] or t < WINDOW or t + WINDOW >= len(xs):
                continue
            for acc, rng in ((b, range(t - WINDOW, t)), (a, range(t + 1, t + 1 + WINDOW))):
                for k in rng:
                    acc[0] += 1
                    acc[1] += xs[k][0]
                    acc[2] += xs[k][1]
        zv, zr = z2(a[1], a[0], b[1], b[0]), z2(a[2], a[0], b[2], b[0])
        z = max(zv, zr)
        out.append(
            {
                "player": p,
                "loss_events_hands": a[0],
                "vpip_after": a[1] / max(a[0], 1),
                "vpip_before": b[1] / max(b[0], 1),
                "pfr_after": a[2] / max(a[0], 1),
                "pfr_before": b[2] / max(b[0], 1),
                "z": z,
                "p": min(1.0, 2 * 0.5 * math.erfc(z / math.sqrt(2))),
            }
        )  # fmt: skip  (one-sided, x2 for two rates)
    for r, h in zip(out, holm([r["p"] for r in out])):
        r["holm"] = h
    return out


def change(hands, me):
    """The first hand where a player's recent VPIP is 3 standard errors from its lifetime VPIP."""
    life = defaultdict(lambda: [0, 0])
    rec = defaultdict(lambda: [0.0, 0.0])
    first, n_seen = {}, defaultdict(int)
    for h in hands:
        vpip, _ = preflop_flags(h)
        for p in h["table"]["players"]:
            if p == me:
                continue
            n_seen[p] += 1
            life[p][0] += 1
            life[p][1] += p in vpip
            rec[p][0] = DECAY * rec[p][0] + 1
            rec[p][1] = DECAY * rec[p][1] + (p in vpip)
            q = min(max(life[p][1] / life[p][0], 0.01), 0.99)
            if life[p][0] >= 60 and rec[p][0] >= 20 and p not in first:
                z = (rec[p][1] / rec[p][0] - q) / math.sqrt(q * (1 - q) / rec[p][0])
                if abs(z) >= 3:
                    first[p] = {"hand": n_seen[p], "z": round(z, 2), "lifetime_vpip": round(q, 3),
                                "recent_vpip": round(rec[p][1] / rec[p][0], 3)}  # fmt: skip
    return {p: first.get(p) for p in sorted(life)}


def main_effects(x, n, iters=200):
    """Logistic main effects for a players x players table of x successes in n trials: logit p_ab = r_a + c_b, fitted
    by Newton steps on each set in turn; returns the expected successes. What is left over is the pair's own."""
    k = x.shape[0]
    r, c = np.zeros(k), np.zeros(k)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(r[:, None] + c[None, :])))
        w = n * p * (1 - p)
        r += ((x - n * p).sum(1)) / np.maximum(w.sum(1), 1e-9)
        p = 1 / (1 + np.exp(-(r[:, None] + c[None, :])))
        w = n * p * (1 - p)
        c += ((x - n * p).sum(0)) / np.maximum(w.sum(0), 1e-9)
    p = 1 / (1 + np.exp(-(r[:, None] + c[None, :])))
    return n * p, n * p * (1 - p)


def soft_play(hands, me):
    """Does A treat B differently from how A treats others and how others treat B? Two tables, A x B: A's bets and
    raises after the flop with B in the pot, and A's folds facing B's bet. Each is fitted with logistic main effects
    (A's own aggression, everyone's aggression against B); a pair's soft play is what is left over, in both directions
    pooled: fewer bets into the partner (z_aggression < 0) and more folds to the partner (z_fold > 0), in each
    direction (the weaker scores the pair). Without the main effects, how others treat a player read as collusion;
    pooled over both directions, a one-way style effect (a limper folding to a tag that bets only strong hands) did."""
    names = sorted({p for h in hands for p in h["table"]["players"] if p != me})
    idx = {p: k for k, p in enumerate(names)}
    K = len(names)
    ax, an = np.zeros((K, K)), np.zeros((K, K))  # A's aggressive actions, A's actions, with B live
    fx, fn = np.zeros((K, K)), np.zeros((K, K))  # A's folds, A facing B's bet
    for h in hands:
        folded, last_aggr, street_now = set(), None, 0
        for p, street, kind in h["table"]["actions"]:
            if street != street_now:
                street_now, last_aggr = street, None
            if street >= 1 and p != me and p in idx:
                for b in h["table"]["players"]:
                    if b != p and b != me and b not in folded and b in idx:
                        an[idx[p], idx[b]] += 1
                        ax[idx[p], idx[b]] += kind in ("bet", "raise", "allin")
                if last_aggr is not None and last_aggr != p and last_aggr != me and last_aggr in idx:
                    fn[idx[p], idx[last_aggr]] += 1
                    fx[idx[p], idx[last_aggr]] += kind == "fold"
            if kind == "fold":
                folded.add(p)
            if kind in ("bet", "raise", "allin"):
                last_aggr = p
    ea, va = main_effects(ax, an)
    ef, vf = main_effects(fx, fn)
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            cells = [(i, j), (j, i)]
            n_a, n_f = sum(an[c] for c in cells), sum(fn[c] for c in cells)
            # soft play between partners runs both ways: each direction must show it, and the pair is scored by the
            # weaker one (one-way effects are style, such as folding to a player who only bets strong hands)
            za = max((ax[c] - ea[c]) / math.sqrt(max(va[c], 1e-9)) for c in cells)
            zf = min((fx[c] - ef[c]) / math.sqrt(max(vf[c], 1e-9)) if fn[c] >= 10 else 0.0 for c in cells)
            score = max(-za, zf)
            pairs.append(
                {
                    "pair": f"{names[i]} + {names[j]}",
                    "aggression": sum(ax[c] for c in cells) / max(n_a, 1),
                    "aggression_expected": sum(ea[c] for c in cells) / max(n_a, 1),
                    "fold_to_partner": sum(fx[c] for c in cells) / max(n_f, 1),
                    "fold_expected": sum(ef[c] for c in cells) / max(n_f, 1),
                    "n": int(n_a),
                    "nf": int(n_f),
                    "z_aggression": round(za, 2),
                    "z_fold": round(zf, 2),
                    "p": min(1.0, math.erfc(score / math.sqrt(2))),
                }
            )  # fmt: skip  (best of two one-sided)
    for r, h in zip(pairs, holm([r["p"] for r in pairs])):
        r["holm"] = h
    return sorted(pairs, key=lambda r: r["p"])


def suited(hands, me):
    """The share of suited hands among each player's shown hands, against everyone's."""
    shown = defaultdict(lambda: [0, 0])
    for h in hands:
        for p, cards in h["table"].get("shown", {}).items():
            if p == me or len(cards) != 2:
                continue
            shown[p][0] += 1
            shown[p][1] += cards[0][1] == cards[1][1]
    tot_n, tot_x = sum(v[0] for v in shown.values()), sum(v[1] for v in shown.values())
    out = []
    for p, (n, x) in sorted(shown.items()):
        z = z2(x, n, tot_x - x, tot_n - n)
        out.append({"player": p, "shown": n, "suited": x / max(n, 1), "others": (tot_x - x) / max(tot_n - n, 1),
                    "z": round(z, 2), "p": p_of(z)})  # fmt: skip
    for r, h in zip(out, holm([r["p"] for r in out])):
        r["holm"] = h
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--connector", default="testplatform")
    ap.add_argument("--out")
    a = ap.parse_args()
    hands = load(a.db, a.connector)
    me = hands[0]["table"]["me"] if hands else ""
    print(f"{len(hands)} hands with the whole table recorded; Kestrel is {me}")
    res = {"hands": len(hands), "tilt": tilt(hands, me), "change": change(hands, me), "soft_play": soft_play(hands, me),
           "suited": suited(hands, me)}  # fmt: skip
    print("\ntilt: VPIP / raise in the 25 hands after the player's own 15bb loss, against the 25 before it")
    for r in res["tilt"]:
        flag = "TILT" if r["holm"] < 0.05 and r["z"] > 0 else ""
        print(f"  {r['player']:12s} {r['loss_events_hands']:4d} hands after a loss: vpip {r['vpip_after']:.2f} (before "
              f"{r['vpip_before']:.2f}) pfr {r['pfr_after']:.2f} ({r['pfr_before']:.2f})  z {r['z']:+.2f}  "
              f"Holm p {r['holm']:.4f} {flag}")  # fmt: skip
    print("\nchange: the first hand where recent VPIP is 3 standard errors from lifetime")
    for p, c in res["change"].items():
        print(f"  {p:12s} {c if c else 'none'}")
    print("\nsoft play, the pairs most like partners (scored by the weaker direction)")
    for r in res["soft_play"][:6]:
        flag = "COLLUSION?" if r["holm"] < 0.05 else ""
        print(
            f"  {r['pair']:24s} aggression {r['aggression']:.2f} (expected {r['aggression_expected']:.2f}, z "
            f"{r['z_aggression']:+.2f}) | folds to partner {r['fold_to_partner']:.2f} (expected {r['fold_expected']:.2f}, "
            f"z {r['z_fold']:+.2f}, {r['nf']} spots)  Holm p {r['holm']:.4f} {flag}"
        )
    print("\nsuited hands among those shown")
    for r in res["suited"]:
        print(
            f"  {r['player']:12s} {r['shown']:4d} shown, suited {r['suited']:.2f} vs others {r['others']:.2f}  "
            f"z {r['z']:+.2f}  Holm p {r['holm']:.4f}"
        )
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
