"""Behavioural analytics on real players (HandHQ, 2009 online cash games): biases, each tested against a null.

Usage: uv run python scripts/behaviour.py HANDHQ_DIR OUTDIR [HANDS_PER_FOLDER] [WORKERS]
Hands are replayed exactly (PokerKit) with their time, table and player ids; ids are stable within a site/stake folder.
A session is one player's run of hands at one table with no gap over 10 minutes.

1. Quitting (house-money / disposition effect, behavioural finance): is a player more likely to leave the table
   when ahead in the session than when behind? Per player: P(this is the last hand | session net so far > +10bb)
   minus P(... | < -10bb), averaged over players. Null: the same with each session's hands in shuffled order (keeps
   every session's length and results, destroys their order).
2. Chasing losses: is a player looser (VPIP) when stuck in the session (< -20bb) than when even or ahead? Paired within
   player; null as above.
3. Bet-size tells: river bets whose maker's cards were shown at showdown, hand strength (share of all holdings on
   that board the hand beats) by bet size relative to the pot. A balanced player's big bets are polarised (nuts and
   bluffs); a strong, monotone relationship is a tell. Showdown-only: called bets are over-represented, so the
   bluff share is a lower bound.
4. Fatigue: within player, VPIP, aggression and big blinds won in a session's hands 1-60 versus 61-180 versus 181+.
"""

import datetime as dt
import glob
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hand_histories as hh  # noqa: E402

GAP = dt.timedelta(minutes=10)


def read_folder(args):
    """Hands of one site/stake folder in time order: time, table, players, net (bb), stacks (bb), rows, shown."""
    folder, limit = args
    from pokerkit import HandHistory

    files = sorted(glob.glob(os.path.join(folder, "**", "*.phhs"), recursive=True))
    out = []
    for f in files:
        if len(out) >= limit:
            break
        for part in re.split(r"(?m)^\[\d+\]\s*$", open(f, errors="replace").read()):
            if not part.strip() or len(out) >= limit:
                continue
            try:
                h = hh.parse_phh(part)
                if "actions" not in h or "players" not in h:
                    continue
                hist = HandHistory.loads(part)
                st = None
                for st in hist:
                    pass
                bb = abs(float(max(h.get("blinds_or_straddles", [1]) or [1]))) or 1.0
                net = [(float(b) - float(a)) / bb for a, b in zip(hist.starting_stacks, st.stacks)]
                if not all(math.isfinite(x) for x in net):
                    continue
                hh_, mm, ss = (int(x) for x in str(h["time"]).split(":"))
                t = dt.datetime(int(h["year"]), int(h["month"]), int(h["day"]), hh_, mm, ss)
                shown = {}
                for a in h["actions"]:
                    p = a.split()
                    if len(p) >= 3 and p[1] == "sm" and "?" not in p[2]:
                        shown[int(p[0][1:]) - 1] = p[2]
                out.append(
                    {
                        "t": t.timestamp(),
                        "table": str(h.get("table")),
                        "players": h["players"],
                        "net": net,
                        "stacks": [float(x) / bb for x in h["starting_stacks"]],
                        "rows": list(hh.replay(h)),
                        "shown": shown,
                    }
                )
            except Exception:  # a hand PokerKit cannot replay is skipped, not guessed
                continue
    out.sort(key=lambda x: x["t"])
    return os.path.basename(folder.rstrip("/")), out


def vpip_pfr_af(hd):
    vp, pr, post_r, post_c = set(), set(), defaultdict(int), defaultdict(int)
    for r in hd["rows"]:
        if r["street"] == "preflop":
            if r["action"] == "cbr" or (r["action"] == "cc" and r["to_call"] > 0):
                vp.add(r["player"])
            if r["action"] == "cbr":
                pr.add(r["player"])
        else:
            post_r[r["player"]] += r["action"] == "cbr"
            post_c[r["player"]] += r["action"] == "cc" and r["to_call"] > 0
    return vp, pr, post_r, post_c


def sessions(name, hands):
    """(folder|player) -> list of sessions; a session is a list of (t, net, vpip, raises, calls)."""
    last = {}
    out = defaultdict(list)
    for hd in hands:
        vp, _, prr, pc = vpip_pfr_af(hd)
        for j, p in enumerate(hd["players"]):
            key = (p, hd["table"])
            row = (hd["t"], hd["net"][j], p in vp, prr[p], pc[p], hd["stacks"][j])
            if key in last and hd["t"] - last[key] <= GAP.total_seconds():
                out[f"{name}|{p}"][-1].append(row)
            else:
                out[f"{name}|{p}"].append([row])
            last[key] = hd["t"]
    return out


def quit_and_chase(sess, rng, shuffle=False):
    """Per player: quit-rate difference ahead vs behind, and VPIP difference stuck vs not."""
    q_diff, c_diff = [], []
    for p, ss in sess.items():
        qa, qb, va, vb = [], [], [], []
        for s in ss:
            if len(s) < 20:
                continue
            s = list(s)
            if shuffle:
                rng.shuffle(s)
            run = 0.0
            for k, (_, net, vp, _, _, stack) in enumerate(s):
                if k >= 10 and net > -0.9 * stack:  # a player who just went broke leaves mechanically: excluded
                    last = k == len(s) - 1
                    if run > 10:
                        qa.append(last)
                    elif run < -10:
                        qb.append(last)
                    if run < -20:
                        va.append(vp)
                    elif run > -5:
                        vb.append(vp)
                run += net
        if len(qa) >= 30 and len(qb) >= 30:
            q_diff.append(np.mean(qa) - np.mean(qb))
        if len(va) >= 30 and len(vb) >= 30:
            c_diff.append(np.mean(va) - np.mean(vb))

    def summ(x):
        x = np.array(x)
        if len(x) < 3:
            return None
        return {
            "players": len(x),
            "mean": round(float(x.mean()), 5),
            "t": round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2),
        }

    return summ(q_diff), summ(c_diff)


def fatigue(sess):
    buckets = [(0, 60), (60, 180), (180, 10**9)]
    per = []
    for p, ss in sess.items():
        acc = {b: [] for b in buckets}
        for s in ss:
            for k, row in enumerate(s):
                for b in buckets:
                    if b[0] <= k < b[1]:
                        acc[b].append(row)
        if all(len(acc[b]) >= 100 for b in buckets):
            per.append(
                [
                    (
                        np.mean([r[2] for r in acc[b]]),
                        sum(r[3] for r in acc[b]) / max(1, sum(r[3] + r[4] for r in acc[b])),
                        100 * np.mean([r[1] for r in acc[b]]),
                    )
                    for b in buckets
                ]
            )
    per = np.array(per)  # players x bucket x (vpip, aggression, bb/100)
    if len(per) < 3:
        return None
    out = {"players": len(per)}
    for j, lab in enumerate(("vpip", "postflop_aggression", "bb100")):
        for b, name in ((1, "61-180_minus_1-60"), (2, "181+_minus_1-60")):
            d = per[:, b, j] - per[:, 0, j]
            out[f"{lab}_{name}"] = round(float(d.mean()), 4)
            out[f"{lab}_{name}_t"] = round(float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))), 2)
    return out


def tells(data):
    import range_inference as ri

    rows = []
    for _, hands in data.items():
        for hd in hands:
            if not hd["shown"]:
                continue
            board = None
            for r in hd["rows"]:
                if r["street"] == "river" and r["action"] == "cbr" and r["seat"] in hd["shown"]:
                    board = [r["board"][i : i + 2] for i in range(0, 10, 2)]
                    rows.append((r["size_x_pot"], board, hd["shown"][r["seat"]], r["raises_before"]))
    cache, out = {}, []
    for size, board, hole, raises in rows:
        k = tuple(board)
        if k not in cache:
            cache[k] = ri.strengths(board)
        h = ri.norm(hole)
        if h in cache[k]:
            out.append((size, cache[k][h][0], raises))
    rep = {"river_bets_with_cards": len(out)}
    for lo, hi, lab in (
        (0, 0.4, "small <40% pot"),
        (0.4, 0.8, "medium 40-80%"),
        (0.8, 1.2, "pot 80-120%"),
        (1.2, 99, "overbet >120%"),
    ):
        s = [x[1] for x in out if lo <= x[0] < hi and x[2] == 0]
        if s:
            rep[lab] = {
                "n": len(s),
                "mean_strength": round(float(np.mean(s)), 3),
                "share_bluffs_below_median_hand": round(float(np.mean([v < 0.5 for v in s])), 3),
                "share_nutted_top_5pct": round(float(np.mean([v > 0.95 for v in s])), 3),
            }
    rs = [x[1] for x in out if x[2] >= 1]
    if rs:
        rep["river_raises"] = {
            "n": len(rs),
            "mean_strength": round(float(np.mean(rs)), 3),
            "share_bluffs_below_median_hand": round(float(np.mean([v < 0.5 for v in rs])), 3),
        }
    return rep


def load(src, per_folder, workers):
    """Replayed hands per folder; cached in $HANDHQ_CACHE (a pickle) so the graph study reuses the same read."""
    import pickle

    cache = os.environ.get("HANDHQ_CACHE")
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    folders = sorted(d for d in glob.glob(os.path.join(src, "*")) if os.path.isdir(d))
    with ProcessPoolExecutor(workers) as pool:
        data = dict(pool.map(read_folder, [(f, per_folder) for f in folders]))
    if cache:
        pickle.dump(data, open(cache, "wb"))
    return data


def main(src, outdir, per_folder=100000, workers=27):
    os.makedirs(outdir, exist_ok=True)
    data = load(src, per_folder, workers)
    sess = {}
    for name, hands in data.items():
        sess.update(sessions(name, hands))
    rng = random.Random(0)
    q, c = quit_and_chase(sess, rng)
    qn, cn = quit_and_chase(sess, rng, shuffle=True)
    lens = [len(s) for ss in sess.values() for s in ss]
    rep = {
        "hands": sum(len(v) for v in data.values()),
        "players": len(sess),
        "sessions": len(lens),
        "median_session_hands": float(np.median(lens)),
        "quit_more_when_ahead_than_behind": {"real": q, "shuffled_null": qn},
        "looser_when_stuck_20bb": {"real": c, "shuffled_null": cn},
        "fatigue": fatigue(sess),
        "bet_size_tells": tells(data),
    }
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:4]))
