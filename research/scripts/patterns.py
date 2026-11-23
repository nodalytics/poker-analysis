"""Pattern scanner: detect behaviour patterns in how models and players play and act.

Usage: uv run python scripts/patterns.py [ARENA_DIR] [OUT_PREFIX]
       (defaults: results/arena, results/patterns/latest) -> OUT_PREFIX.md and OUT_PREFIX.json

Detectors (each finding carries counts and evidence, so it can be checked by hand):
  chess  repetition (moves into a position seen before), shuffles (a piece moved straight back), openings,
         illegal-move failures, how games end, and skill adherence: if a learner's skill file says to avoid
         repetition, its repetition rate before and after it wrote that.
  poker  VPIP (voluntarily put money in), PFR (preflop raise), aggression factor (raises / calls after the flop),
         fold to a raise, all-in rate, favourite raise sizes, VPIP by position, tilt (VPIP in the 3 hands after
         losing 50+ chips vs otherwise), and the nearest personality for every model (by VPIP, PFR, aggression).
  agents tool-call loops (same tool and arguments 3+ times in a row), answers with no tool call, failed tasks
         whose answer still asserted a number, and tool-argument errors, across results/raw.
Detections are descriptive. With few games most rates are noisy; the report gives the counts behind each rate.
"""

import collections
import glob
import json
import os
import re
import sys

import chess
from pokerkit import StandardHighHand

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import poker_selfplay as ps  # noqa: E402


def short(e):
    base, _, tag = e.partition("#")
    name = base.split("/")[-1].split(":")[0].replace("-GGUF", "")
    return f"{name}#{tag}" if tag else name


def rate(n, d):
    return round(n / d, 3) if d else None


# ---------------------------------------------------------------------------------------------- chess
def scan_chess(arena):
    out = collections.defaultdict(lambda: collections.Counter())
    openings = collections.defaultdict(collections.Counter)
    endings = collections.Counter()
    by_round = collections.defaultdict(lambda: collections.Counter())
    for path in sorted(glob.glob(os.path.join(arena, "rounds", "*", "chess.jsonl"))):
        rnd = int(path.split(os.sep)[-2])
        games = collections.defaultdict(list)
        for line in open(path):
            r = json.loads(line)
            if r.get("end"):
                endings[r["how"].split(" at ")[0].split(",")[0]] += 1
                continue
            games[r["game"]].append(r)
        for plies in games.values():
            board, seen = chess.Board(), collections.Counter()
            seen[board.board_fen() + str(board.turn)] += 1
            last_move = {}
            for r in plies:
                e = r["model"]
                mv = board.parse_san(r["san"])
                if r["ply"] <= 2:
                    openings[e][r["san"]] += 1
                prev = last_move.get(e)
                if prev and mv.from_square == prev.to_square and mv.to_square == prev.from_square:
                    out[e]["shuffles"] += 1
                last_move[e] = mv
                board.push(mv)
                key = board.board_fen() + str(board.turn)
                if seen[key]:
                    out[e]["repeats"] += 1
                    by_round[(e, rnd)]["repeats"] += 1
                seen[key] += 1
                out[e]["moves"] += 1
                out[e]["forced"] += int(r["forced_random"])
                by_round[(e, rnd)]["moves"] += 1
    rows = []
    for e, c in sorted(out.items()):
        rows.append(
            {
                "entrant": short(e),
                "moves": c["moves"],
                "repeat_rate": rate(c["repeats"], c["moves"]),
                "shuffle_rate": rate(c["shuffles"], c["moves"]),
                "no_legal_move_rate": rate(c["forced"], c["moves"]),
                "openings": dict(openings[e].most_common(3)),
            }
        )
    adherence = []
    for skill in glob.glob(os.path.join(arena, "notes", "*", "skill-chess.md")):
        text = open(skill).read().lower()
        if "repet" not in text:
            continue
        name = skill.split(os.sep)[-2]
        mine = {r: c for (e, r), c in by_round.items() if e.endswith("#learn") and short(e).startswith(name)}
        if len(mine) >= 2:
            first = min(mine)
            before = rate(mine[first]["repeats"], mine[first]["moves"])
            later = [c for r, c in mine.items() if r > first]
            after = rate(sum(c["repeats"] for c in later), sum(c["moves"] for c in later))
            adherence.append(
                {
                    "entrant": f"{name}#learn",
                    "rule": "skill file mentions avoiding repetition",
                    "repeat_rate_round_1": before,
                    "repeat_rate_later": after,
                    "followed": after is not None and before is not None and after < before,
                }
            )
    return {"players": rows, "endings": dict(endings), "skill_adherence": adherence}


def strength(hole, board):
    """0 high card, 1 pair, 2 two pair, 3+ trips and better; None preflop or when cards are unknown."""
    if not board or board == "-" or not hole or hole == "-":
        return None
    try:
        label = StandardHighHand.from_game(hole.replace(" ", ""), board.replace(" ", "")).entry.label
    except (ValueError, KeyError):
        return None
    return list(type(label)).index(label)


def bet_patterns(bet, sizing, r, e, kind, pre, pf_raiser, checked, opened, first_on_flop):
    a, to_call = r["action"], r["to_call"]
    if pre and kind == "calls" and to_call <= 2 and not opened:
        bet[e]["limps"] += 1
    if pre:
        bet[e]["preflop_decisions"] += 1
    if r["street"] == "flop" and e == pf_raiser and not opened and to_call == 0:
        bet[e]["cbet_chances"] += 1
        bet[e]["cbets"] += int(kind == "raises")
    if not pre and kind == "raises" and e in checked and opened:
        bet[e]["check_raises"] += 1
    if not pre and e in checked:
        bet[e]["check_raise_chances"] += int(opened)
    if kind != "raises":
        return
    amt = int(re.search(r"raises to (\d+)", a).group(1))
    base = max(1, r["pot"] - amt)
    frac = amt / base
    street = r["street"]
    bet[e][f"raises_{street}"] += 1
    bet[e]["overbets"] += int(frac > 1.0 and not pre)
    bet[e]["min_raises"] += int(amt <= 2 * max(2, to_call) and to_call > 0)
    bet[e][f"allins_{street}"] += int("(all-in)" in a)
    sizing.append((round(frac, 2), strength(r.get("hole"), r.get("board")), street))


def betting_summary(c, sz):
    post = [(f, s) for f, s, st in sz if st != "preflop" and s is not None]
    bluffs = sum(1 for _, s in post if s == 0)
    big = [s for f, s in post if f >= 0.75]
    small = [s for f, s in post if f < 0.75]
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
    return {
        "limp_rate": rate(c["limps"], c["preflop_decisions"]),
        "cbet_rate": rate(c["cbets"], c["cbet_chances"]),
        "cbet_chances": c["cbet_chances"],
        "check_raise_rate": rate(c["check_raises"], c["check_raise_chances"]),
        "overbets": c["overbets"],
        "min_raises": c["min_raises"],
        "raises_by_street": {k[7:]: v for k, v in c.items() if k.startswith("raises_")},
        "allins_by_street": {k[7:]: v for k, v in c.items() if k.startswith("allins_") and v},
        "postflop_bets_sized": len(post),
        "bluff_share": rate(bluffs, len(post)),
        "sizing_tell": {
            "mean_strength_big_bets": mean(big),
            "mean_strength_small_bets": mean(small),
            "n_big": len(big),
            "n_small": len(small),
        },
        "median_size_x_pot": sorted(f for f, _, _ in sz)[len(sz) // 2] if sz else None,
    }


# ---------------------------------------------------------------------------------------------- poker
def scan_poker(arena):
    st = collections.defaultdict(collections.Counter)
    sizes = collections.defaultdict(collections.Counter)
    pos_vpip = collections.defaultdict(collections.Counter)
    tilt = collections.defaultdict(collections.Counter)
    bet = collections.defaultdict(collections.Counter)  # betting patterns
    sizing = collections.defaultdict(list)  # (size as fraction of pot, hand strength) per bet or raise
    for path in sorted(glob.glob(os.path.join(arena, "rounds", "*", "poker-table*", "hands.jsonl"))):
        rows = [json.loads(x) for x in open(path)]
        hands = collections.defaultdict(list)
        ends = {}
        for r in rows:
            (ends.__setitem__(r["hand"], r) if r.get("end") else hands[r["hand"]].append(r))
        recent_loss = collections.defaultdict(int)  # hands left in a "just lost big" window
        for h in sorted(hands):
            acts, end = hands[h], ends.get(h)
            seats = [short(x) for x in acts[0]["seats"]] if acts else []
            vp, pr = set(), set()
            street_now, pf_raiser, checked, opened = None, None, set(), False
            for r in acts:
                e, a = short(r["model"]), r["action"]  # action rows carry full names, end rows short ones
                kind = a.split()[0]
                pre = r["street"] == "preflop"
                if r["street"] != street_now:
                    street_now, checked, opened = r["street"], set(), False
                    first_on_flop = r["street"] == "flop"
                bet_patterns(bet, sizing[e], r, e, kind, pre, pf_raiser, checked, opened, first_on_flop)
                first_on_flop = False
                if kind == "checks":
                    checked.add(e)
                if kind == "raises":
                    opened = True
                    if pre:
                        pf_raiser = e
                if pre and kind in ("calls", "raises"):
                    vp.add(e)
                if pre and kind == "raises":
                    pr.add(e)
                if not pre and kind == "raises":
                    st[e]["post_raises"] += 1
                if not pre and kind == "calls":
                    st[e]["post_calls"] += 1
                if not pre and r["to_call"]:  # facing a bet after the flop
                    st[e]["faced_bet"] += 1
                    st[e]["folded_to_bet"] += int(kind == "folds")
                if "(all-in)" in a:
                    st[e]["allins"] += 1
                if kind == "raises" and r["pot"]:
                    amt = int(re.search(r"raises to (\d+)", a).group(1))
                    sizes[e][round(amt / max(1, r["pot"] - amt), 1)] += 1
            for i, e in enumerate(seats):
                st[e]["hands"] += 1
                st[e]["vpip"] += int(e in vp)
                st[e]["pfr"] += int(e in pr)
                pos = (
                    "button"
                    if i == len(seats) - 1
                    else "small blind"
                    if i == 0
                    else "big blind"
                    if i == 1
                    else "early/middle"
                )
                pos_vpip[e][pos + "_hands"] += 1
                pos_vpip[e][pos + "_vpip"] += int(e in vp)
                window = "after_big_loss" if recent_loss[e] > 0 else "normal"
                tilt[e][window + "_hands"] += 1
                tilt[e][window + "_vpip"] += int(e in vp)
                recent_loss[e] = max(0, recent_loss[e] - 1)
            if end:
                for e, p in zip((short(x) for x in end["seats"]), end["payoffs"]):
                    if p <= -50:
                        recent_loss[e] = 3
    players = []
    for e, c in sorted(st.items()):
        players.append(
            {
                "entrant": short(e),
                "kind": ps.kind_of(e),
                "hands": c["hands"],
                "vpip": rate(c["vpip"], c["hands"]),
                "pfr": rate(c["pfr"], c["hands"]),
                "aggression": round(c["post_raises"] / c["post_calls"], 2) if c["post_calls"] else None,
                "fold_to_bet": rate(c["folded_to_bet"], c["faced_bet"]),
                "allin_rate": rate(c["allins"], c["hands"]),
                "favourite_raise_sizes_x_pot": dict(sizes[e].most_common(3)),
                "betting": betting_summary(bet[e], sizing[e]),
                "vpip_by_position": {
                    k.replace("_hands", ""): rate(pos_vpip[e][k.replace("_hands", "_vpip")], v)
                    for k, v in pos_vpip[e].items()
                    if k.endswith("_hands")
                },
                "tilt": {
                    "vpip_after_big_loss": rate(tilt[e]["after_big_loss_vpip"], tilt[e]["after_big_loss_hands"]),
                    "vpip_otherwise": rate(tilt[e]["normal_vpip"], tilt[e]["normal_hands"]),
                    "hands_after_big_loss": tilt[e]["after_big_loss_hands"],
                },
            }
        )
    refs = {p["entrant"]: p for p in players if p["entrant"] in ps.PERSONAS}
    for p in players:
        if p["entrant"] in ps.PERSONAS or p["vpip"] is None or not refs:
            continue

        def dist(q):
            # aggression (raises per call) runs 0-8; scale it to the share range and skip it when missing
            d = (p["vpip"] - q["vpip"]) ** 2 + (p["pfr"] - q["pfr"]) ** 2
            if p["aggression"] is not None and q["aggression"] is not None:
                d += ((min(p["aggression"], 8) - min(q["aggression"], 8)) / 8) ** 2
            return d

        best = min(refs.values(), key=dist)
        p["plays_most_like"] = f"{best['entrant']} ({ps.PERSONAS[best['entrant']]['kind']})"
    return {"players": players}


# ---------------------------------------------------------------------------------------------- agents
def scan_agents():
    found = []
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "raw", "*.json"))):
        try:
            data = json.load(open(path))
        except (OSError, ValueError):
            continue
        if not isinstance(data, list) or not data or "tasks" not in data[0]:
            continue
        for r in data:
            for t in r["tasks"]:
                calls = [s for s in t.get("log", []) if "tool" in s]
                run, worst = 1, 1
                for a, b in zip(calls, calls[1:]):
                    same = a["tool"] == b["tool"] and a.get("args") == b.get("args")
                    run = run + 1 if same else 1
                    worst = max(worst, run)
                errors = sum(1 for s in calls if str(s.get("result", "")).startswith(("ERROR", "### Error")))
                ans = t.get("answer") or ""
                flags = []
                if worst >= 3:
                    flags.append(f"loop: same call {worst}x in a row")
                if not calls and ans:
                    flags.append("answered with no tool call")
                if not t["passed"] and re.search(r"\d", ans):
                    flags.append("failed but asserted a number")
                if errors:
                    flags.append(f"{errors} tool errors")
                if flags:
                    found.append(
                        {
                            "file": os.path.basename(path),
                            "model": short(r["model"]),
                            "task": t["task"],
                            "passed": t["passed"],
                            "patterns": flags,
                            "answer": ans[:100],
                        }
                    )

    def category(flag):
        return "tool errors" if flag.endswith("tool errors") else flag.split(":")[0]

    summary = collections.Counter(category(f) for x in found for f in x["patterns"])
    return {"findings": found, "counts": dict(summary)}


def report(res):
    L = ["# Behaviour patterns", ""]
    c = res["chess"]
    if c["players"]:
        L += [
            "## Chess",
            "",
            "| Entrant | Moves | Repeats | Shuffles | No legal move | Top openings |",
            "|---|---|---|---|---|---|",
        ]
        for p in c["players"]:
            L.append(
                f"| {p['entrant']} | {p['moves']} | {p['repeat_rate']:.0%} | {p['shuffle_rate']:.0%} | "
                f"{p['no_legal_move_rate']:.0%} | {', '.join(f'{k}×{v}' for k, v in p['openings'].items())} |"
            )
        L += ["", "Endings: " + ", ".join(f"{k} {v}" for k, v in c["endings"].items()), ""]
        for a in c["skill_adherence"]:
            L.append(
                f"- **{a['entrant']}** wrote a rule about repetition: repeat rate {a['repeat_rate_round_1']} in "
                f"its first round, {a['repeat_rate_later']} later; {'followed' if a['followed'] else 'not followed'}."
            )
        L.append("")
    p = res["poker"]
    if p["players"]:
        L += [
            "## Poker",
            "",
            "| Entrant | Kind | Hands | VPIP | PFR | Aggr. | Fold to bet | All-in | Tilt VPIP (after loss / else) | Plays like |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        f = lambda v: "–" if v is None else f"{v:.0%}"  # noqa: E731
        for q in sorted(p["players"], key=lambda q: q["kind"]):
            t = q["tilt"]
            L.append(
                f"| {q['entrant']} | {q['kind']} | {q['hands']} | {f(q['vpip'])} | {f(q['pfr'])} | "
                f"{q['aggression'] if q['aggression'] is not None else '–'} | {f(q['fold_to_bet'])} | "
                f"{f(q['allin_rate'])} | {f(t['vpip_after_big_loss'])} / {f(t['vpip_otherwise'])} "
                f"(n={t['hands_after_big_loss']}) | {q.get('plays_most_like', '')} |"
            )
        L += [
            "",
            "### Betting patterns",
            "",
            "| Entrant | Limp | C-bet | Check-raise | Median size ×pot | Overbets | Min-raises | Bluff share | Strength: big bets / small bets |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for q in sorted(p["players"], key=lambda q: q["kind"]):
            b = q["betting"]
            t = b["sizing_tell"]
            L.append(
                f"| {q['entrant']} | {f(b['limp_rate'])} | {f(b['cbet_rate'])} (n={b['cbet_chances']}) | "
                f"{f(b['check_raise_rate'])} | {b['median_size_x_pot'] if b['median_size_x_pot'] is not None else '–'} | "
                f"{b['overbets']} | {b['min_raises']} | {f(b['bluff_share'])} (n={b['postflop_bets_sized']}) | "
                f"{'–' if t['mean_strength_big_bets'] is None else t['mean_strength_big_bets']} / {'–' if t['mean_strength_small_bets'] is None else t['mean_strength_small_bets']} |"
            )
        L += [
            "",
            "Strength: 0 high card, 1 pair, 2 two pair, 3+ better, at the moment of the bet. A big-bet mean well "
            "above the small-bet mean is a sizing tell: bet size gives the hand away.",
            "",
        ]
    a = res["agents"]
    if a["findings"]:
        L += [
            "## Agent tests",
            "",
            "Counts: " + ", ".join(f"{k} {v}" for k, v in a["counts"].items()),
            "",
            "| File | Model | Task | Passed | Patterns |",
            "|---|---|---|---|---|",
        ]
        for x in a["findings"]:
            L.append(
                f"| {x['file']} | {x['model']} | {x['task']} | {'✓' if x['passed'] else '✗'} | {'; '.join(x['patterns'])} |"
            )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    arena = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "arena")
    prefix = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "patterns", "latest")
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    res = {
        "chess": scan_chess(arena),
        "poker": scan_poker(arena),
        "agents": scan_agents(),
    }
    json.dump(res, open(prefix + ".json", "w"), indent=1)
    open(prefix + ".md", "w").write(report(res))
    print(
        f"{prefix}.md: {len(res['chess']['players'])} chess entrants, {len(res['poker']['players'])} poker "
        f"entrants, {len(res['agents']['findings'])} agent findings"
    )
