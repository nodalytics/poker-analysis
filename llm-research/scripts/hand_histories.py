"""Read published hand histories (PHH format) into player statistics and per-decision training rows.

Usage: uv run python scripts/hand_histories.py PHH_DIR OUT_PREFIX [MAX_HANDS] [--population]
(--population pools anonymised players per stake, e.g. HandHQ, into one row per stake)
Writes OUT_PREFIX.stats.json (per player: hands, VPIP, PFR, aggression, fold to a bet, 3-bet) and
OUT_PREFIX.decisions.jsonl.gz (one row per decision: player, street, seat, hole cards when known, board, pot,
amount to call, action, size as a fraction of the pot).

Source: the PHH dataset (github.com/uoftcprg/phh-dataset, MIT, pinned to commit e47fbd581637): 10,000 Pluribus hands
against professionals (all hole cards known), 83 WSOP final-table hands, and 21.6M anonymised 2009 HandHQ hands.
Only published research data: no poker site is scraped.
"""

import ast
import collections
import glob
import gzip
import json
import os
import re
import sys


def parse_phh(text):
    """Minimal PHH reader: the key = value lines this project needs."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+)\s*=\s*(.+)$", line.strip())
        if m:
            try:
                out[m.group(1)] = ast.literal_eval(m.group(2).replace("true", "True").replace("false", "False"))
            except (ValueError, SyntaxError):
                out[m.group(1)] = m.group(2)
    return out


def hands(phh_dir, limit=None):
    """Every hand under phh_dir: .phh files hold one, .phhs files many ([1], [2], ... sections)."""
    files = sorted(glob.glob(os.path.join(phh_dir, "**", "*.phh*"), recursive=True))
    n = 0
    for f in files:
        text = open(f, errors="replace").read()
        parts = re.split(r"(?m)^\[\d+\]\s*$", text) if f.endswith(".phhs") else [text]
        for part in parts:
            if not part.strip():
                continue
            if limit and n >= limit:
                return
            n += 1
            yield f, parse_phh(part)


def replay(h):
    """Yield one decision dict per player action, with the betting state just before it."""
    names = h.get("players") or [f"p{i + 1}" for i in range(len(h["starting_stacks"]))]
    n = len(names)
    blinds = list(h.get("blinds_or_straddles", [0] * n)) + [0] * n
    holes, board, street = {}, [], "preflop"
    bets = [blinds[i] for i in range(n)]
    pot_before_street = 0
    raises_this_street = 0
    for a in h.get("actions", []):
        p = a.split()
        if p[0] == "d":
            if p[1] == "dh":
                holes[int(p[2][1:]) - 1] = p[3]
            elif p[1] == "db":
                board += [p[2][i : i + 2] for i in range(0, len(p[2]), 2)]
                street = {3: "flop", 4: "turn", 5: "river"}.get(len(board), street)
                pot_before_street += sum(bets)
                bets = [0] * n
                raises_this_street = 0
            continue
        if not p[0].startswith("p") or len(p) < 2 or p[1] not in ("f", "cc", "cbr"):
            continue
        i = int(p[0][1:]) - 1
        to_call = max(bets) - bets[i]
        pot = pot_before_street + sum(bets)
        row = {
            "player": names[i],
            "seat": i,
            "players": n,
            "street": street,
            "hole": holes.get(i),
            "board": "".join(board),
            "pot": pot,
            "to_call": to_call,
            "action": p[1],
            "raises_before": raises_this_street,
        }
        if p[1] == "cc":
            bets[i] += to_call
        elif p[1] == "cbr":
            amt = int(float(p[2]))
            row["size_x_pot"] = round((amt - bets[i]) / max(1, pot + to_call), 3)
            bets[i] = amt
            raises_this_street += 1
        yield row


def stake_of(path):
    m = re.search(r"_(\d+)NLH", path)
    return f"{m.group(1)}NL" if m else "unknown"


def main(phh_dir, prefix, limit=None, population=False):
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    st = collections.defaultdict(collections.Counter)
    n_hands = 0
    with gzip.open(prefix + ".decisions.jsonl.gz", "wt") as out:
        for f, h in hands(phh_dir, limit):
            if "actions" not in h or "starting_stacks" not in h:
                continue
            n_hands += 1
            vp, pr, three = set(), set(), set()
            names = h.get("players") or [f"p{i + 1}" for i in range(len(h["starting_stacks"]))]
            if population:  # anonymised players: keep them apart while reading, sum them per stake at the end
                names = [f"{stake_of(f)}|{x}" for x in names]
                h = dict(h, players=names)
            for r in replay(h):
                out.write(json.dumps(r) + "\n")
                c, who = st[r["player"]], r["player"]
                pre = r["street"] == "preflop"
                if pre and r["action"] in ("cc", "cbr") and not (r["action"] == "cc" and r["to_call"] == 0):
                    vp.add(who)
                if pre and r["action"] == "cbr":
                    pr.add(who)
                    if r["raises_before"] >= 1:
                        three.add(who)
                if pre and r["raises_before"] == 1:
                    c["three_bet_chances"] += 1
                if not pre and r["action"] == "cbr":
                    c["post_raises"] += 1
                if not pre and r["action"] == "cc" and r["to_call"] > 0:
                    c["post_calls"] += 1
                if not pre and r["to_call"] > 0:
                    c["faced_bet"] += 1
                    c["folded_to_bet"] += r["action"] == "f"
            for who in names:
                st[who]["hands"] += 1
                st[who]["vpip"] += who in vp
                st[who]["pfr"] += who in pr
                st[who]["three_bets"] += who in three
    if population:
        pooled = collections.defaultdict(collections.Counter)
        for who, c in st.items():
            pooled[who.split("|")[0]].update(c)
        st = pooled
    rate = lambda a, b: round(a / b, 3) if b else None  # noqa: E731
    stats = {
        who: {
            "hands": c["hands"],
            "vpip": rate(c["vpip"], c["hands"]),
            "pfr": rate(c["pfr"], c["hands"]),
            "three_bet": rate(c["three_bets"], c["three_bet_chances"]),
            "aggression": round(c["post_raises"] / c["post_calls"], 2) if c["post_calls"] else None,
            "fold_to_bet": rate(c["folded_to_bet"], c["faced_bet"]),
        }
        for who, c in st.items()
        if c["hands"] >= 50
    }
    json.dump({"source": phh_dir, "hands": n_hands, "players": stats}, open(prefix + ".stats.json", "w"), indent=1)
    print(f"{n_hands} hands, {len(stats)} players with 50+ hands -> {prefix}.stats.json")
    for who, s in sorted(stats.items(), key=lambda x: -x[1]["hands"])[:8]:
        print(
            f"  {who:12} hands {s['hands']:6}  VPIP {s['vpip']:.0%}  PFR {s['pfr']:.0%}  3-bet {s['three_bet']}  "
            f"aggr {s['aggression']}  fold-to-bet {s['fold_to_bet']}"
        )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--population"]
    main(args[0], args[1], int(args[2]) if len(args) > 2 else None, population="--population" in sys.argv)
