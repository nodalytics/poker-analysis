"""A player's Kestrel practice hands as hand histories, for review (2026-11-27).

Kestrel writes every practice hand to its session folder (sessions/<table>-practice/hands.jsonl). Since 2026-11-27
each record also holds a private part the screen never sees: every player's cards, the whole board (the cards still to
come included), the stacks at the start and every decision with the network's probabilities (at the player's own
seat: what Kestrel would have done). This turns the hands of one player into
  OUT.txt   one readable history a hand: seats, stacks and cards, each street's actions with Kestrel's
            probabilities beside the networks' moves and its advice beside the player's, the board, the results;
  OUT.phhs  the same hands in PHH (github.com/uoftcprg/phh-dataset's format, as the HandHQ and Pluribus hands), in
            chips with the big blind 100, so poker tools and our own readers (hand_histories.py) can read them.
Hands from before 2026-11-27 have no private part: the cards nobody showed are left out.

The histories hold a person's play: keep them out of the repository.

Usage (lab: copy the container's sessions first):
  docker cp kestrel-kestrel-1:/data/sessions/. ~/scratch/kestrel_sessions/
  python3 scripts/kestrel_export.py ~/scratch/kestrel_sessions --player Alex --out ~/scratch/review/alex
"""

import argparse
import glob
import json
import os
import re

STREETS = ["Pre-flop", "Flop", "Turn", "River"]
BB = 100


def hands(root, player=None, table=None):
    """Practice hands under root (a sessions folder or one hands.jsonl), in table and hand order; with player, only
    the hands where a seat had that name (case-insensitive)."""
    files = [root] if root.endswith(".jsonl") else glob.glob(os.path.join(root, "*-practice", "hands.jsonl"))
    files.sort(key=lambda f: int(re.sub(r"\D", "", os.path.basename(os.path.dirname(f))) or 0))
    for f in files:
        for line in open(f, errors="replace"):
            try:
                h = json.loads(line)
            except ValueError:
                continue
            if h.get("type") != "practice_hand" or (table is not None and h.get("table") != table):
                continue
            names = [r["name"] for r in h.get("results", [])]
            if player and player.lower() not in (n.lower() for n in names):
                continue
            yield h


def _num(label):
    m = re.findall(r"[0-9.]+", label)
    return float(m[0]) if m else 0.0


def _opts(d):
    return ", ".join(f"{o['label']} {100 * o['prob']:.0f}%" for o in sorted(d["options"], key=lambda o: -o["prob"]))


def text(h):
    res = sorted(h.get("results", []), key=lambda r: r["seat"])
    priv = h.get("private") or {}
    holes = {x["name"]: x["hole"] for x in priv.get("holes", [])}
    stacks = {x["name"]: x["stack_bb"] for x in priv.get("stacks_bb", [])}
    after = {s["name"]: s["chips_bb"] for s in h.get("standings", [])}
    out = [f"Kestrel practice, table {h.get('table')}, tournament {h.get('tournament')}, hand {h.get('hand')}, "
           f"level {h.get('level')}"]  # fmt: skip
    for r in res:
        st = stacks.get(r["name"], after.get(r["name"], 0) - r["result_bb"])
        cards = holes.get(r["name"]) or r.get("hole")
        who = " (the player)" if r.get("you") else ""
        out.append(f"  {r['pos']:<7} {r['name']:<16} {st:6.1f} bb  {' '.join(cards) if cards else '?? ??'}{who}")
    board = priv.get("board") or h.get("board", [])
    decs = priv.get("decisions")
    nb = {0: 0, 1: 3, 2: 4, 3: 5}
    street = -1
    for k, a in enumerate(h.get("log", [])):
        if a["street"] != street:
            street = a["street"]
            shown = board[nb[street - 1] : nb[street]] if street else []
            out.append(f"{STREETS[street]}{' [' + ' '.join(shown) + ']' if shown else ''}:")
        line = f"  {a['pos']:<7} {a['name']:<16} {a['action']}"
        if decs and k < len(decs):
            d = decs[k]
            line += f"   ({'Kestrel would: ' if d.get('human') else 'Kestrel: '}{_opts(d)})"
        out.append(line)
    if h.get("showdown"):
        for s in range(max(street + 1, 1), 4):
            out.append(f"{STREETS[s]} [{' '.join(board[nb[s - 1] : nb[s]])}] (all in)")
    if priv.get("board"):
        out.append(f"  the whole board: {' '.join(priv['board'])}")
    out.append("Results: " + ", ".join(f"{r['name']} {r['result_bb']:+.1f} bb" for r in res))
    return "\n".join(out)


def phh(h):
    """The hand in PHH, or None without the private part (unknown stacks or cards)."""
    priv = h.get("private")
    if not priv:
        return None
    res = sorted(h.get("results", []), key=lambda r: r["seat"])
    n = len(res)
    names = [r["name"] for r in res]
    p_of = {nm: k + 1 for k, nm in enumerate(names)}
    stacks = {x["name"]: x["stack_bb"] for x in priv["stacks_bb"]}
    holes = {x["name"]: x["hole"] for x in priv["holes"]}
    start = [round(stacks[nm] * BB) for nm in names]
    antes = [0] * n
    antes[1] = min(BB, max(start[1] - BB, 0))  # the big-blind ante, posted by the big blind
    contrib = [0.0] * n
    contrib[0], contrib[1] = 0.5, 1.0  # big blinds, the hand so far (Kestrel's labels: "raise to X" is the hand total)
    acts = [f"d dh p{k + 1} {''.join(holes[nm])}" for k, nm in enumerate(names)]
    board = priv["board"]
    nb = {0: 0, 1: 3, 2: 4, 3: 5}
    street, at_street = 0, contrib[:]
    for a in h.get("log", []):
        while a["street"] > street:
            street += 1
            acts.append(f"d db {''.join(board[nb[street - 1] : nb[street]])}")
            at_street = contrib[:]
        p, lab = p_of[a["name"]] - 1, a["action"]
        if lab.startswith("fold"):
            acts.append(f"p{p + 1} f")
        elif lab.startswith(("check", "call")):
            contrib[p] += _num(lab) if lab.startswith("call") else 0.0
            acts.append(f"p{p + 1} cc")
        else:  # raise to X / all-in X: X the player's hand total, PHH wants the street's total
            contrib[p] = _num(lab)
            acts.append(f"p{p + 1} cbr {round((contrib[p] - at_street[p]) * BB)}")
    if h.get("showdown"):
        while street < 3:
            street += 1
            acts.append(f"d db {''.join(board[nb[street - 1] : nb[street]])}")
        for r in res:
            if not r.get("folded"):
                acts.append(f"p{p_of[r['name']]} sm {''.join(holes[r['name']])}")
    after = {s["name"]: s["chips_bb"] for s in h.get("standings", [])}
    lines = ["variant = 'NT'", "ante_trimming_status = true", f"antes = {antes}",
             f"blinds_or_straddles = {[BB // 2, BB] + [0] * (n - 2)}", f"min_bet = {BB}", f"starting_stacks = {start}",
             f"actions = {acts}", f"players = {names}",
             f"finishing_stacks = {[round(after.get(nm, 0) * BB) for nm in names]}",
             f"hand = {h.get('hand')}", f"table = {h.get('table')}"]  # fmt: skip
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", help="a copy of Kestrel's sessions folder, or one hands.jsonl")
    ap.add_argument("--player", help="only the hands with a seat of this name")
    ap.add_argument("--table", type=int)
    ap.add_argument("--out", required=True, help="OUT.txt and OUT.phhs are written")
    a = ap.parse_args()
    hs = list(hands(a.sessions, a.player, a.table))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".txt", "w") as f:
        f.write("\n\n".join(text(h) for h in hs) + "\n")
    n = 0
    with open(a.out + ".phhs", "w") as f:
        for h in hs:
            p = phh(h)
            if p:
                n += 1
                f.write(f"[{n}]\n{p}\n\n")
    won = sum(r["result_bb"] for h in hs for r in h["results"] if a.player and r["name"].lower() == a.player.lower())
    print(json.dumps({"hands": len(hs), "phh_hands": n, "player_result_bb": round(won, 1) if a.player else None,
                      "files": [a.out + ".txt", a.out + ".phhs"]}))  # fmt: skip


if __name__ == "__main__":
    main()
