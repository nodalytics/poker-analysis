"""Read no-limit hold'em hand histories from several sites into one plain format.

Supported:
  text, PokerStars family   PokerStars, GGPoker (exported histories), ACR / Winning Poker Network, Ignition / Bovada,
                            Winamax, and other sites that write the same layout: a header line with the hand number,
                            seats with stacks, blind posts, '*** HOLE CARDS ***', 'Dealt to <hero> [..]', one action
                            per line, '*** FLOP/TURN/RIVER ***' with the board, shows, collected, summary.
  PHH                       the open Poker Hand History standard (.phh, .phhs), as in our research datasets.
Not yet: 888, partypoker and iPoker (different layouts).

Amounts become integer units: cents when the file shows decimal money ($, EUR, GBP), chips otherwise. Kept: hands
with one small and one big blind and no straddle or dead blind (those change the order of betting).
Each hand: id, site, time, table, scale, blinds, players (in PokerKit order: small blind, big blind, ... button;
heads-up: big blind, then button), stacks, hero, hero_cards, actions [('board', street, cards) |
('act', player, fold/check/call/bet/raise, raise-to amount or None)], board, shown, returned, collected, text.
"""

import os
import re

CUR = r"[$€£]?"
AMT = CUR + r"([\d,]+(?:\.\d+)?)(?:\s?(?:€|EUR|USD|GBP))?"
HEAD = re.compile(
    r"^(?:PokerStars (?:Hand|Game)|Poker Hand|Game Hand|Ignition Hand|Bovada Hand|Winamax Poker[^#\n]*HandId:|Hand)"
    r"\s*#\s*([\w-]+)",
    re.M,
)
SEAT = re.compile(r"^Seat (\d+): (.+?) \(" + AMT + r"(?: in chips)?(?:,[^)]*)?\)", re.M)
BUTTON = re.compile(r"Seat #(\d+) is the button")
DEALT = re.compile(r"^Dealt to (.+?) \[(\w\w) (\w\w)\]", re.M)
BOARD = re.compile(r"^\*\*\* (FLOP|TURN|RIVER) \*\*\* (.*)$")
CARDS = re.compile(r"\[([^\]]*)\]")


def site_of(first_line):
    for key, name in (
        ("PokerStars", "PokerStars"),
        ("Poker Hand #", "GGPoker"),
        ("Game Hand #", "ACR / WPN"),
        ("Ignition", "Ignition"),
        ("Bovada", "Bovada"),
        ("Winamax", "Winamax"),
    ):
        if key in first_line:
            return name
    return "other"


def _scale(text):
    return 100 if re.search(r"[$€£]|\d\.\d\d", text) else 1


def _amt(s, scale):
    return int(round(float(s.replace(",", "")) * scale))


def split_hands(text):
    starts = [m.start() for m in HEAD.finditer(text)]
    return [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]


def _strip(name):
    return name.strip().rstrip(":").strip()


def parse_hand(block):
    """One hand of text -> dict, or None when it is not a hand we can replay."""
    first = block.splitlines()[0]
    low = first.lower().replace("'", "")
    if "holdem" not in low or "no limit" not in low.replace("-", " ").replace("(", " ").replace(")", " "):
        return None
    pre, _, rest = block.partition("*** HOLE CARDS ***")
    if not rest:
        rest = block[len(pre) :]
    scale = _scale(first + pre)
    hid = HEAD.search(block).group(1)
    date = re.search(r"(\d{4}[/-]\d\d[/-]\d\d[ T]\d\d:\d\d:\d\d)", first) or re.search(
        r"(\d{4}[/-]\d\d[/-]\d\d[ T]\d\d:\d\d:\d\d)", pre
    )
    seats = [(int(s), _strip(name), _amt(a, scale)) for s, name, a in SEAT.findall(pre)]
    btn = BUTTON.search(pre)
    sb = re.search(r"^(.+?):? posts (?:the )?small blind " + AMT, block, re.M)
    bb = re.search(r"^(.+?):? posts (?:the )?big blind " + AMT, block, re.M)
    if not seats or not btn or not sb or not bb:
        return None
    if re.search(r"posts (?:a )?(?:straddle|dead|small & big|small and big)", block):
        return None
    sb_name, bb_name = _strip(sb.group(1)), _strip(bb.group(1))
    names = [n for _, n, _ in seats]
    if sb_name not in names or bb_name not in names:
        return None
    blinds = (_amt(sb.group(2), scale), _amt(bb.group(2), scale))
    # sitting-out players hold a seat without being dealt in: keep players who post or act
    involved = {sb_name, bb_name}
    for _, n, _ in seats:
        if re.search(r"^" + re.escape(n) + r":? (?:folds|checks|calls|bets|raises|posts)", block, re.M):
            involved.add(n)
    seats = [s for s in sorted(seats) if s[1] in involved]
    k = [n for _, n, _ in seats].index(sb_name)
    clockwise = seats[k:] + seats[:k]
    if len(clockwise) == 2:
        clockwise = [clockwise[1], clockwise[0]]
        if clockwise[0][1] != bb_name:
            return None
    elif clockwise[1][1] != bb_name:
        return None
    players = [n for _, n, _ in clockwise]
    stacks = [st for _, _, st in clockwise]
    dealt = DEALT.search(rest)
    hero, hero_cards = (_strip(dealt.group(1)), dealt.group(2) + dealt.group(3)) if dealt else (None, None)
    actions, board, shown, returned, collected = [], [], {}, {}, {}
    body = rest.split("*** SUMMARY ***")[0]
    by_len = sorted(players, key=len, reverse=True)
    for line in body.splitlines():
        line = line.strip()
        m = BOARD.match(line)
        if m:
            new = CARDS.findall(m.group(2))[-1].split()
            actions.append(("board", m.group(1).lower(), new))
            board += new
            continue
        m = re.match(r"^Uncalled bet \(" + AMT + r"\) returned to (.+)$", line)
        if m:
            p = _strip(m.group(2))
            returned[p] = returned.get(p, 0) + _amt(m.group(1), scale)
            continue
        who = next((p for p in by_len if line.startswith(p + " ") or line.startswith(p + ":")), None)
        if who is None:
            continue
        tail = line[len(who) :].lstrip(":").strip()
        m = re.match(r"^collected " + AMT, tail)
        if m:
            collected[who] = collected.get(who, 0) + _amt(m.group(1), scale)
            continue
        m = re.match(r"^shows \[(\w\w) (\w\w)\]", tail)
        if m:
            shown[who] = m.group(1) + m.group(2)
            continue
        if tail.startswith("folds"):
            actions.append(("act", who, "fold", None))
        elif tail.startswith("checks"):
            actions.append(("act", who, "check", None))
        elif tail.startswith("calls"):
            actions.append(("act", who, "call", None))
        elif tail.startswith("bets"):
            actions.append(("act", who, "bet", _amt(re.search(AMT, tail[4:]).group(1), scale)))
        elif tail.startswith("raises"):
            m2 = re.search(r"to " + AMT, tail)
            actions.append(("act", who, "raise", _amt(m2.group(1), scale) if m2 else None))
    if not collected:  # some sites only report winnings in the summary
        for line in block.split("*** SUMMARY ***")[-1].splitlines():
            m = re.search(r"(?:won|collected) \(?" + AMT, line)
            if m and line.startswith("Seat "):
                who = next((p for p in by_len if (": " + p) in line), None)
                if who:
                    collected[who] = collected.get(who, 0) + _amt(m.group(1), scale)
    tm = re.search(r"Table '([^']+)'", pre) or re.search(r"Table:? ?([^\n]+?)(?: \d+-max| Seat|$)", pre, re.M)
    return {
        "id": hid,
        "site": site_of(first),
        "time": date.group(1) if date else "",
        "table": tm.group(1).strip() if tm else "",
        "scale": scale,
        "blinds": blinds,
        "players": players,
        "stacks": stacks,
        "hero": hero,
        "hero_cards": hero_cards,
        "actions": actions,
        "board": board,
        "shown": shown,
        "returned": returned,
        "collected": collected,
        "text": block.strip(),
    }


def parse_phh(text, hero=None):
    """Hands in the PHH format (one per .phh, many per .phhs with [n] headers). The hero is the named player whose
    hole cards are known (or the only one), since PHH has no 'Dealt to' line."""
    import ast

    out = []
    for part in re.split(r"(?m)^\[\d+\]\s*$", text):
        d = {}
        for line in part.splitlines():
            m = re.match(r"^(\w+)\s*=\s*(.+)$", line.strip())
            if m:
                try:
                    d[m.group(1)] = ast.literal_eval(m.group(2).replace("true", "True").replace("false", "False"))
                except (ValueError, SyntaxError):
                    d[m.group(1)] = m.group(2)
        if "actions" not in d or "starting_stacks" not in d:
            continue
        n = len(d["starting_stacks"])
        players = list(d.get("players") or [f"p{k + 1}" for k in range(n)])
        blinds = [b for b in d.get("blinds_or_straddles", []) if b]
        if len(blinds) != 2 or any(d.get("antes", [0]) or [0]):
            continue
        vals = list(d["starting_stacks"]) + blinds
        scale = 100 if any(isinstance(v, float) and v != int(v) for v in vals) else 1
        holes, actions, board, shown = {}, [], [], {}
        for a in d["actions"]:
            p = a.split()
            if p[0] == "d" and p[1] == "dh":
                holes[int(p[2][1:]) - 1] = p[3]
            elif p[0] == "d" and p[1] == "db":
                cs = [p[2][k : k + 2] for k in range(0, len(p[2]), 2)]
                board += cs
                actions.append(("board", {3: "flop", 4: "turn", 5: "river"}[len(board)], cs))
            elif p[0].startswith("p") and len(p) >= 2:
                who = players[int(p[0][1:]) - 1]
                if p[1] == "f":
                    actions.append(("act", who, "fold", None))
                elif p[1] == "cc":
                    actions.append(("act", who, "call", None))
                elif p[1] == "cbr":
                    actions.append(("act", who, "raise", int(round(float(p[2]) * scale))))
                elif p[1] == "sm" and "?" not in p[2]:
                    shown[who] = p[2]
        known = {k: v for k, v in holes.items() if "?" not in v}
        h = hero if hero in players else (players[next(iter(known))] if len(known) == 1 else None)
        if h is None:
            continue
        win = d.get("winnings") or [0] * n
        out.append(
            {
                "id": str(d.get("hand", len(out) + 1)),
                "site": "PHH",
                "time": f"{d.get('year', 2000)}/{int(d.get('month', 1)):02d}/{int(d.get('day', 1)):02d} {d.get('time', '00:00:00')}",
                "table": str(d.get("table", "")),
                "scale": scale,
                "blinds": tuple(int(round(b * scale)) for b in blinds),
                "players": players,
                "stacks": [int(round(s * scale)) for s in d["starting_stacks"]],
                "hero": h,
                "hero_cards": holes.get(players.index(h)),
                "actions": actions,
                "board": board,
                "shown": shown,
                "returned": {},
                "collected": {p: int(round(w * scale)) for p, w in zip(players, win) if w},
                "text": part.strip(),
            }
        )
    return out


def parse_text(text, hero=None):
    if re.search(r"^actions\s*=", text, re.M):
        return parse_phh(text, hero)
    return [h for h in (parse_hand(b) for b in split_hands(text)) if h]


def parse_files(paths, hero=None):
    out = []
    for p in paths:
        out += parse_text(open(p, encoding="utf-8-sig", errors="replace").read(), hero)
    return out


HAND_FILES = (".txt", ".phh", ".phhs", ".log")


def find_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in names if n.lower().endswith(HAND_FILES)]
        else:
            files.append(p)
    return sorted(files)
