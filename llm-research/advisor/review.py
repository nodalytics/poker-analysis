"""Session review: every decision the hero made in finished hands, graded against the solver stack, plus leaks.

Usage: python -m advisor.review HAND_HISTORY_FOLDER_OR_FILES... [--hero NAME] [--out report.html] [--no-solver]
Reads hand-history files from PokerStars, GGPoker, ACR/WPN, Ignition/Bovada, Winamax or PHH (handhistory.py; only
finished hands, it never watches a table), replays each hand, asks the engine
what it would play at each of the hero's decisions, and grades the hero's action by how often the engine plays it:
  agrees        the engine plays it 35%+ of the time
  questionable  10-35%
  mistake       under 10%
Decisions are ranked by the pot at stake (in big blinds) so the costliest come first. Leak statistics compare the
hero's frequencies with typical winning ranges for 6-max cash games, and two tests from our behavioural study:
playing looser when stuck in a session, and calling river raises (only 8.5% of real river raises are bluffs).
"""

import argparse
import collections
import datetime as dt
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advisor import engine, handhistory  # noqa: E402

CATEGORY = {"fold": "fold", "check": "call", "call": "call", "bet": "raise", "raise": "raise"}
# typical winning 6-max cash ranges (approximate; trackers' common guidance)
REFERENCE = {
    "VPIP": (20, 28),
    "PFR": (16, 23),
    "3-bet": (6, 10),
    "Fold to 3-bet": (40, 60),
    "Flop c-bet": (50, 70),
    "Went to showdown": (24, 32),
    "Won at showdown": (48, 58),
}


def grade(p):
    return "agrees" if p >= 0.35 else "questionable" if p >= 0.10 else "mistake"


def review_hand(h, hero, use_solver=True):
    """Replay one hand; returns (decisions, facts) for the hero, or None if the hand cannot be replayed."""
    if hero not in h["players"]:
        return None
    hi = h["players"].index(hero)
    holes = [h["hero_cards"] if p == hero else h["shown"].get(p) for p in h["players"]]
    try:
        t = engine.Table(h["stacks"], h["blinds"], holes, seed=sum(map(ord, h["id"])), pay=False, exclude=h["board"])
    except Exception:
        return None
    bb = h["blinds"][1]
    decisions = []
    f = collections.Counter()
    pre_raises, hero_raised_pre, faced_3bet, last_pre_raiser = 0, False, False, None
    hero_folded = False
    for ev in h["actions"]:
        if ev[0] == "board":
            if not t.needs_board():
                return None
            try:
                t.deal(ev[2])
            except Exception:
                return None
            continue
        _, p, kind, amt = ev
        i = t.actor()
        if i is None or h["players"][i] != p:
            return None  # our replay and the file disagree: skip the hand rather than guess
        s = t.state
        street = engine.STREETS[t.street]
        to_call = s.checking_or_calling_amount
        if p == hero:
            ctx = t.ctx()
            try:
                adv = engine.advise(s, i, ctx, use_solver=use_solver)
            except Exception as e:  # an advisor failure is reported, not fatal
                adv = {"source": "error", "mix": {"fold": 0, "call": 0, "raise": 0}, "detail": {}, "why": str(e)}
            cat = CATEGORY[kind]
            pct = adv["mix"].get(cat, 0.0)
            decisions.append(
                {
                    "street": street,
                    "action": kind + (f" to {amt / bb:.1f}bb" if amt and kind in ("bet", "raise") else ""),
                    "category": cat,
                    "engine_pct": round(pct, 3),
                    # Monte Carlo ignores later betting, so it can only call a decision questionable, not a mistake
                    "grade": "n/a"
                    if adv["source"] == "error"
                    else min(grade(pct), "questionable")
                    if adv["source"] == "Monte Carlo equity" and grade(pct) == "mistake"
                    else grade(pct),
                    "pot_bb": round(s.total_pot_amount / bb, 1),
                    "to_call_bb": round(to_call / bb, 1),
                    "board": " ".join(repr(c) for c in s.get_board_cards(0)) if s.board_count else "",
                    "advice": adv,
                }
            )
            # frequencies
            if street == "preflop":
                if pre_raises == 1 and not hero_raised_pre:
                    f["3bet_chance"] += 1
                    f["3bet"] += kind == "raise"
                if hero_raised_pre and pre_raises >= 2 and not faced_3bet:
                    faced_3bet = True
                    f["faced_3bet"] += 1
                    f["fold_to_3bet"] += kind == "fold"
                if kind in ("call", "raise", "bet"):
                    f["vpip"] = 1
                if kind in ("raise", "bet"):
                    f["pfr"] = 1
                    hero_raised_pre = True
            if street == "flop" and last_pre_raiser == hero and not f["cbet_seen"]:
                f["cbet_seen"] = 1  # the hero's first flop decision as the pre-flop raiser
                if not any(k == "raise" for k, _ in t.street_actions):  # nobody has bet into them
                    f["cbet_chance"] += 1
                    f["cbet"] += kind in ("bet", "raise")
            if street == "river" and to_call and any(k == "raise" for k, _ in t.street_actions[1:]):
                f["faced_river_raise"] += 1
                f["called_river_raise"] += kind == "call"
            hero_folded = hero_folded or kind == "fold"
        if street == "preflop" and kind in ("raise", "bet"):
            pre_raises += 1
            last_pre_raiser = p
        t.act({"bet": "raise", "check": "check"}.get(kind, kind), amt)
    invested = h["stacks"][hi] - t.state.stacks[hi]  # the pot is never awarded in a replay (pay=False)
    net = h["collected"].get(hero, 0) + h["returned"].get(hero, 0) - invested
    # who was still in at the end, from the file's folds (PokerKit's flags reflect the stand-in cards' mucking)
    folded = {e[1] for e in h["actions"] if e[0] == "act" and e[2] == "fold"}
    live_end = sum(1 for p in h["players"] if p not in folded)
    saw_flop = len(h["board"]) >= 3 and not (hero_folded and t.street == 0)
    showdown = not hero_folded and live_end >= 2 and len(h["board"]) == 5
    facts = {
        "net_bb": net / bb,
        "vpip": bool(f["vpip"]),
        "pfr": bool(f["pfr"]),
        "3bet_chance": f["3bet_chance"],
        "3bet": f["3bet"],
        "faced_3bet": f["faced_3bet"],
        "fold_to_3bet": f["fold_to_3bet"],
        "cbet_chance": f["cbet_chance"],
        "cbet": f["cbet"],
        "saw_flop": saw_flop,
        "showdown": showdown,
        "won_showdown": showdown and h["collected"].get(hero, 0) > 0,
        "faced_river_raise": f["faced_river_raise"],
        "called_river_raise": f["called_river_raise"],
        "players": len(h["players"]),
    }
    return decisions, facts


def detect_hero(hands):
    c = collections.Counter(h["hero"] for h in hands if h["hero"])
    return c.most_common(1)[0][0] if c else None


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def run(paths, hero=None, use_solver=True, progress=None):
    files = handhistory.find_files(paths)
    hands = handhistory.parse_files(files, hero)
    hero = hero or detect_hero(hands)
    rows, skipped = [], 0
    for k, h in enumerate(hands):
        r = review_hand(h, hero, use_solver)
        if r is None:
            skipped += 1
        else:
            rows.append((h, *r))
        if progress:
            progress(k + 1, len(hands))
    return summarise(rows, hero, skipped, len(files))


def summarise(rows, hero, skipped, n_files):
    facts = [f for _, _, f in rows]
    n = len(facts)
    tot = lambda k: sum(f[k] for f in facts)  # noqa: E731
    net = sum(f["net_bb"] for f in facts)
    stats = {
        "VPIP": pct(tot("vpip"), n),
        "PFR": pct(tot("pfr"), n),
        "3-bet": pct(tot("3bet"), tot("3bet_chance")),
        "Fold to 3-bet": pct(tot("fold_to_3bet"), tot("faced_3bet")),
        "Flop c-bet": pct(tot("cbet"), tot("cbet_chance")),
        "Went to showdown": pct(tot("showdown"), tot("saw_flop")),
        "Won at showdown": pct(tot("won_showdown"), tot("showdown")),
    }
    # behavioural: looser when stuck (sessions: same table, gaps under 10 minutes)
    stuck, even = [], []
    sessions = collections.defaultdict(list)
    for h, _, f in rows:
        sessions[h["table"]].append((h["time"], f))
    for _, xs in sessions.items():
        xs.sort(key=lambda x: x[0])
        run_net, last = 0.0, None
        for tm, f in xs:
            try:
                t = dt.datetime.strptime(tm.replace("-", "/"), "%Y/%m/%d %H:%M:%S")
            except ValueError:
                t = None
            if last and t and (t - last).total_seconds() > 600:
                run_net = 0.0
            (stuck if run_net < -20 else even).append(f["vpip"])
            run_net += f["net_bb"]
            last = t
    decisions = []
    for h, ds, _ in rows:
        for d in ds:
            decisions.append({**d, "hand": h["id"], "cards": h["hero_cards"], "text": h["text"]})
    grades = collections.Counter(d["grade"] for d in decisions)
    flagged = sorted(
        (d for d in decisions if d["grade"] in ("mistake", "questionable")),
        key=lambda d: (d["grade"] != "mistake", -d["pot_bb"]),
    )
    by_source = collections.Counter(d["advice"]["source"] for d in decisions)
    return {
        "hero": hero,
        "sites": dict(collections.Counter(h["site"] for h, _, _ in rows)),
        "files": n_files,
        "hands_reviewed": n,
        "hands_skipped": skipped,
        "net_bb": round(net, 1),
        "bb_per_100": round(100 * net / n, 1) if n else None,
        "stats": stats,
        "reference": REFERENCE,
        "behaviour": {
            "vpip_when_stuck_20bb": pct(sum(stuck), len(stuck)),
            "vpip_otherwise": pct(sum(even), len(even)),
            "hands_stuck": len(stuck),
            "river_raises_faced": tot("faced_river_raise"),
            "river_raises_called": tot("called_river_raise"),
        },
        "decisions": len(decisions),
        "grades": dict(grades),
        "advice_sources": dict(by_source),
        "flagged": flagged[:60],
    }


def _status(v, ref):
    if v is None:
        return "—", ""
    lo, hi = ref
    return ("low", "warn") if v < lo else ("high", "warn") if v > hi else ("in range", "ok")


def call_txt(d):
    return f", {d['to_call_bb']}bb to call" if d["to_call_bb"] else ""


def to_html(rep):
    e = html.escape
    rows = []
    for k, v in rep["stats"].items():
        st, cls = _status(v, rep["reference"][k])
        lo, hi = rep["reference"][k]
        rows.append(
            f"<tr><td>{e(k)}</td><td class=num>{'—' if v is None else f'{v:.1f}%'}</td>"
            f"<td class=num>{lo}–{hi}%</td><td class={cls}>{st}</td></tr>"
        )
    b = rep["behaviour"]
    tips = []
    if b["vpip_when_stuck_20bb"] is not None and b["vpip_otherwise"] is not None and b["hands_stuck"] >= 30:
        d = b["vpip_when_stuck_20bb"] - b["vpip_otherwise"]
        tips.append(
            f"When stuck 20bb+ in a session you play {b['vpip_when_stuck_20bb']}% of hands, against "
            f"{b['vpip_otherwise']}% otherwise ({d:+.1f} points; real players average +1.2)."
        )
    if b["river_raises_faced"]:
        tips.append(
            f"You called {b['river_raises_called']} of {b['river_raises_faced']} river raises. In 4,001 real river "
            "raises shown down, only 8.5% were bluffs: call them only with very strong hands."
        )
    cards = []
    for d in rep["flagged"]:
        mix = d["advice"]["mix"]
        detail = ", ".join(f"{k} {v:.0%}" for k, v in d["advice"].get("detail", {}).items() if isinstance(v, float))
        cards.append(
            f"<details class='{d['grade']}'><summary><span class=tag>{d['grade']}</span> "
            f"<b>{e(d['cards'] or '')}</b> · {e(d['street'])} {e(d['board'])} · you <b>{e(d['action'])}</b> · pot "
            f"{d['pot_bb']}bb{call_txt(d)} · engine plays it {d['engine_pct']:.0%}</summary>"
            f"<p><b>{e(d['advice']['source'])}:</b> fold {mix['fold']:.0%}, check/call {mix['call']:.0%}, "
            f"bet/raise {mix['raise']:.0%}{(' (' + e(detail) + ')') if detail else ''}. {e(d['advice']['why'])}</p>"
            f"<pre>{e(d['text'])}</pre></details>"
        )
    g = rep["grades"]
    return f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Session Review</title><style>
:root{{--bg:#f6f7f4;--fg:#15201a;--muted:#56615a;--line:#d9ded8;--ok:#1f6b47;--warn:#b5541c;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#121815;--fg:#e8ede9;--muted:#a6b1aa;--line:#2c3631;--ok:#5cc08f;--warn:#f0905a;--card:#1a221d}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;margin:0}}
main{{max-width:980px;margin:0 auto;padding:32px 16px;display:grid;gap:28px}}
h1{{margin:0;font-size:28px}} h2{{margin:0 0 8px;font-size:20px}} .muted{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}}
.card b{{display:block;font-size:22px}} table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
.ok{{color:var(--ok)}} .warn{{color:var(--warn);font-weight:600}}
details{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin-bottom:8px}}
summary{{cursor:pointer}} .tag{{font-size:12px;text-transform:uppercase;letter-spacing:.05em;padding:2px 6px;border-radius:4px;border:1px solid var(--line)}}
details.mistake .tag{{color:var(--warn)}} pre{{white-space:pre-wrap;font-size:13px;color:var(--muted);overflow-x:auto}}
</style></head><body><main>
<header><h1>Session review: {e(rep["hero"] or "unknown")}</h1>
<p class=muted>{rep["hands_reviewed"]} hands from {rep["files"]} files ({rep["hands_skipped"]} skipped). Finished hands only; reviewed after play.</p></header>
<div class=cards>
<div class=card><span class=muted>Result</span><b>{rep["net_bb"]:+.1f} bb</b></div>
<div class=card><span class=muted>Win rate</span><b>{(rep["bb_per_100"] or 0):+.1f} bb/100</b></div>
<div class=card><span class=muted>Your decisions</span><b>{rep["decisions"]}</b></div>
<div class=card><span class=muted>Agree with engine</span><b>{g.get("agrees", 0)}</b></div>
<div class=card><span class=muted>Questionable</span><b>{g.get("questionable", 0)}</b></div>
<div class=card><span class=muted>Mistakes</span><b>{g.get("mistake", 0)}</b></div>
</div>
<section><h2>Your frequencies</h2><table><tr><th>Stat</th><th class=num>You</th><th class=num>Typical winning range</th><th></th></tr>{"".join(rows)}</table>
<p class=muted>Ranges are approximate for 6-max cash games; heads-up and full-ring differ. A few hundred hands are noisy.</p></section>
<section><h2>Patterns</h2>{"".join(f"<p>{e(t)}</p>" for t in tips) or "<p class=muted>Not enough hands yet.</p>"}</section>
<section><h2>Decisions to look at</h2><p class=muted>Mistakes first, then by pot size. The engine's answer comes from {e(", ".join(f"{k} ({v})" for k, v in rep["advice_sources"].items()))}.</p>
{"".join(cards) or "<p class=muted>None flagged.</p>"}</section>
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--hero")
    ap.add_argument("--out", default="session_review.html")
    ap.add_argument("--no-solver", action="store_true")
    a = ap.parse_args()
    rep = run(a.paths, a.hero, not a.no_solver)
    open(a.out, "w", encoding="utf-8").write(to_html(rep))
    json.dump({k: v for k, v in rep.items() if k != "flagged"}, sys.stdout, indent=1)
    print(f"\nreport: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
