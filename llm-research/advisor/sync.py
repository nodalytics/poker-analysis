"""Hand Sync: collect your finished hands from every site into one library, and review each session after it ends.

Usage: python -m advisor.sync [--idle MINUTES] [--source FOLDER ...]
Each run (the app runs it every few minutes when Auto is on):
  1. reads new or changed hand-history files in the source folders (your sites' folders, and GGPoker's PokerCraft
     zip downloads), parses them (handhistory.py) and adds hands not seen before to the library
     (~/PokerReview/library/<site>/<date>.txt), so nothing is counted twice;
  2. when no source file has changed for IDLE minutes (default 20: the session is over) and new hands are waiting,
     reviews them (review.py), saves the report, and adds the session to the trend table (~/PokerReview/trends.html).
It never reviews while hands are still arriving: that would be advice during play. Only your own files are read;
it does not scrape or collect other players' hands from sites.
"""

import argparse
import datetime as dt
import html
import json
import os
import re
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advisor import handhistory, review  # noqa: E402

HOME = os.path.join(os.path.expanduser("~"), "PokerReview")
DEFAULT_SOURCES = [
    r"C:\ACR Poker\handHistory",
    os.path.expandvars(r"%LOCALAPPDATA%\PokerStars\HandHistory"),
    os.path.expandvars(r"%APPDATA%\winamax\documents\accounts"),
    os.path.join(os.path.expanduser("~"), "Downloads"),  # GGPoker PokerCraft zips, Ignition downloads
]


SITE_DIRS = {
    "PokerStars": "PokerStars",
    "GGPoker": "GGPoker",
    "ACR / WPN": "ACR-WPN",
    "Ignition": "Ignition",
    "Bovada": "Bovada",
    "Winamax": "Winamax",
    "PHH": "PHH",
}
MAX_FILE = 50 * 1024 * 1024  # a hand-history file or zip member larger than this is skipped
MAX_ZIP_TOTAL = 200 * 1024 * 1024
MAX_ZIP_MEMBERS = 5000


class Sync:
    def __init__(self, home=HOME, sources=None, idle_minutes=20, hero=None):
        self.home = home
        self.lib = os.path.join(home, "library")
        self.state_path = os.path.join(home, "sync_state.json")
        os.makedirs(self.lib, exist_ok=True)
        st = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {}
        # Windows defaults whose %VARIABLES% do not expand (other systems) are left out
        defaults = [p for p in DEFAULT_SOURCES if "%" not in p and (os.name == "nt" or not p.startswith("C:"))]
        self.sources = sources if sources is not None else st.get("sources", defaults)
        self.idle = idle_minutes
        self.hero = hero or st.get("hero")
        self.files = st.get("files", {})  # path -> [size, mtime]
        self.seen = set(st.get("seen", []))  # site|hand id
        self.pending = st.get("pending", [])  # hand texts waiting for review
        self.history = st.get("history", [])  # one summary per reviewed session
        self.last_change = st.get("last_change", 0.0)

    def save(self):
        json.dump(
            {
                "sources": self.sources,
                "hero": self.hero,
                "files": self.files,
                "seen": sorted(self.seen),
                "pending": self.pending,
                "history": self.history,
                "last_change": self.last_change,
            },
            open(self.state_path, "w"),
        )

    # ---- import ----
    def _texts(self, path):
        """(label, text) pairs in a file: plain hand-history files, or the .txt members of a zip download."""
        if path.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(path) as z:
                    infos = z.infolist()
                    if len(infos) > MAX_ZIP_MEMBERS:
                        return
                    total = 0
                    for info in infos:
                        # zip-bomb guards: a size cap per member and in total, and a sane compression ratio
                        if not info.filename.lower().endswith(handhistory.HAND_FILES) or info.file_size > MAX_FILE:
                            continue
                        if info.compress_size and info.file_size / info.compress_size > 200:
                            continue
                        total += info.file_size
                        if total > MAX_ZIP_TOTAL:
                            return
                        with z.open(info) as f:
                            data = f.read(MAX_FILE + 1)[:MAX_FILE]
                        yield f"{path}!{info.filename}", data.decode("utf-8-sig", errors="replace")
            except (zipfile.BadZipFile, OSError, RuntimeError):
                return
        elif os.path.getsize(path) <= MAX_FILE:
            yield path, open(path, encoding="utf-8-sig", errors="replace").read()

    def candidates(self):
        out = []
        for src in self.sources:
            if not os.path.isdir(src):
                continue
            downloads = os.path.basename(os.path.normpath(src)).lower() == "downloads"
            for root, _, names in os.walk(src):
                for n in names:
                    low = n.lower()
                    if downloads and not (low.endswith(".zip") or "hand" in low or "hh" in low):
                        continue  # in Downloads, only files that look like hand histories
                    if low.endswith(handhistory.HAND_FILES + (".zip",)):
                        out.append(os.path.join(root, n))
                if downloads:
                    break  # Downloads: top level only
        return out

    def import_new(self):
        added = 0
        for path in self.candidates():
            try:
                st = os.stat(path)
            except OSError:
                continue
            sig = [st.st_size, st.st_mtime]
            if self.files.get(path) == sig:
                continue
            self.files[path] = sig
            self.last_change = max(self.last_change, st.st_mtime)
            for _, text in self._texts(path):
                for h in handhistory.parse_text(text, self.hero):
                    key = f"{h['site']}|{h['id']}"
                    if key in self.seen:
                        continue
                    self.seen.add(key)
                    # path parts come from file contents: only a strict date and a fixed site folder name
                    day = (h["time"] or "")[:10].replace("/", "-")
                    day = day if re.fullmatch(r"\d{4}-\d\d-\d\d", day) else "unknown"
                    d = os.path.join(self.lib, SITE_DIRS.get(h["site"], "other"))
                    target = os.path.realpath(os.path.join(d, f"{day}.txt"))
                    if not target.startswith(os.path.realpath(self.lib) + os.sep):
                        continue
                    os.makedirs(d, exist_ok=True)
                    with open(target, "a", encoding="utf-8") as f:
                        f.write(h["text"] + "\n\n\n")
                    self.pending.append(h["text"])
                    added += 1
        return added

    # ---- review ----
    def session_over(self, now=None):
        return (now or time.time()) - self.last_change >= 60 * self.idle

    def review_pending(self, use_solver=False):
        hands = []
        for text in self.pending:
            hands += handhistory.parse_text(text, self.hero)
        if not hands:
            self.pending = []
            return None
        rep = review.run_hands(hands, self.hero, use_solver)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.home, f"review-{stamp}.html")
        open(path, "w", encoding="utf-8").write(review.to_html(rep))
        g = rep["grades"]
        dec = max(1, rep["decisions"])
        self.history.append(
            {
                "reviewed": stamp,
                "first_hand": min(h["time"] for h in hands),
                "sites": rep.get("sites", {}),
                "hands": rep["hands_reviewed"],
                "bb_per_100": rep["bb_per_100"],
                "net_bb": rep["net_bb"],
                "stats": rep["stats"],
                "mistakes_per_100_decisions": round(100 * g.get("mistake", 0) / dec, 1),
                "report": path,
            }
        )
        self.pending = []
        self.write_trends()
        return path

    def write_trends(self):
        e = html.escape
        rows = "".join(
            f"<tr><td>{e(s['first_hand'])}</td><td>{e(', '.join(s['sites']))}</td><td class=n>{s['hands']}</td>"
            f"<td class=n>{(s['bb_per_100'] or 0):+.1f}</td>"
            + "".join(
                f"<td class=n>{'—' if s['stats'].get(k) is None else s['stats'][k]}</td>"
                for k in ("VPIP", "PFR", "3-bet", "Fold to 3-bet", "Flop c-bet")
            )
            + f"<td class=n>{s['mistakes_per_100_decisions']}</td>"
            f"<td><a href='{e(os.path.basename(s['report']))}'>report</a></td></tr>"
            for s in reversed(self.history)
        )
        hands = sum(s["hands"] for s in self.history)
        net = sum(s["net_bb"] for s in self.history)
        page = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Poker Trends</title><style>
:root{{--bg:#f6f7f4;--fg:#15201a;--muted:#56615a;--line:#d9ded8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#121815;--fg:#e8ede9;--muted:#a6b1aa;--line:#2c3631}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0}}
main{{max-width:1100px;margin:0 auto;padding:28px 16px}} table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}} .n{{text-align:right;font-variant-numeric:tabular-nums}}
a{{color:inherit}} .wrap{{overflow-x:auto}}</style></head><body><main>
<h1>Your sessions</h1><p style="color:var(--muted)">{len(self.history)} sessions, {hands} hands, {net:+.1f} bb in total
({(100 * net / hands if hands else 0):+.1f} bb/100). Each row is one reviewed session; open its report for the details.</p>
<div class=wrap><table><tr><th>Started</th><th>Sites</th><th class=n>Hands</th><th class=n>bb/100</th><th class=n>VPIP</th>
<th class=n>PFR</th><th class=n>3-bet</th><th class=n>Fold to 3-bet</th><th class=n>C-bet</th><th class=n>Mistakes /100 decisions</th><th></th></tr>
{rows}</table></div></main></body></html>"""
        open(os.path.join(self.home, "trends.html"), "w", encoding="utf-8").write(page)

    def step(self, now=None, use_solver=False):
        """One sync: import, and review if the session is over. Returns a status dict."""
        added = self.import_new()
        report = None
        if self.pending and self.session_over(now):
            report = self.review_pending(use_solver)
        self.save()
        return {"added": added, "waiting": len(self.pending), "report": report}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--idle", type=float, default=20)
    ap.add_argument("--source", action="append")
    ap.add_argument("--hero")
    a = ap.parse_args()
    s = Sync(sources=a.source, idle_minutes=a.idle, hero=a.hero)
    print(json.dumps(s.step()))


if __name__ == "__main__":
    main()
