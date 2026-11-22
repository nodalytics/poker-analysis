"""Replay real logged hands as a video: our bot against Slumbot, exactly as Slumbot's server recorded them.

Usage: uv run python scripts/replay_video.py HANDS_DIR OUT.mp4 [N_HANDS] [TITLE]
HANDS_DIR is a slumbot_client output (hands_*.jsonl: our cards, Slumbot's cards revealed at the end, the board, the
action string and the result). Picks the biggest pot won, the biggest pot lost, and a few typical hands, and animates
each: seats, hole cards (Slumbot's shown at showdown or the end), board street by street, every action with the
pot. Nothing is simulated: every card and chip comes from the log.
"""

import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

FELT, RIM, INK, MUTED = "#1f6b47", "#14492f", "#f4f4f0", "#b9c9bf"
WIN, LOSS = "#1baf7a", "#eb6834"
BIG, STACK = 100, 20000
SUIT = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
RED = {"h", "d"}
FPS = 10


def steps(r):
    """[(street, actor 'us'/'slumbot', text, pot after)] from the action string."""
    pos = r["pos"]  # 0: we are the big blind
    out = []
    total = [0, 0]
    for si, st in enumerate(r["action"].split("/")):
        street = [BIG, BIG // 2] if si == 0 else [0, 0]
        actor, last = (1, BIG) if si == 0 else (0, 0)
        for m in re.finditer(r"b(\d+)|[kcf]", st):
            t = m.group()
            who = "us" if actor == pos else "slumbot"
            if t[0] == "b":
                amt = int(m.group(1))
                txt = ("bets " if last == 0 else "raises to ") + f"{amt / BIG:g}bb"
                if amt + total[actor] >= STACK:
                    txt = "all-in"
                last = amt
                street[actor] = amt
            elif t == "c":
                street[actor] = last
                txt = "calls"
            elif t == "k":
                txt = "checks"
            else:
                txt = "folds"
            out.append((si, who, txt, (total[0] + total[1] + street[0] + street[1]) / BIG))
            actor = 1 - actor
        total = [total[0] + street[0], total[1] + street[1]]
    return out


def card_txt(ax, x, y, c, size=20, hidden=False):
    if hidden:
        ax.text(x, y, "▮", fontsize=size, ha="center", va="center", color="#444")
        return
    ax.text(
        x,
        y,
        c[0].replace("T", "10") + SUIT[c[1]],
        fontsize=size,
        ha="center",
        va="center",
        fontweight="bold",
        color="#c0392b" if c[1] in RED else "#111",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#bbb"},
    )


def draw(ax, r, k, label, title):
    """Frame for step k (len(steps) = the end, cards revealed)."""
    ax.clear()
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.add_patch(matplotlib.patches.Ellipse((8, 4.6), 13.5, 6.2, fc=RIM))
    ax.add_patch(matplotlib.patches.Ellipse((8, 4.6), 12.8, 5.6, fc=FELT))
    st = steps(r)
    done = k >= len(st)
    cur = st[min(k, len(st) - 1)] if st else (0, "us", "", 1.5)
    street = cur[0] if not done else max((s[0] for s in st), default=0)
    nb = [0, 3, 4, 5][min(street, 3)]
    if done:
        nb = len(r["board"] or [])
    board = (r["board"] or [])[:nb]
    for j, c in enumerate(board):
        card_txt(ax, 5.6 + 1.2 * j, 4.9, c)
    pot = cur[3] if st else 1.5
    ax.text(8, 3.6, f"pot {pot:g}bb", color=MUTED, ha="center", fontsize=12)
    ax.text(
        8,
        1.0,
        "Blueprint bot (ours)" + (" · big blind" if r["pos"] == 0 else " · button"),
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        8,
        8.4,
        "Slumbot" + (" · button" if r["pos"] == 0 else " · big blind"),
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    for j, c in enumerate(r["hole"]):
        card_txt(ax, 7.4 + 1.2 * j, 1.9, c)
    show = done and r.get("slumbot_hole")
    for j in range(2):
        card_txt(ax, 7.4 + 1.2 * j, 7.4, r["slumbot_hole"][j] if show else "", hidden=not show)
    if st and not done:
        y = 2.9 if cur[1] == "us" else 6.4
        ax.text(
            8,
            y,
            ("Blueprint " if cur[1] == "us" else "Slumbot ") + cur[2],
            ha="center",
            fontsize=13,
            color=INK,
            fontweight="bold",
        )
    if done:
        w = r["winnings_chips"] / BIG
        ax.text(
            8,
            6.2,
            f"Blueprint {'wins' if w > 0 else 'loses'} {abs(w):g}bb",
            ha="center",
            fontsize=16,
            fontweight="bold",
            color=WIN if w > 0 else LOSS,
        )
    ax.text(0.2, 8.7, title, fontsize=13, fontweight="bold", va="top")
    ax.text(0.2, 8.1, label, fontsize=10.5, color="#444", va="top")
    ax.text(
        0.2,
        0.2,
        "Real hand from Slumbot's public API log (cards and bets as its server recorded them).",
        fontsize=8,
        color="#666",
    )


def pick(rows, n):
    rows = [r for r in rows if r.get("slumbot_hole") and r["action"].count("/") >= 1]
    by = sorted(rows, key=lambda r: r["winnings_chips"])
    chosen = [("Biggest pot won", by[-1]), ("Biggest pot lost", by[0])]
    mid = [r for r in by if 300 <= abs(r["winnings_chips"]) <= 3000]
    step = max(1, len(mid) // max(1, n - 2))
    for r in mid[::step][: n - 2]:
        chosen.append(("A typical hand", r))
    return chosen


def main(hands_dir, out, n=4, title="Blueprint bot vs Slumbot"):
    rows = []
    for f in sorted(glob.glob(os.path.join(hands_dir, "hands_*.jsonl"))):
        rows += [json.loads(x) for x in open(f)]
    chosen = pick(rows, int(n))
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    w = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "26"])
    with w.saving(fig, out, dpi=100):
        for label, r in chosen:
            st = steps(r)
            for k in range(len(st) + 1):
                draw(ax, r, k, label, title)
                for _ in range(int(FPS * (1.1 if k < len(st) else 2.6))):
                    w.grab_frame()
    plt.close(fig)
    print(out, os.path.getsize(out) // 1024, "KB", [lab for lab, _ in chosen])


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(a[2:4]))
