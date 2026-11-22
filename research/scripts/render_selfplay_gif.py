"""Render self-play poker from hands.jsonl (spectator view: every hand's cards are shown).

Usage: uv run python scripts/render_selfplay_gif.py hands.jsonl OUT.gif|OUT.mp4 [N_HANDS]
N_HANDS: the N biggest pots (default 3), or 0 for every hand in order. Always ends on the final standings.
A .gif keeps to the AGENTS.md limits (720px, 64 colours); a full session is too long for that, so write .mp4
(H.264 through ffmpeg) for it. Prints the final size.
"""

import json
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

W, H = 720, 405
FELT, RIM, BG, INK, DIM, HI = (21, 94, 60), (12, 56, 36), (14, 18, 24), (236, 240, 244), (150, 160, 170), (250, 204, 21)
SUITS = {"s": ("♠", (20, 20, 20)), "c": ("♣", (20, 20, 20)), "h": ("♥", (200, 30, 40)), "d": ("♦", (200, 30, 40))}
SEATS4 = [(150, 95), (570, 95), (570, 300), (150, 300)]


ABBR = [
    ("Big-Tiger-Gemma", "BigTiger"),
    ("Tiger-Gemma-9B", "Tiger9B"),
    ("qwen2.5vl", "Qwen"),
    ("wizard-vicuna", "Wizard"),
]
TAGS = {"learn": "L", "frozen": "F", "toolkit": "T", "solverplays": "S", "solverfirst": "A", "plain": "P"}


def label(name):
    """Short seat label that keeps who is who: 'Qwen·L' (learner), '·F' (frozen twin), '·T' (toolkit)."""
    base, _, tag = name.partition("#")
    short = next((v for k, v in ABBR if k in base), base)
    return f"{short}·{TAGS.get(tag, tag)}" if tag else short


def seat_xy(i, n):
    """Seat centres: the original four corners for up to 4 players, else evenly round the table's oval."""
    if n <= 4:
        return SEATS4[i]
    import math

    a = -math.pi / 2 + 2 * math.pi * i / n
    return (int(W / 2 + 268 * math.cos(a)), int(H / 2 - 8 + 150 * math.sin(a)))


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/dejavu/", ""):
        try:
            return ImageFont.truetype(d + name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F, FB, FS, FC = font(14), font(15, True), font(12), font(17, True)


def card(d, x, y, c, w=30, h=42):
    d.rounded_rectangle([x, y, x + w, y + h], 4, fill=(250, 250, 250), outline=(90, 90, 90))
    if not c or len(c) < 2:
        return
    sym, col = SUITS.get(c[1], ("?", (0, 0, 0)))
    d.text((x + w / 2, y + 14), c[0], font=FC, fill=col, anchor="mm")
    d.text((x + w / 2, y + 31), sym, font=FC, fill=col, anchor="mm")


def frame(hand, seats, stacks, bets, holes, folded, board, pot, actor, caption):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.ellipse([60, 40, W - 60, H - 40], fill=FELT, outline=RIM, width=8)
    if len(seats) <= 4:
        d.text((W / 2, 18), f"Hand {hand}  ·  play chips  ·  blinds 1/2", font=FS, fill=DIM, anchor="mm")
    else:
        caption = f"Hand {hand}  ·  {caption}"
    cs = board.split() if board and board != "-" else []
    for k in range(5):
        x = W / 2 - 92 + k * 37
        if k < len(cs):
            card(d, x, H / 2 - 38, cs[k])
        else:
            d.rounded_rectangle([x, H / 2 - 38, x + 30, H / 2 + 4], 4, outline=(40, 120, 80))
    d.text((W / 2, H / 2 + 22), f"Pot {pot}", font=FB, fill=INK, anchor="mm")
    for i, name in enumerate(seats):
        cx, cy = seat_xy(i, len(seats))
        half = 92 if len(seats) <= 4 else 70
        box = [cx - half, cy - 34, cx + half, cy + 34]
        d.rounded_rectangle(
            box, 8, fill=(18, 24, 32), outline=HI if i == actor else (60, 70, 80), width=3 if i == actor else 1
        )
        d.text(
            (cx - half + 8, cy - 26),
            label(name)[:11],
            font=FB if half > 80 else F,
            fill=DIM if folded[i] else INK,
        )
        d.text((cx - half + 8, cy - 6), f"{stacks[i]} chips", font=FS, fill=DIM)
        if bets[i]:
            d.text((cx - half + 8, cy + 12), f"bet {bets[i]}", font=FS, fill=HI)
        hc = (holes.get(i) or "").split()
        for k, c in enumerate(hc[:2]):
            if folded[i]:
                x0 = cx + half - 70 + k * 34
                d.rounded_rectangle([x0, cy - 21, x0 + 30, cy + 21], 4, fill=(60, 66, 72))
            else:
                card(d, cx + half - 70 + k * 34, cy - 21, c)
    d.rectangle([0, H - 30, W, H], fill=(8, 10, 14))
    d.text((W / 2, H - 15), caption, font=FB, fill=INK, anchor="mm")
    return im


def standings(chips, start, title):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((W / 2, 50), title, font=FC, fill=INK, anchor="mm")
    for k, (name, c) in enumerate(sorted(chips.items(), key=lambda x: -x[1])):
        y = 110 + k * 46
        d.rounded_rectangle(
            [140, y - 18, W - 140, y + 18], 8, fill=(18, 24, 32), outline=HI if k == 0 else (60, 70, 80)
        )
        d.text((160, y), f"{k + 1}. {name}", font=FB, fill=HI if k == 0 else INK, anchor="lm")
        d.text((W - 160, y), f"{c} chips ({c - start:+})", font=FB, fill=INK, anchor="rm")
    d.text((W / 2, H - 20), f"winner: {max(chips, key=chips.get)}", font=FB, fill=HI, anchor="mm")
    return im


def main(path, out, n):
    rows = [json.loads(line) for line in open(path)]
    ends = [r for r in rows if r.get("end")]
    chips = {}
    for e in sorted(ends, key=lambda e: e["hand"]):
        chips.update(dict(zip(e["seats"], e["stacks"])))
    big = sorted(ends, key=lambda e: -sum(p for p in e["payoffs"] if p > 0))[:n] if n else ends
    step, rest = (600, 1500) if not n else (1100, 2600)
    frames, durations = [], []
    for e in sorted(big, key=lambda e: e["hand"]):
        h, seats = e["hand"], e["seats"]
        acts = [r for r in rows if r.get("hand") == h and not r.get("end")]
        holes = {r["seat"]: r["hole"] for r in acts}
        holes.update({i: x for i, x in enumerate(e["holes"]) if x and x != "-"})
        folded = [False] * len(seats)
        for r in acts:
            if r["action"] == "folds":
                folded[r["seat"]] = True
            cap = f"{r['street']}: {seats[r['seat']]} {r['action']}" + ("  (invalid reply)" if r["invalid"] else "")
            frames.append(
                frame(h, seats, r["stacks"], r["bets"], holes, list(folded), r["board"], r["pot"], r["seat"], cap)
            )
            durations.append(step)
        won = ", ".join(f"{s} {'+' if p > 0 else ''}{p}" for s, p in zip(seats, e["payoffs"]) if p)
        frames.append(
            frame(
                h,
                seats,
                e["stacks"],
                [0] * len(seats),
                holes,
                [not x for x in e["in_at_end"]],
                e["board"],
                sum(p for p in e["payoffs"] if p > 0),
                None,
                f"Result: {won}",
            )
        )
        durations.append(rest)
    frames.append(standings(chips, 200, f"Final standings after {len(ends)} hands"))
    durations.append(5000)
    if out.endswith(".mp4"):
        write_mp4(frames, durations, out)
    else:
        pal = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
        pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print(f"{out}: {len(frames)} frames, {os.path.getsize(out) / 1e6:.2f} MB, hands {len(big)}")


def write_mp4(frames, durations, out):
    with tempfile.TemporaryDirectory() as tmp:
        lines = []
        for k, (f, ms) in enumerate(zip(frames, durations)):
            f.save(os.path.join(tmp, f"{k:05d}.png"))
            lines += [f"file '{k:05d}.png'", f"duration {ms / 1000:.3f}"]
        lines.append(f"file '{len(frames) - 1:05d}.png'")
        with open(os.path.join(tmp, "list.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                os.path.join(tmp, "list.txt"),
                "-vf",
                "fps=10,pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "30",
                "-preset",
                "slow",
                "-movflags",
                "+faststart",
                out,
            ],
            check=True,
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
