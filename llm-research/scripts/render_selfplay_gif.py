"""Render self-play hands from hands.jsonl as a small GIF (spectator view: every hand's cards are shown).

Usage: python3 render_selfplay_gif.py hands.jsonl OUT.gif [N_HANDS]
Picks the N hands with the biggest pots (default 3). Keeps to the media limits in AGENTS.md:
720px wide, 64 colours, about 1 frame per action; prints the final size.
"""

import json
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 720, 405
FELT, RIM, BG, INK, DIM, HI = (21, 94, 60), (12, 56, 36), (14, 18, 24), (236, 240, 244), (150, 160, 170), (250, 204, 21)
SUITS = {"s": ("♠", (20, 20, 20)), "c": ("♣", (20, 20, 20)), "h": ("♥", (200, 30, 40)), "d": ("♦", (200, 30, 40))}
SEATS = [(150, 95), (570, 95), (570, 300), (150, 300)]


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
    d.text((W / 2, 18), f"Hand {hand}  ·  play chips  ·  blinds 1/2", font=FS, fill=DIM, anchor="mm")
    cs = board.split() if board and board != "-" else []
    for k in range(5):
        x = W / 2 - 92 + k * 37
        if k < len(cs):
            card(d, x, H / 2 - 38, cs[k])
        else:
            d.rounded_rectangle([x, H / 2 - 38, x + 30, H / 2 + 4], 4, outline=(40, 120, 80))
    d.text((W / 2, H / 2 + 22), f"Pot {pot}", font=FB, fill=INK, anchor="mm")
    for i, name in enumerate(seats):
        cx, cy = SEATS[i]
        box = [cx - 92, cy - 34, cx + 92, cy + 34]
        d.rounded_rectangle(
            box, 8, fill=(18, 24, 32), outline=HI if i == actor else (60, 70, 80), width=3 if i == actor else 1
        )
        d.text((cx - 84, cy - 26), name[:11], font=FB, fill=DIM if folded[i] else INK)
        d.text((cx - 84, cy - 6), f"{stacks[i]} chips", font=F, fill=DIM)
        if bets[i]:
            d.text((cx - 84, cy + 12), f"bet {bets[i]}", font=F, fill=HI)
        hc = (holes.get(i) or "").split()
        for k, c in enumerate(hc[:2]):
            if folded[i]:
                d.rounded_rectangle([cx + 14 + k * 34, cy - 21, cx + 44 + k * 34, cy + 21], 4, fill=(60, 66, 72))
            else:
                card(d, cx + 14 + k * 34, cy - 21, c)
    d.rectangle([0, H - 30, W, H], fill=(8, 10, 14))
    d.text((W / 2, H - 15), caption, font=FB, fill=INK, anchor="mm")
    return im


def main(path, out, n):
    rows = [json.loads(line) for line in open(path)]
    ends = [r for r in rows if r.get("end")]
    big = sorted(ends, key=lambda e: -sum(p for p in e["payoffs"] if p > 0))[:n]
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
            durations.append(1100)
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
        durations.append(2600)
    pal = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    import os

    print(f"{out}: {len(frames)} frames, {os.path.getsize(out) / 1e6:.2f} MB, hands {[e['hand'] for e in big]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
