"""Render one game from chess games.jsonl as a small GIF.

Usage: uv run python scripts/render_chess_gif.py games.jsonl OUT.gif [GAME] [MAX_FRAMES]
GAME defaults to the decisive game with the fewest plies (the clearest one to watch). At most MAX_FRAMES
moves are drawn (default 90, evenly sampled if the game is longer). Kept to the AGENTS.md media limits.
"""

import json
import os
import sys

import chess
from PIL import Image, ImageDraw, ImageFont

SQ, PAD, CAP = 44, 22, 34
W = SQ * 8 + PAD * 2
BOARD_BOTTOM = CAP + PAD + SQ * 8
H = BOARD_BOTTOM + 20 + 22 + 28
LIGHT, DARK, HI, BG, INK = (238, 238, 210), (118, 150, 86), (246, 246, 105), (14, 18, 24), (236, 240, 244)
GLYPH = {
    "K": "♔",
    "Q": "♕",
    "R": "♖",
    "B": "♗",
    "N": "♘",
    "P": "♙",
    "k": "♚",
    "q": "♛",
    "r": "♜",
    "b": "♝",
    "n": "♞",
    "p": "♟",
}


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in ("/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/dejavu/", ""):
        try:
            return ImageFont.truetype(d + name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FP, FS, FB = font(34), font(11), font(14, True)


def draw(board, last, top, bottom, caption):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((PAD, 8), top, font=FB, fill=INK)
    y0 = CAP
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        x, y = PAD + f * SQ, y0 + PAD + (7 - r) * SQ
        col = LIGHT if (f + r) % 2 else DARK
        if last and sq in (last.from_square, last.to_square):
            col = HI
        d.rectangle([x, y, x + SQ, y + SQ], fill=col)
        p = board.piece_at(sq)
        if p:
            d.text((x + SQ / 2, y + SQ / 2 + 1), GLYPH[p.symbol()], font=FP, fill=(15, 15, 15), anchor="mm")
    for i in range(8):
        d.text(
            (PAD + i * SQ + SQ / 2, y0 + PAD + 8 * SQ + 9), "abcdefgh"[i], font=FS, fill=(150, 160, 170), anchor="mm"
        )
        d.text((PAD / 2, y0 + PAD + i * SQ + SQ / 2), str(8 - i), font=FS, fill=(150, 160, 170), anchor="mm")
    d.text((PAD, BOARD_BOTTOM + 20), bottom, font=FB, fill=INK)
    d.rectangle([0, H - 28, W, H], fill=(8, 10, 14))
    d.text((W / 2, H - 14), caption, font=FB, fill=INK, anchor="mm")
    return im


def main(path, out, game=None, max_frames=90):
    rows = [json.loads(x) for x in open(path)]
    ends = {r["game"]: r for r in rows if r.get("end")}
    if game is None:
        decisive = [e for e in ends.values() if e["result"] in ("1-0", "0-1")] or list(ends.values())
        game = min(decisive, key=lambda e: len(e["moves"]))["game"]
    e = ends[game]
    short = lambda m: m.split("/")[-1].split(":")[0].replace("-GGUF", "")  # noqa: E731
    plies = [r for r in rows if r.get("game") == game and not r.get("end")]
    keep = (
        set(range(len(plies)))
        if len(plies) <= max_frames
        else {round(i * (len(plies) - 1) / (max_frames - 1)) for i in range(max_frames)}
    )
    board = chess.Board()
    frames, durs = [draw(board, None, f"Black: {short(e['black'])}", f"White: {short(e['white'])}", "start")], [900]
    for i, r in enumerate(plies):
        mv = board.parse_san(r["san"])
        board.push(mv)
        if i in keep:
            n = (r["ply"] + 1) // 2
            cap = f"{n}{'.' if r['ply'] % 2 else '...'} {r['san']}  ({short(r['model'])})" + (
                "  random" if r["forced_random"] else ""
            )
            frames.append(draw(board, mv, f"Black: {short(e['black'])}", f"White: {short(e['white'])}", cap))
            durs.append(450)
    frames.append(
        draw(
            board,
            None,
            f"Black: {short(e['black'])}",
            f"White: {short(e['white'])}",
            f"Result {e['result']}: {e['how'][:40]}",
        )
    )
    durs.append(3000)
    pal = [f.quantize(colors=32, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True)
    print(f"{out}: game {game}, {len(frames)} frames, {os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]) if len(sys.argv) > 3 else None,
        int(sys.argv[4]) if len(sys.argv) > 4 else 90,
    )
