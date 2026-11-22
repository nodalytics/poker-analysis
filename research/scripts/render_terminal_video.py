"""Turn a terminal recording (`script -T timing` output) into a small MP4 that replays it with its real timing.

Usage: uv run python scripts/render_terminal_video.py TYPESCRIPT TIMING OUT.mp4 [TITLE]
Long pauses are shortened to 1.5 s (said in the caption); colours from the recording are kept.
"""

import os
import re
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

W, H, ROWS, PAD = 1280, 720, 26, 24
BG, FG = (13, 17, 23), (230, 237, 243)
COLORS = {"32": (63, 185, 80), "33": (242, 204, 96), "31": (248, 81, 73)}


def font(size, bold=False):
    for d in ("/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/dejavu/", ""):
        try:
            return ImageFont.truetype(d + ("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"), size)
        except OSError:
            continue
    return ImageFont.load_default()


F, FB = font(19), font(19, True)


def lines_of(text):
    """Split into lines of (segments of (text, colour, bold)), honouring basic ANSI colour codes."""
    out = []
    for raw in text.replace("\r\n", "\n").split("\n"):
        raw = raw.split("\r")[-1]
        segs, col, bold = [], FG, False
        for part in re.split(r"(\x1b\[[0-9;]*m)", raw):
            m = re.match(r"\x1b\[([0-9;]*)m", part)
            if m:
                codes = m.group(1).split(";")
                bold = "1" in codes
                col = next((COLORS[c] for c in codes if c in COLORS), FG if codes in (["0"], [""]) else col)
            elif part:
                segs.append((re.sub(r"\x1b\[[^a-zA-Z]*[a-zA-Z]", "", part), col, bold))
        out.append(segs)
    return out


def frame(text, title):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 34], fill=(22, 27, 34))
    d.text((PAD, 8), title, font=FB, fill=(139, 148, 158))
    rows = lines_of(text)[-ROWS:]
    for r, segs in enumerate(rows):
        x = PAD
        for s, col, bold in segs:
            d.text((x, 46 + r * 25), s, font=FB if bold else F, fill=col)
            x += d.textlength(s, font=FB if bold else F)
    return im


def main(ts, timing, out, title="Live recording"):
    data = open(ts, "rb").read()
    nl = data.find(b"\n")
    data = data[nl + 1 :] if data.startswith(b"Script started") else data
    pos, buf, frames = 0, b"", []
    for line in open(timing):
        parts = line.split()
        if len(parts) < 2:
            continue
        delay, n = float(parts[0]), int(parts[1])
        if frames:
            frames[-1][1] = min(delay, 1.5) + 0.04
        buf += data[pos : pos + n]
        pos += n
        frames.append([buf.decode("utf-8", errors="replace"), 0.04])
    frames[-1][1] = 3.0
    with tempfile.TemporaryDirectory() as tmp:
        lst = []
        for k, (text, dur) in enumerate(frames):
            frame(text, title).save(os.path.join(tmp, f"{k:05d}.png"))
            lst += [f"file '{k:05d}.png'", f"duration {dur:.3f}"]
        lst.append(f"file '{len(frames) - 1:05d}.png'")
        open(os.path.join(tmp, "list.txt"), "w").write("\n".join(lst) + "\n")
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
                "fps=12,format=yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                "28",
                "-movflags",
                "+faststart",
                out,
            ],
            check=True,
        )
    print(f"{out}: {len(frames)} frames, {os.path.getsize(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:4], sys.argv[4] if len(sys.argv) > 4 else "Live recording")
