"""Local models play chess against each other: a round robin, each pair twice with colours swapped.

Usage: uv run python scripts/chess_selfplay.py OUTDIR MAX_PLIES MODEL [MODEL ...]
Writes OUTDIR/games.log (readable, `tail -f` it), OUTDIR/games.jsonl (every move and reply) and OUTDIR/*.pgn.

python-chess enforces the rules. Each turn the model gets the FEN, the moves so far and its legal moves in SAN,
and must reply with one. Two tries; then a random legal move is played and counted as a failure. A game that
reaches MAX_PLIES is adjudicated on material (P1 N3 B3 R5 Q9), a lead of 3 or more wins, otherwise a draw.
"""

import itertools
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

import chess
import chess.pgn

OLLAMA = "http://127.0.0.1:11434"
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def short(model):
    """Display name; keeps an entrant tag such as '#learn'."""
    base, _, tag = model.partition("#")
    name = base.split("/")[-1].split(":")[0].replace("-GGUF", "")
    return f"{name}#{tag}" if tag else name


def notes_block(notes, who):
    """The entrant's own skill and memory notes, if any, as a prompt prefix. Entrant ids are 'model#tag'."""
    text = (notes or {}).get(who, "").strip()
    return f"Your own notes from earlier games (you wrote these; use them if they help):\n{text}\n\n" if text else ""


def ask(model, prompt):
    body = {
        "model": model.partition("#")[0],
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.3, "num_predict": 60},
    }
    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            text = json.load(r)["message"]["content"]
    except urllib.error.HTTPError:  # e.g. a 2k-context model overflowing: an invalid move, not a crash
        text = ""
    return text, round(time.time() - t0, 2)


def pick(board, text):
    """The first legal SAN or UCI move named in the reply, or None."""
    m = re.search(r'"move"\s*:\s*"([^"]+)"', text)
    candidates = [m.group(1)] if m else []
    candidates += re.findall(
        r"[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O-O|O-O|[a-h][1-8][a-h][1-8][qrbn]?", text
    )
    for c in candidates:
        c = c.strip().rstrip(".")
        for parse in (board.parse_san, board.parse_uci):
            try:
                return parse(c)
            except ValueError:
                continue
    return None


def material(board):
    s = 0
    for piece, v in VALUES.items():
        s += v * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
    return s


def play(white, black, max_plies, log, jl, game_no, rng, notes=None):
    board = chess.Board()
    sans, fails = [], {white: 0, black: 0}
    # Only automatic endings (checkmate, stalemate, fivefold repetition, 75 moves): a threefold draw must be
    # claimed in real chess, and claiming it automatically ended 11 of the first 12 games in knight shuffles.
    while not board.is_game_over() and len(sans) < max_plies:
        mover = white if board.turn == chess.WHITE else black
        legal = []
        for m in board.legal_moves:
            san = board.san(m)
            board.push(m)
            legal.append(san + (" (repeats a position)" if board.is_repetition(2) else ""))
            board.pop()
        prompt = (
            notes_block(notes, mover) + f"You are playing chess as {'White' if board.turn else 'Black'}.\n"
            f"Position (FEN): {board.fen()}\n"
            f"Moves so far: {' '.join(sans) or 'none'}\n"
            f"Your legal moves: {', '.join(legal)}\n"
            'Reply with ONLY JSON: {"move": "<one of your legal moves>"}'
        )
        move, replies, secs = None, [], 0.0
        for _ in range(2):
            text, t = ask(mover, prompt)
            replies.append(text)
            secs += t
            move = pick(board, text)
            if move is not None:
                break
            prompt += "\nThat was not one of your legal moves. Choose exactly one from the list."
        forced = move is None
        if forced:
            fails[mover] += 1
            move = rng.choice(list(board.legal_moves))
        san = board.san(move)
        board.push(move)
        sans.append(san)
        jl.write(
            json.dumps(
                {
                    "game": game_no,
                    "ply": len(sans),
                    "model": mover,
                    "san": san,
                    "fen": board.fen(),
                    "forced_random": forced,
                    "replies": replies,
                    "seconds": round(secs, 2),
                }
            )
            + "\n"
        )
        log.write(f"  {len(sans):3}. {short(mover):22} {san:8}{'  [random: no legal move named]' if forced else ''}\n")
    outcome = board.outcome()
    if outcome is not None:
        result, how = outcome.result(), outcome.termination.name.lower()
    else:
        m = material(board)
        result = "1-0" if m >= 3 else "0-1" if m <= -3 else "1/2-1/2"
        how = f"adjudicated at {max_plies} plies, material {m:+d}"
    return board, sans, result, how, fails


def main(outdir, max_plies, models):
    os.makedirs(outdir, exist_ok=True)
    log = open(os.path.join(outdir, "games.log"), "a", buffering=1)
    jl = open(os.path.join(outdir, "games.jsonl"), "a", buffering=1)
    rng = random.Random(11)
    points = {m: 0.0 for m in models}
    fails = {m: 0 for m in models}
    moves = {m: 0 for m in models}
    pairs = [p for a, b in itertools.combinations(models, 2) for p in ((a, b), (b, a))]
    log.write(f"=== Chess round robin, {len(pairs)} games, max {max_plies} plies ===\n")
    for g, (white, black) in enumerate(pairs, 1):
        log.write(f"\n--- Game {g}: {short(white)} (White) vs {short(black)} (Black) ---\n")
        board, sans, result, how, f = play(white, black, max_plies, log, jl, g, rng)
        for m in (white, black):
            fails[m] += f[m]
        moves[white] += (len(sans) + 1) // 2
        moves[black] += len(sans) // 2
        points[white] += {"1-0": 1, "0-1": 0}.get(result, 0.5)
        points[black] += {"1-0": 0, "0-1": 1}.get(result, 0.5)
        log.write(f"  result {result} ({how})\n")
        jl.write(
            json.dumps(
                {"game": g, "end": True, "white": white, "black": black, "result": result, "how": how, "moves": sans}
            )
            + "\n"
        )
        pgn = chess.pgn.Game.from_board(board)
        pgn.headers.update(
            {
                "Event": "research chess round robin",
                "White": short(white),
                "Black": short(black),
                "Result": result,
                "Round": str(g),
            }
        )
        with open(os.path.join(outdir, f"game{g:02d}.pgn"), "w") as fh:
            fh.write(str(pgn) + "\n")
    log.write("\n=== Standings ===\n")
    for m in sorted(models, key=lambda m: -points[m]):
        rate = fails[m] / max(1, moves[m])
        log.write(
            f"  {short(m):22} {points[m]:4.1f} pts   moves {moves[m]:4}   no legal move named {fails[m]} ({rate:.0%})\n"
        )
    json.dump(
        {"points": points, "fails": fails, "moves": moves}, open(os.path.join(outdir, "summary.json"), "w"), indent=1
    )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
