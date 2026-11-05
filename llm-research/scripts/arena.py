"""Chess and poker ladder for local models, with self-written skills and memories, against frozen twins.

Usage: uv run python scripts/arena.py ARENA_DIR ROUNDS MODEL [MODEL ...]

Every model enters twice: `model#learn` keeps `notes/<name>/skill-chess.md`, `skill-poker.md` and `memory.md`,
sees them in every prompt and rewrites them after every match; `model#frozen` plays with no notes at all.
Same weights, same prompts otherwise, so learner-minus-twin Elo is the measure of whether the notes help.

Each round: chess pairs neighbours on the chess ladder (1v2, 3v4, ...), colours by coin flip, 120 plies max;
poker plays two tables of HANDS hands from 200 chips, each seating two models (both twins, so they share cards
and opponents) plus the round's simulated humans (poker_selfplay.PERSONAS, at most 3 a table: Bob and two
rotating), scored as pairwise results by final chips. The simulated humans never learn: they are Elo anchors.
Elo starts at 1000, K=24 (poker K is divided by the number of opponents). State is in ARENA_DIR/ladder.json, so
a later run continues the ladder. Model-written notes are capped, used only as prompt text, never executed.
"""

import json
import urllib.error
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chess_selfplay as cs  # noqa: E402
import harness  # noqa: E402
import poker_selfplay as ps  # noqa: E402

HANDS, PLIES, K = 24, 120, 24
# A tool-capable variant plays poker with the poker-analysis toolkit as tools, at the table where its base model
# sits, so it faces the same cards and opponents as that model's learner and frozen twin.
TOOLKIT_MODELS = {"qwen2.5vl:32b": "qwen2.5vl-tools:32b"}
SKILL_MAX, MEMORY_MAX = 1200, 900


def slug(entrant):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", cs.short(entrant.partition("#")[0]))


def read(path, limit=None):
    try:
        text = open(path).read()
    except OSError:
        return ""
    return text[-limit:] if limit else text


def notes_for(arena, entrant, game):
    if not entrant.endswith("#learn"):
        return ""
    small = "wizard-vicuna" in entrant  # 2,048-token context: notes must stay short or every prompt overflows
    d = os.path.join(arena, "notes", slug(entrant))
    skill = read(os.path.join(d, f"skill-{game}.md")).strip()
    memory = read(os.path.join(d, "memory.md"), MEMORY_MAX).strip()
    parts = []
    if skill:
        parts.append(f"## Your {game} skill file\n{skill}")
    if memory:
        parts.append(f"## Your memory (most recent last)\n{memory}")
    text = "\n\n".join(parts)
    return text[-500:] if small else text


def reflect(arena, entrant, game, summary, rnd):
    """The learner rewrites its own skill file and adds one memory line. Returns the new skill text."""
    d = os.path.join(arena, "notes", slug(entrant))
    os.makedirs(d, exist_ok=True)
    skill_path = os.path.join(d, f"skill-{game}.md")
    current = read(skill_path).strip() or "(empty: this is your first match)"
    prompt = (
        f"You are improving yourself at {game}. This is your own skill file, which you see before every move:\n"
        f"<skill>\n{current}\n</skill>\n\n"
        f"What happened in your last {game} match:\n{summary}\n\n"
        f"Rewrite your skill file so you play better next time: short, concrete rules you will actually follow. "
        f"At most {SKILL_MAX} characters. Then write one line starting with MEMORY: about this match or this "
        f"opponent that is worth remembering. Reply with the skill file text, then the MEMORY: line, nothing else."
    )
    try:
        text, _ = _ask_long(entrant, prompt)
    except urllib.error.HTTPError as e:  # context overflow on small models: keep the old files
        return "", f"(reflection failed: HTTP {e.code})"
    text = re.sub(r"^```[a-z]*\n|```$", "", text.strip(), flags=re.M).strip()
    mem = ""
    m = re.search(r"^\s*MEMORY:\s*(.+)$", text, flags=re.M)
    if m:
        mem = m.group(1).strip()[:200]
        text = text[: m.start()].strip()
    skill = text[:SKILL_MAX].strip()
    if skill:
        with open(skill_path, "w") as f:
            f.write(skill + "\n")
    if mem:
        mpath = os.path.join(d, "memory.md")
        with open(mpath, "a") as f:
            f.write(f"- round {rnd}, {game}: {mem}\n")
        text = read(mpath)
        if len(text) > MEMORY_MAX * 1.5:  # condense in the model's own words rather than drop the oldest lines
            cap = 450 if "wizard-vicuna" in entrant else MEMORY_MAX
            with open(mpath, "w") as f:
                f.write(harness.compress_memory(entrant.partition("#")[0], text, cap).rstrip() + "\n")
    return skill, mem


def _ask_long(entrant, prompt):
    import urllib.request

    body = {
        "model": entrant.partition("#")[0],
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.4, "num_predict": 500},
    }
    req = urllib.request.Request(
        cs.OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["message"]["content"], 0


def expected(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))


def chess_round(arena, rnd, ladder, rng, log):
    entrants = sorted(ladder["chess"], key=lambda e: -ladder["chess"][e])
    rdir = os.path.join(arena, "rounds", f"{rnd:02d}")
    os.makedirs(rdir, exist_ok=True)
    jl = open(os.path.join(rdir, "chess.jsonl"), "a", buffering=1)
    glog = open(os.path.join(rdir, "chess.log"), "a", buffering=1)
    notes = {e: notes_for(arena, e, "chess") for e in entrants}
    results = []
    for g, (a, b) in enumerate(zip(entrants[0::2], entrants[1::2]), 1):
        white, black = (a, b) if rng.random() < 0.5 else (b, a)
        glog.write(f"\n--- Round {rnd} game {g}: {cs.short(white)} (White) vs {cs.short(black)} (Black) ---\n")
        board, sans, result, how, fails = cs.play(white, black, PLIES, glog, jl, g, rng, notes)
        jl.write(
            json.dumps(
                {"game": g, "end": True, "white": white, "black": black, "result": result, "how": how, "moves": sans}
            )
            + "\n"
        )
        sw = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
        rw, rb = ladder["chess"][white], ladder["chess"][black]
        ladder["chess"][white] = rw + K * (sw - expected(rw, rb))
        ladder["chess"][black] = rb + K * ((1 - sw) - expected(rb, rw))
        glog.write(f"  result {result} ({how})\n")
        log(f"  chess {cs.short(white)} vs {cs.short(black)}: {result} ({how})")
        results.append((white, black, result, how, sans, fails))
    for white, black, result, how, sans, fails in results:
        for me, opp, colour in ((white, black, "White"), (black, white, "Black")):
            if not me.endswith("#learn"):
                continue
            score = (
                {"1-0": "won", "0-1": "lost"}.get(result, "drew")
                if colour == "White"
                else {"1-0": "lost", "0-1": "won"}.get(result, "drew")
            )
            mine = sans[0::2] if colour == "White" else sans[1::2]
            summary = (
                f"You played {colour} against {cs.short(opp)} and {score} ({how}) after {len(sans)} plies. "
                f"You failed to name a legal move {fails[me]} times. Your moves: {' '.join(mine[:40])}"
            )
            skill, mem = reflect(arena, me, "chess", summary, rnd)
            log(f"    {cs.short(me)} rewrote skill-chess.md ({len(skill)} chars){'; memory: ' + mem if mem else ''}")


def poker_round(arena, rnd, ladder, log, models):
    """Two tables; each seats two models (learner and frozen twin side by side) plus the round's bots."""
    order = list(models)
    random.Random(rnd).shuffle(order)
    bots = ps.bots_for_round(rnd)
    old = dict(ladder["poker"])
    for t, pair in enumerate([order[0::2], order[1::2]], 1):
        if not pair:
            continue
        seats = [f"{m}#{tag}" for m in pair for tag in ("learn", "frozen")] + bots
        seats += [f"{TOOLKIT_MODELS[m]}#toolkit" for m in pair if m in TOOLKIT_MODELS]
        random.Random(rnd * 10 + t).shuffle(seats)
        rdir = os.path.join(arena, "rounds", f"{rnd:02d}", f"poker-table{t}")
        notes = {e: notes_for(arena, e, "poker") for e in seats}
        chips, stats = ps.main(rdir, HANDS, seats, notes)
        n = len(seats)
        for a in seats:
            delta = 0.0
            for b in seats:
                if a != b:
                    s_ = 1.0 if chips[a] > chips[b] else 0.0 if chips[a] < chips[b] else 0.5
                    delta += (K / (n - 1)) * (s_ - expected(old[a], old[b]))
            ladder["poker"][a] = old[a] + delta
        ranked = sorted(seats, key=lambda e: -chips[e])
        log(f"  poker table {t}: " + ", ".join(f"{cs.short(e)} {chips[e]}" for e in ranked))
        for e in seats:
            if not e.endswith("#learn"):
                continue
            st = stats[e]
            summary = (
                f"You finished {ranked.index(e) + 1} of {n} with {chips[e]} chips from 200 ({chips[e] - 200:+}) "
                f"over {st['hands']} hands. You voluntarily played {st['vpip_hands']} hands, made "
                f"{st['decisions']} decisions and sent {st['invalid']} invalid replies. Winner: "
                f"{cs.short(ranked[0])} with {chips[ranked[0]]}. The table also had simulated humans: "
                + ", ".join(f"{b} ({ps.kind_of(b)}) {chips[b]}" for b in bots)
                + "."
            )
            skill, mem = reflect(arena, e, "poker", summary, rnd)
            log(f"    {cs.short(e)} rewrote skill-poker.md ({len(skill)} chars){'; memory: ' + mem if mem else ''}")


def write_ladder_md(arena, ladder, rnd):
    lines = [f"# Ladder after round {rnd}", ""]
    for game in ("chess", "poker"):
        lines += [f"## {game.title()}", "", "| Rank | Entrant | Elo |", "|---|---|---|"]
        for k, (e, r) in enumerate(sorted(ladder[game].items(), key=lambda x: -x[1]), 1):
            who = cs.short(e) if "#" in e and not e.endswith("#toolkit") else f"{cs.short(e)} ({ps.kind_of(e)})"
            lines.append(f"| {k} | {who} | {r:.0f} |")
        lines += ["", "| Model | Learner − frozen twin |", "|---|---|"]
        for m in sorted({e.partition("#")[0] for e in ladder[game] if e.endswith(("#learn", "#frozen"))}):
            gap = ladder[game][m + "#learn"] - ladder[game][m + "#frozen"]
            lines.append(f"| {cs.short(m)} | {gap:+.0f} |")
        lines.append("")
    with open(os.path.join(arena, "ladder.md"), "w") as f:
        f.write("\n".join(lines))


def main(arena, rounds, models):
    os.makedirs(arena, exist_ok=True)
    path = os.path.join(arena, "ladder.json")
    entrants = [f"{m}#{t}" for m in models for t in ("learn", "frozen")]
    ladder = json.load(open(path)) if os.path.exists(path) else {"round": 0, "history": []}
    for game in ("chess", "poker"):
        ladder.setdefault(game, {})
        extra = (
            list(ps.PERSONAS)
            + list(ps.pa.ALGOS)
            + [f"{TOOLKIT_MODELS[m]}#toolkit" for m in models if m in TOOLKIT_MODELS]
        )
        for e in entrants + (extra if game == "poker" else []):
            ladder[game].setdefault(e, 1000.0)
    rng = random.Random(ladder["round"] + 17)
    alog = open(os.path.join(arena, "arena.log"), "a", buffering=1)

    def log(s):
        alog.write(s + "\n")

    for _ in range(rounds):
        ladder["round"] += 1
        rnd = ladder["round"]
        t0 = time.time()
        log(f"\n=== Round {rnd} ===")
        chess_round(arena, rnd, ladder, rng, log)
        poker_round(arena, rnd, ladder, log, models)
        ladder["history"].append({"round": rnd, "chess": dict(ladder["chess"]), "poker": dict(ladder["poker"])})
        json.dump(ladder, open(path, "w"), indent=1)
        write_ladder_md(arena, ladder, rnd)
        log(f"  round {rnd} took {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
