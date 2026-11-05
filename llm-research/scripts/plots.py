"""Analysis charts for the environments, agents and personalities.

Usage: uv run python scripts/plots.py [ARENA_DIR] [OUT_DIR]   (defaults: results/arena, media/plots)

Writes PNGs (small enough to commit) from whatever results exist:
  elo-chess.png, elo-poker.png   Elo per round, one panel per model: learner solid, frozen twin dashed
  twin-gap.png                    learner minus frozen twin Elo, per model and game (does self-improvement work?)
  poker-styles.png                how often each entrant plays a hand vs its chip result, per session
  poker-players.png               mean chip result per entrant type (models, toolkit, personalities, algorithms)
  chess-failures.png              share of moves where a model named no legal move
  agent-suites.png                pass rate per model on each agent test (shell, browser, Windows GUI)
  model-speed.png                 decode speed per model
  trading-r.png                   total R after costs per trading entrant, with every episode as a dot
Colours: the validated reference categorical order, fixed per model; every line and bar is direct-labelled
because three of the five colours sit under 3:1 contrast. Data tables: ladder.md and results/raw.
"""

import collections
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODELS = ["Tiger-Gemma-9B", "wizard-vicuna", "Big-Tiger-Gemma-27B", "qwen2.5vl"]
COLOR = dict(zip(MODELS, SERIES))
TYPES = {
    "language model, learner": SERIES[0],
    "language model, frozen": SERIES[1],
    "toolkit (tool calls)": SERIES[2],
    "simulated human": SERIES[3],
    "algorithm": SERIES[4],
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )


def spread_labels(ax, items):
    """Direct labels just right of each line end, nudged apart vertically so none overlap."""
    if not items:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.07
    placed = []
    for x, y, text in sorted(items, key=lambda t: t[1]):
        y2 = max(y, placed[-1] + gap) if placed else y
        placed.append(y2)
        ax.annotate(
            text, (x, y), xytext=(x + 0.05 + 0.03 * x, y2), fontsize=7, color=INK2, va="center", annotation_clip=False
        )


def short(e):
    base, _, tag = e.partition("#")
    name = base.split("/")[-1].split(":")[0].replace("-GGUF", "")
    return f"{name}#{tag}" if tag else name


def model_of(entrant):
    base = entrant.partition("#")[0]
    return next((m for m in MODELS if m in base), None)


def kind(entrant):
    if entrant.endswith("#learn"):
        return "language model, learner"
    if entrant.endswith("#frozen"):
        return "language model, frozen"
    if entrant.endswith("#toolkit"):
        return "toolkit (tool calls)"
    return "algorithm" if entrant in ("monte", "sol") else "simulated human"


def save(fig, out, name):
    fig.tight_layout()
    path = os.path.join(out, name)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"{path}  {os.path.getsize(path) / 1e3:.0f} KB")


def elo(arena, out):
    path = os.path.join(arena, "ladder.json")
    if not os.path.exists(path):
        return
    hist = json.load(open(path))["history"]
    if not hist:
        return
    for game in ("chess", "poker"):
        rounds = [0] + [h["round"] for h in hist]
        present = [m for m in MODELS if any(model_of(e) == m for e in hist[-1][game])]
        bots = [e for e in hist[-1][game] if model_of(e) is None]
        panels = present + (["simulated players"] if bots else [])
        fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 2.9), sharey=True)
        axes = [axes] if len(panels) == 1 else list(axes)
        for ax, panel in zip(axes, panels):
            labels = []
            if panel == "simulated players":
                for b in sorted(bots):
                    ys = [1000] + [h[game].get(b, 1000) for h in hist]
                    ax.plot(rounds, ys, color=MUTED, linewidth=1.4)
                    labels.append((rounds[-1], ys[-1], b))
            else:
                for tag, ls in (("learn", "-"), ("frozen", "--"), ("toolkit", ":")):
                    for e in [e for e in hist[-1][game] if model_of(e) == panel and e.endswith("#" + tag)]:
                        ys = [1000] + [h[game].get(e, 1000) for h in hist]
                        ax.plot(rounds, ys, color=COLOR[panel], linestyle=ls)
                        labels.append((rounds[-1], ys[-1], tag))
            spread_labels(ax, labels)
            ax.set_title(panel, loc="left")
            ax.set_xlabel("round")
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            ax.set_xlim(0, rounds[-1] * 1.25 + 0.3)
            ax.axhline(1000, color=MUTED, linewidth=0.8)
        axes[0].set_ylabel("Elo")
        fig.suptitle(
            f"{game.title()} ladder: learner (solid) vs frozen twin (dashed)", x=0.01, ha="left", fontweight="bold"
        )
        save(fig, out, f"elo-{game}.png")
    last = hist[-1]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    labels, vals, cols = [], [], []
    for game in ("chess", "poker"):
        for m in MODELS:
            lr = [v for e, v in last[game].items() if model_of(e) == m and e.endswith("#learn")]
            fr = [v for e, v in last[game].items() if model_of(e) == m and e.endswith("#frozen")]
            if lr and fr:
                labels.append(f"{m}\n{game}")
                vals.append(lr[0] - fr[0])
                cols.append(COLOR[m])
    bars = ax.bar(range(len(vals)), vals, color=cols, width=0.6)
    for b, v in zip(bars, vals):
        ax.annotate(
            f"{v:+.0f}",
            (b.get_x() + b.get_width() / 2, v),
            xytext=(0, 3 if v >= 0 else -10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=INK,
        )
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(range(len(vals)), labels, fontsize=7)
    ax.set_ylabel("Elo, learner − frozen")
    ax.set_title(f"Do self-written notes help? After {last['round']} rounds", loc="left")
    save(fig, out, "twin-gap.png")


def poker(arena, out):
    sessions = glob.glob(os.path.join(arena, "rounds", "*", "poker-table*", "summary.json"))
    if not sessions:
        return
    pts, by_kind = [], {}
    for p in sessions:
        s = json.load(open(p))
        for e, c in s["chips"].items():
            st = s["stats"][e]
            vpip = st["vpip_hands"] / max(1, st["hands"])
            pts.append((e, vpip, c - 200))
            by_kind.setdefault(kind(e), []).append(c - 200)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k, col in TYPES.items():
        xs = [v for e, v, _ in pts if kind(e) == k]
        ys = [r for e, _, r in pts if kind(e) == k]
        if xs:
            ax.scatter(xs, ys, s=36, color=col, label=k, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    for name in ("bob", "tina", "max", "carl", "nora", "monte", "sol"):
        mine = [(v, r) for e, v, r in pts if e == name]
        if mine:
            v = sum(x for x, _ in mine) / len(mine)
            r = sum(y for _, y in mine) / len(mine)
            ax.annotate(name, (v, r), xytext=(5, 4), textcoords="offset points", fontsize=7, color=INK2)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xlabel("share of hands played voluntarily")
    ax.set_ylabel("chips won or lost in the session")
    ax.set_title("Poker styles: how loose each player is, and what it earned", loc="left")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, out, "poker-styles.png")
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ks = [k for k in TYPES if k in by_kind]
    means = [sum(by_kind[k]) / len(by_kind[k]) for k in ks]
    bars = ax.barh(range(len(ks)), means, color=[TYPES[k] for k in ks], height=0.55)
    for b, v, k in zip(bars, means, ks):
        ax.annotate(
            f"{v:+.0f} (n={len(by_kind[k])})",
            (v, b.get_y() + b.get_height() / 2),
            xytext=(4 if v >= 0 else -4, 0),
            textcoords="offset points",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
        )
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(ks)), ks)
    ax.set_xlabel("mean chips per session (start 200)")
    ax.set_title("Poker results by kind of player", loc="left")
    save(fig, out, "poker-players.png")


def chess_failures(arena, out):
    files = glob.glob(os.path.join(arena, "rounds", "*", "chess.jsonl"))
    if not files:
        return
    moves, forced = {}, {}
    for p in files:
        for line in open(p):
            r = json.loads(line)
            if r.get("end"):
                continue
            m = model_of(r["model"])
            moves[m] = moves.get(m, 0) + 1
            forced[m] = forced.get(m, 0) + int(r["forced_random"])
    ms = [m for m in MODELS if m in moves]
    rates = [forced[m] / moves[m] for m in ms]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    bars = ax.barh(range(len(ms)), rates, color=[COLOR[m] for m in ms], height=0.55)
    for b, v, m in zip(bars, rates, ms):
        ax.annotate(
            f"{v:.0%} of {moves[m]}",
            (v, b.get_y() + b.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(range(len(ms)), ms)
    ax.set_xlabel("moves where no legal move was named (a random one was played)")
    ax.set_title("Chess: how often each model fails to name a legal move", loc="left")
    save(fig, out, "chess-failures.png")


def agent_suites(out):
    raw = os.path.join(ROOT, "results", "raw")
    suites = {
        "shell (JSON text)": ["dc_results.json", "dc_results2.json"],
        "shell (native tools)": ["dc_results_tools.json"],
        "browser": ["br_results.json"],
        "Windows GUI": ["gui_results.json"],
        "Windows GUI + harness": ["gui_results_harness.json"], "Windows GUI + harness + skills": ["gui_results_skills.json"],
    }
    table = {}
    for suite, names in suites.items():
        for n in names:
            p = os.path.join(raw, n)
            if not os.path.exists(p):
                continue
            for r in json.load(open(p)):
                m = model_of(r["model"]) or r["model"]
                if "tools" in r["model"] and suite == "shell (JSON text)":
                    continue
                ts = r["tasks"]
                table[(m, suite)] = sum(t["passed"] for t in ts) / len(ts)
    if not table:
        return
    ms = [m for m in MODELS if any(k[0] == m for k in table)]
    ss = [s for s in suites if any(k[1] == s for k in table)]
    fig, ax = plt.subplots(figsize=(1.35 * len(ss) + 2.2, 0.55 * len(ms) + 1.4))
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#e8f0fb", "#2a78d6", "#0d366b"])
    for i, m in enumerate(ms):
        for j, s in enumerate(ss):
            v = table.get((m, s))
            if v is None:
                ax.text(j, i, "not run", ha="center", va="center", fontsize=7, color=MUTED)
                continue
            ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, color=ramp(v)))
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color="#ffffff" if v > 0.5 else INK)
    ax.set_xlim(-0.5, len(ss) - 0.5)
    ax.set_ylim(len(ms) - 0.5, -0.5)
    ax.set_xticks(range(len(ss)), ss, fontsize=8)
    ax.set_yticks(range(len(ms)), ms)
    ax.grid(False)
    ax.set_title("Agent tests: share of tasks passed (graded against the machine)", loc="left")
    save(fig, out, "agent-suites.png")


def speed(out):
    p = os.path.join(ROOT, "results", "raw", "load_and_refusal.json")
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    ms, vals = [], []
    for r in rows:
        tps = sorted(x["tok_per_s"] for x in r["results"] if x["tok_per_s"])
        ms.append(r["model"].split("/")[-1].split(":")[0].replace("-GGUF", ""))
        vals.append(tps[len(tps) // 2])
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    bars = ax.barh(range(len(ms)), vals, color=SERIES[0], height=0.55)
    for b, v in zip(bars, vals):
        ax.annotate(
            f"{v:.0f} tok/s",
            (v, b.get_y() + b.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(range(len(ms)), ms)
    ax.set_xlabel("median decode speed, tokens per second (one model loaded at a time)")
    ax.set_title("Model speed on the lab", loc="left")
    save(fig, out, "model-speed.png")


def trading(tdir, out):
    eps = sorted(glob.glob(os.path.join(tdir, "episodes", "*", "trades.json")))
    if not eps:
        return
    per = collections.defaultdict(list)
    for p in eps:
        d = json.load(open(p))
        for e, r in d["totals"].items():
            per[e].append(r)
    order = sorted(per, key=lambda e: sum(per[e]))
    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(order) + 1.4))
    for i, e in enumerate(order):
        m = model_of(e)
        col = COLOR.get(m, MUTED) if m else MUTED
        ax.barh(i, sum(per[e]), color=col, height=0.6)
        ax.scatter(per[e], [i] * len(per[e]), s=14, color=INK2, zorder=3)
        ax.annotate(
            f"{sum(per[e]):+.1f}R",
            (sum(per[e]), i),
            xytext=(4 if sum(per[e]) >= 0 else -4, 0),
            textcoords="offset points",
            va="center",
            ha="left" if sum(per[e]) >= 0 else "right",
            fontsize=7,
        )
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(order)), [short(e) for e in order], fontsize=7)
    ax.set_xlabel("total R after costs across episodes (dots: single episodes)")
    ax.set_title(f"Trading: {len(eps)} replayed episodes, same windows for everyone", loc="left")
    save(fig, out, "trading-r.png")


if __name__ == "__main__":
    arena = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "arena")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "media", "plots")
    os.makedirs(out, exist_ok=True)
    style()
    elo(arena, out)
    poker(arena, out)
    chess_failures(arena, out)
    agent_suites(out)
    speed(out)
    trading(sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "results", "trading"), out)
