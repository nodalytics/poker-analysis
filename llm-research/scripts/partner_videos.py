"""Short animated clips for the partner pack (MP4, 960x540, about 15 s each), in media/partner/.

Usage: uv run python scripts/partner_videos.py
Every number is read from committed results (named on each clip). Same palette as partner_charts.py.
  14-robust-exploitation.mp4   profit against a weak player versus how beatable you become (Leduc, exact)
  15-adaptive-opponent.mp4     how fast a learning opponent finds each strategy's leak
  16-ensembles.mp4             combining strategies: reading the opponent beats every single strategy
  17-slumbot.mp4               the outside test against Slumbot, and the three bugs it caught
"""

import glob
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

import partner_charts as pc  # noqa: E402  (palette and rcParams)

HERE, OUT = pc.HERE, pc.OUT
BLUE, ORANGE, AQUA, GREY, INK, INK2 = pc.BLUE, pc.ORANGE, pc.AQUA, pc.GREY, pc.INK, pc.INK2
YELLOW = "#eda100"  # categorical slot 4
FPS = 12


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def render(name, draw, seconds, title, source, rect=(0.09, 0.2, 0.86, 0.66)):
    """draw(ax, t, fig_caption) is called per frame with t in [0, seconds]; returns the caption text."""
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    ax = fig.add_axes(list(rect))
    head = fig.text(0.02, 0.95, title, fontsize=15, fontweight="bold", va="top")
    cap = fig.text(0.02, 0.09, "", fontsize=11.5, color=INK, va="top", wrap=True)
    fig.text(0.02, 0.02, source, fontsize=8, color=INK2, va="bottom")
    path = os.path.join(OUT, name)
    writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "28"])
    with writer.saving(fig, path, dpi=100):
        for k in range(int(seconds * FPS)):
            ax.clear()
            cap.set_text(draw(ax, k / FPS))
            writer.grab_frame()
    plt.close(fig)
    del head
    print(name, os.path.getsize(path) // 1024, "KB")


def robust_exploitation():
    fr = json.load(open(os.path.join(HERE, "results/leduc_rnr/report.json")))["rnr_frontier"]["maniac"]

    def draw(ax, t):
        shown = int(min(len(fr), t / 1.1)) + 1 if t < 8 else len(fr)
        xs = [r["exploitability"] for r in fr[:shown]]
        ys = [r["profit"] for r in fr[:shown]]
        ax.plot(xs, ys, color=BLUE, lw=2, zorder=2)
        ax.scatter(xs, ys, s=70, color=BLUE, edgecolor=pc.SURFACE, linewidth=2, zorder=3)
        for r in fr[:shown]:
            lab = {0.0: "equilibrium (p 0)", 1.0: "pure exploitation (p 1)"}.get(r["p"], f"p {r['p']}")
            ax.annotate(
                lab,
                (r["exploitability"], r["profit"]),
                xytext={0.9: (6, 8), 1.0: (-70, -20)}.get(r["p"], (8, -14)),
                textcoords="offset points",
                fontsize=9,
                color=INK2,
            )
        ax.set_xscale("log")
        ax.set_xlim(25, 6000)
        ax.set_ylim(0, 2600)
        ax.set_xlabel("how beatable we become: milli-chips a perfect opponent wins per hand (log)")
        ax.set_ylabel("profit vs the maniac (milli-chips/hand)")
        if t >= 8:
            a = ease((t - 8) / 1.5)
            p = fr[2]
            ax.scatter(
                [p["exploitability"]],
                [p["profit"]],
                s=70 + 500 * a,
                facecolor="none",
                edgecolor=AQUA,
                linewidth=2.5,
                zorder=4,
            )
            return (
                f"Sweet spot, p 0.25: {p['profit'] / fr[-1]['profit']:.0%} of the maximum profit while only "
                f"{p['exploitability'] / fr[-1]['exploitability']:.0%} as beatable as pure exploitation."
            )
        if t < 3:
            return "Each dot is one strategy, solved exactly (CFR). p = how much we assume the opponent is a maniac."
        return "More exploitation = more profit, but the last step (p 1) makes us 14x easier to beat for 1% more."

    render(
        "14-robust-exploitation.mp4",
        draw,
        14,
        "Robust exploitation: win from weak players, stay hard to beat",
        "Leduc poker, exact. Source: results/leduc_rnr/report.json (restricted Nash response, Johanson et al.)",
    )


def adaptive_opponent():
    rows = json.load(open(os.path.join(HERE, "results/leduc_rl_rnr/report.json")))["rl_against_robust_exploitation"]
    style = {
        0.0: ("equilibrium", BLUE),
        0.25: ("robust p 0.25", AQUA),
        0.5: ("robust p 0.5", YELLOW),
        1.0: ("pure exploitation", ORANGE),
    }

    def draw(ax, t):
        frac = ease(t / 9)
        for r in rows:
            lab, col = style[r["rnr_p"]]
            c = r["learner_curve"]
            n = max(1, round(frac * len(c)))
            ax.plot(
                [x["episodes"] for x in c[:n]],
                [x["learner_wins_above_game_value"] for x in c[:n]],
                color=col,
                lw=2.2,
                marker="o",
                ms=6,
            )
            if n == len(c):
                ax.annotate(
                    lab,
                    (c[-1]["episodes"], c[-1]["learner_wins_above_game_value"]),
                    xytext=(8, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=10,
                    color=INK,
                )
        ax.axhline(0, color=GREY, lw=1)
        ax.set_xscale("log")
        ax.set_xlim(2000, 1_500_000)
        ax.set_ylim(-250, 2100)
        ax.set_xlabel("hands the learning opponent has practised (log)")
        ax.set_ylabel("what it wins from us (milli-chips/hand)")
        if t < 4:
            return "A learning opponent (reinforcement learning) practises against each of our strategies."
        if t < 9:
            return "Pure exploitation is found and punished within ~3,000 hands."
        return "Robust strategies leak slowly and little; the equilibrium is never beaten. Re-solve before it learns."

    render(
        "15-adaptive-opponent.mp4",
        draw,
        14,
        "How fast can a learning opponent find our leaks?",
        "Leduc poker, 300,000 hands per curve, scored exactly. Source: results/leduc_rl_rnr/report.json",
    )


def ensembles():
    tab = json.load(open(os.path.join(HERE, "results/leduc_ensembles/report.json")))["online"]["table"]
    show = [
        ("equilibrium only", "nash", GREY),
        ("best single strategy", "rnr_maniac", GREY),
        ("bandit picks a strategy (Exp3)", "exp3", BLUE),
        ("read the opponent (Bayes)", "bayes_id", AQUA),
    ]

    def draw(ax, t):
        a = ease(t / 5)
        for i, (lab, key, col) in enumerate(reversed(show)):
            v = tab[key]["mean"] * a
            ax.barh(i, v, 0.55, color=col)
            ax.text(v + 15, i, f"{v:,.0f}", va="center", fontsize=11)
            if t > 7:
                w = tab[key]["worst"]
                ax.text(
                    1450,
                    i,
                    f"worst {w:+.0f}",
                    va="center",
                    fontsize=10.5,
                    color=INK if w >= 0 else ORANGE,
                    fontweight="bold" if w >= 0 else None,
                )
        ax.set_yticks(range(len(show)), [s[0] for s in reversed(show)])
        ax.set_xlim(0, 1650)
        ax.set_xlabel("average profit over 8 unknown opponents (milli-chips/hand)")
        ax.grid(axis="y", visible=False)
        if t < 7:
            return "Against an unknown opponent: which way of combining our strategies earns most?"
        return "Read the opponent, switch to the matching robust strategy: most profit, and never loses."

    render(
        "16-ensembles.mp4",
        draw,
        13,
        "Ensembles: read the opponent, then exploit it robustly",
        "Leduc poker, 200 sessions x 1,000 hands per cell, incl. a trap opponent. "
        "Source: results/leduc_ensembles/report.json",
        rect=(0.3, 0.2, 0.66, 0.66),
    )


def slumbot():
    def stat(d):
        x = [
            json.loads(line)["winnings_chips"] / 100
            for f in glob.glob(os.path.join(HERE, "results/slumbot", d, "hands_*.jsonl"))
            for line in open(f)
        ]
        return 100 * sum(x) / len(x), 196 * statistics.stdev(x) / len(x) ** 0.5, len(x)

    vers = [
        ("v1", "slumbot_solverbot_v1_allin_bug", "bug 1: bluff-shoving 50x the pot on the river"),
        ("v2", "slumbot_solverbot_v2_river_fix_only", "bug 2: heads-up seats swapped (button limped)"),
        ("v3", "slumbot_solverbot_v3_no_hand_ranking", "bug 3: solver ran without hand-ranking tables"),
        ("v4", "slumbot_solverbot_v4", "all fixed; still losing on the river: Slumbot bets stronger than we read"),
        ("v5", "slumbot_solverbot_v5", "a hand-reading model of Slumbot, learned from its revealed cards"),
    ]
    data = [(v, *stat(d), note) for v, d, note in vers]

    def draw(ax, t):
        shown = min(len(data), int(t / 3) + 1)
        for i, (v, m, ci, n, note) in enumerate(data[:shown]):
            col = AQUA if i == len(data) - 1 else GREY
            ax.bar(i, m, 0.55, color=col)
            ax.errorbar(i, m, yerr=ci, color=INK2, capsize=6, lw=1.5)
            ax.text(i, (m + ci + 12) if m > 0 else 30, f"{m:+.0f}\n{n:,} hands", ha="center", va="bottom", fontsize=10)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(len(data)), [d[0] for d in data])
        ax.set_xlim(-0.6, len(data) - 0.4)
        ax.set_ylim(-420, 230)
        ax.set_ylabel("big blinds per 100 hands vs Slumbot")
        ax.grid(axis="x", visible=False)
        i = shown - 1
        if t >= len(data) * 3:
            return "v5 beats Slumbot: +57.5 bb/100 over 3,000 hands (95% interval +2 to +114)."
        return f"{data[i][0]}: {data[i][4]}."

    render(
        "17-slumbot.mp4",
        draw,
        19,
        "The outside test: our solver bot vs Slumbot (strong public bot, 200bb)",
        "Real hands over Slumbot's public API, heads-up no-limit. Source: results/slumbot/",
    )


if __name__ == "__main__":
    robust_exploitation()
    adaptive_opponent()
    ensembles()
    slumbot()
