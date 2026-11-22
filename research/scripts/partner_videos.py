"""Short animated clips for the partner pack (MP4, 960x540, about 15 s each), in media/partner/.

Usage: uv run python scripts/partner_videos.py
Every number is read from committed results (named on each clip). Same palette as partner_charts.py.
  14-robust-exploitation.mp4   profit against a weak player versus how beatable you become (Leduc, exact)
  15-adaptive-opponent.mp4     how fast a learning opponent finds each strategy's leak
  16-ensembles.mp4             combining strategies: reading the opponent beats every single strategy
  17-slumbot.mp4               the outside test against Slumbot, and the three bugs it caught
  19-blueprint-vs-everyone.mp4 the blueprint against every opponent: engines, simulated humans, LLMs, pros' clones
  20-lbr-curves.mp4            how hard each training run is to beat (LBR) as it trains, CFR runs and MLP self-play
  21-slumbot-rematch.mp4       the new blueprints and the self-taught MLP against Slumbot, luck-adjusted
Run one clip: uv run python scripts/partner_videos.py 19
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
        ("v5 again", "slumbot_solverbot_v5_confirm", "the same bot, 5,000 fresh hands: the win did not repeat"),
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
            return "Luck-adjusted, v5 and its repeat together: about -14 bb/100 (95% -45 to +17), roughly even."
        return f"{data[i][0]}: {data[i][4]}."

    render(
        "17-slumbot.mp4",
        draw,
        22,
        "The outside test: our solver bot vs Slumbot (strong public bot, 200bb)",
        "Real hands over Slumbot's public API, heads-up no-limit. Source: results/slumbot/",
    )


# blueprint run 2 (purification + defence floor) against everyone, bb/100 with 95% intervals.
# Engines, simulated humans and LLMs: docs/2026-11-08-blueprint-speed-accuracy.md (results table); Slumbot:
# luck-adjusted, the same doc; the professionals' clone: results/vs_humans/blueprint_run2.json.
BLUEPRINT_VS = [
    ("Max (maniac)", 178.7, 91, 271),
    ("Qwen2.5-VL 32B (LLM)", 97.5, -58, 279),
    ("Carl (calling station)", 82.3, 37, 130),
    ("Big-Tiger-Gemma 27B (LLM)", 78.5, 69, 86),
    ("Monte (equity bot)", 73.9, 27, 119),
    ("Nora (nit)", 55.9, 45, 68),
    ("Tina+", 50.0, 17, 83),
    ("Bob (recreational)", 36.4, 5, 69),
    ("Slumbot (luck-adjusted)", 30.9, -10, 72),
    ("Tina", 27.5, -6, 63),
    ("Our solver bot", 26.3, 6, 46),
]


def blueprint_vs_everyone():
    pros = json.load(open(os.path.join(HERE, "results/vs_humans/blueprint_run2.json")))["entrants"]["blueprintbot"]
    rows = sorted(
        BLUEPRINT_VS + [("Pros' clone (Pluribus hands)", pros["bb_per_100"], *pros["ci95"])], key=lambda r: r[1]
    )

    def draw(ax, t):
        shown = min(len(rows), int(t / 0.9) + 1)
        for i, (name, m, lo, hi) in enumerate(rows[:shown]):
            sig = lo > 0
            col = AQUA if "clone" in name else (BLUE if sig else GREY)
            ax.barh(i, m, 0.62, color=col)
            ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], color=INK2, capsize=4, lw=1.3)
            ax.text(max(hi, m) + 6, i, f"{m:+.0f}" + ("" if sig else "  n.s."), va="center", fontsize=9.5)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9.5)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlim(-70, 300)
        ax.set_xlabel("blueprint's winnings, big blinds per 100 hands (95% interval)")
        ax.grid(axis="y", visible=False)
        if t < 11:
            return "Heads-up at 200 big blinds. Grey: the interval crosses zero (n.s.)."
        return "Positive against every opponent; significant against 9 of 12. Level with or slightly ahead of Slumbot."

    render(
        "19-blueprint-vs-everyone.mp4",
        draw,
        16,
        "The blueprint (our self-play strategy) against everyone we have",
        "Blueprint run 2 with purification and the defence floor. Sources: docs/2026-11-08-blueprint-speed-accuracy.md,"
        " results/vs_humans/",
        rect=(0.3, 0.2, 0.66, 0.66),
    )


def lbr_curves():
    def load(run):
        rows = [ln.split("\t") for ln in open(os.path.join(HERE, "results/lbr_curves", run + ".tsv"))][1:]
        return [float(r[1]) for r in rows]

    cfr = [
        ("run4: better buckets", load("run4"), GREY),
        ("run5: + DCFR + pruning", load("run5"), YELLOW),
        ("run6: + predictive CFR+", load("run6"), BLUE),
        ("runvec: vector CFR, 8 threads", load("runvec"), AQUA),
    ]
    mlp = [("MLP, weak pull: stalls", load("runmlp"), ORANGE), ("MLP, strong pull: converges", load("runmlp2"), BLUE)]

    def draw(ax, t):
        phase2 = t >= 9
        series = mlp if phase2 else cfr
        frac = ease((t - 9) / 5 if phase2 else t / 7)
        top = 1000 if phase2 else 200
        for name, ys, col in series:
            k = max(1, int(round(frac * len(ys))))
            xs = [0.5 * (i + 1) for i in range(k)]
            ax.plot(xs, ys[:k], "-o", color=col, lw=2.2, ms=4, label=name)
        ax.axhline(104, color=INK2, lw=1, ls="--")
        ax.text(0.3, 104 + top * 0.02, "run 2's best: +104", fontsize=9, color=INK2)
        ax.set_xlim(0, 6.2)
        ax.set_ylim(0, top)
        ax.set_xlabel("hours of training")
        ax.set_ylabel("LBR's edge, bb/100 (lower is better)")
        ax.legend(loc="upper right", fontsize=9.5, frameon=False)
        if phase2:
            return "From scratch on one GPU: a weak pull stalls near +600; a strong pull reaches run 2's +104 in ~2 h."
        return "Lower is better (0 = nothing found; each point about +-65). Predictive CFR+ (run6) stays near +30."

    render(
        "20-lbr-curves.mp4",
        draw,
        16,
        "How hard is each new blueprint to beat, as it trains?",
        "LBR (local best response), 3,000 luck-adjusted hands every 30-minute checkpoint. Source: results/lbr_curves/",
    )


def slumbot_rematch():
    def adj(run):
        d = json.load(open(os.path.join(HERE, "results/slumbot_rematch", run + ".json")))
        r = next(iter(d.values()))["plus_flop_turn_river_card_luck"]
        return r["bb100"], *r["ci95"]

    rows = [
        ("run 2 (earlier)", 30.9, -10, 72),
        ("runvec", *adj("runvec")),
        ("run 6", *adj("run6")),
        ("MLP from scratch", *adj("runmlp2")),
    ]

    def draw(ax, t):
        shown = min(len(rows), int(t / 2.5) + 1)
        for i, (name, m, lo, hi) in enumerate(rows[:shown]):
            ax.bar(i, m, 0.55, color=GREY if lo < 0 < hi else BLUE)
            ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK2, capsize=6, lw=1.5)
            ax.text(i, hi + 4, f"{m:+.1f}\nn.s.", ha="center", va="bottom", fontsize=10)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(range(len(rows)), [r[0] for r in rows])
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_ylim(-100, 110)
        ax.set_ylabel("bb/100 vs Slumbot, luck-adjusted")
        ax.grid(axis="x", visible=False)
        if t < 10:
            return "5,000 hands each against Slumbot, the strong public heads-up bot."
        return "Every interval crosses zero: none of these is distinguishable from even, or from the others."

    render(
        "21-slumbot-rematch.mp4",
        draw,
        13,
        "Rematch: the newest strategies against Slumbot",
        "Luck-adjusted (AIVAT-lite: all-in, pre-flop and per-street card luck). Source: results/slumbot_rematch/",
    )


if __name__ == "__main__":
    import sys

    clips = {
        "14": robust_exploitation,
        "15": adaptive_opponent,
        "16": ensembles,
        "17": slumbot,
        "19": blueprint_vs_everyone,
        "20": lbr_curves,
        "21": slumbot_rematch,
    }
    for key in sys.argv[1:] or clips:
        clips[key]()
