"""Milestone charts for the partner pack: one PNG per result, in media/partner/.

Usage: uv run python scripts/partner_charts.py
Every number is read from results/ or copied from the write-ups in docs/ (cited next to each chart).
Palette: the reference categorical slots 1-3 (blue, orange, aqua), validated for colour-blind separation;
grey for baselines and controls.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "media", "partner")
SURFACE, INK, INK2, GRID, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#a3a29c"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MODELS = [("Qwen2.5-VL 32B", BLUE), ("Tiger-Gemma 9B", ORANGE), ("Big-Tiger-Gemma 27B", AQUA)]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    }
)


def save(fig, name, note):
    fig.text(0.01, -0.06, note, fontsize=8.5, color=INK2, ha="left", va="top", transform=fig.transFigure)
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(name, os.path.getsize(os.path.join(OUT, name)) // 1024, "KB")


def push_fold():
    c = next(
        x
        for x in json.load(open(os.path.join(HERE, "results/preflop/push_fold_nash.json")))["charts"]
        if x["stack_bb"] == 10
    )
    r = "AKQJT98765432"
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, key, title in (
        (axes[0], "push", "Small blind: shove all-in"),
        (axes[1], "call", "Big blind: call the shove"),
    ):
        rng = set(c[key])
        for i, a in enumerate(r):
            for j, b in enumerate(r):
                hc = a + b if i == j else (a + b + "s" if j > i else b + a + "o")
                on = hc in rng
                ax.add_patch(plt.Rectangle((j, 12 - i), 0.94, 0.94, color=BLUE if on else "#eeede9"))
                ax.text(
                    j + 0.47, 12 - i + 0.47, hc, ha="center", va="center", fontsize=6.5, color="white" if on else INK2
                )
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 13)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{title}: {len(rng) and c[key + '_share']:.0%} of hands", fontsize=12)
    fig.suptitle(
        "Heads-up push/fold equilibrium at 10 big blinds (computed, matches published charts)",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=14,
    )
    save(
        fig,
        "01-push-fold-10bb.png",
        "Fictitious play over 169 hand classes, 20,000 run-outs per pair. Published reference: 59% shove / 38% call. "
        "Source: results/preflop/push_fold_nash.json",
    )


def solver_speed():
    rows = [("Turn", 18.6, 8.5, 5.6, 4.1), ("Flop", 1666, 156, 256, 41)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (spot, o1, o16, f1, f16) in zip(axes, rows):
        x = [0, 1]
        ax.bar([v - 0.2 for v in x], [o1, o16], 0.38, color=GREY, label="Original TexasSolver")
        ax.bar([v + 0.2 for v in x], [f1, f16], 0.38, color=BLUE, label="Optimised build")
        for v, a, b in zip(x, (o1, o16), (f1, f16)):
            ax.text(v - 0.2, a, f"{a:g}s", ha="center", va="bottom", fontsize=9, color=INK2)
            ax.text(v + 0.2, b, f"{b:g}s", ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(x, ["1 thread", "16 threads"])
        ax.set_title(f"{spot} solve")
        ax.set_ylabel("seconds (lower is faster)")
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Same answer, faster: identical exploitability and strategy", x=0.01, ha="left", fontweight="bold", fontsize=14
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save(
        fig,
        "02-solver-speed.png",
        "The original's flop runs overlapped heavy lab load, so the flop gain is overstated; the turn is a fair "
        "comparison (>=3.3x single-thread). Source: results/solver_bench/solver_bench.log",
    )


def flop_model():
    r = json.load(open(os.path.join(HERE, "results/mlflop/report.json")))
    names = {
        "mlp": "Neural net (used live, 325KB)",
        "gradient_boosting": "Gradient boosting",
        "random_forest": "Random forest",
        "extra_trees": "Extra trees (best)",
        "ensemble_trees": "Tree ensemble",
        "ensemble_all": "All-model ensemble",
    }
    rows = [("Average strategy (baseline)", r["baseline"]["top_action_agreement"], GREY)]
    rows += [(names[k], v["top_action_agreement"], BLUE) for k, v in r["models"].items()]
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, (n, v, col) in enumerate(rows):
        ax.barh(i, v * 100, 0.6, color=col)
        ax.text(v * 100 + 0.6, i, f"{v:.1%}", va="center", fontsize=10)
    ax.set_yticks(range(len(rows)), [n for n, *_ in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("how often the model's top action matches the solver's, on 25 flops it never saw (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Flop solves in milliseconds: models trained on 400 TexasSolver solves")
    save(
        fig,
        "03-flop-model-accuracy.png",
        f"{r['examples']:,} hand-decisions from {r['flops']} flops x 4 spots; tested on {r['test_flops']} held-out "
        "flops. A live flop solve takes 41s-26min. Source: results/mlflop/report.json",
    )


def advice_tests():
    runs = [
        ("100bb\nfirst run", (23 / 30, 22 / 29, 21 / 30)),
        ("100bb\n+ flop model", (19 / 30, 20 / 28, 16 / 30)),
        ("100bb\nsolver decides", (16 / 29, 16 / 23, 15 / 30)),
        ("100bb solver decides\n+ deep pre-flop", (15 / 29, 20 / 28, 20 / 30)),
        ("Short stacks\nno chart (invalid)", (11 / 30, 21 / 30, 21 / 30)),
        ("Short stacks\n+ push/fold chart", (13 / 30, 21 / 30, 21 / 30)),
        ("Short stacks\nsolver decides", (17 / 30, 20 / 30, 19 / 28)),
    ]
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.axhline(50, color=INK2, lw=1, ls="--")
    ax.text(len(runs) - 0.55, 51, "coin flip", fontsize=9, color=INK2, ha="right")
    for k, (name, col) in enumerate(MODELS):
        xs = [i + (k - 1) * 0.16 for i in range(len(runs))]
        ys = [r[1][k] * 100 for r in runs]
        ax.scatter(xs, ys, s=80, color=col, edgecolor=SURFACE, linewidth=2, label=name, zorder=3)
    ax.set_xticks(range(len(runs)), [r[0] for r in runs])
    ax.set_ylim(25, 90)
    ax.set_ylabel("sessions where the advised copy\nbeat the plain copy (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_title(
        "Solver with the LLM: significant once the solver decides every street (pooled p 0.007 short, 0.009 at 100bb)"
    )
    save(
        fig,
        "04-advice-tests.png",
        "Each session seats every model twice (plain and advised) at one 9-seat table; 24 hands; ~30 sessions per run. "
        "Source: docs/2026-11-05-solver-stack.md",
    )


def obedience():
    rows = [("Qwen2.5-VL 32B", 1, 16), ("Big-Tiger-Gemma 27B", 0, 8), ("Tiger-Gemma 9B", 2, 2)]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (n, a, b) in enumerate(rows):
        ax.barh(i + 0.2, a, 0.36, color=GREY, label="advice without the chart" if i == 0 else None)
        ax.barh(i - 0.2, b, 0.36, color=BLUE, label="advice says SHOVE / FOLD from the chart" if i == 0 else None)
        ax.text(b + 0.3, i - 0.2, str(b), va="center", fontsize=10)
        ax.text(a + 0.3, i + 0.2, str(a), va="center", fontsize=10, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("pre-flop all-ins in 30 short-stack sessions (720 hands); the chart answered 706 spots")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title('Told "AKs is a SHOVE: go all-in", most models still don\'t')
    save(
        fig,
        "05-models-ignore-advice.png",
        "Why the next test lets the solver act directly and the LLM decide only spots no solver covers. "
        "Source: solverfirst_short table logs",
    )


def tina_plus():
    rows = [
        ("Tina+ (fixed: stack-aware, balanced)", 182, 134, 234, BLUE),
        ("Monte (Monte Carlo bot)", 144, 90, 202.5, GREY),
        ("Tina (original)", 112, 81, 144, GREY),
        ("Max (maniac)", 97.5, 49.5, 151.5, GREY),
        ("Nora (nit)", 60, 33, 90, GREY),
        ("Carl (calling station)", 53, 33, 75, GREY),
        ("Bob (recreational)", 52, 32, 74, GREY),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvline(100, color=INK2, lw=1, ls="--")
    ax.text(102, 0.5, "fair share (100)", fontsize=9, color=INK2, va="center")
    for i, (n, m, lo, hi, col) in enumerate(reversed(rows)):
        ax.plot([lo, hi], [i, i], color=col, lw=2.5, solid_capstyle="round")
        ax.scatter([m], [i], s=70, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(hi + 3, i, f"{m:g}", va="center", fontsize=10)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlabel("mean prize per tournament, with 95% bootstrap interval")
    ax.grid(axis="y", visible=False)
    ax.set_title("Fixing Tina's weaknesses: Tina+ earns 1.6x Tina over 60 tournaments")
    save(
        fig,
        "06-tina-plus.png",
        "18-player multi-table tournaments, rising blinds and antes, top 3 paid. "
        "Source: results/tina_vs_tinaplus/comparison.txt",
    )


def trading():
    rows = [
        ("Qwen given the paper's method", 0.542, 0.105, 0.979, BLUE),
        ("Qwen alone", 0.367, -0.335, 1.097, GREY),
        ("Paper's rule (K-means gate)", -0.009, -0.411, 0.389, GREY),
        ("Paper's signals, random gate", -0.063, -0.527, 0.430, GREY),
        ("Random trader", -0.491, -1.077, 0.074, GREY),
        ("Buy and hold", -0.541, -1.272, 0.207, GREY),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.axvline(0, color=INK2, lw=1, ls="--")
    for i, (n, m, lo, hi, col) in enumerate(reversed(rows)):
        ax.plot([lo, hi], [i, i], color=col, lw=2.5, solid_capstyle="round")
        ax.scatter([m], [i], s=70, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(hi + 0.04, i, f"{m:+.2f}R", va="center", fontsize=10)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlabel("mean R per 3-day window after costs, 95% interval (46 windows, EURUSD and BTC/USD)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Trading arena: the paper's regime gate adds nothing; the model's judgement does")
    save(
        fig,
        "07-trading.png",
        "Windows overlap in time, so intervals are optimistic. Source: results/trading/summary_46.txt",
    )


def ladder():
    rows = [
        ("Tiger-Gemma 9B", 41, 48),
        ("Big-Tiger-Gemma 27B", 31, 50),
        ("Qwen2.5-VL 32B", 30, 50),
        ("Wizard-Vicuna 13B", 27, 50),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axvline(50, color=INK2, lw=1, ls="--")
    for i, (n, k, t) in enumerate(reversed(rows)):
        v = 100 * k / t
        ax.barh(i, v, 0.55, color=BLUE if n.startswith("Tiger") else GREY)
        ax.text(v + 1, i, f"{k} of {t}", va="center", fontsize=10)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlim(0, 100)
    ax.set_xlabel("poker sessions where the self-learning model beat its frozen twin (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Models that write their own skill notes: one clear learner after 58 rounds")
    save(
        fig,
        "08-self-learning-ladder.png",
        "Twins share a table and the same cards; only Tiger-Gemma 9B passes the "
        "multiple-testing correction (p < 0.001). Source: results/arena/ladder.md",
    )


def alphapoker():
    r = json.load(open(os.path.join(HERE, "results/alphapoker/report.json")))["methods"]
    show = [
        ("Random play", "random", GREY),
        ("AlphaGo search, guessing the cards (PIMC)", "AlphaGo search (PIMC)", GREY),
        ("AlphaGo search over information sets", "AlphaGo search (IS-MCTS)", GREY),
        ("Self-play RL, no averaging (Q-learning)", "self-play RL (Q-learning)", GREY),
        ("AlphaPoker v0: network 64 units", "AlphaPoker v0 (network (64,))", ORANGE),
        ("AlphaPoker v0: network + AlphaGo search", "AlphaPoker v0 + AlphaGo search (PUCT, net prior)", ORANGE),
        ("AlphaPoker v0: network 128+64 units", "AlphaPoker v0 (network (128, 64))", ORANGE),
        ("Sampled self-play that averages (MCCFR)", "MCCFR (sampled self-play)", BLUE),
        ("Fictitious self-play", "fictitious self-play", BLUE),
        ("CFR+ (the solver method)", "CFR+", BLUE),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, (label, key, col) in enumerate(reversed(show)):
        v = r[key]["exploitability"]
        ax.barh(i, v, 0.6, color=col)
        ax.text(v * 1.12, i, f"{v:,.1f}" if v < 10 else f"{v:,.0f}", va="center", fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 8000)
    ax.set_yticks(range(len(show)), [x[0] for x in reversed(show)])
    ax.set_xlabel("exploitability: milli-chips a perfect opponent wins per hand (log scale; lower is better)")
    ax.grid(axis="y", visible=False)
    ax.set_title("AlphaPoker on Leduc poker: search alone fails, averaging self-play converges")
    save(
        fig,
        "10-alphapoker.png",
        "Grey: AlphaGo/AlphaZero ingredients as they are. Orange: our network versions. Blue: "
        "self-play that averages (the Libratus/Pluribus family). Source: results/alphapoker/report.json",
    )


def pca_styles():
    pts = json.load(open(os.path.join(HERE, "results/predictive/pca_points.json")))
    rep = json.load(open(os.path.join(HERE, "results/predictive/report.json")))["2_pca"]
    ev = rep["explained_variance"]
    fig, ax = plt.subplots(figsize=(10, 6))
    win = [p for p in pts if p["bb_per_100"] > 0]
    lose = [p for p in pts if p["bb_per_100"] <= 0]
    for grp, col, lab in ((lose, GREY, "lost money"), (win, BLUE, "won money")):
        ax.scatter(
            [p["pc1"] for p in grp], [p["pc2"] for p in grp], s=9, color=col, alpha=0.55, label=lab, linewidths=0
        )
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of variation): plays more hands, raises and 3-bets more, folds to bets less")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%}): more aggressive after the flop")
    ax.legend(frameon=False, loc="upper right", markerscale=2.5)
    ax.set_title(f"Playing styles of {rep['players']:,} real players (500+ hands each): winners and losers overlap")
    save(
        fig,
        "11-pca-styles.png",
        "PCA of five style rates (VPIP, PFR, 3-bet, post-flop aggression, fold to a bet), "
        "HandHQ 2009. Style vs win rate: correlation 0.08 or less. Source: results/predictive/report.json",
    )


def predict_action():
    r = json.load(open(os.path.join(HERE, "results/predictive/report.json")))["1_next_action"]
    rows = [
        ("Majority action per street", "majority per street", GREY),
        ("Logistic regression", "logistic regression", BLUE),
        ("ESN: reads the hand's action sequence only", "ESN only (reservoir + street)", ORANGE),
        ("ESN + table features", "ESN + static features", ORANGE),
        ("MLP (deep learning)", "MLP (deep learning)", BLUE),
        ("Random forest", "random forest", BLUE),
        ("Gradient boosting", "gradient boosting", BLUE),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (lab, k, col) in enumerate(rows):
        v = 100 * r[k]["accuracy"]
        ax.barh(i, v, 0.6, color=col)
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=10)
    ax.set_yticks(range(len(rows)), [x[0] for x in rows])
    ax.set_xlim(55, 72)
    ax.set_xlabel("next action predicted correctly (fold / check-call / bet-raise), 100,000 later decisions")
    ax.grid(axis="y", visible=False)
    ax.set_title("Predicting what real players do next: trees and networks tie, the ESN is close")
    save(
        fig,
        "12-predict-next-action.png",
        "Trained on 300,000 decisions, tested on 100,000 later ones; 1.3M HandHQ "
        "hands, split by time. Axis starts at 55%. Source: results/predictive/report.json",
    )


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (
        push_fold,
        solver_speed,
        flop_model,
        advice_tests,
        obedience,
        tina_plus,
        trading,
        ladder,
        alphapoker,
        pca_styles,
        predict_action,
    ):
        f()
