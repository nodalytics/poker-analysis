# Poker Analysis

Poker research and tools: Texas hold'em only. The current goal is a strong, hard-to-exploit bot for **9-max
tournaments (MTT)**: ICM, 10–25 big-blind stacks, push/fold. There is also a tournament analysis toolkit, an MCP
server for it, and a desktop study advisor.

Progress is live on the watch page, **Poker AI Watch** (https://poker.stbfarm.org). It shows the training runs, the
search, the poker-sense checks and the ICM push/fold results.

**Kestrel**, the bot and advisor, runs at **https://kestrel.stbfarm.org**: its web app, practice tournaments against
the networks, and the bot's sessions on Slumbot and other platforms. It needs an access link (a token) once per
device; see [`kestrel/README.md`](kestrel/README.md).

## Where things are

| Path | Contents |
|---|---|
| `archive/` | The milestone-1 tournament toolkit, kept for reference: `src/` (cards and hand evaluation, equity, ICM and bubble factor, stack categories and push/fold ranges, the analyzer, a solver wrapper), its MCP server (`mcp_server.py`, `requirements-mcp.txt`), the terminal menu, Flask GUI and demo (`poker_tool.py`, `poker_gui.py`, `demo_milestone1.py`), and `tests/` |
| `kestrel/` | Kestrel: the C++ bot and advisor (engine, connectors, server, web and desktop apps, Docker deployment) |
| `research/` | The research: the bots, the solvers, the evaluations and their write-ups (below) |

### `research/`

| Path | Contents |
|---|---|
| `scripts/ninemax_*.py` | The 9-max neural route: the GPU table engine, population self-play training, evaluation, search at play time, poker-sense checks, ICM push/fold checks, export, and the watch server |
| `scripts/icm.py`, `scripts/icm_pushfold.py` | ICM equity (Malmuth-Harville), random tournament tables, and push/fold equilibria under ICM |
| `blueprint/` | C++ CFR solvers and tools: the heads-up blueprint, best response, re-solving, card abstraction, and the dealing library used by the 9-max engine |
| `advisor/` | The study advisor (hand review, spots, trainer); being rewritten in C++/Qt 6 with capture-only session recording |
| `models/` | Exported networks in half precision with a manifest (`models/README.md`); large packs are GitHub releases |
| `web/` | The watch page (served by `scripts/ninemax_watch.py`) |
| `docs/` | Write-ups, one per study, dated |
| `results/` | Raw results of the evaluations |

LLM evaluations (local models as agents, the arena ladder, GUI control) are also under `research/`; see
[`research/README.md`](research/README.md).

## Read first

- [`research/docs/2026-11-11-next-phases.md`](research/docs/2026-11-11-next-phases.md): the roadmap
  (phases A–D) and the decisions so far.
- [`research/docs/2026-11-11-mtt-strategy.md`](research/docs/2026-11-11-mtt-strategy.md): what good
  tournament play looks like, the checks that hold the networks to it, and how they score.
- [`research/docs/2026-11-09-nine-max-plan.md`](research/docs/2026-11-09-nine-max-plan.md): the 9-max plan,
  the training runs, exploitability tests and search results.
- [`research/AGENTS.md`](research/AGENTS.md): rules for anyone (person or agent) working here: the lab,
  sandboxes, git and lint.

## Running

- Archived toolkit, from `archive/`: `python3 poker_tool.py` (terminal) or `python3 poker_gui.py` (Flask).
- Its MCP server, from `archive/`: `pip install -r requirements-mcp.txt`, then `python3 mcp_server.py` (stdio) or
  `--http --host 127.0.0.1 --port 8770`. The HTTP transport has no authentication; keep it on localhost.
- Research scripts: `cd research && uv sync && uv run python scripts/<name>.py`. Training, search and
  evaluation run on the lab's GPUs; each script's docstring has its command.
- Tests: `cd archive && python3 -m pytest tests`; lint the research with `uv run ruff check scripts` and
  `uv run ruff format --check scripts` from `research/`.

## Rules

- No bots, OCR or real-time assistance on real-money poker sites. The advisor is for study away from the table.
- Never commit credentials or keys; the repo is public. Machine addresses and logins go in `.secrets/` (ignored).
