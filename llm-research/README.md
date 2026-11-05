# LLM research

Local LLM evaluations run on the lab, with the scripts, sandboxes and raw results needed to rerun them.

Python is managed with uv: `uv sync`, then `uv run python scripts/<name>.py`.

Agents: read [`AGENTS.md`](AGENTS.md) first (models, lab rules, sandboxes, media limits, git). `CLAUDE.md` imports it.

| Path | Contents |
|---|---|
| `docs/` | Write-ups, one per study, dated |
| `scripts/` | Test harnesses (`llm_test.py`, `dc_agent_test.py`, `browser_agent_test.py`) |
| `sandboxes/` | Docker builds for the sealed test environments, the Windows GUI VM, and Ollama setup (`sandboxes/ollama/`) |
| `results/raw/` | JSON output from each run, including every model answer and tool call |
| `results/logs/` | Console logs from each run |
| `gui-skills/` | Per-application GUI navigation notes, loaded before and updated after each run |
| `media/` | Small GIF evidence of GUI runs (created with the first one; limits in `AGENTS.md`) |

Studies:

- [2026-11-05 — Local "uncensored" LLM evaluation](docs/2026-11-05-local-llm-evaluation.md):
  speed and refusals, Desktop Commander control, browser control; why Mistral Large was dropped.
- [2026-11-05 — Ladders, trading and Windows GUI control](docs/2026-11-05-arena-trading-gui.md): learners vs
  frozen twins at chess and poker, simulated and algorithmic players, the paper's trading method after costs,
  and Windows-MCP results.
- [2026-11-05 — The solver stack under the LLM](docs/2026-11-05-solver-stack.md): solver advice makes every model
  play better (p 0.003-0.021); preflop charts heads-up and multi-way, multi-way Monte Carlo, the solver library and
  flop network, and human baselines from published hand histories.
- [2026-11-05 — MCP servers for another service](docs/2026-11-05-mcp-for-another service.md):
  options for autonomous another service control by models through MCP (API and GUI), the tools, modes and risk limits.

- [Play-money and practice poker platforms](docs/poker-play-money-platforms.md): candidates for testing agents
  outside our own tables, and which ones allow bots.
- [Poker platforms with an API for agents](docs/poker-api-platforms.md): bot arenas and engines, and the
  play-money-only and untrusted-input rules for connecting ours.

To do:

- **Post-session hand review:** load your own saved hand histories after play ends and have TexasSolver mark the
  mistakes, as a study tool. Offline only: no live client is read while a hand is being played.
- **TexasSolver GPU** (github.com/bupticybee/TexasSolverGPU): the author says 20x+ faster, which could make live
  flop solves real time. Closed source (binary distribution, no open licence), Windows with an NVIDIA GPU only:
  needs one of the lab's idle GPUs passed through to the Windows VM (VFIO, needs root on the lab). Then benchmark
  it against the CPU solver on the same spots, including the textbook river game.

Sources:

- [Q&A: Is a Totally Uncensored LLM Technically Possible?](https://gist.github.com/danielrosehill/a631ca5a97718564944b5bdcb7fb12b6)
  (Daniel Rosehill, May 2025): the gist that named the models tested in the 2026-11-05 study.

Never commit credentials: the Windows login stays in `~/windows11/credentials.env` on the lab.
