# AGENTS.md — research

Instructions for any agent (or person) working in this folder. Read this first; it saves re-discovering the lab.

## What this folder is

Evaluations of local LLMs as agents: speed and refusals, shell control (Desktop Commander), browser control
(Playwright MCP), and GUI control of a Windows 11 VM (Windows-MCP). Each study keeps its write-up, the scripts
that produced it, the sandbox it ran in and its raw output, so anyone can rerun it.

| Path | Contents |
|---|---|
| `docs/` | One write-up per study, named `YYYY-MM-DD-topic.md` |
| `scripts/` | Test harnesses; each grades against the machine, never against the model's claim |
| `sandboxes/` | Docker builds and the Windows VM script, including `windows/oem/install.bat` |
| `gui-skills/` | Navigation notes per application, so a new model or a fresh session does not re-explore |
| `results/raw/`, `results/logs/` | JSON output (every answer and tool call) and console logs |
| `media/` | Small GIF evidence of GUI runs, created with the first one (rules below) |

The poker MCP server is archived with the milestone-1 toolkit (`../archive/mcp_server.py`, tests in
`../archive/tests/test_mcp_server.py`).

## Models (Ollama 0.40.5 on the lab, `~/ollama`, `127.0.0.1:11434`)

| Model | Role | Notes |
|---|---|---|
| `hf.co/TheDrummer/Big-Tiger-Gemma-27B-v3-GGUF:Q4_K_M` | **Default agent** (text, UI Automation tree) | Only model to pass all 4 shell tasks; 61 tok/s, 22GB. No native tool calls: use the JSON text protocol |
| `qwen2.5vl:32b` | **Vision** (screenshots), **best browser agent** | 3/4 browser tasks, the only form fill; 70 tok/s, 55GB loaded. No native tool calls |
| `hf.co/TheDrummer/Tiger-Gemma-9B-v3-GGUF:Q4_K_M` | Fast, weak | 183 tok/s; claimed a script's output without writing the script |
| `wizard-vicuna-uncensored:13b` | Not for agent work | 2,048-token context overflows within a few steps; 0/4 shell, 0/4 browser |
| `big-tiger-gemma-tools:27b`, `qwen2.5vl-tools:32b`, `tiger-gemma-tools:9b` | **Native tool calls** | Same weights, tool templates from `sandboxes/ollama/tool-models/`; 27B and Qwen 4/4 on shell tasks, 9B 2/4 |
| ~~`mistral-large:123b`~~ | **Retired 2026-11-05, deleted** | 16.5 tok/s across all 3 GPUs, made up a price, Mistral Research Licence (no commercial use) |

- Never trust a number a model recalls; give it a tool (the poker MCP server, a search tool, the bridge).
- Before calling a model "better", run the same harness on it; scores are in `docs/`.

## Python: uv

This folder is a uv project (`pyproject.toml`, `uv.lock`). Use `uv run python scripts/<name>.py` and
`uv run ruff check scripts` / `uv run ruff format scripts`; add dependencies with `uv add`, pinned. The agent
harnesses are standard library only, so they also run with a bare `python3` on the lab.
Ollama setup (install, serve, models, tool templates) is in `sandboxes/ollama/`.

## The lab

- The address, login and key are not in the repo: they are in `../.secrets/lab.env` (the `.secrets/` folder is ignored by
  git), which sets `LAB` (user@host) and `LAB_KEY` (the key file): `. .secrets/lab.env` from the repo root. Log in with
  `ssh -o IdentitiesOnly=yes -i "$LAB_KEY" "$LAB"`. No sudo.
- **Other services run there.** Never stop, restart, rebuild or exec into a container you did not start, and never
  bind the lab's port **8000**, which another service uses.
- GPUs: RTX PRO 6000 (96GB), RTX 4090, RTX 3090. Ollama keeps one model loaded at a time.
- Bind every new service to `127.0.0.1` and reach it through an SSH tunnel.
- Run long jobs detached (`setsid nohup ... &`) and write results to a file. Never `pkill -f` in the same SSH
  command that started the job: the pattern matches the command line itself and kills the session.
- Analyse results on your own machine, not on the lab.
- **Other people's jobs run there too.** Leave every process you did not start alone.
- Ollama there runs with `OLLAMA_CONTEXT_LENGTH=8192` and up to 6 models resident; without the context cap
  Qwen2.5-VL took 55GB instead of 22GB.
- In a remote command, `a && b && setsid nohup c &` backgrounds the whole chain and holds the SSH session
  open; wrap the detached part in parentheses: `a && (setsid nohup c ... &)`.

## Sandboxes and grading

- Agent tests run in throwaway containers with `--network none`, no host mounts, 1 CPU, 1GB RAM,
  `--cap-drop ALL`, an unprivileged user. Keep it that way; a model with a shell must not reach the lab.
- Grade every task against the environment (`docker exec`, the test site's own logs, the bridge), never the
  model's answer alone. Two models claimed success without doing the work, and only this caught them.
- Record harness choices that make a test harder or easier (truncated descriptions, missing snapshots) as
  caveats in the write-up.

## Windows 11 VM (`win11-gui`)

- `sandboxes/windows/run-win11.sh`: Windows 11 Enterprise Evaluation (free, 90 days), `dockurr/windows:6.06`,
  8 cores, 16GB (needs `RAM_CHECK=N`), 128GB. Web desktop `127.0.0.1:8006`, RDP `127.0.0.1:3389`.
- Login: `~/windows11/credentials.env` on the lab. **Never commit it, print it or put it in a doc.**
- **One command brings it up:** `sh sandboxes/windows/windows-up.sh [--reinstall]` from your machine copies the
  files to the lab and runs `provision.sh`, which creates the login and the Windows-MCP key, stages `oem/`, starts
  the VM, and only succeeds once an authenticated MCP handshake lists Windows-MCP's tools.
- `oem/install.bat` runs once at the end of a fresh install: hash-checked Python 3.14.8, Windows-MCP 0.8.8, and a
  logon task running `oem/start-mcp.bat`, which it also starts immediately. Keep both files CRLF.
- **Windows-MCP requires a key** (it refuses a network bind without one). The key is `~/windows11/oem/mcp.key` on the
  lab; send `Authorization: Bearer <key>`. Port: the lab's `127.0.0.1:8765` only. Never print or commit the key.
- **The evaluation expires after 90 days, and it is rebuilt, not activated.** Cron on the lab runs
  `rebuild-if-expiring.sh` daily; it reinstalls after 80 days. Anything driving the VM must `touch
  ~/windows11/in-use` for the run and delete it after, so a rebuild never lands mid-test. Keep nothing in the VM
  that is not reproducible from this repo.
- No activation tools, KMS emulators (AutoKMS, KMSAuto) or key generators: piracy, and common malware carriers.
  Use the evaluation rebuild, or a real licence applied with `slmgr /ipk` when there is one.

## Arena: chess and poker ladder (`scripts/arena.py`)

- Every model enters twice: `#learn` reads and rewrites its own `notes/<model>/skill-<game>.md` and `memory.md`
  after every match; `#frozen` never sees notes. **Learner minus frozen twin Elo is the measure of
  self-improvement**; without the twin, a rising Elo could be luck. Twins share a poker table (same cards).
- Poker tables also seat non-learning anchors (`poker_selfplay.PERSONAS`, capped at 2 a table: Bob always, one
  rotating), Monte (eval7 Monte Carlo), Sol (TexasSolver heads-up on turn and river, Monte otherwise), and
  `qwen2.5vl-tools:32b#toolkit`, which calls the poker-analysis toolkit as tools.
- Chess ends only on automatic outcomes (fivefold repetition, not threefold); repeating moves are marked in the
  move list. Claiming threefold automatically ended 11 of the first 12 games in knight shuffles.
- An Ollama error mid-game is an invalid move, and a failed reflection keeps the old notes; never let one model's
  context overflow stop the ladder. Wizard-Vicuna (2k context) gets notes trimmed to 500 characters.
- Analysis: `scripts/plots.py` (charts into `media/plots/`) and `scripts/patterns.py` (behaviour and betting
  patterns into `results/patterns/`), both from the arena directory.

## Context harness (`scripts/harness.py`)

Long tool output is paged through `more_output`, conversations are compacted by the model's own summary before
they overflow, and learner memories are condensed instead of truncated. Agent harnesses take `--harness`; always
keep a baseline run without it, so its effect is measured.

## GUI skills (`gui-skills/`)

- **Before driving an application, load its file** from `gui-skills/`. If there is none, copy
  `gui-skills/TEMPLATE.md` and fill it in as you go.
- **After a run, update the file** with what worked: UI Automation names, keyboard shortcuts, dialogs that
  appeared, what failed and why. Mark each entry `verified YYYY-MM-DD` or `unverified`.
- Prefer keyboard shortcuts and UI Automation names over screen coordinates; coordinates break with
  resolution, scaling and window position.

## Media evidence (GIFs and videos)

GUI runs should leave visual evidence, kept small because it is pushed to GitHub and stays in git history
even after deletion.

- **Format:** GIF for inline viewing on GitHub. At most **2MB**, **20 seconds**, **8 fps**, **960px wide**,
  128 colours. Cut idle time. Build it from the step screenshots of the run rather than screen-recording:
  ```
  ffmpeg -framerate 2 -i step_%03d.png -vf "scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer" -loop 0 run.gif
  ```
- **Name:** `media/YYYY-MM-DD-<study>-<model>-<task>.gif`, and link it from the study's write-up.
- **Keep only the latest** GIF per study, model and task. Delete the superseded file in the same commit
  that adds its replacement, and delete anything no write-up links to.
- Raw recordings, full-size screenshots and anything over the limits stay on the lab, not in git.
- Prune old raw results the same way when a study is rerun: keep the latest run, note in the write-up
  which run it is.

## Git

- Remote `nodalytics` → `github.com/nodalytics/poker-analysis` (**public**), branch `main`; the only remote.
- Commit as `ojies` (`68871517+ojies@users.noreply.github.com`); this repo's local identity is a bot.
- **No `Co-Authored-By` or other attribution trailers**, ever, even if a tool suggests one.
- The repo ignores `*.log`; add result logs with `git add -f`.
- Never commit credentials, the lab key, `credentials.env`, anything in `.secrets/` or API keys (the repo is public). Check `git diff --cached` before committing.
- Lint before committing: `uv run ruff check scripts` and `uv run ruff format --check scripts` (ruff 0.14.0, line length 120).
