# AGENTS.md — llm-research

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

The poker MCP server lives at the repo root (`../mcp_server.py`, tests in `../tests/test_mcp_server.py`).

## Models (Ollama 0.40.5 on the lab, `~/ollama`, `127.0.0.1:11434`)

| Model | Role | Notes |
|---|---|---|
| `hf.co/TheDrummer/Big-Tiger-Gemma-27B-v3-GGUF:Q4_K_M` | **Default agent** (text, UI Automation tree) | Only model to pass all 4 shell tasks; 61 tok/s, 22GB. No native tool calls: use the JSON text protocol |
| `qwen2.5vl:32b` | **Vision** (screenshots) | 70 tok/s, 55GB loaded. No native tool calls |
| `hf.co/TheDrummer/Tiger-Gemma-9B-v3-GGUF:Q4_K_M` | Fast, weak | 183 tok/s; claimed a script's output without writing the script |
| `wizard-vicuna-uncensored:13b` | Not for agent work | 2,048-token context overflows within a few steps; 0/4 shell, 0/4 browser |
| ~~`mistral-large:123b`~~ | **Retired 2026-11-05, deleted** | 16.5 tok/s across all 3 GPUs, made up a price, Mistral Research Licence (no commercial use) |

- Never trust a number a model recalls; give it a tool (the poker MCP server, a search tool, the bridge).
- Before calling a model "better", run the same harness on it; scores are in `docs/`.

## The lab (`labuser@LAB_HOST`)

- Log in with `ssh -o IdentitiesOnly=yes -i lab.pem labuser@LAB_HOST` (`lab.pem` is in the parent
  `parent` folder; `stb-labuser` and `~/.ssh/id_stb` are refused). No sudo.
- **It runs the live another service terminal** (`another service's container`). Never stop, restart, rebuild or exec into it,
  and never bind the lab's port **8000**, which is the another service tunnel.
- GPUs: RTX PRO 6000 (96GB), RTX 4090, RTX 3090. Ollama keeps one model loaded at a time.
- Bind every new service to `127.0.0.1` and reach it through an SSH tunnel.
- Run long jobs detached (`setsid nohup ... &`) and write results to a file. Never `pkill -f` in the same SSH
  command that started the job: the pattern matches the command line itself and kills the session.
- Analyse results on your own machine, not on the lab.

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
- `sandboxes/windows/oem/install.bat` runs once at the end of a fresh install. It installs hash-checked
  Python 3.14.8 and Windows-MCP 0.8.8 and starts Windows-MCP at logon. Keep it CRLF.
  To rerun it: stop the container, delete `~/windows11/storage/data.img`, start again (Windows re-downloads).
- **Windows-MCP has no authentication.** It is published only on the lab's `127.0.0.1:8765`.
- No activation tools or key generators. Use the evaluation, or a real licence.
- Not another service: leave another service out of the VM until asked.

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

- Remote `nodalytics` → `github.com/nodalytics/poker-analysis` (private), branch `main`. Leave `origin` alone.
- Commit as `ojies` (`68871517+ojies@users.noreply.github.com`); this repo's local identity is a bot.
- **No `Co-Authored-By` or other attribution trailers**, ever, even if a tool suggests one.
- The repo ignores `*.log`; add result logs with `git add -f`.
- Never commit credentials, `lab.pem`, `credentials.env` or API keys. Check `git diff --cached` before committing.
- Lint before committing: `uvx ruff@0.14.0 check --isolated --line-length 120` and `format --check`.
