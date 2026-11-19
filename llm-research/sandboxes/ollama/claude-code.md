# Claude Code with a local Ollama model (2026-11-19)

**Claude itself cannot run in Ollama:** Anthropic does not publish Claude's weights. What can run is the **Claude Code
tool** (the CLI and its agent loop) with an open model served by Ollama doing the thinking. Claude Code talks to an
Anthropic-style Messages API, so it only needs an endpoint that speaks it.

## Two ways to connect

1. **Directly**, if the Ollama version serves the Anthropic Messages API (`POST /v1/messages`):

   ```sh
   ANTHROPIC_BASE_URL=http://127.0.0.1:11434 ANTHROPIC_API_KEY=none claude --model big-tiger-gemma-tools:27b
   ```

2. **Through a translating proxy** (LiteLLM), if it does not: the proxy speaks the Anthropic API to Claude Code and
   Ollama's own API to the model.

   ```sh
   uv tool install 'litellm[proxy]'
   litellm --model ollama/big-tiger-gemma-tools:27b --port 4000      # on the lab, bound to 127.0.0.1
   ANTHROPIC_BASE_URL=http://127.0.0.1:4000 ANTHROPIC_API_KEY=none claude
   ```

The lab's Ollama listens on `127.0.0.1:11434` only (`serve.sh`), so from another machine use an SSH tunnel, as for every
lab service: `ssh -N -L 11434:127.0.0.1:11434 <lab>`, then the commands above against `127.0.0.1` locally.

## Which model

Use one of the **tool-calling builds** (`tool-models/`): Claude Code works by calling tools (read, edit, run commands),
and a model without native tool calls cannot drive it. From this folder's agent studies (`docs/`):

| Model | Shell tasks passed | Note |
|---|---|---|
| `big-tiger-gemma-tools:27b` | 4 of 4 | the default agent model |
| `qwen2.5vl-tools:32b` | 4 of 4 | also reads screenshots |
| `tiger-gemma-tools:9b` | 2 of 4 | fast, weak |

What to expect: the Claude Code tool, steered by a much weaker model. Our studies caught two models claiming success
without doing the work, so grade a local-model session against the machine (tests, files, the environment), never
against its own report. Fine for cheap, simple jobs; not a replacement for Claude on multi-step engineering.

## Status on the lab, 2026-11-19

- Our Ollama (0.40.5, `~/ollama`) is **not running**: it did not come back after the two lab reboots of 2026-11-18
  (`serve.sh` restarts it). Nothing listens on 11434.
- Whether 0.40.5 answers `POST /v1/messages` directly is **untested** for that reason. When it runs:

  ```sh
  curl -s 127.0.0.1:11434/v1/messages -H 'content-type: application/json' -H 'x-api-key: none' \
    -H 'anthropic-version: 2023-06-01' \
    -d '{"model":"tiger-gemma-tools:9b","max_tokens":20,"messages":[{"role":"user","content":"Say hi"}]}'
  ```

  A JSON reply with a `content` list means option 1 works; a 404 means use the proxy (option 2).
- A separate `ollama serve` process runs on the lab without a port on the host (likely another user's container).
  It is not ours; leave it alone.
