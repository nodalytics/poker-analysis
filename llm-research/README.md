# LLM research

Local LLM evaluations run on the lab, with the scripts, sandboxes and raw results needed to rerun them.

| Path | Contents |
|---|---|
| `docs/` | Write-ups, one per study, dated |
| `scripts/` | Test harnesses (`llm_test.py`, `dc_agent_test.py`, `browser_agent_test.py`) |
| `sandboxes/` | Docker builds for the sealed test environments, and the Windows GUI VM script |
| `results/raw/` | JSON output from each run, including every model answer and tool call |
| `results/logs/` | Console logs from each run |

Studies:

- [2026-11-05 — Local "uncensored" LLM evaluation](docs/2026-11-05-local-llm-evaluation.md):
  speed and refusals, Desktop Commander control, browser control; why Mistral Large was dropped.

Never commit credentials: the Windows login stays in `~/windows11/credentials.env` on the lab.
