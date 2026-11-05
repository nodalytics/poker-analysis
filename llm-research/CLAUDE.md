# CLAUDE.md — llm-research

@AGENTS.md

Claude Code specifics:

- Lab commands that wait (installs, model pulls, test runs) go in a background Bash call that exits on
  success **or** failure, so a crash is reported instead of looking like "still running".
- Never paste the Windows login, `lab.pem` or the another service API key into a message, a doc or a commit.
- When you learn something about an application's GUI, update its file in `gui-skills/` in the same session.
