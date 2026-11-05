#!/bin/sh
# Start the user-space Ollama server on 127.0.0.1:11434, detached, at low CPU priority.
# MAX_LOADED: 1 for single-model tests (honest load times), 4 for self-play (all four models resident;
# 7 + 9 + 22 + 55GB fits across the lab's 96 + 24 + 24GB GPUs).
# Stop it with: kill "$(pgrep -u "$USER" -x ollama)"   (-x matches the process name, never this command line)
set -eu
MAX_LOADED=${MAX_LOADED:-1}
KEEP_ALIVE=${KEEP_ALIVE:-2m}
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_MODELS="$HOME/ollama/models" OLLAMA_MAX_LOADED_MODELS="$MAX_LOADED" \
  OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE="$KEEP_ALIVE" \
  setsid nohup nice -n 15 "$HOME/ollama/bin/ollama" serve >> "$HOME/ollama/serve.log" 2>&1 < /dev/null &
for i in $(seq 1 30); do curl -fs 127.0.0.1:11434/api/version && echo && exit 0; sleep 1; done
echo "ollama did not start; see ~/ollama/serve.log" >&2
exit 1
