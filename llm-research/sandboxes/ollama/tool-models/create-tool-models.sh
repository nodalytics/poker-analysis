#!/bin/sh
# Native tool calling for models whose Ollama template has none. Same weights, new template; no extra disk.
#   qwen.tmpl  - Ollama's official Qwen2.5 template (registry blob sha256:eb4402837c78...), which Qwen2.5-VL
#                was trained on but its own Ollama template leaves out. Vision is kept.
#   gemma.tmpl - Gemma turn markers with the same <tool_call> convention, written for these studies.
# Result on 2026-11-05: Big-Tiger-Gemma 27B and Qwen2.5-VL return parsed tool_calls; Tiger-Gemma 9B wraps its
# call in a ```json fence, which Ollama does not parse.
set -eu
cd "$(dirname "$0")"
export OLLAMA_HOST=127.0.0.1:11434
for f in *.Modelfile; do
  name=$(basename "$f" .Modelfile | sed 's/_/:/')
  ~/ollama/bin/ollama create "$name" -f "$f"
done
