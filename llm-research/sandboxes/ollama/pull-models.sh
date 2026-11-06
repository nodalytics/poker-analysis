#!/bin/sh
# The models in use (see ../../AGENTS.md); the last three are embedding models for the RAG tests. Ollama verifies each download's sha256.
# Retired: mistral-large:123b (deleted 2026-11-05; see docs/2026-11-05-local-llm-evaluation.md).
set -eu
export OLLAMA_HOST=127.0.0.1:11434
for m in \
  hf.co/TheDrummer/Tiger-Gemma-9B-v3-GGUF:Q4_K_M \
  wizard-vicuna-uncensored:13b \
  hf.co/TheDrummer/Big-Tiger-Gemma-27B-v3-GGUF:Q4_K_M \
  qwen2.5vl:32b \
  nomic-embed-text \
  mxbai-embed-large \
  bge-m3
do
  nice -n 15 ~/ollama/bin/ollama pull "$m"
done
~/ollama/bin/ollama list
