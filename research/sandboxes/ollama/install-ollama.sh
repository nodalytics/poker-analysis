#!/bin/sh
# Install Ollama 0.40.5 in user space (~/ollama), no root. This is what runs on the lab.
# 0.40.5 was published 2026-10-14; the checksum is from the release's sha256sum.txt and is checked here.
set -eu
V=0.40.5
SHA=dc5eabc292f9949546c90a97c820f924b035e05243b0de13c7f366182562169c
mkdir -p ~/ollama/dl ~/ollama/models
cd ~/ollama/dl
curl -fsSLO "https://github.com/ollama/ollama/releases/download/v$V/ollama-linux-amd64.tar.zst"
echo "$SHA  ollama-linux-amd64.tar.zst" | sha256sum -c -
tar --zstd -xf ollama-linux-amd64.tar.zst -C ~/ollama
~/ollama/bin/ollama --version
