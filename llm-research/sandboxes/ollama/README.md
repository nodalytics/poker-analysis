# Ollama setup

How the lab's model server was set up for these studies, so it can be rebuilt from nothing.

| File | Does |
|---|---|
| `install-ollama.sh` | Installs Ollama 0.40.5 in `~/ollama`, checksum-verified, no root |
| `serve.sh` | Starts it on `127.0.0.1:11434`; `MAX_LOADED=4` for self-play |
| `pull-models.sh` | Pulls the four models in use |
| `tool-models/` | Templates and Modelfiles that add native tool calling; `create-tool-models.sh` builds them |
| `docker-compose.yml` | Docker alternative, **untested**, not what the studies used |

Order: `install-ollama.sh`, `serve.sh`, `pull-models.sh`, then `tool-models/create-tool-models.sh`.
