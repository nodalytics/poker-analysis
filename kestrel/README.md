<p align="center"><img src="assets/logo.svg" width="112" alt="Kestrel"></p>

# Kestrel

Kestrel is the C++ poker advisor and self-playing bot (phase D). It covers 9-max tournaments, with heads-up as a second mode. It plays by itself on platforms that offer a bot API, through connectors described in a file. It serves REST, WebSocket, TCP and MCP on `127.0.0.1`, with a web page and a Qt app on top.

Design, boundary and order of work: [`docs/design.md`](docs/design.md). Kestrel stands alone at the repo root; the
research that feeds it (training, the models, the Python reference) is in `llm-research/`.

## Start it

Double-click `kestreld` (`kestreld.exe` on Windows) in the release's `bin/` folder, or run it with no arguments. It starts the app and opens it in your browser. If it's already running, it just opens it. On Linux, `share/kestrel/install-desktop.sh` adds Kestrel to the applications menu.

It finds its web app, logo and connectors next to itself, and your own connectors and models in `~/.config/kestrel/connectors` and `~/.local/share/kestrel/models`. The heads-up data pack goes in `~/.local/share/kestrel/pack`, or set `KESTREL_PACK`.

**From other devices on your network:** Kestrel listens on all interfaces (0.0.0.0) when started by a double-click, `kestreld start` or the desktop app. The "On your network" button shows the links to open on a phone or another PC. Each link carries an access token, and devices without it are refused. The WebSocket and TCP interfaces stay on this computer. For this computer only, put `{"lan": false}` in `~/.config/kestrel/settings.json` (or untick it in the desktop app's Settings); `kestreld serve` listens on the network only with `--lan on`.

## Talk to it

Kestrel serves the same methods on four interfaces, all on this computer:

| Interface | Address |
|---|---|
| REST | `http://127.0.0.1:8790/v1/...` |
| WebSocket | `ws://127.0.0.1:8791/`: JSON-RPC 2.0; send `{"method": "subscribe"}` for the live events |
| TCP | `127.0.0.1:8792`: one JSON-RPC request per line |
| MCP | `kestreld mcp`: for LLMs and agents |

`POST /v1/icm {stacks, payouts, model}` gives prize equity under ICM (`mh`, Malmuth-Harville, what the networks read) or Malmuth-Weitzman (`mw`), for analysis. The methods are `health`, `list_connectors`, `list_engines`, `start_session`, `list_sessions`, `session_status`, `stop_session`, `start_practice`, `practice_act`, `decide` and `icm_equity`. To add Kestrel to Claude Code: `claude mcp add kestrel -- /path/to/kestreld mcp`.

## Play

```
kestreld check kestrel/connectors/slumbot.yaml        # what it talks to and which KESTREL_* variables it reads
kestreld play kestrel/connectors/slumbot.yaml --hands 1000 --out sessions/slumbot --pack llm-research/blueprint/pack
kestreld list                                          # every connector in ~/.config/kestrel/connectors
kestreld play kestrel/connectors/slumbot.yaml --hands 100 --out sessions/ts --pack llm-research/blueprint/pack \
    --solver texassolver --texassolver tools/TexasSolver/build/console_solver \
    --texassolver-resources tools/TexasSolver/resources   # model + solver: the blueprint with TexasSolver
kestreld serve --pack llm-research/blueprint/pack --connectors kestrel/connectors --web kestrel/web --assets kestrel/assets
                                                       # the web app at http://127.0.0.1:8790/ with live tables
```

**9-max tables.** A table connector (`codec: structured`) is played by a 9-max model at 2–9 handed tables, any bet sizes. The test platform is a PokerKit tournament against rule bots on this computer:

```
uv --project llm-research run python kestrel/testplatform/ninemax_platform.py &       # 127.0.0.1:8793
kestreld play kestrel/connectors/testplatform.yaml --model llm-research/models/ninemax/nm25 \
    --tournaments 20 --hands 100000 --out sessions/test9 --pack llm-research/blueprint/pack
```

Other connectors: `pokerpi.yaml` (a self-hosted bot tournament server, long polling), `testplatform-ws.yaml` (the test platform over a WebSocket, speaking Open Poker's protocol: `uv run --with websockets==14.1 python kestrel/testplatform/ws_platform.py`) and `openpoker.yaml` (a draft until the account exists). The connector format is in [`docs/design.md`](docs/design.md#connector-files).

Advice for one spot, in the platform's own chips (`to` is the street total):

```
curl -s -X POST http://127.0.0.1:8790/v1/decide -H 'Content-Type: application/json' -d '{"game": "ninemax",
  "players": [{"id": "a", "stack": 1500}, {"id": "me", "stack": 900}, {"id": "c", "stack": 2200}],
  "button": "a", "me": "me", "sb": 25, "bb": 50, "ante": 50, "hole": ["Ah", "Kd"],
  "actions": [{"player": "a", "street": 0, "kind": "raise", "to": 125}], "payouts": [0.5, 0.3, 0.2]}'
```

The web app shows every running table live, each on its own screen. Switch screens with tabs or keys `1`–`9`, press `g` for a grid of all tables, `n` to start a new table and `t` to change theme. Each screen shows:
- the cards, the pot and the betting as it happens;
- Kestrel's decision, with its probabilities;
- the session's results.

## The desktop app

`Kestrel` (built beside `kestreld` when Qt 6 is installed: `qt6-base-dev` on Debian) shows the same tables as the web app, drawn live, with a grid of all of them and your moves at a practice seat. It installs models, checks and adds connectors, and is the admin for Kestrel's settings: ports, LAN access and its token, the default model, TexasSolver and folders, saved to `settings.json` and applied by restarting the server. It starts Kestrel itself when it isn't running. `Kestrel --tab settings` opens on a tab.

## Build and self-test

```
cmake -S kestrel -B kestrel/build && cmake --build kestrel/build -j
kestrel/build/kestrel_selftest --vectors kestrel/testdata --model llm-research/models/ninemax/nm17 \
    --pack llm-research/blueprint/pack          # or: ctest --test-dir kestrel/build
```

The self-test replays reference data written by the Python code in `llm-research/scripts/`: `advisor_vectors.py` for 9-max, and `kestrel_hu_vectors.py` for 1,160 logged Slumbot decisions. For each decision it checks:
- the actor and the street;
- the legal actions and the raise sizes;
- the features and the network's output.

It also checks the payoffs and ICM, and for heads-up the blueprint's node, bucket, strategy and raise sizes. For platform tables it re-describes 1,922 self-played decisions as a platform would and must get the network's own answer exactly. It prints `PASS` only if all of them match.

The data it reads (the network `models/ninemax/nm17/net.bin` and the heads-up pack `blueprint/pack/`) defaults to `../llm-research`. Set `-DKESTREL_DATA=...` to point elsewhere.

| Path | Contents |
|---|---|
| `core/` | game state (port of `ninemax_engine.Tables`), ICM, network features and forward pass, heads-up blueprint and buckets |
| `connectors/` | connector files, the codecs (`acpc` heads-up, `structured` tables), the session runners, the server and the JSON-RPC gateway |
| `testplatform/` | a PokerKit 9-max tournament platform for testing table connectors |
| `docs/design.md` | the design |
| `assets/` | the logo: `logo.svg` (app icon), `logo-mark.svg` (the bird), `logo-animated.svg` (hovering, for busy states) |
| `connectors/*.yaml` | connector files; `slumbot.yaml` is the worked example |
| `tools/selftest.cpp` | the self-test |
| `testdata/` | reference vectors from `scripts/advisor_vectors.py` |
