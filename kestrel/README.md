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
kestreld play kestrel/connectors/testplatform.yaml --model llm-research/models/ninemax/nm25 --solver search \
    --search-samples 1024 --tournaments 20 --hands 100000 --out sessions/test9s --pack llm-research/blueprint/pack
```

**Search** (`--solver search` or `search-robust`, or the Solver box of a new 9-max table) plays nm25 with search at
decision time, before the flop and in pots of up to three players after it: about 2 s a decision on 8 cores at the
default 1,024 deals (`search_samples` in the settings, 1,024 to 4,096: with 256 deals search lost 5-6.5 bb/100 to
the plain network, because its estimates of each move's value are too noisy), 3× that for `search-robust`. Without it, a 9-max table plays
the network's own policy.

Search assumes every opponent plays like the network. That holds against strong, balanced players (+4 to +6 bb/100
over the plain network against nm25-like tables) and fails against loose ones: on the test platform's calling
stations it lost -96.5 bb/100 where the plain network won +75.5. Against unknown or loose opponents use the plain
network, `search-exploit` (plays each opponent as its profile says) or `search-robust`.

**Models** (`--model`, or the Model box of a new table):

| Model | Game | Notes |
|---|---|---|
| `nm25` | 9-max tournaments, 10-25bb, ICM | the default; never shoves a real stack first in from the first two seats (Rules::early_shove_bb, 15bb) |
| `nm17` | 9-max tournaments | the previous champion |
| `nm30` | progressive knockout (bounty) tournaments | calls wider as an opponent's bounty grows (42.9%, 44.0%, 46.0% for none, small and large) |
| `nm37` | 6-max cash, 50-250bb | **experimental**: a 30-minute exploiter still wins +5.6 bb/100 against it |

Other connectors: `pokerpi.yaml` (a self-hosted bot tournament server, long polling), `testplatform-ws.yaml` (the test platform over a WebSocket, speaking Open Poker's protocol: `uv run --with websockets==14.1 python kestrel/testplatform/ws_platform.py`) and `openpoker.yaml` (a draft until the account exists). The connector format is in [`docs/design.md`](docs/design.md#connector-files).

Advice for one spot, in the platform's own chips (`to` is the street total):

```
curl -s -X POST http://127.0.0.1:8790/v1/decide -H 'Content-Type: application/json' -d '{"game": "ninemax",
  "players": [{"id": "a", "stack": 1500}, {"id": "me", "stack": 900}, {"id": "c", "stack": 2200}],
  "button": "a", "me": "me", "sb": 25, "bb": 50, "ante": 50, "hole": ["Ah", "Kd"],
  "actions": [{"player": "a", "street": 0, "kind": "raise", "to": 125}], "payouts": [0.5, 0.3, 0.2]}'
```

Add `"solver": "search"` (or `search-robust`, and optionally `"samples": 512`) to search the spot; each option then
also carries `q_bb`, its searched value in big blinds of ICM equity, and the answer says `searched`, `samples` and
`ess` (the effective number of deals after weighting by the opponents' moves).

The web app shows every running table live, each on its own screen. Switch screens with tabs or keys `1`–`9`, press `g` for a grid of all tables, `n` to start a new table and `t` to change theme. Each screen shows:
- the cards, the pot and the betting as it happens;
- Kestrel's decision, with its probabilities;
- the session's results.

## The database

Every hand Kestrel plays, at a platform's table or a practice table, goes into one SQLite file,
`~/.local/share/kestrel/kestrel.db` (`db` in the settings to move it, `"db": false` to turn it off), with its session
and the opponents' statistics. Each session's `hands.jsonl` and `summary.json` are still written as an export.

| Table | Rows |
|---|---|
| `sessions` | one per session or practice table: kind, connector, model, solver, folder, start and end, `summary` (JSON) |
| `hands` | one per hand: session, order, connector, the platform's hand id, model, solver, `result_bb`, and `data`, the whole record (JSON) |
| `opponents` | one per platform and username: hands and `stats` (JSON: VPIP, raise, 3-bet, fold to c-bet, ...) |
| `spots` | one per decision at a 9-max table: its hand, model, street, position, the move played, and `vec`, the network's embedding of the spot (128 numbers, length 1) |

```
kestreld db stats                                  # how much it holds
kestreld db sessions                               # every session, newest first, with hands and bb/100
kestreld db import ~/.local/share/kestrel/sessions # history recorded before the database (each folder once)
sqlite3 ~/.local/share/kestrel/kestrel.db "SELECT json_extract(data, '$.hole'), json_extract(data, '$.action'),
  result_bb FROM hands WHERE connector = 'testplatform' ORDER BY result_bb LIMIT 10"   # the biggest losses
```

Over the API: `GET /v1/db` (counts), `GET /v1/db/sessions?limit=&offset=`, `GET /v1/hands?session=&connector=&limit=&offset=`.

**Similar spots and opponent profiles.** `GET /v1/spots/similar?spot=ID&k=10`, or `"similar": k` in a `POST /v1/decide`,
returns the stored spots the network sees as most alike (cosine similarity of its embeddings), with how Kestrel
played them and the result; the desktop app's Spot study tab lists them under its answer. Each opponent gets the
closest player type (nit, tag, lag, maniac, station, loose-passive) by cosine similarity of their statistics, shown
from 30 hands in the HUD line and the Opponents tab, and `GET /v1/opponents?platform=P&player=U` adds the players
met before who play most like them. Both are for study: Kestrel's play does not change on them
([design](docs/design.md#embeddings-and-similarity-2026-11-16)).
The database uses write-ahead logging, so it can be read (sqlite3, a notebook) while Kestrel plays. SQLite is built
into Kestrel, so there is nothing to install.


`Kestrel` (built beside `kestreld` when Qt 6 is installed: `qt6-base-dev` on Debian) shows the same tables as the web app, drawn live, with a grid of all of them and your moves at a practice seat. It installs models, checks and adds connectors, and is the admin for Kestrel's settings: ports, LAN access and its token, the default model, 9-max search deals per decision, TexasSolver and folders, saved to `settings.json` and applied by restarting the server. It starts Kestrel itself when it isn't running. `Kestrel --tab settings` opens on a tab.

## Docker (the lab)

`kestrel/Dockerfile` builds the server with nm25, nm17, nm30 and nm37 and the web app; CI (`kestrel-docker.yml`)
self-tests every image (nm17, nm30's bounties, nm37's cash) and pushes it to `ghcr.io/nodalytics/kestrel`.
`kestrel/docker/compose.yml` runs it beside the test platform for unattended bot sessions:

```
sh kestrel/docker/deploy.sh            # the image CI pushed (main), after a one-time docker login ghcr.io
sh kestrel/docker/deploy.sh --local    # or build it on this machine
```

It listens on the host's `127.0.0.1:8890` only (the inside port is the same: the server checks the Host header,
port included). Everything it keeps is in the `kestrel-data` volume: `kestrel.db`, sessions and the access token.
Every visitor needs the token, since requests reach the container from the Docker network, not from its own
loopback. A domain in front of it (a tunnel or proxy) needs `--public-host NAME`: that name is served with the token
only, and writes only from `https://NAME`. A new image ends the bot sessions in play; start them again.

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

`ctest` also replays nm30's vectors (bounties) and nm37's (6-max cash, deep stacks, three empty seats).

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
