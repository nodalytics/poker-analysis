# Kestrel, the C++ advisor: core, interfaces and connectors (phase D, started 2026-11-14)

**Kestrel** is the advisor's new name (2026-11-14), after the small hawk that hovers, watches and strikes. The
daemon is `kestreld` and the app is `Kestrel`; Kestrel is a standalone project at the repo root (`kestrel/`); the research that feeds it (training, models, the
Python reference) stays in `llm-research/`.

Phase D of [`llm-research/docs/2026-11-11-next-phases.md`](../../llm-research/docs/2026-11-11-next-phases.md), widened on 2026-11-14: besides advising, the
advisor **plays by itself** on platforms that offer a bot API, through **connectors described in a file** (YAML or
JSON) that are added after the build, without recompiling. You give it a connector file and a session plan, start
it, and leave it: it plays, logs every hand, and reports results with confidence intervals.

## Boundary: API platforms only

Connectors talk to a platform's **published API** (HTTP or WebSocket). That covers bot arenas such as Slumbot,
our own lab simulators, and any site that invites bots.
- **Real-money clients are out of scope.** ACR and similar sites ban bots and real-time assistance, and this project
  never automates them (the standing rule since phase A).
- A connector file has to declare `real_money: false` and `bots_permitted: true`, or the runner refuses to load it.
- Capture-only screenshots stay as planned: they pause while a real-money client is running.
- There is no screen reading and no clicking: a connector has no way to drive a desktop client.

## Layout

| Part | What | Depends on |
|---|---|---|
| `core/` (library) | Card and hand types; the single-table game state (a port of `ninemax_engine.Tables`: menus, legality, the big-blind ante, empty seats, side pots); ICM (`icm.py`); the push/fold charts (`results/ninemax/icm_pushfold*.json`); the 9-max network's features and forward pass (a port of `features()` and `NineNet`); bucket lookup through `bp` (`bp_bucket`); the heads-up blueprint through `bp` | C++20, nlohmann/json, and the 7-card evaluator and hand indexer at the pinned commits the trainer uses |
| `engines` | One interface, `decide(state) -> action distribution`. The router picks by game: 9-max tournament at 25bb or less goes to the tournament network (`models/ninemax/nm17`), or push/fold at 10bb or less; deep heads-up cash goes to the heads-up blueprint pack, which played Slumbot level or slightly ahead: +12.4 bb/100 [−16, +41], and +30.9 [−10, +72] with the defence floor, 5,000 hands each, luck-adjusted (`llm-research/docs/2026-11-08-blueprint-speed-accuracy.md`) | core |
| `api/` | Kestrel's own **local API** (REST, WebSocket, TCP, MCP; see below), bound to `127.0.0.1`: `POST /v1/decide`, `GET /v1/health`, `GET/POST /v1/connectors`, `POST /v1/sessions` (start), `GET /v1/sessions/{id}` (progress, bb/100 with an interval), `DELETE /v1/sessions/{id}` (stop) | cpp-httplib, IXWebSocket, standalone asio, nlohmann/json |
| `connectors/` | The runtime that reads connector files and plays sessions over HTTP, WebSocket or MCP | cpp-httplib (with OpenSSL for https), yaml-cpp |
| `kestreld` | The headless daemon: the API plus the session runner. **Set and forget** runs here | the above |
| `Kestrel` (app) | Qt 6 GUI: review, trainer, spot study, hand sync, connector and session panels, and capture-only screenshots. It is a client of the same core | Qt 6 |

## Interfaces: REST, WebSocket, TCP and MCP

Kestrel serves the same methods over four interfaces, all bound to `127.0.0.1`. The methods are `decide`,
`icm_equity`, `pushfold`, `list_connectors`, `start_session`, `session_status`, `stop_session` and `health`.

| Interface | Shape | For |
|---|---|---|
| REST | `POST /v1/decide`, `GET /v1/sessions/{id}`, ... | scripts, dashboards, anything with HTTP |
| WebSocket | `ws://127.0.0.1:<port>/v1/ws`, JSON-RPC 2.0 messages, plus pushed events (`hand_finished`, `session_progress`, `error`) | live views, the GUI, long-running clients |
| TCP socket | newline-delimited JSON-RPC 2.0 on a plain socket | the lowest overhead, other languages without HTTP, simulators |
| MCP | the same methods as MCP tools, over stdio or streamable HTTP | LLMs and agents |

Connectors use the same set of transports towards a platform: `http`, `websocket`, `tcp` (line-delimited JSON) and
`mcp`.

## Two GUIs: desktop and web

- **Desktop:** `Kestrel`, a Qt 6 app with tabs for review, trainer, spot study, hand sync, connectors and sessions,
  plus capture-only screenshots. It talks to the same core.
- **Web:** `kestreld` serves a web page at `http://127.0.0.1:<port>/`, built on the REST and WebSocket interfaces.
  From it you can ask for advice on a spot, manage connectors, start and stop sessions, and watch them play live.
  Hands, bb/100 and errors stream over the WebSocket. Reach it from another machine through an SSH tunnel.
- **The Poker AI Watch page** (`web/index.html`, served publicly) gets a **read-only** Kestrel tab: session
  progress and results. Controls (start, stop, connectors) stay on the local page, because the watch page is public
  and the backend is bound to `127.0.0.1` (decision of 2026-11-10). Remote controls would first need
  authentication.

## MCP, both ways

- **As a connector transport:** a connector can name an MCP server instead of an HTTP base URL. That covers a
  platform, or our own simulator, exposed as MCP tools. The file maps the hand steps to tool calls and their
  results to fields:
  `transport: {mcp: {command: [python, mcp_server.py]}}` (stdio) or `{mcp: {url: http://127.0.0.1:8770/mcp}}`
  (streamable HTTP), with `hand.start: {tool: new_hand, args: {...}}`, `hand.act: {tool: act, args: {...}}`.
- **As a server:** `kestreld --mcp` serves the advisor's own tools (`decide`, `icm_equity`, `pushfold`,
  `start_session`, `session_status`, `stop_session`, `list_connectors`). Any MCP client, an LLM agent among them,
  can then ask for advice or run sessions. It binds `127.0.0.1` only. The repo's Python MCP server
  (`../mcp_server.py`) stays the toolkit reference.

The same boundary applies: an MCP connector is still an API connector, under the same `real_money` and
`bots_permitted` rules.

## Connector files

A connector describes the transport (endpoints, requests, response fields) and names the **codec** that turns the
platform's notation into our game state. Transport is pure configuration. Notation needs a codec, because a platform's
action strings carry rules (Slumbot's `b200c/kk/b400` is street totals in chips), and those cannot be guessed from
an API spec. Built-in codecs cover the common notations: `acpc` (Slumbot and the ACPC servers), `structured` (JSON
actions with type and amount fields, mapped by path), and more as platforms need them. A platform with a notation of
its own needs one new codec. Everything else about it is a file.

```yaml
# connectors/slumbot.yaml
name: slumbot
real_money: false
bots_permitted: true          # slumbot.com publishes its API for bots
game: {variant: nlhe, players: 2, blinds: [50, 100], stack: 20000, reset_every_hand: true}
transport:
  base_url: https://slumbot.com/api/
  content_type: application/json
  retry: {tries: 6, backoff_s: 2}
  rate_limit_per_s: 2
session:
  login: null                  # optional: {path: login, body: {username: ${env:SLUMBOT_USER}, password: ${env:SLUMBOT_PASS}}, token: /token}
  token_field: /token          # JSON pointer, carried into the next request
hand:
  start: {path: new_hand, body: {token: ${token}}}
  act:   {path: act, body: {token: ${token}, incr: ${action}}}
  fields:                      # JSON pointers into each response
    action: /action
    my_seat: /client_pos
    hole: /hole_cards
    board: /board
    result: /winnings          # present when the hand is over
    error: /error_msg
codec: acpc                    # action strings: k c f b<street total>, streets separated by /
seat_map: {0: big_blind, 1: small_blind}
```

A session plan:

```yaml
connector: slumbot
engine: auto                   # or headsup-blueprint, nm17, pushfold
hands: 10000
workers: 1                     # parallel sessions; keep them few out of courtesy to free services
stop: {max_errors: 20, until: "2026-11-15T08:00"}
log: sessions/slumbot-{date}/  # hands.jsonl, summary.json (bb/100 with a bootstrap interval; luck-adjusted where possible)
```

## Verified against the Python reference

Each port gets test vectors written by the Python code it replaces, and the C++ build must reproduce them:
- the game state: random hands at 2–9 players, with and without the ante, as `ninemax_check.py` checks against
  PokerKit;
- features and network output: the same states, `features()` and logits to 1e-4;
- ICM: `icm.py`'s self-test values;
- the Slumbot codec: the action strings in `results/slumbot*/hands.jsonl`, replayed to the same pots and legal ranges
  as `slumbot_client.py`.

`kestreld --selftest` runs all of them.

## Order of work

1. **Core, headless (done 2026-11-14):** the network weights in a flat file (`scripts/ninemax_export_bin.py`), the test vectors, the
   game state, features, the forward pass and ICM. The self-test passes on the first build: 120 hands, 1,118
   decisions. Actors, legal actions and raise sizes match exactly; features to 1.6e-5, logits to 5e-5, payoffs to
   2.4e-4 chips, ICM to 7e-7. The heads-up engine (a port of `blueprint_bot.py` over the data pack, with its
   own bucket lookup) and the Slumbot codec reproduce 1,160 logged Slumbot decisions: nodes, buckets, strategy
   (to 5e-7) and raise sizes match exactly. Not yet ported: the defence floor (the +30.9 version).
2. **API and runner:** `kestreld`, the local REST API, the connector runtime, and the `acpc` codec with the Slumbot
   file. First real test: the heads-up engine against Slumbot, compared with the Python client's results.
3. **Tournament play:** our own tournament simulator, served through the same connector interface. No public
   tournament bot arena is known. Playing networks against each other and against the clones through the advisor's
   own API tests the whole path.
4. **GUIs:** the web page served by `kestreld` (first, since it needs no Qt), the read-only tab on the watch page,
   then the Qt 6 app.

## Build needs

CMake 3.28 or later and a C++20 compiler (g++ 13 on the lab, 14 here). The small libraries (cpp-httplib,
nlohmann/json, yaml-cpp) are fetched by CMake. HTTPS needs the OpenSSL headers and the GUI needs Qt 6. On Debian:
`sudo apt install libssl-dev qt6-base-dev`.
