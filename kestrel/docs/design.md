# Kestrel, the C++ advisor: core, interfaces and connectors (phase D, started 2026-11-14)

**Kestrel** is the advisor's new name (2026-11-14), after the small hawk that hovers, watches and strikes. The
daemon is `kestreld` and the app is `Kestrel`; Kestrel is a standalone project at the repo root (`kestrel/`); the research that feeds it (training, models, the
Python reference) stays in `llm-research/`.

Phase D of [`llm-research/docs/2026-11-11-next-phases.md`](../../llm-research/docs/2026-11-11-next-phases.md), widened on 2026-11-14: besides advising, the
advisor **plays by itself** on platforms that offer a bot API, through **connectors described in a file** (YAML or
JSON) that are added after the build, without recompiling. You give it a connector file and a session plan, start
it, and leave it: it plays, logs every hand, and reports results with confidence intervals.

## How it reaches platforms

Kestrel plays through a platform's API (HTTP, WebSocket, TCP or MCP), described in a connector file. It has no
screen reading and no clicking, so it cannot drive a desktop poker client. Connector files carry no declarations
about money or bot policy (decided 2026-11-14); which platforms to connect is the user's call.

**Connector files are untrusted input.** Anyone can write one and share it, so a file may only read environment
variables named `KESTREL_*` (for example `KESTREL_SLUMBOT_USER`). Anything else is refused, so a shared file cannot
read cloud keys and send them to its server. `kestreld check FILE` shows the hosts a file talks to and the variables
it reads, before it runs.

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
- **Live tables, one screen each:** every running session is a table on its own screen. Screen 1 can play Slumbot
  while screen 2 plays another platform, up to screen N. Each screen shows the table as it plays: seats, stacks,
  cards, the pot, every action as it happens, Kestrel's decision with its probabilities, and the session's results.
  Switch screens with tabs or keys 1–9, or see them all as a grid of live mini-tables. `kestreld` runs the sessions
  side by side and streams each table's events over the WebSocket.
- **The Poker AI Watch page** (`web/index.html`, served publicly) gets a **read-only** Kestrel tab: session
  progress and results. Controls (start, stop, connectors) stay on the local page, because the watch page is public
  and the backend is bound to `127.0.0.1` (decision of 2026-11-10). Remote controls would first need
  authentication.

### Look and feel (both GUIs)

**Logo.** A kestrel, the small falcon that hovers in the wind before it strikes, drawn as a geometric mark: the
GUIs show the mark, not the name (`assets/logo.svg`, `assets/logo-mark.svg`). An animated version
(`assets/logo-animated.svg`), with the wings beating and the bird hovering, plays while Kestrel is thinking, connecting
or starting a session; it stops when the GUI is idle and respects reduced-motion settings.

Modern and sleek, as one design across the web page and the app:
- **Dark first, with a light theme**, both from one set of colour tokens; a restrained palette, a single accent
  colour for actions, and green and red only for results.
- **Typography:** Inter (UI) and JetBrains Mono (numbers, cards, logs), with tabular figures so columns of bb/100 do
  not jitter; a clear type scale and generous spacing.
- **Cards and tables drawn, not pictured:** vector suits in four colours, a felt-free table view with seats, stacks,
  the pot and the action history.
- **Live without noise:** session cards with a sparkline of results and a confidence band, short transitions
  (150–200 ms), no spinners where a skeleton will do.
- **Advice as a bar:** each action's probability as a segmented bar, with the EV and the reason (network, push/fold
  chart, blueprint) in one line.
- Keyboard first (command palette, shortcuts), responsive down to a phone width, readable contrast (WCAG AA).
- The Qt app uses the same tokens through a QSS theme and Qt Quick-like spacing, so it does not look like a default
  widget app.

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

Any connector a user writes is accepted if it is complete. Load a file (`kestreld check FILE`, `kestreld play FILE`)
or drop it into the connectors folder (`~/.config/kestrel/connectors`, `kestreld list`). Nothing is assumed about a
platform: the user lists its routes, the request and response structure of each, and which route plays each step.

| Section | What the user writes |
|---|---|
| `game` | variant (`nlhe`), players, blinds, stack |
| `codec` | the platform's action notation: `acpc` (Slumbot's `b200c/kk/b400`) today; `structured` (JSON actions) next |
| `transport` | `base_url`, retries, a rate limit, headers sent with every route (e.g. an API key) |
| `routes` | **every route, as the platform has it:** name, `method` (GET/POST/PUT/PATCH/DELETE), `path` (relative to the base URL or absolute), and its `request` (body of any shape, `query`, `headers`, `content_type`) and `response` (fields read from this route's answers) |
| `flow` | which route plays each step: `start` (deal a hand), `act` (send our move), optionally `login` (once per session) |
| `session` | `token_field`: a value carried from each response into the next request as `${token}` |
| `fields` | the response structure: where each answer is (JSON pointers), e.g. `hole: /hole_cards`. A field can be `{at, scale, map}`: amounts in other units, or the platform's values mapped to ours (`{BB: 0, SB: 1}`). Cards are read from a list (`["As","Kc"]`) or a string (`"AsKc"`, `"As Kc"`) |

Placeholders in requests: `${token}`, `${action}` (our move in the codec's notation), `${env:KESTREL_*}`. A value that is
a lone placeholder with nothing to fill in is left out of its object, so the first request goes without a token.

The Slumbot file, [`connectors/slumbot.yaml`](../connectors/slumbot.yaml), is the worked example. It played its first
hands by itself on 2026-11-14: 40 hands, no platform errors, no refused moves, no hands off the blueprint's tree.

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
