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
| `codec` | `acpc`: heads-up action strings (Slumbot's `b200c/kk/b400`); `structured`: a 2-9 handed table described in JSON (below) |
| `transport` | `base_url`, retries, a rate limit, headers sent with every route (e.g. an API key) |
| `routes` | **every route, as the platform has it:** name, `method` (GET/POST/PUT/PATCH/DELETE), `path` (relative to the base URL or absolute), and its `request` (body of any shape, `query`, `headers`, `content_type`) and `response` (fields read from this route's answers) |
| `flow` | which route plays each step: `start` (deal a hand), `act` (send our move), optionally `login` (once per session) |
| `session` | `token_field`: a value carried from each response into the next request as `${token}` |
| `fields` | the response structure: where each answer is (JSON pointers), e.g. `hole: /hole_cards`. A field can be `{at, scale, map}`: amounts in other units, or the platform's values mapped to ours (`{BB: 0, SB: 1}`). Cards are read from a list (`["As","Kc"]`) or a string (`"AsKc"`, `"As Kc"`) |

Placeholders in requests: `${token}`, `${action}` (our move in the codec's notation), `${env:KESTREL_*}`. A value that is
a lone placeholder with nothing to fill in is left out of its object, so the first request goes without a token.

The Slumbot file, [`connectors/slumbot.yaml`](../connectors/slumbot.yaml), is the worked example. It played its first
hands by itself on 2026-11-14: 40 hands, no platform errors, no refused moves, no hands off the blueprint's tree.

### Table connectors (`codec: structured`): 2 to 9 players, played by a 9-max model

A platform that describes its table in JSON is played by polling: `flow.state` reads the table, and when it is our
turn `flow.act` sends the move (`flow.start` joins a tournament, again after each one). `fields` say where each part
of the table is: `me`, `hand` (an id), `players` (`{at, items: {id, stack}}`, seat order, chips at the hand's start),
`button`, `sb`, `bb`, `ante` (a big-blind ante), `to_act`, `hole`, `board`, `actions` (`{at, items: {player, street,
kind, to}}`, kinds fold/check/call/bet/raise/allin, others such as blinds skipped; `to` is the street total),
`payouts` (the prizes still to play for), and at the end `finished`, `place`, `prize`, `chips`. A field can also be a
constant: `{value: "${env:KESTREL_..._USER}"}`. `answer` says how a move is sent: `${action}` through a map of action
names, and `${amount}` as the street total (`amount: to`) or the chips put in now (`add`); `"${num:amount}"` sends it
as a number.

The adapter (`core/ninemax_spot`) replays the hand into Kestrel's table: seats placed by position from the button,
chips rescaled to the big blind, and every bet at **its own size**, since the network reads the table state (pot,
stacks, contributions, raise count, aggressor) rather than menu slots. Mapping sizes to the nearest menu size broke on
real tables: a 1bb flop bet nearer an all-in than the smallest menu bet became an all-in, and a raise leaving 3 chips
behind read as all-in, so players were all-in in the replay while still acting on the platform. Stacks under 2bb are
seated (a blind that cannot be covered is posted all-in), opponents may limp, and Kestrel's own move uses the model's
rules and menu; its choice goes back at the same street total in the platform's chips.

**Platforms that push (WebSocket).** With `transport: {websocket: wss://..., headers, rate_limit_per_s,
reconnect_s}`, routes are message templates (`message: {...}`) and the platform's events build the table document
that `fields` read. `events.type` says where an event's type is, and `events.apply.<type>` how it updates the
document: `reset` and `unset` (paths), `set`, `append` (board cards), `push` (an action, read with the usual field
syntax), `merge` (a map such as final stacks), `index` (a list of players into a seat-to-stack map), `copy` (the stacks
at a hand's start); paths may name event fields in braces (`/stacks/{seat}`). An event marked `turn: true` gets an
answer; `reject: true` (a refused move) is answered with a check, then a fold, and counts as a fallback; `send: join`
sends a route (after busting or a closed table); `flow.start` is sent whenever the socket opens. `${nonce}` is new for
every message (Open Poker's `client_action_id`). An action can give its size three ways: the street total (`to`),
the chips it put in (`add`), or the stack left after it (`left`), so a platform's optional fields are not needed.
The WebSocket uses IXWebSocket with TLS through the same OpenSSL as https, verifying the server's certificate.

Tested against `testplatform/ws_platform.py`, the PokerKit platform speaking Open Poker's protocol (turn tokens,
hand ids, `client_action_id`, `action_rejected` codes, busting and rejoining): 1,000 hands with 0 errors, 0 replay
errors and 0 refused moves; 800 hands sizing every action from the stack left (no `add`), also clean; and a connector
that sent fractional amounts had 69 of its moves refused in 100 hands, each answered by the fallback, with the session
playing on. `connectors/openpoker.yaml` is the same mapping pointed at `wss://openpoker.ai/ws`: a draft until the
account exists.

Checked two ways: the self-test re-describes 1,922 self-played decisions at 2-9 players as a platform would (its own
seat order and button, 40 chips a big blind) and must get the network's own probabilities and amounts exactly; and a
PokerKit tournament platform (`testplatform/ninemax_platform.py`, rule bots that limp, min-raise and bet odd sizes,
rising blinds, short stacks, heads-up finishes) played 120 tournaments with nm17 on 2026-11-14: 4,488 decisions, 19%
of the bets it replayed at sizes off the network's menu, 0 replay errors, 0 refused moves; mean prize 0.150 and 0.158
of the pool on two seeds against an even share of 0.111 (rule bots, not a strength measure).

## Verified against the Python reference

Each port gets test vectors written by the Python code it replaces, and the C++ build must reproduce them:
- the game state: random hands at 2–9 players, with and without the ante, as `ninemax_check.py` checks against
  PokerKit;
- features and network output: the same states, `features()` and logits to 1e-4;
- ICM: `icm.py`'s self-test values;
- the Slumbot codec: the action strings in `results/slumbot*/hands.jsonl`, replayed to the same pots and legal ranges
  as `slumbot_client.py`.

`kestreld --selftest` runs all of them.

## Status (2026-11-14)

| Part | State |
|---|---|
| Core (9-max game, ICM, network; heads-up blueprint and buckets) | done, self-tested against the Python reference |
| Connectors: user-written YAML/JSON, user-listed routes with request and response structure, HTTP(S) | done; Slumbot plays by itself |
| `kestreld serve`: REST API and the live stream (server-sent events), tables side by side | done |
| Web app: live tables, screens 1–N, grid, new table, light and dark themes, logo | done |
| WebSocket, TCP and MCP interfaces | done: one JSON-RPC method table (health, list_connectors, list_engines, start/list/status/stop sessions, start_practice, practice_act, decide, icm_equity) over WebSocket `ws://127.0.0.1:8791/` (with `subscribe` for pushed events), TCP `127.0.0.1:8792` (a request per line) and MCP over stdio (`kestreld mcp`, which joins a running Kestrel or starts one quietly). Each is a front door to the REST API, so the same checks apply: the WebSocket refuses pages from other origins, TCP drops a connection whose first line is not JSON (a web page's HTTP request), and malformed requests get JSON-RPC errors. WebSocket as a connector transport: done (see Table connectors); TCP and MCP as connector transports when a platform needs them |
| `structured` codec: 9-max models at 2-9 handed platform tables | done 2026-11-14 (see Table connectors): `kestreld play FILE --model <9-max folder> --tournaments N`, sessions from the web app with the table drawn live (seats, stacks, ICM, the hand's log, Kestrel's options), and `POST /v1/decide` with `game: ninemax` for advice on any spot. A test platform, `testplatform/ninemax_platform.py` with `connectors/testplatform.yaml` |
| Kestrel against Slumbot, with the defence floor (2026-11-14) | 3,000 hands: −1.9 bb/100 [−67, +56], even with Slumbot; no platform errors; 19 hands left the blueprint's tree (checked or called). The Python bot read +12.4 and +30.9 (with the floor) over 5,000 hands each; all within each other's intervals. Telling the versions apart needs tens of thousands of hands or luck adjustment (AIVAT) |
| Defence floor in the heads-up engine | done: on 5,853 logged decisions it fires 36 times and agrees with Python on all but 5, river spots where the order of tied hands decides (Python's sort leaves it arbitrary) |
| Practice tournaments (9-max models) | done: a final table from 2 to 9 players, every seat played by the chosen 9-max model (watch every decision with its probabilities) or you against it (your seat waits for your move; the model's cards and probabilities stay hidden; optional advice for your spot). Stacks carry over, blinds rise every 15 hands, busted players finish in places, ICM recomputed each hand. `POST /v1/practice`, `POST /v1/practice/{id}/act`. Practice games do not train the model: models are trained on the lab and shipped through the pipeline |
| River buckets for 9-max models | the pack rebuilds river buckets from hand-strength cut points; against the training tables 7% of river states read the neighbouring bucket (all off by exactly one), pre-flop, flop and turn match exactly. The exact 123 MB `river_buckets.u8` can be put in the pack folder and is used when present |
| Qt 6 app (`Kestrel`, `app/`) | started 2026-11-15: a client of the same local API. **Tables:** every table drawn as the web app draws it (seats round the oval with stacks, ICM, cards and last actions; heads-up as two seats; board and pot), this hand's log, Kestrel's decision with its probabilities, your move at a practice seat (with optional advice), a live grid of all tables, the web app's keys (1-9, g, n, t); new tables and practice tournaments. **Models:** the list, install a package (checked against its .sha256). **Connectors:** the list, check a file, add one (checked before it is copied). **Settings, as admin:** every server option in `settings.json` (ports, LAN and the access links, a new access token, the default 9-max model, re-solve iterations, TexasSolver, folders), applied by restarting the server (`POST /v1/admin/shutdown`, this computer only). **Log:** kestreld's output. It starts `kestreld start` when no server is running. **Review:** session logs from the sessions folder, each hand drawn with its result, the session's bb/100 with an interval, and the luck-adjusted result (`kestreld luck`). **Spot study:** any spot (a 2-9 handed table or heads-up) through `/v1/decide`, drawn, with every legal move's probability. **Trainer:** random pre-flop tournament spots (6-9 players, 8-30bb, ICM payouts, first in or facing an open): your move, then the model's distribution, scored by agreement. Built when Qt 6 (Widgets, Network) is found; CI builds it on Linux, macOS and Windows and the release archives carry it with Qt's libraries (`KESTREL_DEPLOY_QT`). Still to come: hand sync |
| A database for hands, sessions and opponents | done 2026-11-16: SQLite built in (3.54.1), `kestrel.db` in the data folder; sessions, practice tables and the opponent book write to it beside their files; `kestreld db stats\|sessions\|import`, `GET /v1/db`, `/v1/db/sessions`, `/v1/hands`; checked on the test platform (32 hands, a search session, practice, imports idempotent) |
| Opponent tracking by username, then bounded exploitation | **statistics done 2026-11-15; exploitation to do.** A book per platform (`<sessions>/opponents.json`, `GET /v1/opponents[?platform=]`): hands, VPIP, pre-flop raise, 3-bet (of the chances), fold to a continuation bet, post-flop aggression, went to showdown, the last 20 hands shown. Recorded after every hand, from the finished hand's whole history when the connector gives it (`last_hand`, `last_actions`, `last_shown`, `last_board`; Open Poker's `hand_result` has it), else from what Kestrel saw (which stops at its own last move, so folds after it are missed: fold to c-bet read 0% for every bot until the history was used). Shown under each seat (VPIP/raise/3-bet · hands) in the web and desktop apps, and in the desktop app's Opponents tab. Checked against the test platform's rule bots: the station reads 89% VPIP and 0% raising, the tag 14% and 11%. Requested 2026-11-14. Per platform and username, kept across sessions: hands seen, VPIP, pre-flop raise and 3-bet rates, fold to continuation bet, aggression, went-to-showdown and the hands shown; shown per seat in the web app. Exploitation only within limits that grow with the sample, and only after it measures better than the model's own play on Slumbot and the practice tables: the research's adaptive bot lost (−22.9 bb/100) when its call-down rule met a strong player, and Slumbot-specific hand reading read +12 after luck adjustment (+57 raw). A connector names where usernames are in its responses (`fields`). **Not losing to strong players** is designed in, three ways:
  1. **Safe exploitation** (Ganzfried & Sandholm): risk only what the opponent has already given away through their mistakes. Against a strong player who gives nothing away, Kestrel plays its own strategy.
  2. **Weight by the evidence** (data-biased and restricted responses, Johanson & Bowling): the opponent model counts in proportion to the sample, so a few dozen hands barely move the strategy.
  3. **Exploit measured leaks only:** deviate only where a player's statistics differ from the model's own play in that spot by more than their noise. A balanced player gives no such signal, and the call-down rule that lost to Slumbot would never have fired. |
| Open Poker (openpoker.ai, 6-max cash over WebSocket): an account, a 6-max cash model and a connector | waiting on the user's sign-in (magic link to the account's email); protocol read from the lab (see Platforms) |
| Search at play time: the heads-up safe river re-solve, then 9-max search | heads-up re-solve done: matches Python's `libresolve` on 121 logged river decisions (119 within 7e-4; one spot at 3e-3, float vs double normalisation through 500 CFR+ iterations); made faster on 2026-11-14: one-pass strategies, reused scratch memory, the root's subtrees on 4 threads (summed in order, so the result is the serial solve's), optional `-DKESTREL_NATIVE=ON`. On the lab: 0.68 s → 0.31 s per river decision over all 121, still matching Python. Opt-in per table. 9-max once it beats the plain network |
| Model and solver chosen per table | done: heads-up `hu-blueprint` with `none`, `kestrel-river` or `texassolver`; 9-max models (`nm25` the default, `nm17`) with `none`, `search` or `search-robust` (2026-11-16); in the web and desktop apps, the API (`GET /v1/engines`, `POST /v1/sessions`, `POST /v1/decide` with `solver`) and `kestreld play --solver` |
| TexasSolver as a solver | done: the Python bot's script and tree walk, from the blueprint's ranges on the turn and river; matches the Python bot on 12 logged decisions to 5e-7 (single-threaded, deterministic). Live it is slow, tens of seconds per turn solve here, so it suits Slumbot more than platforms with turn timers. Run as a separate program (AGPL); configured with `--texassolver BIN --texassolver-resources DIR`, and refused without the hand-ranking tables |
| CI: Linux, macOS, Windows builds, self-test, GitHub Releases on `kestrel-v*` tags | done |

## Platforms (searched 2026-11-14)

Platforms where a bot can play other bots or a strong bot through an API, besides Slumbot:

| Platform | Game | How a bot plays | Status for Kestrel |
|---|---|---|---|
| **Slumbot** (slumbot.com) | heads-up NLHE, 200bb, reset each hand | HTTP JSON API, no account needed | **plays** (`connectors/slumbot.yaml`) |
| **Open Poker** (openpoker.ai) | 6-max NLHE cash, virtual chips, 14-day seasons, public leaderboard | WebSocket (`wss://openpoker.ai/ws`), API key from the dashboard | waiting on the account; reachable from the lab only (see below) |
| **poker-pi** (github.com/ArtART241/poker-pi, MIT) | NLHE tournaments and practice tables for bots, 2-9 seats | REST long poll (`GET /api/v1/bot/turn`, `POST /api/v1/bot/action`), bearer token, self-hosted | **plays** (`connectors/pokerpi.yaml`), self-hosted on the lab (see below) |
| **Kestrel test platform** (`testplatform/ninemax_platform.py`) | 9-max tournaments against rule bots, on PokerKit | REST polling | **plays** (`connectors/testplatform.yaml`) |
| **Open Poker** (openpoker.ai) | 6-max NLHE cash, virtual chips, 14-day seasons, public leaderboard | WebSocket (`wss://openpoker.ai/ws`), API key from the dashboard | connector drafted (`connectors/openpoker.yaml`, WebSocket transport built and tested locally); waiting on the account and a 6-max cash model (see below) |
| **GTO Wizard AI benchmark** | heads-up NLHE, 200bb, at least 2,500 hands | API access by application | to do: needs the user's application (see below) |
| **pokbot** (github.com/baobabtr33/pokbot) | bot knockouts every 30 minutes, 6-seat tables | the bot is submitted and runs on their server (stdin/stdout JSON, 5 ms per decision); live play and an MCP server too | not planned (see below) |

**Open Poker, 2026-11-14.** Its site answers this machine with Cloudflare's block page (HTTP 403) but serves the
lab normally, so its documentation was read from the lab (`https://docs.openpoker.ai/llms-full.txt`, revision
2026-09-02). What it is:
- **Game:** NLHE 6-max cash with free virtual chips. Blinds 10/20, buy-in 1,000–5,000 chips (50–250bb), rebuys of
  1,500, 14-day seasons from 5,000 chips, scored on chips held. Prizes are sponsor-funded; Pro is optional
  ($5/season, USDC).
- **Protocol:** WebSocket `wss://openpoker.ai/ws` with `Authorization: Bearer <key>`. JSON messages: client
  `join_lobby`, `action` (with `hand_id`, `turn_token`, `client_action_id`), `set_auto_rebuy`, `resync_request`,
  `leave_table`; server `connected`, `hand_start`, `hole_cards`, `your_turn` (with `valid_actions`),
  `player_action`, `community_cards`, `hand_result`, `table_closed`, `season_ended`. REST at
  `https://api.openpoker.ai/api`.
- **Limits:** 20 WebSocket messages per second, 10 connections per minute per IP.
- **Rules:** a free account connects one public bot; same-owner bots may not sit together; no extra accounts,
  collusion or chip dumping.

**Account:** the dashboard signs in by a magic link emailed to the address. The user chose the account's email, so
the user signs in, creates a bot with **Self Host**, and puts its key on the lab as `KESTREL_OPENPOKER_KEY`; Kestrel
plays Open Poker from the lab.

**The model:** Open Poker is 6-max cash at 50–250bb. The tournament networks (9-max, 10–25bb, ICM) and the heads-up
blueprint (200bb) were trained for other games; the nearest is `nm4` (9-max cash, 100bb), which never trained with
empty seats. A competitive entry needs a short 6-max cash run of the 9-max trainer (`--players 6`, cash stacks).

**The connector** needs the WebSocket transport, an event-stream flow (the server pushes `your_turn`, unlike Slumbot's
request-response), a `structured` codec for its JSON actions, and the 6-max engine.

**poker-pi, 2026-11-14.** A FastAPI tournament server for bots (MIT), cloned and run on the lab in its own venv at
`127.0.0.1:8794` (its default port is 8000, the lab's another service tunnel, so `POKER_PORT` is always set), reached
from this machine through the SSH connection. A bot long-polls `GET /turn?wait=25` (a 204 with no body when nothing is
due), gets the whole table (seats, stacks, the hand's event log with stacks at its start, legal moves) and answers
with the turn's id. Every player posts the ante. The connector needed four general features, now in the structured
codec: an empty reply reads as "nothing yet", `session.vars` carries a value from each reply into the next request
(`turn_id`), `game.ante: each` posts the ante from every stack as dead money, and `flow.status` asks whether a table
is over when no turn comes (`{at: /status, equals: finished}`). Amounts go as whole chips (`answer.decimals` for
others), and an all-in is rounded down, never past the stack; the first run sent 109.5 and had 10 moves refused.
Practice tables against its house bots (tight, calling station, random): at 75bb nm17 busted all 8 tables (it trained
at 10-25bb); at 20bb it won 2 of 8, +111 bb/100 [+6, +234] over 166 hands. 0 replay errors and 0 refused moves over
both. Results per table only give a place when the bot wins (the status says "your bot busted" otherwise).

**Open Poker: what remains.** Done on 2026-11-14: the WebSocket transport (the key as a header, reconnects no
faster than every 6 s against its 10 a minute, at most 20 messages a second) and the event flow, tested against a
local platform that speaks its protocol; `connectors/openpoker.yaml` is the draft. Still to do:
1. Resync: after a reconnect Kestrel joins the lobby again; it should send `resync_request` for a seat it still holds
   (`already_seated`), and a turn pending during a disconnect is folded by the server after 45 s.
2. Check the draft against the live protocol (the field names come from its documentation, not a session).
3. A model for 6-max cash at 50-250bb: a 6-max run of the 9-max trainer with cash stacks and no ICM, checked like the
   tournament networks (sense, post-flop, duplicate matches, an exploiter) before it plays.
4. The account: the user signs in by the magic link, creates a Self Host bot and puts its key on the lab as
   `KESTREL_OPENPOKER_KEY`. Kestrel plays from the lab, which Cloudflare serves (this machine gets a 403).
One public bot per free account; same-owner bots may not sit together, so it is one table at a time.

**GTO Wizard AI benchmark: what remains.** Heads-up NLHE at 200bb against GTO Wizard's AI, at least 2,500 hands,
reported with its own variance reduction. It is the strongest measure available for the heads-up side (the blueprint
alone, with the defence floor, and with each solver). Needs: the user applies for API access; then a connector for
its API (unknown until access is granted) with the `acpc` codec or a structured one. Run it from the lab with the
same settings as the Slumbot sessions, luck-adjusted, so the two results compare.

**pokbot: not planned.** Bots are uploaded and run on the organiser's server within 5 ms a decision (stdin/stdout
JSON), so Kestrel would have to be shipped to a third party and fit that budget (the river re-solve takes 310 ms),
and its installer is a `curl | sh` from a domain that is a bare IP address. It is reconsidered if it gains a remote
API like the others.

## Search

Search is a play-time technique: Libratus and Pluribus re-solved the spot in front of them when it was their turn,
and that is where it pays. It belongs in Kestrel, not only in training.
- **Heads-up:** the safe river re-solve (`llm-research/blueprint/resolve.cpp`, already C++) is linked into Kestrel
  next. It is solved from the start of the river with both blueprint ranges, and its gadget keeps every opponent
  hand at or below its blueprint value. It is off by default until it measures better than the blueprint against
  Slumbot or LBR.
- **9-max tournaments (in Kestrel 2026-11-16, solvers `search` and `search-robust`):** a C++ port of
  `llm-research/scripts/ninemax_search.py` (`core/src/ninemax_search.cpp`), checked with `kestrel_search_match`:
  nm25 tables drawn as in training, 4,000 deals, the hero in every seat of every deal once plain and once with
  search, the gain scored deal by deal (bb/100 of ICM equity, 95% interval):

  | Opponents | Deals a search | Gain over plain nm25 |
  |---|---|---|
  | raise-biased | 1,024 | **+6.16** [+3.35, +8.98] (Python, 2,048 deals: +6.39) |
  | raise-biased | 256 | +0.50 [−2.91, +3.91] |
  | the network | 256 | **−5.21** [−8.65, −1.78] |
  | call-biased | 256 | **−6.54** [−10.21, −2.88] |
  | fold-biased | 256 | **−5.16** [−8.52, −1.80] |
  | the network, robust | 256 | −2.27 [−5.03, +0.50] |

  The port reproduces Python at 1,024 deals; at 256 it loses, because it plays the move with the best estimated
  value and 256 deals leave those estimates too noisy (it then picks noise). So Kestrel offers 1,024 to 4,096 deals
  (default 1,024); 1,024 against the other opponents and 2,048 against the network are running.

## Embeddings and similarity (2026-11-16)

The networks already work with embeddings: every card bucket enters as a learned vector of 32 numbers, and the last
hidden layer (512 numbers) is the network's own picture of the spot. Similarity on top of them is used where it
answers a question; none of it changes how Kestrel plays until a measured test says it helps, as with search.

**Built:**

1. **Opponent profiles** (`connectors/src/profile.cpp`). Six rates per player (VPIP, pre-flop raise, 3-bet, fold to
   a continuation bet, post-flop aggression frequency, went to showdown), each pulled towards the average of the
   reference types by 8 chances of prior so a short sample cannot look extreme, centred on that average and scaled
   by the types' spread, then compared by cosine similarity with six reference types (nit, tag, lag, maniac,
   station, loose-passive; approximate HUD-guide rates) and with every player met before. Shown from 30 hands: in
   the HUD line under each seat, the desktop app's Opponents tab (type, similarity, the runner-up, who the player
   plays most like) and `GET /v1/opponents?platform=P&player=U` (`profile`, `like`). Statistics only: Kestrel does
   not change its play on them yet.
2. **Similar spots** (`core/src/net.cpp` `embed_hidden`, `connectors/src/store.cpp`). Every decision at a 9-max
   table stores the network's last hidden layer, projected to 128 numbers by a fixed random Gaussian matrix (the
   same on every machine) and scaled to length 1, in the database's `spots` table with its hand, position and the
   move played. The nearest stored spots of the same model by cosine similarity are returned by
   `GET /v1/spots/similar?spot=ID&k=10` and by `POST /v1/decide` with `"similar": k`, and shown under the answer in
   the desktop app's Spot study tab with how Kestrel played them and the result. A random projection keeps cosines
   to about ±0.1; the search is a scan of the newest 200,000 spots.

**To do, in order:**

3. **Reusing search answers** (Kestrel). A spot almost identical to one already searched (cosine above a threshold,
   same legal moves and stack depths) reuses that answer instead of 1-2 s of search. Only if a paired match shows
   it costs nothing: two spots close in the network's picture can still call for different moves, and search exists
   for exactly those.
4. **A varied training pool** (training). Training keeps earlier snapshots as opponents, chosen by time. Instead,
   measure how differently each plays on a fixed set of probe spots, by the Jensen-Shannon divergence of their move
   probabilities (the right measure between two distributions, where cosine is not), and keep the most different
   ones. The same number flags a run that has stopped changing or is cycling between strategies. Check: the
   exploiter's winnings against runs trained each way.
5. **Grouping leaks** (training and evaluation). Embed the decisions where the exploiter or LBR wins, cluster them
   (k-means on the embeddings) and report the winnings per cluster, so a leak reads as "facing a 3-bet at 15bb from
   the small blind" rather than one total.
6. **Card abstraction by distribution distance** (training; the costliest, last). The card buckets come from a
   similarity measure too. Libratus and Pluribus cluster hands by the earth mover's distance between their
   distributions of future strength (potential-aware), where ours use percentiles of one strength number. It means
   a new abstraction and a retrained network, and our test of more and fewer buckets made no difference, so it waits
   until the other leaks are closed.

## Models and solvers, chosen per table

A table plays with a **model** (what decides each spot) and, optionally, a **solver** (what re-solves a spot at
decision time), chosen when the table starts. Any model goes with any solver that fits its game. A default for each
is set in Kestrel's settings, and a connector may name its own (`engine: {model: ..., solver: ...}`).

| Models | Game |
|---|---|
| `hu-blueprint` (the data pack, or a training run with `avg.f32`) | heads-up, 200bb |
| `nm25` (the default), `nm17` (`models/ninemax/<name>`) | 9-max tournaments, 10–25bb, ICM |
| a 6-max cash network (to train) | 6-max cash, 50–250bb |

| Solvers | Where | Notes |
|---|---|---|
| `none` | | the model's own policy |
| `kestrel-river` | heads-up river | the safe re-solve built in (CFR+ with the gadget); iterations and minimum reach are settings |
| `texassolver` | heads-up turn and river | TexasSolver as a separate program, as the Python bot ran it. It is AGPL-3.0, so Kestrel calls it and does not link it; its path is a setting, and the user installs it |
| `search` | 9-max: before the flop, and after it in pots of up to three players | deals the unknown cards (1,024 by default, a setting), weights each deal by how likely the network was to make every move the opponents made with those cards, plays every legal move out to the end with the network for everyone, and takes the move with the best ICM value (see Search) |
| `search-robust` | the same spots | the same, each deal played four times with the opponents continuing as the network or tilted towards folding, calling or raising, each move valued at its worst case (after Pluribus); about 3× slower |

The model and solver are chosen in the new-table dialog, the API (`POST /v1/sessions {connector, hands, model,
solver}`) and `kestreld play --model --solver`. Each hand's log records which ones played it, so their results can be
compared on the same platform.

## The trainer's dealer

The networks train on deals from `bp_deal_n`: a partial Fisher–Yates with one seeded splitmix64 stream per deal, so
deals are fast and reproducible. On 4,000,000 nine-handed deals it matched exact odds in every test: positions,
repeated cards, starting-hand classes, seven-card categories, independence, and seeds (lowest p = 0.07;
`llm-research/scripts/dealer_check.py`). Provably fair shuffles such as CoinPoker's prove a real-money deal to its
players and add nothing to training. Kestrel itself never deals: the platform does.

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
