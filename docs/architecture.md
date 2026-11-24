# Architecture

How the parts of this repository fit together: the research that trains the bots, the pipeline that judges and
ships them, Kestrel (the engine that plays and advises), and where each runs. The diagrams are Mermaid and render on
GitHub. Details live in the linked docs; this page is the map.

## 1. The whole system

```mermaid
flowchart LR
    subgraph GH["GitHub (nodalytics/poker-analysis, public)"]
        MAIN["main branch<br/>code, run files, results"]
        CI["Actions<br/>kestrel · kestrel-docker<br/>kestrel-release · research-pipeline"]
        REL["Releases<br/>kestrel-v*, model-*-v*,<br/>headsup-pack-run2"]
        GHCR["ghcr.io<br/>Kestrel images"]
    end

    subgraph LAB["The lab (3 GPUs)"]
        PIPE["research pipeline service<br/>scripts/pipeline.py poll"]
        TRAIN["9-max training<br/>ninemax_selfplay.py"]
        CHECKS["checks<br/>seqtest · opponents · styles<br/>bluffers · exploiters"]
        KD["kestreld (Docker)<br/>127.0.0.1:8890"]
        TP["test platform (Docker)<br/>scripted bots"]
        WATCH["watch server<br/>ninemax_watch.py"]
    end

    subgraph USERS["People and platforms"]
        WEB["browser: web app<br/>kestrel.stbfarm.org"]
        DESK["Kestrel desktop app<br/>(Qt), kestreld CLI"]
        AGENTS["LLMs / agents (MCP)"]
        PLAT["bot-API platforms<br/>Slumbot, test platform"]
        WATCHW["watch page<br/>poker.stbfarm.org"]
    end

    MAIN -- "merged run file" --> PIPE
    PIPE --> TRAIN --> CHECKS
    CHECKS -- "results PR: adopted / not adopted" --> MAIN
    MAIN --> CI
    CI -- "Linux · macOS · Windows builds" --> REL
    CI -- "self-tested image" --> GHCR
    GHCR -- "deploy.sh" --> KD
    REL -- "models install (sha256)" --> KD
    REL --> DESK
    KD <--> TP
    KD -- "nginx + Cloudflare, token" --> WEB
    KD <-- "connectors" --> PLAT
    KD <-- "MCP" --> AGENTS
    WATCH --> WATCHW
```

- **Research** (`research/`) trains and measures the bots. Python with PyTorch on the GPUs, plus C++ CFR solvers.
- **The pipeline** turns a run file merged into `main` into a trained, judged network, with the checks fixed
  before training starts (`research/pipeline/README.md`).
- **Kestrel** (`kestrel/`) is the C++ engine that plays and advises with the adopted models. It is shipped as
  desktop packages and as a Docker image.
- **The lab** runs the training, the checks and the live Kestrel. GitHub only stores and builds; nothing on GitHub
  can send the lab a command (the lab reads `main` by itself).

## 2. Training a 9-max network

```mermaid
flowchart TB
    INIT["start from a checkpoint<br/>(e.g. blueprint/nm25)"]
    subgraph ENGINE["GPU table engine (ninemax_engine.py)"]
        DEAL["dealer: bp_deal_n<br/>(libbp7.so, seeded per deal)"]
        TABLES["thousands of 9-max tables at once<br/>blinds, ante, side pots, empty seats"]
        CTX["tournament context per table<br/>stacks, payouts, stage, bounties"]
    end
    subgraph SEATS["who sits at each table (only on seated chairs)"]
        SELF["the learner, in most seats"]
        POOL["rule players: reg, tag, maniac, station<br/>+ styles: archetypes, generated, human-fitted"]
        EXP["exploiter: trained to beat the learner,<br/>reset every hour"]
        LEAGUE["league: frozen past exploiters,<br/>picked by how often they win (PFSP)"]
    end
    REWARD["reward = change in prize equity<br/>end_equity: ICM (Malmuth-Harville)<br/>+ knockout term in bounty games"]
    UPDATE["NeuRD policy update<br/>+ KL pull to the start, entropy"]
    EVAL["every 30 min: 1-vs-8 duplicate matches<br/>vs rule players, previous checkpoint, exploiter"]
    OUT["blueprint/nmXX<br/>checkpoint.pt, snapshots, progress.log"]

    INIT --> TABLES
    DEAL --> TABLES
    CTX --> TABLES
    SELF & POOL & EXP & LEAGUE --> TABLES
    TABLES --> REWARD --> UPDATE --> SELF
    UPDATE --> EVAL --> OUT
```

- The network (`NineNet`) reads the table as features: cards (bucketed by the abstraction in `blueprint/out`),
  positions, stacks, the betting, and the ICM context. It outputs a policy over the action menu and a value.
- Every chair an opponent takes must have a player in it, and each bust's bounty goes to whoever won the pot that
  took the player's last chips (fixes B1 and B2 in `research/docs/2026-11-23-review-and-plan.md`).

## 3. From a run file to Kestrel

```mermaid
flowchart LR
    RF["research/pipeline/runs/nmXX.json<br/>question, why, training, checks"]
    VAL["CI: pipeline.py validate"]
    MERGE["merged into main<br/>= the pre-registration"]
    POLL["lab: pipeline.py poll<br/>(every 2 min, needs 25 GB free)"]
    TR["train<br/>(minutes, GPU, args, opponents)"]
    CK["checks, all must pass"]
    PR["PR: nmXX adopted / not adopted<br/>results/pipeline/nmXX, prereg result"]
    GATE["model_pipeline.py gate<br/>poker-sense, push/fold, match vs champion,<br/>Kestrel C++ self-test"]
    PKG["package: .tar.gz + .sha256"]
    PUB["publish: GitHub release model-NAME-vV<br/>+ research/models/manifest.json"]
    INST["kestreld models install"]

    RF --> VAL --> MERGE --> POLL --> TR --> CK --> PR
    PR -- "adopted" --> GATE --> PKG --> PUB --> INST
```

The checks a run can carry (`research/pipeline/README.md`):

| Check | Asks |
|---|---|
| `seqtest` | does it lose to the champion (sequential duplicate match)? |
| `opponents` | against rule players, is it within a tolerance of the baseline (paired, with an interval rule)? |
| `styles` | across generated and archetype players, worst case and average? |
| `bluffers` | do scripted bluff jams and raises lose against it? |
| `exploit` / `exploiters` | how much can a learner trained to beat it win (mean over seeds)? |
| `patterns` | do the patterns people found against the bot still win? |
| `consistency`, `opponent-model` | reports for the reviewer, not gates |

## 4. The heads-up route

```mermaid
flowchart LR
    MCCFR["mccfr_vec.cpp<br/>vector MCCFR (public chance sampling), DCFR<br/>(workers paused for discount and save)"]
    RUN["blueprint/run2<br/>regrets, average strategy"]
    PACK["advisor_pack.py<br/>the data pack"]
    HREL["release headsup-pack-run2"]
    HU["Kestrel heads-up engine<br/>(headsup.cpp)"]
    RS["kestrel-river: safe river re-solve<br/>CFR+ with the gadget, defence floor"]
    TS["texassolver: turn and river<br/>(study only, unsafe)"]
    PY["Python reference bot<br/>blueprint_bot.py (LBR, Slumbot)"]

    MCCFR --> RUN --> PACK --> HREL --> HU
    HU --> RS
    HU --> TS
    RUN --> PY
    PY -. "parity: hu_check, resolve_check,<br/>1,160 logged Slumbot decisions" .- HU
```

Off-tree bets (sizes the abstraction doesn't have) map to a nearby abstract action. The draw is fixed per hand, so a
replay of the same hand maps the same way. TexasSolver gets the effective stack (the smaller of the two).

## 5. Inside Kestrel

```mermaid
flowchart TB
    subgraph IF["interfaces (127.0.0.1; a public name only with the token)"]
        REST["REST /v1/..."]
        WS["WebSocket JSON-RPC<br/>+ live events"]
        TCP["TCP JSON-RPC"]
        MCP["MCP tools<br/>(stdio, HTTP)"]
    end
    subgraph APPS["front ends"]
        WEBAPP["web app (kestrel/web)"]
        QT["desktop app (Qt 6)"]
    end
    subgraph SERVER["kestreld"]
        SESS["session runner<br/>unattended bot sessions, bb/100 with intervals"]
        PRAC["practice tables<br/>tournaments vs the networks, PKO"]
        CONN["connectors (YAML/JSON files)<br/>http · websocket · tcp · mcp"]
        DB["kestrel.db<br/>hands, sessions, opponent model"]
    end
    subgraph ENG["engines: model + solver per table"]
        ROUTER["router by game"]
        NET9["9-max / 6-max networks<br/>nm25 (default), nm17, nm30 (PKO), nm37 (cash)"]
        SRCH["search / search-robust<br/>deals weighted by the network, ICM value,<br/>common random numbers, ESS floor"]
        HUENG["heads-up blueprint pack<br/>+ river re-solve / TexasSolver"]
        PF["ICM + push/fold charts"]
    end
    subgraph CORE["core library (C++20)"]
        TABLE["table state, menus, side pots"]
        FEAT["features + forward pass<br/>(ports of the Python trainer)"]
        EQ["ICM, end_equity, knockouts"]
        EVAL7["7-card evaluator, buckets"]
    end

    WEBAPP --> REST & WS
    QT --> REST & WS
    REST & WS & TCP & MCP --> SERVER
    SESS --> CONN
    SESS & PRAC --> ROUTER
    ROUTER --> NET9 & SRCH & HUENG & PF
    NET9 & SRCH & HUENG & PF --> CORE
    SESS & PRAC --> DB
```

- **Parity with the trainer.** The core is a port of the Python engine. `kestrel_selftest` replays the Python
  reference's vectors (features, logits, values, payoffs, ICM, search, heads-up decisions) to within tolerance.
  A build that fails it is not packaged.
- **Connector files are untrusted.** They may read only `KESTREL_*` environment variables, and `kestreld check`
  shows the hosts and variables a file uses before it runs.
- **No real-money play.** Kestrel has no screen reading or clicking, and no connector for a real-money site.

### 5.1 Kestrel's source map

| Layer | Files (`kestrel/`) | Role |
|---|---|---|
| **core** (library, no I/O) | `core/src/table.cpp` | one table's state: blinds, antes, menus, legality, side pots, payoffs |
| | `net.cpp` | the 9-max network: features, forward pass (AVX2/FMA), `end_equity` with the knockout rule |
| | `icm.cpp` | ICM (Malmuth-Harville and Malmuth-Weitzman), start equity |
| | `ninemax_spot.cpp` | a platform's 2-9 handed spot replayed into Kestrel's table, and the answer turned back into its chips |
| | `ninemax_search.cpp` | search at play time: weighted deals, rollouts, CRN, the ESS floor |
| | `headsup.cpp`, `buckets.cpp` | the heads-up blueprint walk, off-tree mapping, card buckets |
| | `resolve.cpp`, `texassolver.cpp` | the safe river re-solve; TexasSolver as an external program (time-limited) |
| | `practice.cpp` | practice tournaments: stacks carry over, blinds rise, places and bounties paid |
| | `luck.cpp` | luck-adjusted heads-up results (AIVAT-style corrections) |
| | `models.cpp` | installed model packages, each file checked against its sha256 |
| **connectors** (I/O) | `connectors/src/server.cpp` | the REST API and web app: Host/Origin checks, token, routes, version |
| | `gateway.cpp` | the same methods over WebSocket, TCP and MCP, each a front door to REST |
| | `connector.cpp`, `codec_acpc.cpp` | connector files, transports with retries, the ACPC and structured codecs |
| | `session.cpp` | the session runner: plays N hands, waits out outages, logs every hand |
| | `store.cpp` | `kestrel.db` (SQLite): sessions, hands, opponents, spots |
| | `opponents.cpp`, `profile.cpp` | opponent statistics, and profiles compared to player types by cosine similarity |
| **programs** | `tools/kestreld.cpp` | the daemon and CLI: `serve`, `play`, `check`, `sessions`, `stats`, `import`, `models install/list/verify`, `mcp` |
| | `app/` | the Qt 6 desktop app, a client of `kestreld` |
| | `web/index.html` | the web app `kestreld` serves |
| **tests** | `tools/selftest.cpp`, `hu_check.cpp`, `resolve_check.cpp`, `search_check.cpp`, `server_check.sh` | parity with the Python reference, heads-up, re-solve, search, server |
| **deployment** | `Dockerfile`, `docker/` (compose, deploy, nginx), `testplatform/`, `packaging/` | the image, the lab's compose with the test platform, desktop install |

### 5.2 How the modules depend on each other

```mermaid
flowchart BT
    subgraph CORE["kestrel_core (library)"]
        TABLE["table"]
        ICM["icm"]
        BK["buckets<br/>(7-card evaluator, indexer)"]
        NET["net<br/>(features, forward, end_equity)"]
        SPOT["ninemax_spot"]
        SEARCH["ninemax_search"]
        HU["headsup"]
        RES["resolve"]
        TSV["texassolver"]
        PRACT["practice"]
        LUCK["luck"]
        MODELS["models"]
    end
    subgraph CONN["connectors"]
        CODEC["codec_acpc"]
        CONNECTOR["connector<br/>(transports, retries)"]
        SESSION["session"]
        STORE["store (SQLite)"]
        OPP["opponents, profile"]
        SERVER["server (REST, web)"]
        GATEWAY["gateway<br/>(WS, TCP, MCP)"]
    end
    KESTRELD["kestreld"]
    APP["Kestrel app (Qt)"]

    NET --> TABLE & ICM
    SPOT --> TABLE & NET
    SEARCH --> NET & BK
    HU --> BK & RES & TSV
    PRACT --> NET & ICM
    SESSION --> CONNECTOR & CODEC & SPOT & SEARCH & HU & STORE & OPP & LUCK
    SERVER --> SESSION & PRACT & MODELS & STORE
    GATEWAY --> SERVER
    KESTRELD --> SERVER & GATEWAY
    APP -. "HTTP / WebSocket" .-> KESTRELD
```

### 5.3 A request's path, and the checks on it

```mermaid
flowchart LR
    CL["client<br/>browser, app, script, agent"]
    CF["Cloudflare<br/>(public name only)"]
    NG["lab nginx<br/>only Cloudflare's addresses"]
    subgraph K["kestreld"]
        HOST["Host check<br/>loopback names, or a configured public host"]
        ORI["Origin check on writes<br/>same origin only"]
        TOK["token<br/>cookie or ?token=, constant-time compare"]
        JSON["body parsed and validated<br/>bad input → 400"]
        RT["route<br/>/v1/decide, /v1/sessions, /v1/practice,<br/>/v1/hands, /v1/opponents, /v1/events, ..."]
        SHUT["shutdown<br/>refused through the public name"]
    end
    CL --> CF --> NG --> HOST
    CL -- "on the same machine" --> HOST
    HOST --> ORI --> TOK --> JSON --> RT
    RT --> SHUT
```

The server binds `127.0.0.1` only. Another machine reaches it through an SSH tunnel, or through the public name, which
is served only with the token. Inside Docker every request comes from the Docker network, so every visitor needs the
token there.

### 5.4 Data Kestrel keeps

| Where | What |
|---|---|
| `kestrel.db` (SQLite) | `sessions`, `hands` (key columns plus the whole record as JSON), `opponents` (statistics by platform and name), `spots`, `meta` |
| `hands.jsonl` | the same hands as lines, an export |
| `settings.json` | defaults: model, solver, search samples, TexasSolver path and limits |
| `models/` | installed packages (`ninemax/<name>/`: `model.json`, `net.bin`, test vectors), each file sha256-checked |
| the heads-up pack | the blueprint data from the `headsup-pack-run2` release |
| the token | created on first start; in Docker, all of the above live in the `kestrel-data` volume |

### 5.5 A bot session's life

```mermaid
stateDiagram-v2
    [*] --> Checking: start_session(connector, hands, model, solver)
    Checking --> Connecting: connector file valid, reads only KESTREL_* variables
    Checking --> Refused: invalid file, or it reads another variable
    Connecting --> Playing: platform answers
    Playing --> Playing: hand → decide → act → log to kestrel.db
    Playing --> Waiting: platform unreachable (after the transport's retries)
    Waiting --> Playing: answers again (pauses double up to a limit)
    Waiting --> Failed: down past the limit
    Playing --> Finished: hands played, or stop_session
    Finished --> [*]
    Failed --> [*]
    Refused --> [*]
```

## 6. One decision, end to end

```mermaid
sequenceDiagram
    participant P as Platform (bot API)
    participant C as Connector
    participant S as Session runner
    participant R as Engine router
    participant N as Network / search
    participant D as kestrel.db
    P->>C: table state (cards, stacks, actions)
    C->>S: normalised spot
    S->>R: decide(spot, model, solver)
    R->>N: features → policy (→ search if enabled)
    N-->>R: action probabilities, value
    R-->>S: chosen action
    S->>C: action
    C->>P: send action
    S->>D: log hand, update results and opponent model
```

## 7. Build, release and deployment

```mermaid
flowchart LR
    PUSH["push to main"] --> KY["kestrel.yml<br/>build + ctest: Linux, macOS, Windows"]
    PUSH --> KDY["kestrel-docker.yml<br/>image + self-test → ghcr.io"]
    PUSH --> RPY["research-pipeline.yml<br/>validate run files"]
    HAND["kestrel-release (by hand)<br/>version = server.cpp VERSION"] --> TAG["tag kestrel-vX.Y.Z"]
    TAG --> KY2["kestrel.yml on the tag<br/>→ GitHub release with 3 packages"]
    TAG --> KDY2["kestrel-docker.yml<br/>→ ghcr.io :X.Y.Z, :latest"]
    KDY2 --> DEP["lab: kestrel/docker/deploy.sh<br/>compose: kestreld + test platform"]
    DEP --> NGX["lab nginx → Cloudflare<br/>kestrel.stbfarm.org (token per device)"]
```

The tests every build runs (`ctest`):

| Test | Covers |
|---|---|
| `selftest` | 9-max parity with the Python trainer (nm17), ICM, heads-up decisions |
| `hu_check` | heads-up: stable off-tree mapping, TexasSolver's effective stack, the floor switch |
| `server_check` | the HTTP server: Host/Origin checks, token, malformed input, public-host shutdown refusal |
| `selftest_pko` | bounty features and equity (nm30); practice PKO tournaments add up |
| `selftest_cash6` | 6-max cash (nm37) |
| `search_check` | search scoring, common random numbers, the ESS floor |

## Where to read more

| Topic | Doc |
|---|---|
| Kestrel's design, interfaces, connectors, search | [`kestrel/docs/design.md`](../kestrel/docs/design.md), [`kestrel/README.md`](../kestrel/README.md) |
| The pipeline and its checks | [`research/pipeline/README.md`](../research/pipeline/README.md) |
| Models and the manifest | [`research/models/README.md`](../research/models/README.md) |
| The heads-up solvers and re-solving | [`research/blueprint/model.md`](../research/blueprint/model.md) |
| The current plan, fixes and results | [`research/docs/2026-11-23-review-and-plan.md`](../research/docs/2026-11-23-review-and-plan.md) |
| Rules for working in the repo and on the lab | [`research/AGENTS.md`](../research/AGENTS.md) |
