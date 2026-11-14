<p align="center"><img src="assets/logo.svg" width="112" alt="Kestrel"></p>

# Kestrel

Kestrel is the C++ poker advisor and self-playing bot (phase D). It covers 9-max tournaments, with heads-up as a second mode. It plays by itself on platforms that offer a bot API, through connectors described in a file. It serves REST, WebSocket, TCP and MCP on `127.0.0.1`, with a web page and a Qt app on top.

Design, boundary and order of work: [`docs/design.md`](docs/design.md). Kestrel stands alone at the repo root; the
research that feeds it (training, the models, the Python reference) is in `llm-research/`.

## Start it

Double-click `kestreld` (`kestreld.exe` on Windows) in the release's `bin/` folder, or run it with no arguments. It starts the app and opens it in your browser. If it's already running, it just opens it. On Linux, `share/kestrel/install-desktop.sh` adds Kestrel to the applications menu.

It finds its web app, logo and connectors next to itself, and your own connectors and models in `~/.config/kestrel/connectors` and `~/.local/share/kestrel/models`. The heads-up data pack goes in `~/.local/share/kestrel/pack`, or set `KESTREL_PACK`.

**From other devices on your network:** put `{"lan": true}` in `~/.config/kestrel/settings.json` and start Kestrel again. The "On your network" button shows the links to open on a phone or another PC. Each link carries an access token, and devices without it are refused.

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

The web app shows every running table live, each on its own screen. Switch screens with tabs or keys `1`–`9`, press `g` for a grid of all tables, `n` to start a new table and `t` to change theme. Each screen shows:
- the cards, the pot and the betting as it happens;
- Kestrel's decision, with its probabilities;
- the session's results.

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

It also checks the payoffs and ICM, and for heads-up the blueprint's node, bucket, strategy and raise sizes. It prints `PASS` only if all of them match.

The data it reads (the network `models/ninemax/nm17/net.bin` and the heads-up pack `blueprint/pack/`) defaults to `../llm-research`. Set `-DKESTREL_DATA=...` to point elsewhere.

| Path | Contents |
|---|---|
| `core/` | game state (port of `ninemax_engine.Tables`), ICM, network features and forward pass, heads-up blueprint and buckets |
| `connectors/` | platform codecs (`acpc`: Slumbot's action strings); transports and the session runner next |
| `docs/design.md` | the design |
| `assets/` | the logo: `logo.svg` (app icon), `logo-mark.svg` (the bird), `logo-animated.svg` (hovering, for busy states) |
| `connectors/*.yaml` | connector files; `slumbot.yaml` is the worked example |
| `tools/selftest.cpp` | the self-test |
| `testdata/` | reference vectors from `scripts/advisor_vectors.py` |
