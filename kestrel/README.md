# Kestrel

Kestrel is the C++ poker advisor and self-playing bot (phase D). It covers 9-max tournaments, with heads-up as a second mode. It plays by itself on platforms that offer a bot API, through connectors described in a file. It serves REST, WebSocket, TCP and MCP on `127.0.0.1`, with a web page and a Qt app on top.

Design, boundary and order of work: [`docs/design.md`](docs/design.md). Kestrel stands alone at the repo root; the
research that feeds it (training, the models, the Python reference) is in `llm-research/`.

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
| `tools/selftest.cpp` | the self-test |
| `testdata/` | reference vectors from `scripts/advisor_vectors.py` |
