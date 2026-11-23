---
name: kestrel-release
description: Ship Kestrel or a model for it — rebuild Kestrel, run its self-tests and the resolver parity check, gate and package a 9-max network with model_pipeline.py, publish the model release, bump Kestrel's version and run the release workflow that tags kestrel-v<version> and builds Linux, macOS and Windows packages. Use it whenever the user says ship, release, publish, package, tag, bump the version, cut a build, or "get nmXX into Kestrel", and before any change to what Kestrel plays by default.
---

# Releasing Kestrel and its models

There are two kinds of release:
- **Models:** GitHub releases named `model-<name>-v<version>`, holding a `.tar.gz` with a `.sha256`.
- **Kestrel builds:** GitHub releases named `kestrel-v<version>`, holding the Linux, macOS and Windows packages.

The heads-up data pack (`headsup-pack-run2`) is a release that the builds download. Nothing ships unless its gate
passed: a model that fails a gate stays in `research/blueprint/`.

## A model (from a trained run)

Run these on the lab, from `research/`, with `PYTHONPATH=scripts .venv-gpu/bin/python` (see lab-runner):

1. **Adopt first, by the run's own registration.** Every check in its `nmXX.json` must pass, judged as in
   model-judging. "Better on one check" isn't adoption.
2. **Gate:**
   `scripts/model_pipeline.py gate blueprint/nmXX [--snapshot avg_Nh.pt]`. It covers the poker-sense checks, push/fold
   agreement, a duplicate match against the current champion in `research/models/manifest.json`, and Kestrel's C++ self-test on
   the network. Its output goes to `results/models/<name>_gate.json`.
3. **Package:**
   `scripts/model_pipeline.py package blueprint/nmXX --name <name> --version <V>`. This writes
   `dist/models/<name>-<V>.tar.gz` and its `.sha256`, and it refuses if the gate failed.
4. **Publish,** on your own machine (it needs gh):
   `uv run python scripts/model_pipeline.py publish dist/models/<name>-<V>.tar.gz`. This creates the release and
   records it in `research/models/manifest.json`. Commit the manifest change (publish-safety).
5. **Check the install:** `kestreld models install <file>` (it verifies every file's sha256). Then play a short
   practice table to see that it loads and acts.

## Kestrel itself

1. **Rebuild from current code:** `cmake --build kestrel/build -j8`. Before testing, compare a binary's date with
   `git log -1 -- kestrel/`.
2. **Tests, all of them:**
   - `ctest --test-dir kestrel/build --output-on-failure`;
   - `kestrel_selftest --vectors DIR --model MODEL_DIR --pack PACK_DIR` (Python-C++ parity on the network's test
     vectors);
   - `kestrel_resolve_check --vectors FILE --run DIR --iters 500` (the C++ re-solver against the reference);
   - a live smoke test: `kestreld serve --port <free> --lan off --pack research/blueprint/pack --models <dir>`, then
     `/v1/health`, `/v1/engines` (the models available, practice true) and one practice table.

   Stop the test server by its port's PID.
3. **Version:** bump `VERSION` in `kestrel/connectors/src/server.cpp`, and kestreld's MCP version to match. The
   release workflow refuses a mismatch or an existing tag.
4. **Release:** run the `kestrel-release` workflow by hand on GitHub:
   `gh workflow run kestrel-release.yml -f version=<V> [-f ref=main] [-f dry_run=true]`.
   - Do a dry run first.
   - It tags `kestrel-v<V>` and starts the build workflow, which runs ctest and attaches the three packages to the
     release.
   - To redo a failed release: `gh release delete kestrel-v<V> --cleanup-tag`, then run it again.
5. **Watch it finish:** `gh run list --workflow kestrel.yml`. Then check that the release has all three platforms
   (`gh release view kestrel-v<V>`).

## Rules that ship with it

- Kestrel's advice is for study and its own bot tables. Don't add a connector for a real-money site; the README
  forbids bots, OCR and real-time assistance there.
- A play rule (like the early-shove rule) is changed only on replicated evidence (model-judging), and its doc entry is
  updated in the same commit.
