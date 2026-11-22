# Model pipeline

From a recipe to a model Kestrel can install: `scripts/model_pipeline.py`. Its docstring has the full usage.

| Step | Where | What |
|---|---|---|
| `train RECIPE --run blueprint/nmXX --gpu N` | lab | Launches the trainer with the recipe's settings, detached. The recipe is copied into the run as `pipeline.yaml` |
| `gate RUN --snapshot avg_12h.pt` | lab | Runs the quality gates listed below, with the recipe's thresholds, into `results/models/<run>_gate.json` |
| `package RUN --name N --version V` | lab | Builds `dist/models/<name>-<version>.tar.gz` and its `.sha256`. Refused unless the gate passed |
| `publish PACKAGE` | your machine (gh) | Creates the GitHub Release `model-<name>-v<version>`. The release is recorded in `models/manifest.json` |
| `kestreld models install PACKAGE` | anywhere | Kestrel checks the archive against its `.sha256` and every file against `model.json`, then installs it |

The quality gates:
- poker-sense checks within their bands;
- premium hands raised first in;
- push/fold agreement with the ICM chart;
- not clearly worse than the game's champion (`models/manifest.json` → `champions`) in a 1-vs-8 duplicate match;
- Kestrel's C++ self-test on the new network.

The package holds:
- the fp16 weights and the run's config;
- Kestrel's `net.bin` and `net.json`;
- the test vectors;
- `model.json`, which carries the recipe, training statistics, gate results and every file's sha256.

Recipes live in `recipes/`. `mtt9.yaml` is nm17's recipe.

## First run (2026-11-14): nm17 shipped as mtt9 version 1

`gate blueprint/nm17 --snapshot avg_12h.pt` passed all five gates:

| Gate | Result | Needed |
|---|---|---|
| Poker-sense checks in band | 8 of 14 | at least 7 |
| Premiums raised first in | yes | yes |
| Push/fold agreement | 0.904 | at least 0.85 |
| Against the champion (nm17's export) | −0.7 [−1.3, −0.1] bb/100 | not worse by more than 1 bb/100 |
| Kestrel's self-test | PASS | PASS |

The package was 8.5 MB, published as the release `model-mtt9-v1`. From that download, `kestreld models install` checked every file, and Kestrel's self-test passed on the installed network. A tampered copy was refused.

Two lessons from this run:
- The self-test gate has to check the network as it ships, in half precision. Reference data from the full-precision snapshot is off by up to 5e-3 in the logits.
- The champion gate needs a margin. A model played against its own export read −0.7 [−1.3, −0.1] over 20,000 deals.
