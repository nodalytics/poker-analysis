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
