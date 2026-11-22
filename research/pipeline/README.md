# The research pipeline

Two parts: **runs** (`runs/`, `scripts/pipeline.py`) train and judge a network on the lab when a pull request is
merged; the **model pipeline** (`recipes/`, `scripts/model_pipeline.py`) packages an adopted network for Kestrel.

## Runs

Train, test, judge and report a 9-max network without logging in to the lab. **A recipe merged into `main` is a
run**: the lab picks it up, trains it, runs every check the recipe lists, writes the verdict and opens a pull
request with the results. Progress shows on the dashboard (https://poker.stbfarm.org, "Pipeline").

### Propose a run

1. Copy [`example.json`](example.json) to `runs/<name>.json`. The name is `nm` and a number not used before
   (the last ones are in `../models/manifest.json` and `../results/prereg/`).
2. Say what it asks (`question`) and why (`why`), what to train (`train`), and how it is judged (`checks`).
3. Open a pull request. CI checks the recipe (`scripts/pipeline.py validate`); a reviewer reads it.
4. Merge it. Within two minutes the lab starts the run. The merge commit is the pre-registration: the checks
   and their pass rules cannot change once it runs, and a failed run is reported, not re-run until it passes.
5. When it finishes, the lab opens a pull request titled `<name>: adopted` or `<name>: not adopted` with the
   results (`results/pipeline/<name>/`, `results/prereg/<name>.result.json`). Merge it to keep the record. An adopted
   network then ships through the model pipeline below.

One run at a time. A run is refused when the lab has less than 25 GB of disk free.

### The run file

```json
{
 "name": "nm44",
 "question": "What the run answers, in a sentence.",
 "why": "What made us ask: a finding, a player's report, a leak.",
 "train": {"init": "nm25", "minutes": 360, "gpu": "any", "args": {"early": 0.25}},
 "checks": [ ... ],
 "adopt": "all"
}
```

| Field | Meaning |
|---|---|
| `train.init` | the run it starts from (its `blueprint/<init>/checkpoint.pt`) |
| `train.minutes` | training time, 10 to 720 |
| `train.gpu` | `any` (the first free: PRO 6000, then 4090, then 3090), `pro6000`, `4090` or `3090` |
| `train.args` | options of `scripts/ninemax_selfplay.py` over nm25's settings (`--early 0.25` is `"early": 0.25`; a flag is `true`, `false` drops one of nm25's) |
| `train.opponents` | extra opponents in the training population (below) |

#### Training opponents

The stronger and more varied the opponents a network trains against, the less it can get away with habits that
only work against itself. `train.opponents` adds, to nm25's rule players, past snapshots and exploiters:

| Field | Adds |
|---|---|
| `versions` | earlier networks, frozen: a list (`["nm43", "nm17"]`) or `"auto"`: the 10 newest that read the same inputs, from nm17 on |
| `generated` | that many generated styles (`seed` picks them): the network's own running average with its tendencies pushed, so each is a strong player with a leak, and it gets stronger as training goes |
| `archetypes` | the five archetypes as styles: nit, station, maniac, shover, loose-passive |
| `human` | styles fitted to real player types from 2009 online hands (below); built automatically the first time a run asks for them |
| `patterns` | every pattern people found that beats the bot (`pipeline/patterns/`, and the ones mined from the practice tables, re-mined before training): played by the network's own average except in the pattern's spot |

With `--pfsp` (opponents picked by how often they beat the network) the hardest of them come up most. Example:
`"opponents": {"versions": "auto", "generated": 12, "archetypes": true, "human": true}`.

#### Human styles

`scripts/human_styles.py` turns published hand histories into opponents, in three steps the pipeline runs by
itself when `results/human_styles/styles.json` is missing:
1. **players:** each player's VPIP, PFR, 3-bet, post-flop aggression and fold to a bet, from 150,000 hands of each
   of the 27 HandHQ stakes (2009, six or more players): 4,633 players with 500 hands or more.
2. **types:** the players in 8 types by k-means (`types.json`): from nits (VPIP 16%, folding to 64% of bets) and
   tight-aggressive regulars (most hands) to loose-passive (36%) and loose-aggressive (51%) players.
3. **fit:** for each type, the style of nm25 whose own statistics come closest (`styles.json`, with each type's
   target and what the style achieves). The fit is approximate: nm25's styles fold to bets less than the tightest
   humans do. The humans played ~100bb cash games, the networks 10-25bb tournaments.

The `styles` check plays them too with `"human": true`.

#### Patterns people find

When a friend or partner finds something that beats the bot, write it down in `pipeline/patterns/` (format in
[`patterns/README.md`](patterns/README.md)), or let the miner find it: `scripts/exploit_patterns.py mine` reads every
hand played at Kestrel's practice tables and keeps the spots where the human's move won reliably (15 hands or more,
2 standard errors above zero). The first: from the heads-up button at 25bb+, all-in when unopened, +2.08 bb a hand
over 15 hands. Every pattern is then tested against each new network and can be trained against.

#### After every run

The pipeline re-runs the version consistency report (`scripts/ninemax_consistency.py`: every version on the same
spots, the new one included) into `results/ninemax/consistency.json`.

nm25's settings (`BASE_TRAIN` in `scripts/pipeline.py`): `--tourney --no-open-limp --menu mtt --depths 10,25
--ante 1 --short 0.5`, population training with the rule players (`--pool reg,reg,tag,maniac,station
--pool-share 0.5`) and an exploiter in training (`--exploit-share 0.25`), NeuRD with a KL pull of 0.3.

#### Checks

Every check must pass for `adopted`. Each has an `id` (your name for it) and a `type`:

| Type | What it does | Fields | Passes when |
|---|---|---|---|
| `seqtest` | sequential duplicate match, the run vs 8 copies of `against` (`seqtest.py`) | `against`, `early` (share of early-stage tables, 0-1), `max` deals, `alpha`, `margin` | `pass`: `ahead` (stops for a difference, ahead), `equivalent_or_ahead`, or `not_behind` (not stopped behind, and not ending below -0.25) |
| `opponents` | the run and `baseline` each against 8 copies of rule players that play nothing like it, on the same deals (`pipeline_opponents.py`) | `against` (some of `reg`, `tag`, `maniac`, `station`, `random`), `baseline`, `hands`, `tolerance` | it is no more than `tolerance` bb/100 below the baseline against every one |
| `styles` | generated opponents: nm25-like players with their tendencies pushed by random amounts (looser, tighter, more aggressive, folding more or less to bets, shoving, bluffing weak hands), plus five archetypes (nit, station, maniac, shover, loose-passive); the run and `baseline` each against 8 copies of every one, on the same deals | `count` (generated, 0-60), `seed` (the same seed is the same crowd for every run), `hands`, `baseline`, `worst_tolerance`, `mean_tolerance`, `human` (also the human-fitted styles) | it is within `worst_tolerance` of the baseline against its worst style, and within `mean_tolerance` on average |
| `consistency` | where the run decides differently from `reference` on the same few thousand spots, by street, position, stack, facing a bet and players seated, with the most disputed spots (`ninemax_consistency.py`) | `reference`, `deals` | always: it is a report for the reviewer, not a gate |
| `bluffers` | scripted bluffers in one seat vs 8 copies of the run (`ninemax_hybrid.py`) | `bluffers` (some of `jam-any`, `jam-40`, `raise-any`), `early`, `unit` (`chips` or `equity`) | every bluffer loses (95% interval below 0) |
| `patterns` | every pattern people found (`pipeline/patterns/*.json`, written by hand) or mined from the practice tables (`results/exploit_patterns/mined.json`, re-mined first): each played by the run itself except in its spot, against 8 copies of the run (`exploit_patterns.py test`) | `baseline`, `hands`, `limit` (default 0), `mine` (default true) | no pattern wins more than `limit` bb/100 against the run: a pattern found once stays closed |
| `opponent-model` | does Kestrel's opponent model help? Kestrel itself (the image the lab runs) with the run, against the test platform's bots, which keep their names and styles (calling station, limper, tilter, style-switcher, two allies who soft-play each other, odd sizers): no solver, search, and search with the opponent model (`search-exploit`), each with a fresh opponent database (`kestrel_opponent_check.py`) | `hands` per arm and seed, `seeds`, `samples` | always: a report of what the model adds over search and over the network alone |
| `exploiters` | a battery of exploiters that attack differently: `copy` (the standard one), `explore` (keeps trying odd moves, entropy 0.02), `fast` (learns fast and commits, lr 1e-3) and `outsider` (starts from nm17 and searches elsewhere); or your own `profiles` (`id`, `lr`, `entropy`, `init`, `minutes`) | `minutes` each, `limit`, `profiles` | the best of them wins at most `limit` bb/100 |
| `exploit` | a learner trains for `minutes` to beat the run (`ninemax_eval.py exploit`) | `minutes`, `limit` | the exploiter's last reading is at most `limit` bb/100 |

Matches against copies of nm25 alone only say how a run does against nm25. A good run file also has
`opponents` and `styles` checks and an `exploit` check, so a run cannot pass by learning to beat its own family.

### Switching it on (once, on the lab)

1. **Start the service** (needs sudo, so the lab's owner runs it in a terminal on the lab):
   ```
   sudo sh ~/research/pipeline/deploy/install.sh
   ```
   It installs `research-pipeline.service`, starts it now and at every boot, and prints its status, which
   should read `active (running)`. `sudo sh ~/research/pipeline/deploy/install.sh --remove` takes it out.
2. **Optional: let the lab open the results pull requests.** On GitHub, create a fine-grained personal access
   token limited to `nodalytics/poker-analysis`, with **Contents** and **Pull requests** read and write. Save it on
   the lab:
   ```
   mkdir -p ~/.config/research-pipeline
   (umask 077; cat > ~/.config/research-pipeline/token)    # paste the token, Enter, then Ctrl-D
   chmod 600 ~/.config/research-pipeline/token
   ```
   Without it the results stay on the lab, and the dashboard's Pipeline panel says so. Never commit or paste it.

After a change to `pipeline/deploy/` or the poll loop, run step 1 again; a run in progress is not stopped.

Check it: `systemctl status research-pipeline` (running), `tail ~/research/results/pipeline/poll.log` (errors, if
any), and the Pipeline panel on the dashboard.

### On the lab

The service (`deploy/research-pipeline.service`) runs `scripts/pipeline.py poll`. It fetches `main` from GitHub every two minutes (the repo is
public: it needs no access to fetch). Before a run it copies `main`'s code (`scripts/`, `pipeline/`, `web/`)
over the lab's copy, so the code that runs is the merged code; results on the lab are never overwritten.
Nothing on GitHub can send the lab a command: it only reads `main`.

- State and logs: `results/pipeline/<name>/` (`state.json`, `train.log`, one `.json` and `.log` per check).
- Status: `python3 scripts/pipeline.py status`.
- Stop a run: `touch results/pipeline/<name>/STOP`.
- Rerun the checks of a run whose training finished: `python3 scripts/pipeline.py run pipeline/runs/<name>.json --from checks`.
- Results pull requests need the token from step 2 above.

## The model pipeline: shipping a network

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

### First run (2026-11-14): nm17 shipped as mtt9 version 1

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
