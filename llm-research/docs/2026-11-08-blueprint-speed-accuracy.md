# Blueprint: results so far, and making it faster and more accurate — 2026-11-08

All figures are bb/100 at 200bb heads-up, with 95% intervals; Slumbot and LBR figures are luck-adjusted.

## Results so far

| Opponent | Blueprint result | Notes |
|---|---|---|
| Slumbot (5,000 hands each) | +1.0 [−30, +32] (run 2 early); +12.4 [−16, +41] (run 2 final); **+30.9 [−10, +72] (run 2 + defence floor)** | All positive, none yet significant on its own |
| Max (maniac) | +178.7 [+91, +271] | |
| Carl (calling station) | +82.3 [+37, +130] | |
| Monte (equity bot) | +73.9 [+27, +119] | |
| Nora (nit) | +55.9 [+45, +68] | |
| Tina+ | +50.0 [+17, +83] | |
| Bob (recreational) | +36.4 [+5, +69] | |
| Tina (tight-aggressive) | +27.5 [−6, +63] | |
| Our solver bot | +26.3 [+6, +46] | |
| Big-Tiger-Gemma 27B (LLM) | +78.5 [+69, +86] | 400 hands |
| Qwen2.5-VL 32B (LLM) | +97.5 [−58, +279] | 400 hands |
| Adaptive bot, Tiger-Gemma 9B, the Qwen toolkit agent, the Big-Tiger + solver hybrid | running | |

**LBR, a best-responding opponent: lower is better, and 0 means LBR finds nothing.**

| Blueprint bot | LBR's edge |
|---|---|
| Run 2 final, no purification | +218 [+150, +287] |
| Run 2 final, 5% purification | +135 [+68, +202] |
| **Run 2 final, purification + minimum-defence floor** | **+104 [+44, +164]** |
| Run 2 over training (4.7 → 21.5 billion iterations) | levels off at about +140 |
| Run 3 (1,000 buckets) at 19.5 billion | +173 [+89, +258]: finer average-strength buckets did not help |
| **Unsafe live re-solving + floor** | **+996 [+733, +1,260]** |

### What these show

1. **The blueprint beats every engine and LLM we have.** It is level with or slightly ahead of Slumbot.
2. **The limit is the abstraction, not training time:**
   - more iterations stopped helping after about 5 billion;
   - five times finer average-strength buckets did not help either.

   Hence run 4 (distribution-aware buckets) and run 5 (better training).
3. **The big leak was over-folding to large bets:** 61% / 71% / 79% on flop / turn / river, against a 50% limit.
   The minimum-defence floor fixed most of it.
4. **Unsafe re-solving is a trap.** Solving a spot with the blueprint's idea of the opponent's range lets a real
   opponent who plays differently exploit the solution. That is why Libratus used *safe* subgame solving. Re-solving
   stays off until the safe version is built.

## Speed and accuracy: ideas from solvers, now in the trainer

`mccfr.cpp` builds as `mccfr3`. The new behaviour is behind flags, so old runs stay reproducible.

| Idea | Source | What it does | Flag |
|---|---|---|---|
| **Discounted CFR** | Brown & Sandholm 2019; how modern solvers weight iterations | Each period (60 s of sampling), positive regrets × t^1.5/(t^1.5+1), negative regrets × ½, average strategy × (t/(t+1))². Early, noisy iterations fade, so it converges in far fewer iterations | `--dcfr --dcfr-seconds 60` |
| **Regret-based pruning** | Pluribus | After a warm-up, 95% of the time the traverser skips actions with regret below −C (folding the nuts, for example). Regret matching gives those actions probability 0, so the value is unchanged and only wasted work goes. Typically 2–3× more iterations a second | `--prune-after-minutes 30 --prune-threshold 100000` |
| Vector updates over whole ranges (public chance sampling) | TexasSolver's core; Johanson et al. 2012 | Update every hand for a dealt board at once, not one hand pair per iteration | next: a rewrite of the traversal |
| Safe subgame solving | Libratus | Re-solve so the result cannot be worse than the blueprint against any opponent | roadmap item 2 |

### The A/B test running now

| | Run 4 | Run 5 |
|---|---|---|
| Abstraction | distribution-aware, 500 buckets per street (`out_pa`) | the same |
| Training | MCCFR, linear discounting | **DCFR + pruning** |
| Length | 24 h, 16 threads | 24 h, 16 threads |
| Measured by | iterations per second; the LBR curve every checkpoint | the same |

## Ideas from PioSolver

PioSolver is a commercial, closed-source GTO solver. These ideas come from its public documentation and common
practice, not its code.

| Idea | What it is | How we use it |
|---|---|---|
| **Node locking** | Fix the opponent's strategy at some spots to how they really play, then re-solve the rest. It is the standard tool for exploiting a known leak | **Phase B, opponent adaptation.** Lock the opponent's observed frequencies (from Hand Sync or opponent models) in the re-solve, blended with the blueprint for safety |
| **Accuracy target as % of pot** | Solve until exploitability falls below, say, 0.3% of the pot, and report it | Report the re-solver's exploitability in % of pot; stop when it is reached instead of after a fixed iteration count. Faster where spots are easy |
| **Representative flop subsets** | About 184 flops, weighted to stand in for all 1,755, for analysis | Evaluate and report strategies on a weighted subset instead of every flop. A faster LBR and abstraction check |
| **Bet-tree rules** | Add all-in when stack-to-pot is low; force all-in when little would remain; geometric sizes | Our tree already turns raises that leave under a third of the stack into all-in. Add an SPR-based all-in rule and a geometric size where LBR finds leaks |
| **Rake** | Solve with rake taken out of the pot | Model real-site rake in the advisor (5% up to a cap) so its advice matches real games |
| **Compact strategy storage** | Strategies stored in fewer bits | Store the blueprint's average strategy as 8-bit probabilities (about 4× smaller) so the Windows app can carry it |
| **Aggregate reports** | Frequencies and EV over many flops at once | Blueprint reports: c-bet and fold-to-bet frequencies by texture, and where LBR wins |

## Next, in order

1. **Read the run 4 vs run 5 A/B:** iterations per second, and the LBR curve on each.
2. **Vector traversal** (public chance sampling) as `mccfr4`, if run 5's results justify the rewrite.
3. **Safe re-solving** (roadmap item 2): our own vectorised turn/river solver with a safety gadget. Then **node
   locking** on top of it for opponent adaptation (phase B).
4. **8-bit strategy storage**, so the blueprint ships in the advisor app.
5. **GPU tracks:** PyTorch is installing on the lab; AlphaHoldem-style RL first, then Deep CFR / ReBeL.

## Ideas from other fields (2026-11-08)

| Field | Idea | What it would do for the blueprint |
|---|---|---|
| Maths: online convex optimisation | **Predictive / optimistic regret matching (PCFR+)**: predict the next regret update from the last | Often converges several times faster than CFR+/DCFR in published tests. Speed; cheap to try as run 6 |
| Numerical analysis: multigrid | Solve a coarse version first, then use it to start the fine one | Start the 500/1,000-bucket blueprint from the solved 200-bucket one instead of from zero. Speed |
| Particle physics / Monte Carlo | **Quasi-Monte Carlo and stratified sampling**: deal cards from low-discrepancy sequences, balanced across board textures | Less noise per iteration, the same accuracy in fewer iterations |
| Maths: optimal transport | **True potential-aware abstraction**: Wasserstein distance between distributions over next-street buckets | The next step beyond run 4's one-dimensional version (Johanson et al. 2013). Accuracy |
| Control theory | **Kalman filter** for drifting opponent tendencies; **bandits** (Thompson sampling) to choose how hard to exploit | Phase B: fast, risk-controlled adaptation |
| Ecology / evolution | **PSRO** (a population of strategies evolving against each other) and **population-based training** | GPU deep-RL track: robust bots that do not overfit to one opponent |
| Signal processing | Low-rank and 8-bit compression of strategy tables | Shrinks the blueprint so the Windows app can carry it |
| Statistical physics | Temperature / annealing (quantal response, entropy-regularised mirror descent) | Smoother early training; modest gain |
| Information theory | Minimum description length to choose bucket counts per street | Principled abstraction sizes |
| Topology / geometry | Persistent homology on hand-strength space | Weak here: our earlier topology studies found nothing beyond average strength; skipped |

Proposed next as A/B runs: PCFR+ (run 6), a multigrid start, and stratified quasi-random dealing.

## Engineering: towards 1 million iterations per second

`mccfr4` adds these, with the same maths unless a flag says otherwise:

- **Lazy buckets and ranks.** A hand's flop, turn and river buckets and its showdown rank are computed only when the
  traversal reaches that street. Many hands end before the flop.
- **A faster random generator:** xoshiro256** instead of mt19937_64.
- **2 MB huge pages** for the regret and strategy tables (`madvise`), for fewer slow address lookups on random
  access.
- **Two traversals per deal:** both players update on each dealt hand, sharing its lookups.
- **New maths flags:**
  - `--pcfr`: predictive CFR+;
  - `--stratified`: every thread cycles through all 1,326 starting hands;
  - `--init-from RUN --init-abs DIR`: multigrid start from a coarser solved run.

**Measured so far:**

| Build | Iterations per second | Conditions |
|---|---|---|
| Run 2, 24 threads | 520k (22k per thread) | lab moderately busy |
| Run 3, 16 threads | 353k (22k per thread) | 1,000 buckets |
| `mccfr4`, 16 threads | 250–307k | lab oversubscribed: load 94 on 64 cores, so not a fair comparison |

**Path to 1M per second:**

1. Re-benchmark on a quiet lab.
2. Then one run on about 56 dedicated threads.
3. Pruning (run 5) should add up to 2–3× once regrets settle.
4. Vector traversal per dealt board (public chance sampling) is the next multiplier if still short.

## Chaos theory and fractals (2026-11-08)

| Idea | Relevance | Verdict |
|---|---|---|
| **Chaos in learning dynamics** (Sato, Galla & Farmer; Bailey & Piliouras) | Regret-based learning in games can cycle or turn chaotic instead of converging. That is why CFR averages strategies, and why predictive CFR+ damps the cycling | Relevant. Measure the Lyapunov exponent of our training (do nearby starts diverge?) to check that averaging and optimism do their job |
| **Self-similarity of the game tree** | Bet and raise subtrees repeat on every street | Already used: abstraction and strategy sharing |
| **Hurst exponent** (rescaled range, DFA) on players' result and style series | Do tilt and style regimes persist longer than chance? | A useful niche test; extends the regime study on HandHQ |
| **Fractal dimension of strategy landscapes** | | Little practical value; skipped |

## One million a second on one core (2026-11-08)

**The target as first stated cannot be met.** One external-sampling iteration walks one dealt hand down the tree,
and most of its time is spent waiting on memory:

- a regret-table read at every decision, a random address in 1.1 GB;
- at roughly 100 ns a miss and some 20–40 decisions an iteration, a core tops out near 25–50k iterations a second;
- we measure 22k.

No instruction-level trick (SIMD, assembly) helps a program that is waiting on memory. One million such
iterations a second on one core is about 20× past that bound.

**Three routes were considered:**

| Route | Idea | Verdict |
|---|---|---|
| **A. Vector updates** (public chance sampling) | Deal a board, then update every hand of both players at once: about 1,081 × 1,081 hand pairs a traversal, as array arithmetic | **Chosen.** Measured in hand pairs a second per core, and judged by LBR's edge per core-hour |
| B. Smaller tables | 8-bit regrets, fewer buckets, so more of the table fits in cache | Helps a little; costs accuracy |
| C. Many cores | Already done: 16–56 threads a run | Scales total speed, not per core |

### Route A built: `mccfr_vec`

`blueprint/mccfr_vec.cpp` uses the same tree (now shared as `tree.hpp`), abstraction and table layout as `mccfr`,
so `blueprint_bot.py` and LBR read its runs unchanged.

- **Fold values** with card removal by inclusion–exclusion over each hand's two cards.
- **Showdown values** by two sorted passes with per-card sums and tie groups.
- A **built-in self-test** checks both against brute force on every start. Maximum error 0.0000 (fold) and 0.0002
  (showdown, float rounding).
- **Zero-reach skipping:** a node the opponent cannot reach with any hand is skipped. External sampling never visits
  such nodes either, so the two trainers compute the same quantities.
- DCFR discounting, checkpoints and resume as in `mccfr`.

**Measured (lab load about 84 on 64 cores, so not a clean number):**

| Trainer | Per core | Unit |
|---|---|---|
| `mccfr` external sampling | 13–22k | hands (one hand pair) a second |
| **`mccfr_vec`** | **0.48–0.56M** | **hand pairs a second** |

Hand pairs are not the same unit as external-sampling iterations: a vector traversal also walks every action at
every node instead of sampling one. **The honest comparison is LBR's edge per core-hour.** `runvec` (8 threads,
24 h, DCFR) runs alongside runs 4–6 with its own LBR curve.

### Profile, and the assembly question

`perf` is blocked on the lab (no root), so the profile is from `gprof` on a single-core benchmark:

| Part | Share of time |
|---|---|
| The tree walk: strategy from regrets, regret and average updates per hand | 66% |
| Showdown values | 25% |
| Fold values | 9% |

The walk's time goes on reads and writes into the regret table, gathered by each hand's bucket, which is memory
again. Hand-written assembly is not worth it:

- the compiler already emits AVX-512 for the plain loops (`-O3 -march=native` on the lab's Zen 5 Threadripper);
- the hot part waits on memory, not arithmetic.

If a later profile shows compute-bound loops, AVX-512 *intrinsics* give the same instructions as assembly and stay
maintainable.

The bigger wins are algorithmic:

- compute each strategy once per **bucket** instead of once per hand (about 400 buckets against 1,081 hands on a
  river);
- skip actions whose regret is very negative (the pruning of run 5);
- sample turn and river cards per street, so the preflop and flop work is shared across many run-outs.

## Status of every run (2026-11-08, 09:30 lab time)

| Run | Trainer | State |
|---|---|---|
| run2 | `mccfr`, 200 buckets | Finished: 21.5 billion iterations; the bot in use |
| run3 | `mccfr2`, 1,000 buckets | Stopped at a 20.3 billion checkpoint (resumable); LBR +173 at 19.5 billion, worse than run 2 |
| run4 | `mccfr2`, out_pa (A/B baseline) | Restarted 08:59, 24 h, 16 threads; about 200k iterations a second under load |
| run5 | `mccfr3`, out_pa, DCFR + pruning | Restarted, 24 h, 16 threads |
| run6 | `mccfr4`, out_pa, PCFR+ | Restarted, 24 h, 16 threads |
| runvec | `mccfr_vec`, out_pa, DCFR | Started 09:27, 24 h, 8 threads |
| Flop library | TexasSolver | Paused at 122 of 240 flops (resumable) |

Each run has `lbr_trend.sh` scoring every 30-minute checkpoint with 3,000 LBR hands. The GPU environment
(`.venv-gpu`, PyTorch 2.9.1 with CUDA) is installed and working.

## Answers given in conversation, recorded

- **Is every move legal?** Every action goes through PokerKit's rules engine, and Slumbot's server checks every
  action we send. No illegal action was recorded in 92,780 self-play decisions or 31,000 Slumbot hands.
- **Does the blueprint learn by itself?** Yes, by self-play: MCCFR plays the simplified game against itself
  billions of times, and regret minimisation moves it toward an equilibrium. It does not learn from opponents
  during play; that is phase B (opponent adaptation) and phase C (continual learning from Hand Sync), in
  `2026-11-07-blueprint-plan.md`.
- **Ideas from LLMs:** tested and listed in `2026-11-08-llm-ideas-for-the-blueprint.md`. Attention did not beat a
  small MLP at predicting Slumbot's actions, and end-to-end learning from tokens failed at our data scale.

## LBR curves, 3 hours in (2026-11-08, 12:25 lab time)

LBR's edge in bb/100 (lower is better), 3,000 hands a point, so each interval is about ±65. These are all with
purification and the defence floor, so compare with run 2's **+104** and its plateau near +140.

| Run | Trainer | Threads | Points so far (oldest → newest) | Recent level |
|---|---|---|---|---|
| run4 | MCCFR, distribution-aware buckets | 16 | +89, +98, +130, +121, +81, +108 | about +100 |
| run5 | + DCFR + pruning | 16 | +91, +46, +54, +44, +87, +70 | about +60 |
| **run6** | **+ PCFR+** | 16 | +27, +18, +10, +44, +63, +21 | **about +30** |
| **runvec** | **vector CFR + DCFR** | **8** | +105, +71, +27, +30, +40 | **about +35** |
| runmlp | MLP self-play, GPU | 1 GPU | +165, +607, +567 | getting worse |

- **The distribution-aware buckets alone (run 4) already match run 2's best.**
- **PCFR+ (run 6) and the vector trainer (runvec) are the best so far,** about +30.
- **The vector trainer gets there on half the threads.** Per core-hour it leads, which is what route A was chosen
  for.
- The intervals overlap, so the ranking needs the full 24 hours.

Iterations a second: run 4 177k, run 5 321k (pruning nearly doubles it), run 6 191k, runvec 3.7 vector iterations
a second (each about 1.2 million hand pairs).

**A mirror-match check:** the blueprint against itself (1,500 sessions, data for distillation) read +19.9
[+0.7, +38.8] for one copy. Two identical bots must be level, so this is a 1-in-20 fluke at the 95% level, or a
seat asymmetry in bot_sim worth checking before trusting a small head-to-head edge.
