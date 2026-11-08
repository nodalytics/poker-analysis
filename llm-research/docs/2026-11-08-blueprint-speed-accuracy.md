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
