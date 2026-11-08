# Blueprint plan: re-solving, then opponent adaptation, then continual learning — 2026-11-07

## Where we are

- **Blueprint v1** (`blueprint/run2`): heads-up no-limit at 200bb.
  - Abstraction: 200 card buckets per street; open 2–3bb; 3-bet 3×/4×; post-flop ⅓, ⅔, 1× and 2× pot; all-in.
  - Training: sampled CFR self-play at about 520k hands per second.
  - Measured after about 5–8 billion iterations:

    | Measure | Result (bb/100) |
    |---|---|
    | Against Slumbot, luck-adjusted, 5,000 hands | +1.0 [−30, +32] |
    | Against our solver bot, 20,000 hands | +26.3 [+6, +46] |
    | LBR, luck-adjusted | +152 → +122 → +119 → +124 |

    LBR's edge has levelled off.
- **Blueprint v2** (`blueprint/run3`): 1,000 buckets per street (16-bit tables, 2.2 GB), training for 24 hours.
- **Integration:** the blueprint drives the advisor for heads-up spots with deep stacks (60bb or more), and is a
  trainer opponent at a 200bb table.

## Phase A: live re-solving on the turn and river

This is the Libratus/Pluribus step: play the blueprint until the turn, then solve the actual spot.

1. **Ranges from the blueprint.** Each player's range is every holding, weighted by the product of the blueprint's
   probabilities for that player's actions so far, each read at the holding's bucket on the board of that street.
   Blocked cards are removed.
2. **The solve.** TexasSolver solves from the start of the current street with the real pot and stacks, using those
   exact-combo ranges. The actual actions on the street are walked to the nearest sizes, and our hand's strategy is
   sampled.
3. **Fallback:** the blueprint, when no solver is installed or the solve fails.
4. **Measure:**
   - head-to-head against the plain blueprint;
   - Slumbot, luck-adjusted;
   - LBR, where one solve gives the strategy for every holding.

   Done means the re-solver beats or ties the plain blueprint and lowers LBR.

## Phase B: opponent adaptation during play

These steps build on phase A's solver.

1. **Read the opponent.** Use the generic hand-reading model tilted to their observed action rates
   (`opp_adjust.py`), or a model learned from their revealed hands when every hand's cards are known, as with
   Slumbot.
2. **Exploit robustly.** In the re-solve, give the opponent the range our model reads instead of the blueprint's.
   Blend it with the blueprint's range at weight p (robust exploitation): p = 0 is the plain re-solve, and p grows
   with the number of hands seen.
3. **Measure:**
   - profit against our weak bots (maniac, calling station, nit) and against Slumbot;
   - LBR, which checks how much exploitability the adaptation costs. Earlier results said robust exploitation keeps
     most of the profit at a fraction of the risk.

## Phase C: continual learning

1. **Keep training.** Runs resume from checkpoints, so the blueprint improves between sessions. Measure each
   checkpoint with `blueprint/lbr_trend.sh`.
2. **Learn from real hands.** Feed Hand Sync's library (your hands) and our own bots' logs into the opponent models
   used in phase B.
3. **Refine the abstraction where LBR finds leaks.** Add bet sizes or buckets where LBR wins most. Today that is
   all-ins and pot-size bets.

## Ground rules

- Texas hold'em only. All computation runs on the lab.
- Every claim is luck-adjusted and comes with a 95% interval.
- Never use real-money tables. The advisor works on finished hands, practice and study only.

## Added 2026-11-08: the other recipes, all of them, in this order

| # | Recipe | What we build | Measured by |
|---|---|---|---|
| 1 | **Potential-aware card abstraction** (Johanson et al. 2013) | Turn and flop buckets by k-means over the *distribution* of future strength (earth mover's distance, which in one dimension is the L1 distance between CDFs). River buckets by strength against 8 opponent-hand clusters (OCHS). 500 buckets per street, then blueprint run 4 | LBR against run 2 and run 3; Slumbot |
| 2 | **Safe subgame solving** (Libratus) and **depth-limited solving with continuation strategies** (Modicum) | Our own C++ turn/river subgame solver (vectorised CFR+ over hands) with a safety gadget, and depth-limited leaves where the opponent chooses among blueprint continuation strategies | LBR and Slumbot against the plain blueprint and the TexasSolver re-solver |
| 3 | **DeepStack / Supremus** | Train the value network on many solved spots for depth-limited lookahead, on top of 2 | As 2 |
| 4 | **GPU track A: AlphaHoldem-style deep RL** | PPO self-play on the RTX PRO 6000 with a card/action tensor encoding; a millisecond-decision policy | Head-to-head against the blueprint; Slumbot; LBR (its policy for every holding) |
| 5 | **GPU track B: Deep CFR / DREAM, then ReBeL** | Neural regret networks without card abstraction, then RL plus search over public belief states | As 4 |

Items 1 and 2 run on the CPUs alongside current jobs. 4–5 use the GPU, which is idle when no LLM is loaded.
Each item ships only after it beats or ties the one before it on luck-adjusted measures.

## The recipes surveyed (2026-11-08)

Dates and findings are from the published papers, as recalled. Check the papers before relying on fine details.

| Recipe | Core idea | Fit for us |
|---|---|---|
| **DeepStack** (Moravčík et al., 2017) | No full blueprint: re-solve every decision live, a few actions ahead, with a trained value network estimating the rest | We have a value network and depth-limited search; the value net needs training on many more solves |
| **Modicum / depth-limited solving** (Brown & Sandholm, 2018) | Re-solve live only a few actions deep. At the leaves, the opponent picks among several blueprint continuation strategies, keeping the search safe; cheap enough for a laptop | The best next step: builds on the blueprint and fixes the re-solver's weak spot, turns and rivers after calling a flop bet |
| **Safe / nested subgame solving** (Libratus) | Re-solve so the result cannot be worse than the blueprint (a max-margin safety constraint) | Upgrades our current, unsafe re-solver; small and principled |
| **Potential-aware card abstraction** (Johanson et al., 2013) | Group hands by their whole distribution of future strength (k-means with earth mover's distance), not just average strength | LBR's edge levelled off with 200 or 1,000 buckets, which points to the *kind* of bucketing; strong bots use this |
| **Deep CFR / Single Deep CFR / DREAM** (Brown et al. 2019; Steinberger 2019–20) | Neural networks learn regrets directly, with no card abstraction | GPU research track; strength at full heads-up no-limit is less proven than blueprints |
| **ReBeL** (Facebook AI, 2020) and **Student of Games** (DeepMind, 2023) | AlphaZero for imperfect information: self-play RL plus search over public belief states | The most general, and "AlphaPoker" done right; heavy, with days to weeks of GPU training |
| **AlphaHoldem** (2022) | End-to-end deep RL (a PPO variant) by self-play; millisecond decisions without search; reported strong against Slumbot-class bots at modest compute | A good GPU track: a different style of bot to test against the blueprint and Slumbot |
| **Supremus** (2020) | A faster, stronger DeepStack | Same track as DeepStack |
| **NFSP** (Heinrich & Silver, 2016) | Deep RL self-play with an averaged policy | Weaker at heads-up no-limit; our own RL results agree; skipped |

The order chosen is the roadmap table above. The user asked to do all of them on 2026-11-08.

## Next eight, in order (2026-11-08, evening)

The user approved all eight.

| # | Item | What we build | Done means |
|---|---|---|---|
| 1 | **A sharper yardstick: best response in the simplified game** | A vector best-response tool (`blueprint/br_vec`): street by street from the river back, it estimates each betting-node-and-bucket's action values over thousands of sampled boards, picks the best action, then plays that best response against the table to measure it. It gives each checkpoint's exploitability *inside its own abstraction*, with intervals a few bb/100 wide instead of LBR's ±65. It is a lower bound that tightens with more boards. It complements LBR, which plays the real game and catches abstraction and bet-mapping leaks | Runs 4/5/6/vec and the MLP runs ranked with non-overlapping intervals |
| 2 | **Opponent adaptation (phase B)** | Turn a professional clone into a table in the simplified game. Fine-tune the copied MLP against it, mixed with self-play and anchored by the KL pull to the blueprint copy (a restricted Nash response with a KL anchor). Train against a clone of half the hands, and test against a clone of the *other* half at the bot table, so the test is not the training opponent | Wins clearly more than the blueprint's +64 against the held-out clone, with LBR not much worse |
| 3 | **Slumbot rematch** | 5,000 hands each for runvec and run 6 (4.5–5 h checkpoints) and `runmlp2` (from scratch, 1.8 billion hands, LBR +104) | Compare with run 2's +30.9 [−10, +72] |
| 4 | **Safe re-solving** (Libratus) | Our own vector turn/river solver (built from `mccfr_vec`'s fold and showdown code) with the max-margin safety gadget, so a re-solve cannot do worse than the blueprint | Beats the plain blueprint head-to-head and lowers LBR (unsafe re-solving: +996) |
| 5 | **Ship the blueprint in the advisor** | The copied MLP (3.5 MB) instead of the 0.22 GB table, plus today's fixes; rebuild the Windows exe in the lab VM | The exe gives blueprint advice on a heads-up spot |
| 6 | **One combined run** | After the 24-hour A/B: the winning trainer on about 56 threads with the distribution-aware buckets | A new best on items 1 and 3 |
| 7 | **Vector trainer speed** | Strategy computed once per bucket, not once per hand; regret pruning | About 2× hand pairs a second per core, same self-test |
| 8 | **Partner videos and page** | Refresh with today's results: blueprint vs everyone, the professional clones, and the MLP learning curves | Page republished |

### Progress on the eight (2026-11-08, 15:30 lab time)

| # | State |
|---|---|
| 1 | **Built, verified, weaker than hoped.** `br_vec` (in `model.md` §7.2) decides level by level, and the in-sample check passes (+227). A plain argmax overfits: 40 boards gave −53 out of sample. So deviations are cross-validated (chosen on even boards, kept only if odd boards agree). At 400 boards it bounds run 2's abstract exploitability at **+2.3 [−1.1, +5.6]**: honest but loose. A 3,000-board run (about 3 h) is running to see how the bound grows. A GPU version is the way to make it routine |
| 2 | **Running.** Clones of disjoint halves of the professionals' hands (`pros_a`, `pros_b`) became tables in the simplified game (`clone_table.py`). Run 2's table beats them by +42.2 and +40.2 [+38.9, +41.4]: the bar. `runexploit` copies run 2 (60 min), then trains with half of each batch against `pros_a`, anchored to the blueprint copy, and reports against `pros_b` every 30 min. Live baseline (blueprint vs `clone#pros_b` at the bot table) is running |
| 3 | **Done.** Luck-adjusted against Slumbot, 5,000 hands each: runvec **+18.4 [−17, +54]**, run 6 **+6.0 [−36, +48]**, `runmlp2` (MLP from scratch, 2 h) **−31.8 [−79, +16]**. Run 2 with the floor read +30.9 [−10, +72]. None is distinguishable from the others or from zero at this sample size |
| 4 | **Built; measuring.** `resolve.cpp`: exact-card CFR+ on the river subgame, with the re-solve gadget. 200 sampled river spots (re-solved from the street's start, 1,000 iterations): the opponent's best response falls by **161 ± 17 chips per hand pair (about 36%)**, and no opponent hand gains more than +1.4 over the blueprint. The unsafe re-solve cuts 573 but leaves one hand +1,179 better off. **The first live version re-solved from the current node**, after the opponent's bet, so it trusted the blueprint's betting range. LBR exploited that: **+176 [+98, +254]**, worse than the blueprint. Re-solving now starts at the beginning of the river; LBR, a head-to-head against the plain blueprint and the held-out clone are rerunning |
| 5 | **Done.** Data pack (80 MB: 8-bit table, flop/turn buckets, river cut points) and `bp.dll` cross-compiled with MinGW. The Windows exe was rebuilt in the lab VM: 108 MB, self-test passed in Windows with the blueprint answering a 200bb heads-up spot. Shipped as the 8-bit table rather than the copied MLP, which LBR rated worse (+190 against the table's +104) |
| 6 | After the 24-hour A/B (runs 4–6 and runvec end about 09:00 lab time on 2026-11-09) |
| 7 | **Done at 1.6×.** Strategies and regret writes per distinct bucket: 1.6× on one core. **Pruning does not carry over to vector updates.** In runvec's checkpoint a third of individual regrets are below −10⁵, but an action can only be skipped when it is dead in *every* bucket on the board, which is rare across a few hundred buckets; the benchmark showed no clear gain. The flag stays, off by default |
| 8 | **Done.** Clips 19–21 (blueprint vs everyone, the LBR curves, the Slumbot rematch) and partner page version 8 |

### Update, 2026-11-08 17:00 lab time

- **Phase B works on a held-out opponent.** `runexploit` was copied from run 2, then fine-tuned for 30 minutes with
  half of each batch against the `pros_a` clone, anchored to the copy. In the simplified game (2 million hands
  each):

  | Opponent | Run 2's table | runexploit |
  |---|---|---|
  | `pros_a` (trained against) | +42.2 | +57.3 [+55.5, +59.2] |
  | `pros_b` (held out) | +40.2 [+38.9, +41.4] | +48.2 [+46.6, +49.9] |
  | run 2 | — | −3.3 |

  That is **+8 bb/100 more from an opponent it never saw**, at almost no cost against the blueprint. LBR's curve
  on it is now running.
- **Safe re-solving, live:**

  | Version | LBR's edge | vs the plain blueprint | vs `pros_b` clone |
  |---|---|---|---|
  | From the street's start (v2) | +146 [+76, +215] | −18.6 [−48, +11] | +36.5 [+11, +64] (plain: +20.0 [−10, +50]) |

  No measurable gain yet. LBR still wins with river bets into folds. Next: v2 with the defence floor on top
  (running).
