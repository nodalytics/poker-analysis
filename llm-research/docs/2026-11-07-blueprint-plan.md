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
