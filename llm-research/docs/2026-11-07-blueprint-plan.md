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
