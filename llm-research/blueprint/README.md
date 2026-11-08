# Blueprint: a heads-up no-limit hold'em strategy learned by self-play

## The goal

Build a heads-up no-limit Texas hold'em bot that is **as hard as possible to beat**, using the recipe behind
Libratus (2017) and Pluribus (2019):

1. **Simplify the game.** Group similar hands into buckets, and allow only a handful of bet sizes.
2. **Solve the simplified game by self-play.** Sampled counterfactual regret minimisation (MCCFR) plays the
   simplified game against itself billions of times and converges toward an equilibrium: a strategy that no
   opponent can exploit for much.
3. **Play the real game from that solution.** Map real bets onto the simplified ones; optionally re-solve the turn
   and river live with TexasSolver.
4. **Then exploit, safely.** Read an opponent and lean toward their mistakes without becoming beatable. This is
   robust exploitation, phase B of the plan.

The full game has about 10¹⁶¹ decision points, far too many to solve directly
(`../docs/2026-11-07-game-size-and-solving.md`). The blueprint is the simplified game's solution, and it is the
foundation everything else plays on.

## How we measure it

The aim is to make **LBR's edge go down** and **results against strong opponents go up**.

- **LBR (local best response, `../scripts/lbr.py`).** A best-responding opponent that sees our bot's whole range and
  bets to exploit it. Its winnings are a lower bound on how beatable the bot is. **Lower is better; 0 would mean
  LBR finds nothing.** Results are luck-adjusted: both hands are known, so card luck is removed exactly.
- **Slumbot.** A strong public heads-up bot, played at 200bb through its API. Results are luck-adjusted by
  `../scripts/aivat_lite.py`.
- **Our other engines and players.** The solver bot, adaptive bot, Monte, Tina+, simulated humans, and LLMs and
  LLM agents.
- **Legality.** Every action goes through PokerKit's rules engine, and Slumbot's server checks every action we send.
  No illegal action has been recorded in 92,780 decisions in self-play or 31,000 hands against Slumbot.

## Results so far

| Measure | Result (bb/100, 95% interval) |
|---|---|
| Against Slumbot, luck-adjusted (5,000 hands, run 2 early) | **+1.0** [−30, +32]: even |
| Against our solver bot (20,000 hands, 200bb) | **+26.3** [+6, +46] |
| LBR against run 2's final checkpoint, purification and defence floor | **+104** [+44, +164]; was +622 with a mapping bug |
| LBR across run 2's training | levels off at about +140 from 5 billion iterations on |
| Live re-solving against the plain blueprint (10,000 hands) | +2.6 [−28, +33]: no measurable gain yet |

Latest results, run comparisons and the speed/accuracy work: `../docs/2026-11-08-blueprint-speed-accuracy.md`.
Summary as of 2026-11-08:

- **Slumbot:** +1.0 / +12.4 / +30.9 across three 5,000-hand runs (the last with the defence floor).
- **Our engines and LLMs:** it beats every one tested.
- **LBR:** +104 at best.
- **Re-solving:** unsafe re-solving is exploitable (LBR +996) and stays off.

## The pieces

| File | What it does |
|---|---|
| `abstraction.cpp` | Card abstraction. For every suit-isomorphic hand and board: river hand strength against all 990 opponent hands, turn and flop expected strength and strength², cut into equal-population buckets (200 → `out/`, 1,000 → `out1000/`). Verified against eval7. |
| `mccfr.cpp`, `tree.hpp` | The trainer: builds the betting tree (`tree.hpp`, shared) and runs external-sampling MCCFR with lock-free threads, linear-CFR discounting, checkpoints and resume. Flags add DCFR, pruning, PCFR+, stratified dealing and a multigrid start. |
| `mccfr_vec.cpp` | The vector trainer (public chance sampling): deals a board and updates every hand of both players at once, with card-removal-exact fold and showdown values. It checks itself against brute force on start; `--bench S` gives hand pairs a second on one core. |
| `bp.cpp` → `libbp.so` | Bucket lookup for Python (`libbp2.so` for the 1,000-bucket tables). |
| `check.cpp`, `preclass.cpp`, `inspect_blueprint.py` | Correctness check against eval7; pre-flop class names; reading a strategy. |
| `abstraction_pa.cpp` | Distribution-aware abstraction: k-means on the CDF of next-street strength (flop, turn) and on strength against 8 opponent groups (OCHS, river); 500 buckets → `out_pa/` |
| `lbr_trend.sh` | Runs LBR on every new checkpoint, giving the convergence curve in `RUN/lbr_trend.tsv`. |
| `../scripts/blueprint_bot.py` | Plays the blueprint. It walks the real hand down the tree with pseudo-harmonic bet mapping, plays the average strategy with 5% purification, applies the minimum-defence floor, and can re-solve turns and rivers live (`BLUEPRINT_RESOLVE=1`). |

**Game and abstraction:**

- Heads-up, 200bb stacks, blinds 50/100 (Slumbot's game).
- Pre-flop: open to 2, 2.5 or 3bb; 3-bet to 3× or 4×; 4-bet to 2.3×; all-in.
- Post-flop: bets of ⅓, ⅔, 1 and 2 pots; raises to pot or 2.5×; all-in.
- The tree has 277,406 nodes.

**Training runs:**

| Run | Buckets per street | Tables | Iterations |
|---|---|---|---|
| `run2` | 200 | 0.44 GB | 21.5 billion in 12 h, about 500k per second |
| `run3` | 1,000 | 2.2 GB | 24 h |
| `run4` | 500, distribution-aware | 1.1 GB | 24 h, MCCFR (the A/B baseline) |
| `run5` | 500, distribution-aware | 1.1 GB | 24 h, **DCFR + pruning** (`mccfr3 --dcfr --prune-after-minutes 30`) |
| `run6` | 500, distribution-aware | 1.1 GB | 24 h, **PCFR+** (`mccfr4 --pcfr`) |
| `runvec` | 500, distribution-aware | 1.1 GB | 24 h on 8 threads, **vector CFR + DCFR** (`mccfr_vec --dcfr`); judged on LBR per core-hour |
| `runmlp`, `runmlp2` | 200 (`out`) | 0.22 GB exported | 24 h each on one GPU: **an MLP learning from scratch by self-play** (`../scripts/mlp_selfplay.py`), about 300k hands a second, exported in this table layout |

## Building and running (on the lab)

```
cd ~/llm-research/blueprint
make -j                              # fetches the two libraries at pinned commits into deps/ and builds everything
./check 300                          # hand strengths must match eval7
./abstraction out 200                # card abstraction, about 70 s on 32 threads
./mccfr out run2 --dry-run           # tree size and memory
./mccfr out run2 --threads 24 --minutes 720 --checkpoint-minutes 30    # resumes if run2/ has a checkpoint
sh lbr_trend.sh blueprint/run2 3000 12                                 # from ~/llm-research
```

Third-party code is fetched at pinned commits into `deps/` and never committed:

- **PokerHandEvaluator** by Henry Lee, Apache-2.0: 7-card hand evaluation.
- **hand-isomorphism** by Kevin Waugh, BSD-style licence: suit-isomorphic hand indexing.

## Next

Plan: `../docs/2026-11-07-blueprint-plan.md`.

- **Phase A, re-solving:** in place; measuring it against LBR and Slumbot.
- **Phase B, opponent adaptation on top of the re-solver:** read the opponent, then lean toward their mistakes
  robustly.
- **Phase C, continual learning:**
  - keep training: runs resume from checkpoints;
  - feed real hands from Hand Sync into the opponent models;
  - refine the abstraction where LBR finds leaks.
