# A 9-max blueprint: plan (2026-11-09)

The user shared a summary of Pluribus (Brown & Sandholm, *Science* 2019), the bot that beat professionals at
6-max no-limit hold'em, and said: **"They built it for 6max. We need 9max."** This note compares what we have with
what Pluribus did, says what changes at nine players, and sets out the build order.

## Decisions (2026-11-09)

- **9-max is the product.** The heads-up blueprint is the prototype that proved the methods.
- **Cash games and tournaments.** Cash first at **100bb**. Tournaments add ICM: chips are converted into prize
  equity, which changes the payoffs (below).
- Ideas from the Pluribus summary the user sent are folded in below: search at play time, the pre-flop-only
  average, and coarse-blueprint/fine-search.

## What Pluribus did, and where we stand

| Pluribus (6-max, 100bb) | Ours today (heads-up, 200bb) | For 9-max |
|---|---|---|
| Self-play by MCCFR (external sampling), from random play | The same (`mccfr.cpp`), plus vector CFR (`mccfr_vec.cpp`) and GPU self-play | External sampling generalises to N players. **Vector CFR does not**: its showdown compares two ranges, and N ranges cost about 1,326^N |
| Linear CFR early, negative-regret pruning | Have both (DCFR, `--prune`; run 5) | Carries over |
| Card abstraction (hands bucketed per street) | 200 percentile and 500 distribution-aware buckets | Buckets depend only on a player's hand and the board, so they carry over. Strength against one opponent is a weaker guide in a multiway pot |
| Action abstraction: several raise sizes on the first pre-flop actions, fewer later | 3–6 actions per node, 277,000-node tree | **The big change**: nine players multiply the betting sequences. Sizes must be cut hard in multiway pots |
| Average strategy kept for the first round only; later rounds use the current strategy, and search covers them | Average strategy everywhere | Adopt it: it halves memory after the flop |
| Real-time depth-limited search; at its leaves each player picks one of 4 continuation strategies (blueprint, and versions biased to fold, call, raise) | River-only safe re-solving, heads-up (`resolve.cpp`) | Needs an N-player depth-limited solver: the largest new component |
| Training: 64 cores, 8 days (12,400 core-hours), under 512 GB RAM, no GPUs, about $150 | Lab: 64 cores, **125 GB RAM**, 3 GPUs | Same CPU budget as Pluribus, a quarter of its training memory |
| Play: 2 CPUs, under 128 GB RAM | The advisor ships an 80 MB pack | A 9-max pack will be much larger, or the advisor must search at play time |
| Measured against professionals, with AIVAT | LBR, Slumbot, clones of professionals | **LBR is heads-up only.** 9-max needs new yardsticks (below) |

With more than two players, CFR has no guarantee of reaching an equilibrium, and an equilibrium would not
guarantee a profit anyway. Pluribus showed it works in practice; we have to measure it the same way, by results
against strong opponents.

## What changes at nine players

1. **Tree size.** Each of nine seats can fold, call or raise before the flop, so the number of pre-flop betting
   sequences grows quickly with each extra raise size. The memory budget sets the abstraction. Regrets
   (4 bytes) for every (betting node × bucket × action), plus averages for the pre-flop only, must fit in about
   100 GB of the lab's 125 GB.
2. **Multiway play.** Side pots when several players are all in; showdowns among 3–9 hands, with split pots.
3. **Positions.** Nine positions (UTG to the big blind) instead of two; the pre-flop strategy differs strongly by
   position.
4. **Evaluation.** No exact best response is practical, and LBR is two-player. The yardsticks will be:
   - **One bot against eight copies of a baseline**, with duplicate seating (every deal played from every seat)
     and luck adjustment. Pluribus was evaluated the same way: one AI against five copies, and five AIs against
     one human.
   - **Tables of opponents:** our personas, the solver bot, and clones of professionals at full-ring tables
     (PokerKit already plays nine seats).
   - **Heads-up checks:** in pots that are down to two players, the existing LBR logic can still probe the
     blueprint's play.
5. **Data.** The professionals' hands we cloned (Pluribus dataset) are 6-max. HandHQ has full-ring hands: in a
   sample of 134,000 hands, 14% had 7–10 players and 5% exactly nine. Extrapolated to all 21.6 million hands, that
   is roughly **3 million hands with 7+ players and 1 million nine-handed**. That is enough for full-ring opponent
   models and clones of human play.

## Build order

| # | Step | Done means |
|---|---|---|
| 1 | **Tree-size calculator** for N-player action abstractions: count nodes and table entries for candidate bet-size menus (by street, raise level and number of players in the pot) | A menu for 9-max, 100bb, that fits about 100 GB with 200 buckets, chosen by measured size |
| 2 | **N-player game engine** in C++: the tree, side pots, multiway showdown with split pots, folded seats; checked against PokerKit on random hands | Payoffs agree with PokerKit on a million random hands |
| 3 | **N-player external-sampling MCCFR**: linear CFR early, negative-regret pruning, average kept for the pre-flop only, 16-bit or 32-bit regrets | A 3-player smoke run converges sensibly; a 9-max run starts |
| 4 | **The 9-max blueprint run**: 64 cores for up to 8 days (Pluribus's budget), once run 7 has finished | A strategy that beats the baseline bots at 9-max tables |
| 5 | **Evaluation harness**: 1-vs-8 duplicate matches with luck adjustment; personas, the solver bot and full-ring human clones at the table | Results with confidence intervals for every checkpoint |
| 6 | **Full-ring data**: extract HandHQ's 7–10-player hands; clones and opponent models for full ring | Clones that predict held-out players about as well as the 6-max clones (77–82%) |
| 7 | **N-player depth-limited search** with Pluribus's four continuation strategies at the leaves | Beats the plain blueprint in 1-vs-8 matches |
| 8 | **Advisor**: a 9-max data pack (8-bit pre-flop table, post-flop from the blueprint or search), Windows build | The exe gives 9-max advice |

Heads-up work continues alongside: run 7, the network bake-off and the learning-rule tests use the GPUs and some
CPUs, and the 9-max run would need all 64 cores. Steps 1–3 are code and short tests, and can start now.

## What has to be reworked

Almost everything that encodes "two players" is rebuilt; the methods and most of the tooling carry over.

| Component | Heads-up today | For 9-max |
|---|---|---|
| Card abstraction (`abstraction*.cpp`, `bp.cpp`) | 200 / 500 buckets per street | **Keep.** Buckets depend on one player's cards and the board. A multiway-strength feature may be added later |
| Betting tree (`tree.hpp`) | Two players, 200bb | **Rebuild** for N players, 100bb: positions, side pots, a menu that shrinks in multiway pots (`blueprint/tree_size.py` sizes it) |
| External-sampling MCCFR (`mccfr.cpp`) | Two players | **Generalise** to N players; keep linear/discounted CFR and pruning |
| Vector CFR (`mccfr_vec.cpp`, run 7) | Two-range showdown | **Cannot extend**: N ranges cost about 1,326^N. Heads-up only |
| GPU self-play (`mlp_selfplay.py`) | Two seats | **Extend** to N seats. Neural play does not need a table per node, which matters because 9-max tables do not fit our memory (below) |
| Network and learning-rule tests (bake-off, learners, map) | Heads-up | **Transfer**: they pick the network and the update for the 9-max neural trainer |
| Re-solving (`resolve.cpp`) | Heads-up river, safe gadget | **Rebuild** as N-player depth-limited search with Pluribus's four continuation strategies |
| Yardsticks: LBR, `br_vec`, Slumbot | Heads-up | **Replace**: 1-vs-8 duplicate matches with luck adjustment, tables of clones and bots. LBR stays for pots that reach two players |
| Bot and advisor (`blueprint_bot.py`, pack) | Declines any table other than heads-up | **Rebuild** for nine seats |

## First measurements

**Baseline at a 9-handed table** (`results/ninemax/baseline/`; 400 sessions × 24 hands, 100bb dealt fresh every
hand). The current bot has no 9-max blueprint, so it plays its fallback (charts, flop model, Monte Carlo):

| Seat | bb/100 [95% interval] |
|---|---|
| Current bot (`blueprintbot`, fallback) | +120 [+78, +160] |
| Solver bot (same fallback logic) | +76 [+37, +112] |
| Monte (Monte Carlo equity) | +91 [+42, +140] |
| Clone of the professionals (`pros`, `pros_b`) | +8 [−23, +40], +17 [−16, +52] |
| Personas: bob, nora, tina | +132, +83, +52 |
| Persona max (maniac) | −579 [−677, −485] |

Nearly everyone profits from the maniac. Against the professional clones' level of play the current bot's edge is
unknown; this is the number a 9-max blueprint has to beat, and the next baseline should drop the maniac.

**Tree sizes** at 100bb (`blueprint/tree_size.py`; memory = 4-byte regrets + 4-byte averages, or averages pre-flop
only as Pluribus did):

| Players | Menu | Decision nodes (pre-flop / flop / turn / river) | GB at 200 buckets | Pre-flop-only average | GB at 500 buckets | Pre-flop-only average |
|---|---|---|---|---|---|---|
| 2 | wide | 180 / 2.8 k / 24 k / 148 k | 0.8 | 0.4 | 1.9 | 1.0 |
| 6 | lean | 289 k / 4.1 M / 13 M / 28 M | 154 | **77** | 383 | 192 |
| 6 | medium | 1.8 M / 11 M / 37 M / 97 M | 508 | 257 | 1,262 | 634 |
| 9 | lean | (counting) | | | | |

Menus: *lean* is one open size, one 3-bet and one 4-bet size, two post-flop bet sizes heads-up and one multiway.
*medium* has three post-flop sizes heads-up and two multiway. Even 6-max with the lean menu fits the lab's 125 GB
only with Pluribus's pre-flop-only average at 200 buckets. 9-max is larger again, so on this hardware a 9-max table
must be coarse and lean, with **search** doing the fine work at play time, or the post-flop strategy must be a
**network** instead of a table.

## Pluribus ideas, and making it learn faster, further and wider

| Idea | What it gives us |
|---|---|
| **Search at play time** (depth-limited, four continuation strategies at the leaves; off-tree bets added to the search tree) | LBR now wins mainly with flop and turn bets, which the river-only re-solve cannot reach; search from the turn and depth-limited from the flop attacks that. Adding opponents' real bet sizes removes bet-mapping leaks. At 9-max it is what lets a coarse table play well |
| **Average strategy pre-flop only**; current strategy plus search after the flop | Halves memory: the difference between a 6-max table fitting or not |
| **Coarse blueprint, fine search** | Wider without more memory |
| **Faster: CFR on the GPU** | Vector CFR is 1,326-wide array arithmetic per board, which suits GPUs. It applies to heads-up and to pots that reach two players. For 9-max, the GPU route is neural self-play, which already plays 300,000–560,000 hands a second and extends to nine seats |
| **Further: warm starts** | Start a finer or larger abstraction from a trained coarse one instead of from zero |

## Tournaments: ICM

In a tournament, chips are not money: the Independent Chip Model (Malmuth–Harville) turns every player's stack into
an expected share of the prize pool. The terminal payoffs of the game then become changes in ICM equity rather
than in chips, which makes calling all-ins tighter near the money and at the final table. The plan:
- **Cash blueprint first**, at 100bb.
- **Tournament spots at 10–60bb by short-stack blueprints and search** with ICM payoffs at the terminal nodes.
  Search is where ICM fits, because the payoff depends on every player's stack and the payout structure, which a
  single precomputed table cannot cover.
- A push/fold solver with ICM for short stacks (≤ 15bb) is small and exact, and comes first.
