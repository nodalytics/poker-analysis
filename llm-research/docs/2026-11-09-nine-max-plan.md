# A 9-max blueprint: plan (2026-11-09)

The user shared a summary of Pluribus (Brown & Sandholm, *Science* 2019), the bot that beat professionals at
6-max no-limit hold'em, and said: **"They built it for 6max. We need 9max."** This note compares what we have with
what Pluribus did, says what changes at nine players, and sets out the build order.

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
