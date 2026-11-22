# How big is poker, and can we solve it all? — 2026-11-07

## How many scenarios are there?

These counts are for heads-up Texas hold'em.

| What | How many |
|---|---|
| Starting hands (two cards) | 1,326 combinations (169 strategically different: AKs, AKo, 77, …) |
| Flops | 22,100 (1,755 strategically different once suits are interchangeable) |
| 5-card poker hands | 2,598,960 |
| 7-card hands (your 2 + the 5 on the board) | 133,784,560 |
| Ways to deal one heads-up hand (both players' cards and the board) | about 2.8 trillion (1,326 × 1,225 × 1,712,304) |
| Heads-up **limit** hold'em: decision situations (information sets) | about 3.2 × 10¹⁴ |
| Heads-up **no-limit** hold'em (200bb, the Slumbot game): decision situations | about **10¹⁶¹** |

For comparison, the observable universe has about 10⁸⁰ atoms. No-limit is that large because every possible bet size
opens a new branch of the game.

## Can we simulate and solve them all?

**No, and nobody can.** Storing even one bit for each of 10¹⁶¹ situations would take more memory than there are atoms
in the universe, many times over. The limit is the number itself, not computing power.

What has been done:

- **Heads-up limit hold'em** (about 3 × 10¹⁴ situations, fixed bet sizes) was essentially solved in 2015 by the Cepheus
  team (Bowling et al.). It took around two months on a couple of hundred machines with thousands of CPU cores.
  No-limit is about 10¹⁴⁶ times larger.
- **Libratus and Pluribus did not solve no-limit.** They solved a simplified version once, the "blueprint", then
  solved the current part of the game precisely during play.

## What we can do: the same recipe

| Part | What it is | Status |
|---|---|---|
| Exact solutions where the game is small | Short-stack push/fold; any single river; most turns | Done: push/fold charts (`poker_math.py`), live TexasSolver on turns and rivers (`poker_agents.py`) |
| A **blueprint** of the whole game, simplified | A handful of bet sizes; similar hands and boards grouped into buckets (about 10⁹–10¹² situations instead of 10¹⁶¹); solved by sampled CFR self-play | **Not built.** The flop model, deep pre-flop charts and rule-based defences stand in for it, and LBR keeps finding leaks in them |
| Real-time re-solving | During play, solve the current spot precisely, starting from the blueprint | Partly: turn and river solves, the value network (`value_net.py`), depth-limited search (`depth_limited.py`) |

**The blueprint is the missing piece.**

- **Scale:** Pluribus's blueprint took about 8 days on one 64-core server, the same size as our lab. A heads-up
  blueprint is smaller than Pluribus's 6-player one.
- **What it replaces:** most of the current patchwork.
- **What it needs:**
  - card bucketing: hands grouped by strength and potential on each street;
  - a sampled-CFR trainer that runs for days across the lab's cores;
  - measurement against LBR and Slumbot.
- **Cost:** about a week to build, then a week or more of lab time. It would use most of the lab's cores and a lot
  of its memory, so it has to be scheduled around other users and the another service terminal.

## Why simplification needs leak-hunting

Every bot that plays no-limit, ours included, simplifies the game. LBR (`scripts/lbr.py`) measures where the
simplifications leak. Our flop model knew only 66%-pot bets, and LBR won +459 bb/100 from that until the defences
and the bet-size-aware flop model were added. After the fixes it is +79 [−9, +167], no longer significantly
positive. See `2026-11-07-edges-and-real-tests.md`.
