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

## The route: neural on the lab (chosen), a table on a 512 GB machine (documented)

**Chosen: the neural route on the lab** (the user, 2026-11-09). The strategy is a network trained by self-play on
the GPUs, not a table per betting node. Memory then grows with the network (a few million weights), not with the
tree, and the bet menu can be wider than any table could afford. Built so far:

| Piece | File | State |
|---|---|---|
| N-player dealing with buckets, showdown ranks and cards (about 3 million 9-player deals a second) | `blueprint/bp.cpp` `bp_deal_n`, `libbp5.so` | Done |
| A vectorised N-player no-limit engine on the GPU: positions, a menu by raise level and pot size, min-raise rules, incomplete all-ins that do not reopen the betting, side pots | `scripts/ninemax_engine.py` | Done |
| A check of the engine against PokerKit: random legal games replayed card for card, actor, legality and payoffs compared | `scripts/ninemax_check.py` | **0 mismatches in 11,000 tables** at 3, 4, 6 and 9 players and all three menus, almost all ending in multiway all-in showdowns with side pots (odd chips of split pots allowed) |
| The 9-max trainer: one residual network for all seats, regularised actor-critic with a magnet; evaluated by 1-vs-8 duplicate matches (the network in each of the nine seats in turn, same deals) against a rule-based tight-aggressive player and against its own previous checkpoint | `scripts/ninemax_selfplay.py` | **Running**: `blueprint/nm2` (population training) on the RTX PRO 6000, resumed 2026-11-10 01:37 lab time; `nm1` (self-play control) stopped at 120 minutes |

Smoke test (2 minutes, shared GPU): about 18,000 nine-handed hands a second; against eight rule-based players
−1,234 then −988 bb/100, and +275 and +356 against its own previous checkpoint, so it learns. Next: the real-game
adapter (PokerKit state → the network's inputs), so the 9-max network can sit at `bot_sim` tables against the
clones and the personas, and in the advisor.

**First result, and the self-play blind spot** (`nm1`, pure self-play, 30 minutes, 85 million hands at 47,000 a
second; 1-vs-8 duplicate matches, 5,000 deals × 9 seatings):

| Hero (1 seat) vs 8 copies of | bb/100 [95% interval] |
|---|---|
| Trained network vs the rule-based tight-aggressive player | **+47** [+26, +68] (+52 [+41, +63] in the run's own log) |
| Rule-based player vs the trained network | −73 [−92, −53] |
| Rule-based player vs an untrained (random) network | +1,210 [+1,099, +1,322] |
| **Trained network vs an untrained network** | **+8 [−5, +22]** |

The network beats a sensible player but hardly profits from eight random players who shove 30% of the time. It
learned only against itself: it folds 72% under the gun and almost never shoves, so it never met wild play. That
is what pure self-play by policy gradient gives (CFR avoids it by exploring every action). **Fix: population
training** (`--pool-share`): half of each batch at tables where the network holds 1–8 random seats against one
opponent from a population of rule players (tight-aggressive, maniac, calling station, random) and its own last
five checkpoints, learning from its own seats only. `nm2` runs that from 2026-11-09 14:35 lab time alongside `nm1`
(the pure self-play control), and both now report results against the maniac, the station and random players too.

**Population training works** (1-vs-8 duplicate matches, bb/100, from the runs' logs):

| Hero vs 8 copies of | `nm1` self-play, 60 / 90 / 120 min | `nm2` population, 30 / 60 / 90 min |
|---|---|---|
| Tight-aggressive rule player | +50 / +51 / +36 | −77 / +65 / **+127** |
| Its own checkpoint 30 minutes earlier | +38 / +20 / +9 | +691 / +95 / +34 |
| Maniac | — | +868 / +1,120 / +1,127 |
| Calling station | — | +1,543 / +3,293 / +4,102 |
| Random network | +8 at 30 min (separate check) | +697 / +777 / +808 |

`nm2` overtook the self-play control against the sensible player within an hour and exploits every wild type,
while `nm1` stalled and then slipped. The cost is speed: 11,000 hands a second against 41,000, because the rule
players run outside the network. The rule players are weak by design, so these numbers show robustness, not
strength; strength is measured at real 9-handed tables once the adapter exists.

**Lab reset, 2026-11-09 16:09 lab time.** The lab rebooted without a clean shutdown (the journal stops mid-stream)
at 120 minutes into `nm1` and 90 into `nm2`. `nm2` was resumed from its checkpoint at 01:37 on 2026-11-10 for the
remaining 1,350 minutes; `nm1` was not restarted, since the comparison it was for is settled.

**A ramped pull learns much faster** (`nm3`: `nm2`'s settings, but the KL pull rises from 0.03 to 0.3 over the first
3 hours, from the trainability map). Against the tight-aggressive player, bb/100:

| Minutes | 30 | 60 | 90 | 120–240 |
|---|---|---|---|---|
| `nm2`, constant pull 0.3 (fresh start, then after resuming) | −77 | +65 | +127 | +180 to +193 after resuming |
| `nm3`, ramped pull | **+205** | +200 | +192 | +173 to +189 |

`nm3` reached in 30 minutes what `nm2` needed about 3 hours of training for. Both then sit at the same level
(about +185 against the tight-aggressive player, +1,150 maniac, +4,950 station, +900 random) and score about 0
against their own checkpoint of 30 minutes earlier. These rule opponents no longer separate the runs: a stronger
yardstick (the networks against each other, and real 9-handed tables) is needed next. `nm2` was stopped at 06:12 on
2026-11-10 to make room for `nm4`: `nm3`'s settings with NeuRD, the least exploitable rule in the heads-up test.

### Stronger yardsticks (2026-11-10)

**Networks against each other** (`scripts/ninemax_eval.py cross`, 20,000 deals × 9 seatings, bb/100):

| Hero vs 8 copies of | `nm1` (self-play, 297 M hands) | `nm2` (population, 295 M) | `nm3` (ramped pull, 234 M) |
|---|---|---|---|
| `nm1` | | −5.2 [−9.1, −1.4] | −1.4 [−5.9, +3.1] |
| `nm2` | +7.3 [+3.4, +11.2] | | −8.3 [−13.1, −3.6] |
| `nm3` | +3.2 [−0.9, +7.3] | −2.8 [−7.4, +1.9] | |
| tight-aggressive rule player | +47 | +183 | +177 |

Against each other they are within a few bb/100. The self-play network is slightly better at a table of its peers;
the population networks are far better against weak players.

**Real 9-handed tables** (`bot_sim`, 400 sessions × 24 hands, 100bb dealt fresh every hand, the maniac `max`
replaced by the network; `nine#RUN` seats via `scripts/ninemax_agent.py`):

| Seat | with `nine#nm3` | with `nine#nm1` |
|---|---|---|
| **9-max network** | **+44** [+11, +76] | +27 [+7, +48] |
| Current bot (`blueprintbot`, fallback logic) | +32 [+5, +62] | +36 [+11, +63] |
| Solver bot | +14 | −17 |
| Monte | −8 | +23 |
| Professional clones (`pros`, `pros_b`) | −18, −12 | −16, −12 |
| Personas bob, tina, nora | −53, −14, +13 | −33, −8, −1 |

The first time the network played real hands: it wins at both tables, and the population network (`nm3`) topped
its table, ahead of the current bot, though the intervals overlap.

**2,000 sessions** (48,000 hands per seat), same line-ups:

| Seat | with `nine#nm3` | with `nine#nm1` |
|---|---|---|
| **9-max network** | **+36.6** [+24.1, +49.8] | **+34.4** [+25.8, +43.4] |
| Solver bot | +26.2 [+13.5, +39.3] | +16.8 [+5.1, +28.2] |
| Current bot (`blueprintbot`) | +21.8 [+9.2, +34.8] | +21.4 [+10.4, +33.1] |
| Monte | +8.9 | +5.5 |
| Nora | +0.6 | +10.1 |
| Professional clones | −13.1, −7.8 | −9.9, −10.3 |
| Bob, Tina | −34.3, −39.0 | −26.1, −41.8 |

Both networks top their tables, about 13–15 bb/100 ahead of the current bot. Self-play only and population
training are level here: this field has no maniac, which is where population training paid off in the engine.

**Exploiters in training (`nm5`, from 2026-11-10 07:05 lab time, 4090):** `--exploit-share 0.25`: a quarter of
every batch seats an exploiter network once against eight copies of the main network; both learn from those
tables. Each hour the exploiter joins the population frozen and restarts as a copy of the main network. Each
evaluation also reports how much the current exploiter wins. Otherwise `nm3`'s settings, population share 0.4.
The single-loop trainer runs it at 43,000 hands a second on the 3090 alone (smoke test), against 10,000–16,000
for population training before.

The adapter rebuilds the engine's inputs from PokerKit's operations; `scripts/ninemax_agent_check.py` replays
random engine games in PokerKit and found **0 differences** in features or legal actions over 6,116 decisions.

**Exploitability** (`scripts/ninemax_eval.py exploit`): a copy of `nm3` trains against eight frozen copies of `nm3`
(plain policy gradient, no pull) and plays it 1-vs-8 by duplicate every 15 minutes. It started at −1.8, as it
should by symmetry, and **won +191 bb/100 [+180, +202] after 15 minutes**, with its entropy near 0: a near-
deterministic counter-strategy. `nm3` is highly exploitable by a player who adapts to it, even though it beats every
fixed opponent tried. This is the 9-max counterpart of the heads-up LBR finding and the main weakness to fix:
the population needs **exploiters** (networks trained to beat the current one, as in AlphaStar's league and PSRO),
so the main network learns to close the holes they find.

**Speed:** the trainer now plays a whole batch (self-play and population tables together) in one loop with a
per-table seating (`play(..., assign=...)`). Before, it ran one loop per seating; the time a loop takes is set by
its length, not its table count. 3,800 → 6,700 hands a second on the same shared 3090 (1.75×). Running runs keep
the old code; new runs get it.

The 9-player count of the lean menu (`blueprint/tree_size.py`) did not finish within 40 GB and an hour: the
betting states alone exceed what the 6-player count needed (about 10 GB), before any table is built.

**Documented alternative: a table on a 512 GB machine.** The Pluribus recipe as published: external-sampling MCCFR
for N players with linear CFR and pruning, a lean bet menu, 100–200 buckets, the average strategy kept pre-flop only,
and search at play time. Training needs about 512 GB of RAM (Pluribus: under 512 GB, 64 cores, 8 days) and up to
1 TB for a richer menu. Playing needs about 32 GB. The memory table below sizes it. If a 512 GB machine
(the Threadripper 9975WX in the summary the user shared) becomes available, steps 2–4 of the build order become
this route, and the neural network can then serve as its search's leaf values.

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

## Nine players before the flop, few after it (2026-11-10)

The user was told that at 9-max, even in tournaments, more than six players in a hand is extremely rare, and was
sent advice to design the solver around "at most six players after the flop". Measured on the 96,000 nine-handed
hands of the 2,000-session tables above (`results/ninemax/table2k_*`, bots at 100bb, no maniac):

| Players acting after the flop | Share of all hands | Share of hands that see a flop |
|---|---|---|
| none (won pre-flop) | 44.4% | |
| 2 | 40.1% | 72.2% |
| 3 | 12.0% | 21.6% |
| 4 | 2.2% | 4.0% |
| 5 | 0.9% | 1.6% |
| 6 | 0.33% | 0.6% |
| 7 | 0.02% | 0.04% |
| 8 or 9 | never | never |

Players calling or raising pre-flop: 1 in 31%, 2 in 39%, 3 in 17%, 4 in 3.3%, 6 or more in 0.4% of hands.

**The claim holds:** 9-way pots do not happen, and 6 or more after the flop is about 1 hand in 300. Real
tournament tables are similar or tighter (short stacks late on push or fold); only very soft games, where many
players limp, see 5–7 players in a pot now and then.

**Design consequences:**

- **Before the flop the game is truly 9-handed.** Positions, the folds so far and who is left to act all matter;
  the network plays it with all nine seats and the full action history.
- **After the flop the folded players are gone.** A pot reached by k players is a k-player game; the folded seats
  matter only through the dead money and what their folds say. No 9-player post-flop solution is ever needed.
- **"At most six" is a rarity, not a rule.** Seven or more see a flop about 1 time in 5,000, so the bot must
  still act; the network plays any number of players, so no special case is built for them.
- **The real line is lower than six.** Solving cost climbs steeply with each player: heads-up and 3-way pots can be
  searched during a hand in seconds, 5–6-way pots cannot (Pluribus had the same limit). So:

| Players after the flop | Share of flops | Handled by |
|---|---|---|
| 2 | 72% | the network, plus search during the hand (planned) |
| 3 | 22% | the network, plus search during the hand (planned) |
| 4–6 | 6% | the network alone (search maybe later) |
| 7+ | under 0.1% | the network alone |

  Search therefore covers 94% of flops, and the expensive quality (search, finer bet sizes) goes into heads-up
  and 3-way pots, where nearly all the money is played.

**On the rest of that advice** (pre-flop for the full table, post-flop up to six, ICM over all remaining players,
individual opponent ranges, precomputed plus real-time strategy, a Threadripper 9975WX with 256 GB, an RTX PRO
6000 and two 4090s):

- Pre-flop for nine, precomputed plus real-time, and opponent ranges match this plan (the 9-max network, search
  during the hand, the opponent models and clones).
- **ICM over all remaining players** is not done in practice: exact ICM costs grow exponentially with the player
  count, and early in a large field ICM is close to chip value anyway. ICM matters near the bubble and at the
  final table, computed over the players left there. This plan does that: chip value early, ICM push/fold near the
  bubble and at the final table (section below).
- **256 GB would not hold a 9-max table blueprint** at useful detail. A 6-player table needs 154 GB with the
  lean menu (77 GB with the average kept pre-flop only) and 508 GB with the medium one; 9-max pre-flop feeding
  post-flop pots of up to six is larger again, since each way of reaching the flop has its own post-flop game.
  Pluribus trained in about 512 GB. The neural route needs GPU time, not RAM; the lab's RTX PRO 6000, 4090 and
  3090 with 125 GB already cover it. A 512 GB machine matters only for the documented table route.

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

## Watching it learn and play (`web/`, served by `scripts/ninemax_watch.py`, 2026-11-10)

A web page served on the lab (127.0.0.1:8790 only), reached with
`ssh -N -L 8790:127.0.0.1:8790 labuser@LAB_HOST` and http://localhost:8790. The training runs need no change.

| View | Default | What it shows |
|---|---|---|
| 1. Training | on | Every run's evaluations charted against hands trained: vs the tight-aggressive player, the exploiter's winnings (exploitability), vs its previous checkpoint, vs maniacs, stations and random players, speed, entropy |
| 2. Hands | on | For every new checkpoint, the watcher plays 48 deals with it (nine seats, all cards) and keeps the 8 busiest plus a quiet one (`results/ninemax/watch/<run>/<hands>.json`); a table replayer steps through them, so its play can be compared across checkpoints |
| 3. Real tables | from the page | bot_sim sessions (`results/ninemax/table*/`): the network against the solver bot, the current bot, the clones and personas |
| 4. Live | from the page | Fresh hands from the newest checkpoint, dealt continuously, against itself or eight rule players |

Started 2026-11-10 07:50 lab time: `PYTHONPATH=scripts .venv-gpu/bin/python scripts/ninemax_watch.py` (CPU, 2 threads).

## Stacks, bubbles and final tables (2026-11-10)

**The question** (from the user's partner): will it understand the number of big blinds, its position, the payouts,
bubble situations and final tables?

**What the 9-max network understands as of `nm1`–`nm5`:**

| Concept | Understood? | Why |
|---|---|---|
| Position | Yes | Its seat (one-hot, SB to button), who has folded and who is still to act are inputs; the real-table results come mostly from pre-flop |
| Big blinds (stack depth) | Only 100bb | Every training hand started with nine 100bb stacks |
| Payouts, bubble, final table | No | It maximises chips; it does not know that busting on the bubble costs more than doubling up gains |

**The terms, made precise:**

- **Bubble:** with 500 entrants and 200 paid, the bubble is the moment 201 are left: one more bust and everyone left
  is paid. The pressure starts earlier, from roughly 220–230 left. Busting then wins nothing, so short and medium
  stacks tighten.
- **Final table:** the last nine. Each place pays clearly more than the one below, so the same pressure is on every
  hand.
- **"Worse hands are way more valuable" there** is true for the player **betting** and false for the player
  **calling**. A big stack can shove much weaker hands, because medium stacks cannot afford to call off before the
  next pay jump. A player facing an all-in needs a much stronger hand to call than in a cash game. A tournament bot
  needs both: wider shoves when others are under pressure, tighter calls when it is under pressure.

**The design (built 2026-11-10; to be tested on the lab):**

1. **Any stack depth** (`ninemax_engine.Tables(..., stacks=[G, P])`). Every seat has its own stack, at least 2bb;
   calls and all-ins are capped by it, shoves below a full raise do not reopen the betting, side pots follow the
   contribution levels as before. `ninemax_check.py ... mix` checks it against PokerKit with every seat on its own
   2–200bb stack. The trainer's `--stacks LO,HI` draws each stack log-uniform in that range, and the network also
   sees its stack, every player's chips behind and its effective stack.
2. **ICM** (`scripts/icm.py`). Malmuth-Harville: a player finishes first with probability stack / total chips, the
   next places likewise among those left; equity = the sum of P(place) × prize. Exact for nine players: the
   probability of every set of players taking the top places is carried place by place (2^9 = 512 sets), batched
   over whole training batches. Checked against brute force over all finishing orders (3, 5, 7 players); equities
   sum to the prize pool; winner-take-all gives the chip share. Players busted in the same hand are placed by
   their starting stacks, the usual rule.
3. **The tournament network** (`--tourney`). Every training table is a tournament situation (`icm.scenarios`):
   - stacks: an average of 10–60bb (log-uniform), each seat spread around it, at least 2bb;
   - **final table** (40%): places 1–9 paid, top-heavy (prize ∝ place^−α, α 0.6–1.4);
   - **bubble** (40%): the next 1–3 players to bust get nothing, the rest a nearly flat ladder (the min-cash and the
     first steps): a table near the money;
   - **in the money** (20%): every place paid, a gentle ladder.

   The network sees the payout ladder, every player's ICM equity at the start of the hand and its own equity per
   chip, on top of the stack inputs. **Its reward is the change in its ICM equity**, in big-blind equivalents (the
   equity change × the table's chips / BB): with prizes proportional to chips this is exactly the chip result, so
   cash and tournament play share one scale. It is expected to learn the bubble and final-table adjustments by
   itself: shoving wider as the big stack, calling tighter as a medium stack.

**Limits of this first version:**

- One table is the whole world: at the bubble, the other tables' players are not modelled. The ladder at this
  table (who busts first gets nothing) stands in for the field.
- Short-handed final tables (fewer than nine left) and antes are not yet in the scenarios.
- No push/fold Nash solver yet: the network covers short stacks through its all-in action. A Nash push/fold chart
  for nine players under ICM is the planned check of its short-stack play.
- The real-table adapter plays mixed-stack networks at any depth from 2bb, but declines tournament networks
  (bot_sim tables have no payouts) and lets Monte play those hands.

**Runs to start when the lab is reachable:** `nm6`, cash with mixed stacks (`--stacks 5,200`), and `nm7`, the
tournament network (`--tourney`), both with population and exploiter training and the ramped pull. First the
engine check with mixed stacks and the ICM self-test.
