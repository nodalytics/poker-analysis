# An audit, checked, and the plan that follows (2026-11-23)

An audit ("CFR precision and 9-max tournament refinement roadmap", 2026-11-23) proposed bug fixes in
`research/` and `kestrel/`, then five pre-registered runs (nm45-nm49). Before planning on it, every claim was checked
against the code and the result files. The code claims are worth having. Several numbers were misread, the headline
heads-up fix changes nothing, and the roadmap repeats the pattern its own summary warns against. This document
records the checks, then the plan: measurement first, the confirmed fixes, then training aimed at the opponents
nm25 actually loses to.

## 1. The audit, claim by claim

### Numbers

| Claim | What the files say |
|---|---|
| nm25 misses 4 of 14 pre-flop sense checks; post-flop 8/8 | `sense2_nm25.json` has 16 checks, 5 warn; `postflop2_nm25.json` passes 12/12 (4/14 and 8/8 are an older count) |
| shoves 6.8% UTG, 21.9% CO at 8bb | shares of hands **played** (16% and 31%); about 1.1% and 6.8% of all hands |
| 0.89 push/fold agreement | that is the chip-EV scenario. Under ICM: final table 0.60, bubbles 0.47-0.76, mixed stacks 0.87 |
| Nash jams 100% (button, bubble) | the chart (`icm_pushfold.py`) is fictitious play with at most one caller; the bubble scenarios pay `[1]*8` (satellite-shaped). The final-table chart is plausible; the bubble chart is not a normal MTT bubble |
| nm25 over-calls SB jams 27.3% vs 7.1% | matches. **This and the final-table under-shove are the fair findings** |
| exploiter leaks: ITM <12bb +0.594 (Holm 0.021); bubble 18-25bb +2.795 (Holm 0.068) | numbers match; the second is **not significant**; overall the exploiter wins +0.30 [−0.14, +0.74] |
| river TV 0.22 across "nm25-nm43", 26.6% flips | mean pairwise TV among 8 models (25, 26, 27, 36, 38, 40, 42, 43) on 206 river spots; T5s 3.4bb is the most disputed spot, picked |
| nm25 folds too much to bluff jams | the bluffs win chips and **lose prize equity** (−12 to −49): the folds are right under ICM |
| six single-knob runs within ±0.4 of nm25, exploitability rising | eight runs; nm32/nm33 were judged against nm31, not nm25; 4-5 have exploiter readings; the nm25 reference is two readings of the same setup (+0.2 [−0.4, +0.7], +0.2 [−0.3, +0.8]), while one nm43 run read +1.0, +0.7 and +2.1 and was judged on the last |
| search −96.5, search-exploit −41.7, nm25 +75.5 bb/100 vs rule bots | match; one seed, 2,000 hands; search-exploit is not distinguishable from zero |
| run2 vs Slumbot +12.4, +30.9 with MDF | match, but 5,000 hands: [−16.0, +40.7] and [−10.1, +71.8]. Neither is a win |
| re-solve −20.0 [−38.2, −1.8] over 20k | matches; which re-solver it ran is not recorded |
| re-solve bleeds +996 to LBR | that was the **unsafe TexasSolver** re-solve (2026-11-08); the safe re-solve reads +105 [+42, +168] |
| champions nm25 and run2/run6 | nm25 and **run2** (shipped); run6 is not a champion (Slumbot +6.0 [−36, +48]) |

### Code

| Claim | Verdict | Action |
|---|---|---|
| **HU** re-solver passes `rg` without `enter(i)` in `cfr(opp)` (P0) | **Wrong as a bug.** The traverser's own reach only weights its average strategy; `enter(i)` belongs in the opponent's reach while we traverse, which is what the code does. The mis-weighted average is the opponent's subgame strategy, which is never output. `resolve --test` already shows the safety property: worst-hand gain over CBV +1.4 (unsafe +1,179). **Tested 2026-11-23:** the re-solver built as is and with the audit's change gave byte-identical `--test` output on 24 spots (seed 7) and 300 spots (seed 11) | none |
| purify + MDF floor on top of the re-solve | true, deliberate and measured (+146 without the floor, +105 with it, `model.md`); it does void the formal guarantee | Kestrel gets the opt-out Python has |
| Kestrel re-samples off-tree bet mappings every replay | **confirmed**; `det=false` in production, selftests use `det=true` and cannot see it | fix |
| unsafe TexasSolver path | no gadget and dropped zero-reach hands: true and known ("stays off"). Kestrel's effective stack `my_stack + my_bet` (`headsup.cpp:375`): **confirmed bug**. "Incremental chips into raise": **wrong**, TexasSolver's raise is chips added | fix the stack; warn on the `texassolver` solver in Kestrel |
| `mccfr_vec` discount/checkpoint race | **confirmed** (master thread rescales and saves while workers write); PCFR+ is in `mccfr.cpp`, not missing | barrier |
| `advisor_pack` assumes `river_buckets.u8` | partly: a distribution-aware dir fails loudly (FileNotFoundError), it does not mis-pack | support `.u16` or refuse clearly |
| **9-max** empty seats in seating | **real, broader and milder**: `seats_used()` only knows `--seats`, so exploiter, league and population all seat on empty seats (`stacks=0`) at short tables, ~9% of seats. An empty seat never acts (u = 0): the cost is wasted seats and PFSP scores pulled toward 0, not corrupted learning. nm44 ran this path | fix |
| training lacks Kestrel's 15bb early-shove rule | true and **intentional**: a post-hoc mask, measured +0.21 (n.s.) | none |
| PKO bounty to `u.argmax(1)` | **confirmed** (ties and side pots; one more side-pot case than the auditor found); bounty runs only | fix + brute-force test |
| `<= 0.5` busts sub-chip survivors | **wrong**: stacks are chips, any survivor has ≥ 1. **Tested:** the engine's own `payoffs()` on 200,000 random tables (ties, side pots, folds, antes): smallest survivor stack 1.0 chip, none in (0, 1) | none |
| search: board dropped / no bounty / no CRN / ESS collapse / greedy | board: confirmed (texture nets only). Bounty: confirmed in Python and C++. CRN: deals shared, rollout draws not. ESS: computed, never acted on. Greedy: intentional | fix the first four |
| pipeline hardcodes `--menu mtt`, `libbp7.so`; thin validation; live checkpoints scored | first two confirmed (harmless until a recipe overrides them); checks **do** score `avg`. New: the exploit check takes the exploiter's **last** reading | read menu/lib from the run; judge the exploiter on the mean over seeds (A1) |
| nm44 "20 styles vs 25 players" | wrong: 5 archetypes + 20 generated; nm44's `styles.json` holds all 25 | none |

### Our own docs, checked the same day

| Doc claim | Check |
|---|---|
| `model.md` re-solve safety table (200 spots, 1,000 iterations): unsafe −573 ± 46 / +1,179, safe −161 ± 17 / +1.4 | **reproduced exactly** (seed 1): −573.0 ± 45.5 / +1,178.7 and −160.8 ± 17.0 / +1.4 |
| without the floor, worst single-hand gain +808 | today's build reads **+718** (same spots, `--floor 0`); same conclusion, the number in the doc is stale |
| LBR: blueprint +135, with the MDF floor +104; unsafe re-solve +996 | match `lbr_run2final_purify0.05` (134.9), `lbr_run2final_mdf` (104.1), `lbr_bp_resolve_mdf` (996.2, luck-adjusted; raw 895.5) |
| LBR: safe re-solve from the node +176, from the street's start +146, with the floor +105 | match, but the reports are **only on the lab** (`lbr_safe_resolve_v1_current_node`, `_v2_street_root`, `lbr_safe_floor`): copy them into `results/` |
| nm36-nm43 verdicts and intervals in `2026-11-17-upgrades.md` | match `results/prereg/*.result.json` |
| early-shove rule +0.21 bb/100, z +1.31 over 40,000 deals | **no saved result** in the repo or on the lab; the doc and the commit message are the only record. Re-run it (cheap) and save it |

## 2. What we already knew, and what is new

The plateau is ours (`2026-11-17-upgrades.md`): eight training changes, none moving nm25 by more than ±0.4 bb/100 in
its own game. Our conclusion then: gains are likelier **against opponents unlike nm25**, and nm25's exploiter
reference must be measured properly before judging fine-tunes on it. The audit adds three things that hold up: the
seating waste, the search defects, and the Kestrel heads-up bugs. Its roadmap (four more single-change fine-tunes of
nm25, judged partly on a one-caller push/fold chart) does not follow from them.

nm44 (the league) passed four of five checks and failed **other-players** on one opponent: station +98.9 against
nm25's +126.3, with a tolerance of 1.0 bb/100. ~~Against a calling station at 20,000 hands that tolerance is far
inside the noise.~~ **Corrected the same day:** the two networks play the same deals, so the difference is paired, and
its interval is about ±2 bb/100 (section 5). nm44's station gap is real; the check lacked an interval, not precision.

## 3. The plan

### Phase A: measure before training (no training, about a day)

A0, housekeeping first: copy the three lab-only LBR reports into `results/`, re-run the early-shove match
(`seqtest.py --early-shove-bb 15`, 40,000 deals) and save it, and correct `model.md`'s +808.

| # | What | Why |
|---|---|---|
| A1 | **Exploiter baseline**: the 30-minute exploiter against nm25, nm43 and nm44, 4 seeds each, same game; report mean and spread across seeds, and judge on the mean, not a run's last line | every adoption limit rests on nm25's +0.2 from one setup; one nm43 run read +1.0, +0.7, +2.1 and was judged on the last |
| A2 | **Paired other-players check**: the same deals for candidate and baseline (duplicate seating), a confidence interval on the difference, tolerance stated as an interval, not 1.0 bb/100; first measure the spread of the current check by running nm25 against itself on several seeds | the check that failed nm44 reports no interval; nm44's own checkpoints read +36 to +121 against the station |
| A3 | **ICM reference that can be a target**: push/fold with up to two callers and real MTT payouts on the bubble; then play a chart pusher against nm25 at 10bb final tables, in prize equity | before training toward a chart, show the chart beats nm25 |
| A4 | **Heads-up re-measure**: run2 against Slumbot, plain and with the safe re-solve, 20,000+ hands each, and record which re-solver ran; log every Slumbot bet above 2x pot and our answer (our tree stops at 2x, Slumbot's first bets go to 50x; `2026-11-24-slumbot-nl.md`) | every heads-up "win" so far has an interval through zero |

### Phase B: the confirmed fixes (each with a test that fails before it)

9-max, in `research/`:

1. Seating: draw each table's context before seating; exploiter, league, population and PFSP use seated players only.
2. PKO: each bust's bounty to the winner(s) of the highest pot level the player paid into, split on ties; a test
   against brute-force settlement, including the side-pot case.
3. Search (Python and Kestrel): pass the board; score bounties with the knockout term; seed rollouts per (deal,
   step) so every candidate move sees the same draws; fall back to the network when ESS is below a floor.
4. Pipeline: menu and library from the run's config; validate patterns and opponent-model values; the exploit check
   runs several seeds and reports their mean and spread.

Heads-up and Kestrel (each with `kestrel_selftest` and `kestrel_resolve_check` passing after it):

5. Effective stack in Kestrel's TexasSolver call; a warning on the `texassolver` table solver.
6. Off-tree mappings cached per action history in Kestrel (Python's behaviour); a selftest with `det=false`.
7. `mccfr_vec`: barrier around the discount and the checkpoint save.
8. `advisor_pack`: pack `.u16` distribution-aware tables, or refuse with a clear message.
9. Kestrel: the floor-off option for re-solved nodes the Python bot has.

Not changed: the re-solver's `rt` (not a bug), the bust threshold, the early-shove rule in training, the styles count.

### Phase C: training, pre-registered, after A and B

| Run | Question | Gate |
|---|---|---|
| nm44 re-judged | with A1's baseline and A2's paired check, does the league (already trained) pass? Registered as a re-test of a failed run, with that caveat stated | all five checks, the new other-players interval |
| nm45 | the league re-trained with fixed seating (B1): does it beat nm44 where nm44 was weak, calling stations? | no-harm; paired other-players; styles; exploiter within A1's nm25 spread |
| nm46 | only if A3 shows the chart pusher beats nm25 at short ICM final tables: nm25 trained with those tables weighted (nm42 showed weighting alone moves leaks around, so A3's gain is the target, not chart agreement) | the A3 match reversed; no-harm; exploiter |
| HU | Kestrel's heads-up bot after B5, B6 and B9 plays A4's Slumbot match too | Kestrel level with the Python bot (intervals overlap); any blueprint retrained with `mccfr_vec` waits for B7 |

Dropped from the audit's roadmap: search distillation (nm36 already tried it: equivalent) until search itself
beats the plain network against rule bots; R-NaD KL regularisation (another single-knob run on a plateau, held for
after nm45); training to push/fold chart agreement (A3 first).

### Phase D: where the remaining gain is

Unchanged from the plateau finding: opponents unlike nm25. Search with B3 is re-tested against the rule-bot lineup
(seeds 21 and two more), `search-robust` with it; real human hands when Open Poker's sign-in arrives. Platform
integrations and the real-money and RTA rules are unchanged.

## 4. Order and cost

A1, A2 and the B fixes run in parallel (A on the PRO 6000 and the 3090, while a coding-model evaluation holds the
4090; B on CPU and locally). Then the nm44 re-judgement (checks
only, hours), then nm45 (6 hours of training plus checks). nm46 waits on A3. The heads-up work is independent.
Every run goes through `research/pipeline` with its `nmXX.json` registered before it starts; no re-reading of a
failed run except as a new, stated registration.

## 5. Results so far (2026-11-23)

**A1, the exploiter baseline** (30-minute exploiter, pipeline settings, seeds 7-10, two seeds on the PRO 6000 and
two on the 3090 per model; logs in `results/exploit_baseline/`):

| Model | Final readings | Mean | SD |
|---|---|---|---|
| nm25 | +1.1, +0.8, +0.4, +0.7 | **+0.75** | 0.29 |
| nm44 | +0.7, +0.4, +0.7, +0.2 | **+0.50** | 0.24 |
| nm43 | +1.4, +1.2, +0.9, +1.8 | **+1.32** | 0.38 |

nm25's reference was +0.2; on four seeds it is +0.75, so "nm25 + 1" limits were set from a low draw. nm44 is the
hardest of the three to exploit; nm43 is still about 0.6 above nm25. Limits are now stated against these means.

**A0, the early-shove rule** (`results/early_shove/`). The documented +0.21 bb/100 had no saved result. Re-run with
the default command it read −0.50 (z −2.66, "difference" at the sixth look). Five replicates on separate seeds,
40,000 deals each, with a control (nm25 against itself, no rule):

| | Mean | Range |
|---|---|---|
| nm25 with the rule | −0.11 | −0.33 to +0.11 |
| control, no rule | −0.04 | −0.24 to +0.16 |

The rule is neutral: +0.21 and −0.50 were single draws from a spread the control shows too. Kestrel keeps it. Lesson
for every single sequential match: `seqtest.py` seeds the deals but not the context or the action sampling, so one run
is one draw; claims near the boundary need replicates.

**A tester's report: "it folds to almost all pre-flop aggression"** (`scripts/ninemax_aggressor_check.py`,
`results/aggressors/`). One scripted aggressor seat against eight copies, by duplicate, 20,000 deals x 9 seatings,
in prize equity and in chips, bb/100 for the aggressor:

| Stage, depth | jam any two | jam top 40% | 3x raise or 3-bet any two | nm25 folds facing an all-in |
|---|---|---|---|---|
| ICM tables, 10-25bb | −49.2 equity, **+7.9 chips** | −11.1, **+10.9 chips** | −12.3, −15.5 | 87% |
| early stage, 10-25bb | −137.2, −140.5 | −25.7, −24.8 | −26.9, −25.8 | 71% |
| ICM tables, 25-60bb | −112.3, −22.9 | −29.5, +1.2 | −7.2, −8.5 | 91% |
| early stage, 25-60bb | −362.3, −362.4 | −82.4, −81.9 | −39.0, −35.9 | 78% |

nm25 does fold to most all-ins, and at short ICM tables a jammer wins **chips** against it, which is what a player at
the table sees. But every aggressor loses **prize equity** at every stage and depth: the folds are correct, and calling
more would cost money. nm43 and nm44 read the same within a few bb/100. A thinking human picking spots could do better
than these scripts; the learning exploiter (+0.75 on nm25) is the measure of that, and it is small.

**A2, the paired other-players check** (`scripts/pipeline_opponents.py` now reports `difference_ci`;
`results/paired_opponents/`). nm44 minus nm25, 1-vs-8 against each rule player, 20,000 deals x 9 seatings, seeds 13-15:

| Against 8 x | Seed 13 | Seed 14 | Seed 15 |
|---|---|---|---|
| reg | −0.40 [−1.04, +0.24] | −0.38 [−1.05, +0.29] | −0.55 [−1.23, +0.12] |
| tag | −1.05 [−1.84, −0.26] | −0.48 [−1.27, +0.32] | −0.53 [−1.30, +0.25] |
| maniac | −3.35 [−4.32, −2.38] | −3.31 [−4.24, −2.38] | −3.23 [−4.20, −2.26] |
| station | **−27.40 [−29.20, −25.60]** | **−27.44 [−29.14, −25.73]** | **−28.56 [−30.36, −26.76]** |

nm44's failure stands: the league costs about 22% of nm25's win against calling stations and a little against
maniacs, while making it the hardest of the three to exploit (+0.50). So **nm44 is not re-judged into adoption**, and
the planned "nm44 re-judged" run is dropped. nm45 (the league on fixed seating) must keep nm25's station win rate,
judged on this paired interval, not only tie nm25 and resist the exploiter. A league trained against itself loses value
against players who never fold; adding stations and other non-folding styles to the league's pool is the obvious lever.

**B1 smoke test:** 10 minutes of training on the fixed seating with nm44's settings (league, exploiter, short tables;
`results/paired_opponents/smoke_seating_progress.log`): 38,000 hands a second, no errors, checkpoint written, exploiter
−0.2 and −0.1. Its station result fell from +125.9 to +117.9 within the 10 minutes, the same direction as nm44's.

**B5-B9, heads-up and Kestrel fixes** (each with a check that failed before it):

| Item | Fix | Check, before → after |
|---|---|---|
| B5 | TexasSolver's effective stack is `min(mine + my bet, theirs + their bet)`, as Python's; `texassolver` is labelled unsafe (`"unsafe": true` in `/v1/engines`, `kestrel/docs/design.md`) | `kestrel_hu_check`: unequal stacks (19,800 vs 4,800) gave `set_effective_stack 396` → 96. With ACPC's equal stacks it never mattered |
| B6 | the off-tree mapping's draw is a hash of the hand so far (our cards, the board, the betting up to the bet) and a per-engine salt, so a hand keeps one mapping and hands still differ | `kestrel_hu_check`, `det=false`, one hand replayed 26 times: 4 of 56 flop bet sizes stable → 56 of 56; across 40 other hands 51 sizes still reach two or more nodes |
| B7 | `mccfr_vec`: the master raises a flag, all threads park at a barrier, discount their own slices, the master saves, all resume | lab, abs_200g2, 8 threads, 1 minute, DCFR every 5 s: before, the last checkpoint holds 544 iterations but training stopped at 558 (the save raced the workers); after, 534 and 534. Both resume from their checkpoint. Throughput the same within noise (8.2-9.3 it/s). A race cannot be shown numerically beyond that; TSan would also flag the deliberate lock-free (Hogwild) regret updates |
| B8 | `advisor_pack` refuses a distribution-aware (OCHS) abstraction by name, reads `.u16` tables, and refuses river labels not ordered by hand strength | `scripts/advisor_pack_check.py`: FileNotFoundError on all three cases → refused, packed exactly, refused |
| B9 | `--resolve-floor on|off` (`kestreld serve` and `play`, `settings.json` `resolve_floor`); default on | `kestrel_hu_check`, 40 re-solved river spots facing a bet: floored 6 with the default, 0 with it off |
| server | malformed input is a 400 (exception handler plus typed checks), `/v1/admin/shutdown` refuses a public Host, the token is compared in constant time | `kestrel/tools/server_check.sh`: 11 failures (500s, and shutdown 200 through `kestrel.test`) → all pass |

**B3, search** (Python `scripts/ninemax_search.py`, Kestrel `core/src/ninemax_search.cpp`; checks
`scripts/ninemax_search_check.py` and `kestrel_search_check` (a ctest), logs in `results/search_b3/`). On the old
search the Python check fails 9 checks; on the new one all pass.

| Defect | Fix | Check, before → after |
|---|---|---|
| board dropped (texture nets) | the samples carry their dealt board (`deal_fixed` returns the cards); `match()` and the distillation tables carry the real one | nm41's texture inputs saw a board in 0 of 43 search calls and never in match → every call; nm25's Q bit-identical with and without the board |
| bounties ignored | rollouts scored with the trainer's own reward (`end_equity`, factored out of `play()`; C++ `end_equity` beside `start_equity`); `with_pay` uses `start_equity`, so the network's equity inputs match training too | nm30, 4,000 hands: gap to the trainer's reward up to 14.6 bb (mean 1.66) → 0; nm25 bit-identical to the old score |
| rollout draws independent per move | one uniform per (sample, step), shared by every move and continuation, inverse-CDF sampling (`--no-crn`: the old draws); C++ the same (`SearchConfig::crn`, `rollout_seed`) | the draws are shared (checked on them); variance below |
| ESS never acted on | below `min_ess` x samples (default 0.02) the decision is the network's own policy, and counted (`ess_fallback`; C++ `fell_back`, `ess_fallback` in the advice) | no floor → fallen-back decisions play the network's policy exactly, the rest greedy |

Common random numbers, nm25, 24 spots x 24 searches of 256 samples, mean variance of Q(a) − Q(b) over 135 move pairs:
with the same deals (rollout noise only) 0.0762 → 0.0614 bb² (x0.81, lower in 67% of pairs); with fresh deals (all
of search's noise) 0.2097 → 0.1714 (x0.82, lower in 60%). On the old search, where both arms are the same algorithm,
the same reading gave x1.10 and x1.05 (lower in 47% and 41%). A real but modest gain: most of the noise is the deals
and their weights, which were already shared, and after different moves the players face different spots, so shared
draws couple them only loosely.

Kestrel (`kestrel_search_check`, `results/search_b3/kestrel_search_check.log`): folding to a heads-up shove, whose
value is exact, read −0.30 bb against the trainer's −1.30 on nm30 before (plain ICM) and −1.30 after; nm25 −0.60
both. With the same deals, CRN took the rollout variance of Q(a) − Q(b) from 0.429 to 0.347 bb² (x0.81; 6 spots x 8
searches, 48 pairs), the same as Python. Texture inputs do not exist in Kestrel's features (no Kestrel model has
them), so the board item is Python only.

ESS: in a 300-deal nm25 match (95 decisions searched at 256 samples) the ESS share's 1st, 5th and 10th percentiles are
0.008, 0.010 and 0.015 (median 0.12). The floor of 0.02 (5 effective deals of 256, 20 of Kestrel's 1,024) sends 13.7%
of searched decisions back to the network; 0.05 would send 29% and 0.1 43%. 0.02 is the conservative choice: it removes
only decisions whose values rest on a handful of deals. Whether it helps in play is not measured yet: run
`ninemax_search.py` with `--min-ess 0` and with the default on the same seed, and compare with `search_paired.py`.

**Merged 2026-11-23 (B1-B9 all in `main`).** Two notes from the merge:
- B3's `end_equity` (the trainer's end-of-hand equity, now shared with search) still gave every bounty to the hand's
  biggest winner; on merging it uses B2's `knockouts`, and Kestrel's C++ `end_equity` was ported to the same rule, so
  search scores bounty games exactly as the trainer rewards them. Kestrel's ctest passes 6/6 on the merged code
  (selftest, hu_check, server_check, selftest_pko, selftest_cash6, search_check); the PKO, seating and search checks
  pass on the lab.
- Common random numbers reduce the noise between moves only on the same deals (x0.73-0.81); with fresh deals the
  ratio read x0.82 in one run and x1.12 in another, so treat CRN as a small gain at best: the deals, already shared,
  are the main noise. The ESS floor's effect on win rate is still to be measured (a paired match with and without it).


**A second audit, checked (2026-11-24).** A line-by-line audit of `kestrel/core/src/` and the training code
lists six bugs; they are B5 (effective stack), B7 (mccfr_vec), B6 (off-tree mapping), B1 (seating), B2 (bounties) and
B3 (search), described against the code before this plan's fixes, which were already in `main`. It adds two points:
- Kestrel's practice tables still paid each knockout to the hand's biggest winner (the audit called this correct). They
  now use the rule training and search use: the best live hand(s) among the players who covered the busted player,
  split on ties. `kestrel_selftest` on a PKO model now plays three bounty practice tournaments and checks that the
  prizes plus the bounty cash collected add up to the prize pool (worst error 1e-16).
- `blueprint_bot.py`'s off-tree mapping cache cleared itself whole at 200,000 entries, which could re-draw an earlier
  bet's reading in the middle of a hand. It now drops the least recently used half, so a reading the hand has used
  is never dropped.

**nm45 drafted.** `pipeline/runs/nm45.json`: the league (as nm44) on the fixed seating, with the five archetypes
(nit, station, maniac, shover, loose-passive) added to its pool and the rule players' share back at nm25's 0.5. Its
other-players check uses the new `"rule": "interval"`: on 100,000 paired hands, the lower end of the difference's 95%
interval must be above -2.0 bb/100 against every rule player (nm44's station reading, -27.4 [-29.2, -25.6], fails it).
The exploiter is judged on the mean of four seeds with a limit of +1.25 (nm25's +0.75 plus 0.5).

**A3 results (2026-11-24): the ICM push/fold chart against nm25** (`scripts/icm_pushfold.py`,
`scripts/equity3_169.py`, `scripts/ninemax_chart_match.py`; `results/pushfold_a3/`, pre-registered in
`results/pushfold_a3/prereg.json` before any deal was played).

The reference first. `icm_pushfold.solve2` allows up to two callers after a shove (the first caller may be followed by
one overcaller; everyone after folds). Three-way all-ins are valued from a new table of three-hand equities
(`results/preflop/equity3_169.npz`, 9 MB): every unordered triple of the 169 classes, 4,000 Monte Carlo deals each with
card removal among the three hands, the six strength orders stored, so side pots with unequal stacks are settled
exactly. The one-caller solver is kept. `--selftest` adds: AA wins 0.670 against KK and QQ; a hand-built side-pot case;
a three-handed table whose values (everyone's, and the shove, call and overcall values by class) match a direct
enumeration of every class triple; the batched three-way sums match a direct sum; with overcalls forbidden the
two-caller solver reproduces the one-caller one; winner-take-all equities sum to 1 and AA always overcalls, 72o never.
Two fixes came out of the checks, and both change the one-caller charts slightly: a player busted by an all-in had a
stack of exactly 0, which Malmuth-Harville pays nothing at all, so at a final table every all-in's downside was
overstated by the lowest prize (now a vanishing stack, as the engine's reward does); and the 169x169 table's two
entries for a pair are separate estimates that disagree by up to 0.019, now averaged so both sides of an all-in agree.

The match. The chart player is nm25 except before the flop when it is first in (shove or fold), facing a first-in
shove (call or fold) or facing a shove and one call (overcall or fold); there it plays the two-caller chart solved
for the table's exact stacks. Anywhere else (a limp or a smaller raise ahead, a re-shove, after the flop) it plays
nm25. Tables: 40 fixed stack configurations per stage, 9 players, average stack log-uniform 8-12.5bb, each stack x
lognormal (sd 0.7, as training's tables) and at least 2bb, the big-blind ante of nm25's game; every chart converged to
a best-response gain under 0.01bb (median 0.003-0.004). Payouts: a final table paying 26, 17.5, 12.6, 9.4, 7.2, 5.6,
4.4, 3.6, 3.0 (% of the pool), and a bubble with 9 players and 8 paid (the same ladder's top 8, ninth nothing). The
match is duplicate and paired: the chart player in each of the 9 seats against 8 x nm25, and nm25 in the same seat on
the same deals, with common random numbers for every action draw, so the two arms play identical hands until the
chart acts differently. 500,000 deals x 9 seatings per arm and stage; prize equity in bb/100-equivalent:

| Stage | Chart − nm25, prize equity | Replicate (seed 22) | Chips | Control (nm25 vs itself) |
|---|---|---|---|---|
| final table | **−0.14 [−0.22, −0.07]** | −0.13 [−0.21, −0.05] | +1.66 [+1.51, +1.81] | +0.03 [−0.04, +0.09] |
| bubble | **+0.65 [+0.55, +0.75]** | +0.64 [+0.54, +0.74] | +4.23 [+4.08, +4.37] | +0.01 [−0.07, +0.08] |

By the hero's own stack (per seat-hand; seats of one deal share cards, so these intervals are approximate), final
table / bubble: under 8bb +0.27 / +0.83, 8-15bb −0.26 / +0.66, over 15bb −1.12 / +0.11. The chart helps where
push/fold is the right game (short stacks, and the bubble's stronger ICM pressure) and costs equity at the final
table's deeper stacks, where nm25's smaller raises do better than shoving. First in, the chart shoves 25% (final
table) and 31% (bubble) of the time against nm25's 11% and 17%; facing a shove it calls 13% and 9% against nm25's 11%
and 8%. In chips the chart wins at both stages: chips overstate it, as with the aggressors above.

Verdict, by the pre-registered rule (Holm over the two stages): the bubble result is positive and survives, so
**nm46 is worth registering**, with two limits on what it may claim. The gain is small (+0.65 bb/100 at the bubble)
and the chart loses at the final table, so nm46's target is the bubble and stacks under about 15bb, not chart
agreement everywhere: pulling deeper final-table stacks toward the chart would cost equity. nm46's gate stays as
planned: the A3 match reversed (nm46 at least level with the chart player at the bubble, and no worse than nm25 at
the final table), no-harm, the exploiter. Caveats: the intervals cover the deals, not the choice of 40 table
configurations or of the payout ladders; the push/fold model ignores limps and small raises, which the chart player
meets by playing nm25.

**The ESS floor in play (2026-11-24): no measurable effect.** nm25 searching at 256 samples, 200,000 deals per arm,
seed 4242, with the floor (`--min-ess 0.02`) and without (`0`), against each opponent bias
(`results/search_b3/ess_floor/`: `run.sh`, the eight result files, `analyse.py` and its output `paired.txt`). The floor
sends 5.9-10.6% of searched decisions back to the network. Floor minus none, bb/100:

| Opponents | No floor | Floor | Floor − none |
|---|---|---|---|
| net | +0.30 | +0.35 | +0.05 [−0.17, +0.27] |
| call | +0.52 | +0.71 | +0.19 [−0.06, +0.44] |
| fold | +0.17 | +0.12 | −0.05 [−0.26, +0.16] |
| raise | +1.73 | +1.61 | −0.12 [−0.35, +0.12] |
| all four, pooled | | | +0.02 [−0.10, +0.13] |

Every interval includes zero, and the pooled one rules out an effect larger than about 0.13 bb/100 either way. The
floor stays (it costs nothing and keeps decisions resting on a handful of deals out of play), and it is not a lever
worth more work. One finding about the method: the two arms' per-deal results are uncorrelated (r = −0.001 to 0.021,
no deal identical), so "the same seed" does not give the same deals in `ninemax_search.py`. The intervals above are
therefore unpaired; a paired design would need the deals drawn from their own generator, apart from search's draws.

**Paired matches and four Kestrel follow-ups (2026-11-24).**
- `ninemax_search.match` is paired now: the tables' stacks and payouts follow `--seed`, and every action is drawn
  from `Draws`, one uniform per (deal, seat, step of the hand) from a generator of the match's own, apart from search's
  rollout draws (torch's global generator, never seeded there, drew both before). `ninemax_search_check.py` part 5:
  the network twice is identical on all 64 deals; search with the floor at 1.0 (always the network's policy) matches
  the network on 100% of deals; with and without the floor, 92% of deals are identical and the per-deal correlation is
  0.85 (0.00 in the match above). `ninemax_lbr.py` uses the same draws for its two arms, and its tables now carry the
  board (texture networks read it).
- Kestrel's settings page has the defence floor on re-solved river spots (on by default; `resolve_floor`).
- `hu_check` re-solves each river spot once and decides it with the floor on and off from the cache, and ctest runs it
  at 30 iterations: 18 s, from 103 s.
- The release pack stays 8-bit (`avg.u8`, 55 MB, against `avg.f32`'s 222 MB). Measured on run2's 20.3 M rows
  (`scripts/pack_u8_cost.py`): mean total variation 0.066%, 99th percentile 0.32%, worst 0.94%; the actions it rounds
  to zero hold 0.014% of the probability, and Kestrel drops every action under 5% before it plays.

**Outside reading (2026-11-24).** `2026-11-24-slumbot-nl.md` reads Slumbot NL (Jackson, AAAI 2013 workshop) in full
and checks DyypHoldem and the papers around Slumbot. Three items join the plan:
- A4 logs Slumbot's overbets (above 2x pot) and our answers; Slumbot NL allows first bets up to 50x pot, our tree 2x.
- `mccfr_vec` trial on a small blueprint, scored by LBR: about 50 boards an iteration instead of one, and boards walked
  through a shuffled list of all boards instead of drawn at random (Slumbot NL's two sampling choices).
- Read AV-AIVAT (Li, Chen, Huang 2026: luck correction with a test valid at any stopping time) and estimate how many
  deals it would have saved on nm45's checks, which ran 40,000 deals for no-harm alone.

## 6. To do (updated 2026-11-27)

Done this round: the blueprint sampling and tree trials (`2026-11-24-slumbot-nl.md`: flop cycle and batched updates
not adopted, wide tree not shown); the card-vs-translation diagnostic (translation is the leak: off-menu LBR +249 vs
on-menu +122); nm46 run through nm45's checks (`results/ninemax/nm46_byhand/`); nm45 and nm46 on the lab's Kestrel;
Kestrel 0.4.1 deployed on the lab with practice matches between models, with or without the user at the table.

**9-max**
- [x] nm47 (nm46's recipe, pre-registered): **not adopted** (2026-11-27). It failed no-harm (inconclusive after 40,000
  deals at −0.39 bb/100: not shown to be not behind nm25) and styles (mean −0.14 against nm25, needs at least 0); it
  passed other players, bluffers and the exploiter (+1.1, limit +1.25). With nm46's 200,000-deal −0.14, the recipe is a
  shade behind nm25 head to head. nm25 stays the default.
- [ ] nm46's exploiter check: the fourth seed (first three +1.0, +1.5, +0.9; limit +1.25).
- [ ] Bet sizing, in this order (requested 2026-11-27: more sizes are harder for people to read and open strategies the
      network cannot play now):
  - [ ] **A. More sizes on the menu.** The `wide` menu (opens 2/2.5/3.5bb, two 3-bet and 4-bet sizes, heads-up bets
        0.25-2x pot, two raise sizes; 8 actions instead of 6), never trained. Build: per-seat menus in the table
        engine (so a wide network and nm25 can share a table for the no-harm check and in Kestrel); a warm start from
        nm25 that maps its 6 actions onto the 8 by nearest size; a size-usage report (which new sizes it plays, where)
        and a size-tells report (how well its hand strength can be read from its size, against nm25). Then a
        pre-registered run after nm47.
  - [ ] **B. A size head:** fold, call or raise, then a size from a learned distribution over pot fractions. Needs a
        learning rule for a continuous action (NeuRD works on a fixed list) and an exploiter that can try sizes. After A.
  - [x] **A built** (2026-11-27): per-seat menus (trainer and Kestrel), the nearest-size warm start, the sizes report;
        nm48 registered (PR #6), queued after nm47.
  - [ ] **B. A size head** (built 2026-11-27, `SizeHeadNet`, menus `size8` and `grid16`, `ninemax_sizehead_init.py`;
        a 3-minute run trains at nm25's level). Still: the check suite on a size-head network, the size head in
        Kestrel's C++, a full warm start (3,000 steps), and a pre-registered run (nm49) after nm48.
  - [ ] **C. Search picks the size:** Kestrel's 9-max search scores extra sizes by rollouts at play time. After B.
- [ ] **A machine that keeps learning** (requested 2026-11-27): practice games feed training automatically. Already
      built: Kestrel stores every practice hand (`kestrel.db`, `sessions/`); before a run with `"patterns": true` the
      pipeline mines the live Kestrel's hands for spots where people beat the bot (`exploit_patterns.py mine`, one
      mined so far, plus two the partner found) and trains against them, and a `patterns` check tests them. Missing:
      (1) a standing recipe that registers the next run by itself when enough new practice hands have come in, with
      the same fixed checks (no hand-written run file); (2) an adopted network shipped to Kestrel automatically
      (the model pipeline, then a Kestrel redeploy); (3) mining the hands people lost too, and model-against-model
      practice tables. Learning stays offline and gated: a network that updated itself during play could be steered
      into a leak by the person playing it, so nothing reaches Kestrel without passing the checks.
- [ ] **Playing from depth** (requested 2026-11-27: "the models and Kestrel deeply understand poker, and play from
      depth"). Today the networks play by reflex, one pass from the spot to a move with the hand reduced to a bucket.
      The strong systems (DeepStack, Libratus, Pluribus, ReBeL) have four things more, and they map onto items 1 and 2
      below:
  1. *Thinking in ranges:* a belief about what each opponent holds, updated with every action. **Started:**
     `ninemax_belief.py` (a network over each opponent's 1,326 combinations, from the public state), used as search's
     proposal (`ninemax_search.py --proposal belief --over k`), measured by `ninemax_belief_eval.py`.
  2. *Looking ahead:* search at each decision under those beliefs (Kestrel's 9-max search, today with uniform guesses).
  3. *Knowing what a situation is worth:* a value network over everyone's ranges, so search can stop early
     (DeepStack); the heads-up turn re-solve test (C against D) is measuring whether it pays.
  4. *Seeing the real cards and the whole action history:* blockers, board texture, what a line of bets means.
     **Started:** `--history 24 --arch attn`, a small transformer over the hand's last 24 actions and every seat
     (each seat's actions attend to the others'), feeding the trunk; warm-started from nm25 with its summary at zero
     weight. Built and checked (a 3-minute run plays nm25 level); needs a pre-registered run (`--chunks 8`: memory),
     and Kestrel's C++ (a transformer encoder) once a run passes.
- [x] **Search oversampling in Kestrel** (2026-11-27): 8 x the deals, weighted, resampled; move-value error 1.39 to
      0.66 bb on 363 spots (`results/search_belief/`). Belief guesses cut it to 0.48; their C++ port is next. A paired
      Kestrel match (oversampling 1 against 8, 1,500 deals) is measuring the gain in play.
- [ ] **A poker foundation model** (requested 2026-11-27, after Kronos: github.com/shiyu-coder/Kronos, MIT, a family of
      decoder-only transformers pretrained on candlesticks from over 45 exchanges, a tokenizer turning OHLCV into
      discrete tokens, 4.1 M to 499.2 M parameters). The poker version: a tokenizer for hands (seats and stacks, cards
      when seen, every action with its size bucketed, board, showdown) and a decoder-only transformer pretrained by
      next-token prediction on billions of self-play hands (every hole card known), the HandHQ human hands we hold and
      the Pluribus hands. What it is for: an opponent model (a player's next action from their recent hands, the basis
      of bounded exploitation), a belief model (hidden cards, against `ninemax_belief.py`'s 0.73 bits), a start for
      fine-tuning, and realistic training opponents. What it is not: a strong player by itself; predicting play is not
      playing well, and every system that beat professionals got its strength from game-theoretic training and search
      (Libratus, Pluribus, DeepStack, ReBeL, Student of Games), with plain feed-forward networks. Stage 0: tokenizer and
      data; stage 1: a ~25 M model, scored on next-action log-loss against the current opponent models and on hidden
      cards against the belief network; stage 2: into search if it wins either.
- [ ] **RLGF** (requested 2026-11-28; `2026-11-28-rlgf.md`): reinforcement learning from game feedback. Experiments in
      order: label quality by rollout count; a game reward model; an RLGF fine-tune of nm25 as a pipeline run; DPO of
      the foundation model on game pairs.
- [ ] **Foundation model, the next steps** (picked 2026-11-28 from the user's ideas; `2026-11-27-poker-foundation-model.md`):
      (1) long-horizon opponent modelling on HandHQ players' consecutive hands; (2) token arrangement (suit-canonical
      cards, fused action tokens); (3) Mamba and a Mamba-attention hybrid (a build environment of their own, started);
      (4) self-play RL of the transformer through the pipeline. RLHF not picked: game results are the better signal.
- [ ] **Seat angles** (requested 2026-11-27, the "π" question): seats sit in a circle, so each seat's position among
      the players seated, counted from the button, enters as an angle, $(\cos 2\pi k/n, \sin 2\pi k/n)$ for position
      $k$ of $n$, for me and each other seat. Every table size reads alike and the button sits next to the small blind.
      An input change (`--seat-angles`), warm-started from nm25 with the new inputs at zero, judged by a run.
- [ ] **Expanding the model** (agreed 2026-11-27), in this order after bet sizing:
  1. **Search with beliefs** (ReBeL-style): search at play time over the opponents' predicted ranges rather than
     random cards, so the network works each spot out afresh; builds on Kestrel's 9-max search (+3.7 to +6.2 bb/100
     at 1,024 samples) and needs item 2's belief output.
  2. **A richer game state**: the cards themselves rather than buckets; the hand's action history read as a sequence
     (a transformer) rather than hand-made summaries; the belief output (`--belief-weight`) switched on.
  3. **Opponent modelling, then bounded exploitation**: a network conditioned on Kestrel's opponent profile, exploiting
     only within limits that grow with the evidence (the adaptive bot lost −22.9 bb/100 when its read was wrong).
  4. **Richer opponents, PSRO-style**: each round a best response to the current network joins the pool; with the
     generated and human styles, the partner's patterns and the practice-game loop, so self-play keeps finding new
     strategies (the league, nm45, lost the station: the pool must keep the rule players).
  5. **One network for every format**: 9-max tournaments, bounties, 6-max cash and heads-up, conditioned on the format.
  6. **The research loop itself**: a queue of hypotheses that writes the next run files, ranked by past results, under
     the same fixed checks.
- [ ] A3 follow-up: the push/fold target at the bubble and under about 15bb, from the better of nm25 and nm47.
- [ ] Partner's practice games: tally them by spot (`kestrel_practice_tally.py`) against nm25, nm45 and nm46.
- [ ] **Partner's notes on playing Kestrel** (2026-11-27): "still playing a high level of poker"; (1) with medium to
      large stacks it avoids unnecessary confrontation, probably ICM; (2) under about 10bb its ranges change to calling
      more bluffs; (3) it raises or goes all in first to win without a showdown. Judged "a small problem we can ignore
      for now". Against what is measured: (1) is ICM's advice for medium stacks, but the chip leader should do the
      opposite and pressure them, so measure entry and raising by chip rank at the table; (2) shorter stacks should call
      wider, and A3 found nm25's short-stack play off the ICM chart (the chart beat it under 8bb, +0.27 final table,
      +0.83 bubble) while calling shoves less often overall (11% against 13%), so the leak may be which hands, not how
      often: compare its calling range under 10bb with the chart's, hand by hand; (3) is fold equity, correct at
      10-25bb, and nm25 shoves first in less than the chart (11% against 25% final table, 17% against 31% bubble).
      None is a pattern that beats the bot (not for `pipeline/patterns/`); the A3 follow-up's target covers (2).
- [ ] Read AV-AIVAT; estimate the deals it would save on the 40,000-deal checks.
- [ ] Kestrel desktop app: the practice table editor (the web app has it).

**Heads-up**
- [ ] The turn re-solve's paired C-vs-D rerun (running): whether exact rivers beat blueprint leaves; if they do,
      stage 0 of the river value network.
- [ ] Off-tree bets: the same off-menu LBR with the turn and river re-solve on, against +249.
- [ ] A4: run 2 against Slumbot, plain and re-solved, logging Slumbot's bets above 2x pot.
- [ ] Batched updates and the wide tree stay options; a wide solve as long as the standard one only if A4's overbet
      count calls for it.
