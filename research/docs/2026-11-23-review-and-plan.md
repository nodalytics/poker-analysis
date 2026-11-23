# A teammate's audit, checked, and the plan that follows (2026-11-23)

A teammate's audit ("CFR precision and 9-max tournament refinement roadmap", 2026-11-23) proposed bug fixes in
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
nm25's +126.3, with a tolerance of 1.0 bb/100. Against a calling station at 20,000 hands that tolerance is far
inside the noise (nm44's own training evals against station swung +36 to +121). The check, not only the network,
needs fixing.

## 3. The plan

### Phase A: measure before training (no training, about a day)

A0, housekeeping first: copy the three lab-only LBR reports into `results/`, re-run the early-shove match
(`seqtest.py --early-shove-bb 15`, 40,000 deals) and save it, and correct `model.md`'s +808.

| # | What | Why |
|---|---|---|
| A1 | **Exploiter baseline**: the 30-minute exploiter against nm25, nm43 and nm44, 4 seeds each, same game; report mean and spread across seeds, and judge on the mean, not a run's last line | every adoption limit rests on nm25's +0.2 from one setup; one nm43 run read +1.0, +0.7, +2.1 and was judged on the last |
| A2 | **Paired other-players check**: the same deals for candidate and baseline (duplicate seating), a confidence interval on the difference, tolerance stated as an interval, not 1.0 bb/100; first measure the spread of the current check by running nm25 against itself on several seeds | the check that failed nm44 reports no interval; nm44's own checkpoints read +36 to +121 against the station |
| A3 | **ICM reference that can be a target**: push/fold with up to two callers and real MTT payouts on the bubble; then play a chart pusher against nm25 at 10bb final tables, in prize equity | before training toward a chart, show the chart beats nm25 |
| A4 | **Heads-up re-measure**: run2 against Slumbot, plain and with the safe re-solve, 20,000+ hands each, and record which re-solver ran | every heads-up "win" so far has an interval through zero |

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

**B5-B9, heads-up and Kestrel fixes** (each with a check that failed before it):

| Item | Fix | Check, before → after |
|---|---|---|
| B5 | TexasSolver's effective stack is `min(mine + my bet, theirs + their bet)`, as Python's; `texassolver` is labelled unsafe (`"unsafe": true` in `/v1/engines`, `kestrel/docs/design.md`) | `kestrel_hu_check`: unequal stacks (19,800 vs 4,800) gave `set_effective_stack 396` → 96. With ACPC's equal stacks it never mattered |
| B6 | the off-tree mapping's draw is a hash of the hand so far (our cards, the board, the betting up to the bet) and a per-engine salt, so a hand keeps one mapping and hands still differ | `kestrel_hu_check`, `det=false`, one hand replayed 26 times: 4 of 56 flop bet sizes stable → 56 of 56; across 40 other hands 51 sizes still reach two or more nodes |
| B7 | `mccfr_vec`: the master raises a flag, all threads park at a barrier, discount their own slices, the master saves, all resume | lab, abs_200g2, 8 threads, 1 minute, DCFR every 5 s: before, the last checkpoint holds 544 iterations but training stopped at 558 (the save raced the workers); after, 534 and 534. Both resume from their checkpoint. Throughput the same within noise (8.2-9.3 it/s). A race cannot be shown numerically beyond that; TSan would also flag the deliberate lock-free (Hogwild) regret updates |
| B8 | `advisor_pack` refuses a distribution-aware (OCHS) abstraction by name, reads `.u16` tables, and refuses river labels not ordered by hand strength | `scripts/advisor_pack_check.py`: FileNotFoundError on all three cases → refused, packed exactly, refused |
| B9 | `--resolve-floor on|off` (`kestreld serve` and `play`, `settings.json` `resolve_floor`); default on | `kestrel_hu_check`, 40 re-solved river spots facing a bet: floored 6 with the default, 0 with it off |
| server | malformed input is a 400 (exception handler plus typed checks), `/v1/admin/shutdown` refuses a public Host, the token is compared in constant time | `kestrel/tools/server_check.sh`: 11 failures (500s, and shutdown 200 through `kestrel.test`) → all pass |
