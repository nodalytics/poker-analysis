# 9-max: the next phases (2026-11-11)

Where the 9-max work stands, and the plan from here. The detailed results are in
[`2026-11-09-nine-max-plan.md`](2026-11-09-nine-max-plan.md); this page is the roadmap.

> **Scope (2026-11-11): 9-max tournaments (MTT) only.**
> - The cash runs (`nm1`–`nm9`, `nm11`) are finished or stopped, and kept as reference. What they taught carries
>   over: NeuRD is the hardest to exploit, about 8 hours of training is enough, and search needs about 2,000 samples.
> - Everything from here targets tournament play: ICM, 10–25bb stacks and push/fold.

## Where we are

The neural route has covered steps 1–5 of the original build order (blueprint plus evaluation):

| Done | Result |
|---|---|
| 9-max networks: cash at 100bb (`nm1`–`nm5`), mixed stacks 5–200bb (`nm6`), tournament with ICM (`nm7`, `nm8`) | All plateau within 2–3 hours of training; afterwards they gain +1 to +4 bb/100 per half hour against their own previous checkpoint |
| Real 9-handed tables (`bot_sim`, 2,000 sessions) | Every network tops its table, at +30 to +38 bb/100; the four cash runs cannot be told apart this way |
| Exploitability (a fresh exploiter, 30 minutes) | `nm3` +219, `nm5` +125, **`nm4` (NeuRD) +44**: the measure that separates the runs |
| ICM push/fold equilibrium for 9 players (`scripts/icm_pushfold.py`) | Matches the chip-EV solver exactly under winner-take-all |
| The tournament network against those charts (`scripts/ninemax_pushfold_check.py`) | 90% agreement at chip value, but only 74–81% at final tables and real bubbles: **it calls 3–6× too often under ICM** |
| The old "bubble" tables were satellites (a flat prize ladder) | Fixed: a top-heavy ladder above the bust spots; `nm7` continues as `nm8` |
| The watch page, https://poker.stbfarm.org | Live behind Cloudflare and nginx; systemd service `poker-watch` |
| Heads-up table solver run 7 (CPU) | Finished at 2.82 M iterations. **Final LBR +104.5 [+84, +125]** on 30,000 hands, level with run 2's +104: more training on a finer abstraction did not help |

## Phase A: finish the blueprint (GPUs, about 2 days)

1. **`nm9`: NeuRD with exploiters.** It combines the two changes that each lowered exploitability.
   - It starts automatically when `nm3` finishes on the RTX PRO 6000.
   - It keeps a snapshot every 4 hours (`--keep-hours 4`), so later checkpoints can be played against earlier ones
     (`ninemax_eval.py cross blueprint/nm9/avg_4h.pt blueprint/nm9`). That answers whether extra hours help, which
     the tag yardstick cannot show.
   - **Done means:** a fresh 30-minute exploiter wins less than `nm4`'s +44.
2. **`nm11`: cash at random 20–50bb stacks.** Stopped after 1.5 hours when the scope became tournaments only.
3. **`nm10` and `nm12`: the tournament runs** (restarted 2026-11-11 15:05 lab time, 24 hours).
   - **Depths:** most mid- and late-tournament decisions are made at 10–25bb.
     - `nm10` (RTX PRO 6000): three tables in four average 10–25bb, and one in four 5–10bb (pure push/fold),
       via `--depths "10,25;10,25;10,25;5,10"`. By seat, 49% are under 10bb, 38% are 10–25bb and 13% are over
       25bb, because stacks spread around each table's average.
     - `nm12` (4090): 10–25bb tables only, so the two runs show whether the 5–10bb share helps.
   - **No open-limping** (`--no-open-limp`, a rule in `ninemax_engine.RULES`): first in before the flop, from
     UTG to the button, a player folds, opens or shoves. The small blind may still complete.
     - Measured before the change, first in at UTG: `nm8` limped AA 90%, KK 45% and AKo 38%; `nm4` limped
       KK, QQ and AK about 47% and shoved the rest.
     - Limping strong hands lets weak hands see a cheap flop (the partner's AK vs Q9/KT example). The habit
       came from the training pool's rule players (maniacs, stations), against whom trapping pays.
     - The rule players follow it too: a call they would have made becomes the smallest open.
     - The PokerKit check still finds 0 mismatches.
   - **Open sizes: the `mtt` menu** (`--menu mtt`, in `ninemax_engine.MENUS`). It opens 2×, 2.5× or 3×; 3-bets,
     4-bets and the bets after the flop are as in `medium`, with the same number of action slots. The PokerKit
     check finds 0 mismatches over 2,000 tables, at equal and at mixed stacks. Restarted with it 2026-11-11 15:10
     lab time.
   - Recipe: NeuRD, exploiters on 25% of tables, a pull ramped from 0.1, entropy bonus 0.02, a snapshot every
     4 hours. Bubble tables are top-heavy above the bust spots.
   - The earlier versions are kept or dropped as follows:
     - `nm10a`, 1.5 hours at 5–15bb and 20–50bb tables, is kept.
     - The 10–25bb restart without the no-limp rule ran only minutes and was replaced.
   - **Done means:** at least 85% agreement with the ICM charts at final tables and real bubbles (`nm8`: 74–81%),
     the big blind's calling near the chart's, and no open-limping by construction.
4. **Pick the tournament champion**, by exploitability first and the ICM charts second.
5. **Tournament realism** (next):
   - **antes:** a big-blind ante, as most MTTs now use;
   - **short-handed final tables:** 2–8 players.

   Both need the engine and `icm.scenarios` extended, and the PokerKit check rerun.

## Phase B: search at play time (step 7 of the build order, about a week, CPUs)

**Status (2026-11-11):**
- Version 1 (root rollouts with network-weighted ranges) beats `nm4` by **+20 bb/100** at 2,048 samples. At 256
  samples it loses −27, because only about 14 of the samples count.
- **Next: score tournament hands by ICM equity** instead of chips, and search the tournament networks.

- When a hand reaches the flop heads-up or 3-way (94% of flops), the bot re-solves that spot during play instead
  of trusting the blueprint, as Pluribus did. The network supplies the ranges and the leaf values, and Pluribus's
  four continuation strategies sit at the depth limit.
- This is the fix for exploitability that more training does not give.
- **Done means:** it beats the plain network 1-vs-8, and a fresh exploiter's winnings fall sharply.

## Phase C: realistic opponents (step 6, alongside B)

- Extract HandHQ's 7–10-player hands, and build full-ring clones and opponent models, so the test tables look
  like the real player pool and not the 6-max clones.
- A tournament mode for `bot_sim` (stacks plus a payout ladder), so tournament networks can play the bots; the
  real-table adapter declines them today.
- **Done means:** clones that predict held-out players about as well as the 6-max clones (77–82%).

## Phase D: the advisor, rewritten in C++ (step 8)

The advisor (`advisor/`, Python and Tkinter today) is rewritten in **C++ with Qt 6**. Design in
[`advisor/README.md`](../advisor/README.md#c-rewrite-qt-6-planned-2026-11-11).

- The same tabs: session review, trainer, spot study, hand sync.
- The blueprint engine (`bp.dll` / `libbp`) is linked directly instead of through Python.
- New 9-max data: the cash and tournament networks exported to ONNX or LibTorch, the ICM layer, the push/fold
  charts, and search (phase B) once ready.
- **Periodic screenshots, capture only:**
  - The interval is set from milliseconds to minutes.
  - Screenshots are saved to a folder for review after the session, and are **never analysed while you play**.
  - Capture **pauses itself while a real-money poker client is running**, except for the advisor's own Trainer
    table.
  - This keeps the advisor out of real-time assistance, which real-money sites ban.
- **Done means:** a native build for Windows (and macOS and Linux) that self-tests, gives 9-max advice in review,
  trainer and spot study, and records sessions at the chosen interval.

## Open items (smaller)

- The automatic commit security review fails with HTTP 401: re-authenticate the security plugin.
- Optional: a sampling estimate of the full 9-max tree size, which only matters if a 512 GB machine appears.

## Decisions taken

| Date | Decision |
|---|---|
| 2026-11-09 | The neural route on the lab, not a 512 GB table |
| 2026-11-10 | poker.stbfarm.org is public; the backend stays bound to 127.0.0.1 |
| 2026-11-10 | Bubble tables are top-heavy above the bust spots (satellites excluded) |
| 2026-11-11 | The advisor is rewritten in C++ with Qt 6 |
| 2026-11-11 | Cash training moves to random 20–50bb stacks (`nm11`); 100bb is not what real tables look like |
| 2026-11-11 | `nm10`, the tournament curriculum, at 5–15bb and 20–50bb tables |
| 2026-11-11 | **Scope: 9-max tournaments (MTT) only.** `nm11` (cash) stopped; the cash runs are reference |
| 2026-11-11 | Tournament tables at 10–25bb (`nm10` also 5–10bb on 1 table in 4; `nm12` 10–25bb only) |
| 2026-11-11 | No open-limping from UTG to the button: fold, open or shove (the partner's advice, confirmed by the measured leak) |
| 2026-11-11 | Tournament menu `mtt`: opens of 2×, 2.5× and 3× |
| 2026-11-11 | Advisor screenshots are capture-only and pause while a real-money client runs; no live table reading |

| Pending | Decision |
|---|---|
| Antes and short-handed final tables | Next tournament run after `nm10` |

## Poker sense from the partner, and what it means for the models

The principles of tournament play and the checks that hold the networks to them are in
[`2026-11-11-mtt-strategy.md`](2026-11-11-mtt-strategy.md) (`scripts/ninemax_sense.py`).

The advice (2026-11-11): with strong hands (AA, KK, QQ, AK, AQ, KJ), raise before the flop instead of calling.
Weak hands such as Q9 or KT should not see a cheap flop: some of them hit, and they take the pot from AK. Raising
makes them pay to see the flop or fold, and they rarely call a raise with nothing. The size need not be 2.5×.

**Why the networks broke this rule, and the general lesson:**
- A network learns whatever wins against the opponents it trains against. The training pool includes rule-based
  maniacs (who raise most of the time) and calling stations. Against them, limping a big hand to trap pays, so
  the networks learned to limp AA and AK.
- Real tournament players are neither, and a fresh exploiter beats these networks by +44 to +219 bb/100 partly
  for this reason. **A network is only as sound as the opponents it learns from.**

**How this shapes the work:**
1. **Rules that good players never break go into the game itself**, so no training time is spent learning them
   and no pool can teach them away. No open-limping is the first. Candidates checked the same way (measure
   first, then decide): no open-shoving deep stacks (over about 25bb) and no tiny raises facing a shove.
2. **Measure behaviour by hand**, not only overall results. The premium-hand check (first-in action for AA, KK,
   QQ, AK, AQ, KJ by position) found this leak in seconds. It joins the ICM push/fold check as a standing test
   for every tournament run.
3. **Train against realistic opponents** (phase C): clones of real full-ring tournament players in the pool,
   instead of maniacs and stations, so the network learns what wins against people.
4. **Sizes from practice:** a tournament menu with the opens real players use (2×, 2.5×, 3×), learned per
   position and stack.

