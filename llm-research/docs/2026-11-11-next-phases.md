# 9-max: the next phases (2026-11-11)

Where the 9-max work stands, and the plan from here. The detailed results are in
[`2026-11-09-nine-max-plan.md`](2026-11-09-nine-max-plan.md); this page is the roadmap.

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
2. **`nm11`: cash at random 20–50bb stacks** (started 2026-11-11 13:36 lab time, 4090, 24 hours).
   - Real cash tables rarely sit at exactly 100bb, so every seat's stack is drawn between 20 and 50bb
     (log-uniform, `--stacks 20,50`).
   - Otherwise it uses `nm9`'s recipe: NeuRD, exploiters on 25% of tables, and a snapshot every 4 hours.
   - **Done means:** it is as hard to exploit as `nm9` at these depths, and it beats the bots at bot_sim tables
     dealt 20–50bb.
3. **`nm10`: a tournament curriculum** (restarted 2026-11-11 15:00 lab time at 10–25bb, RTX PRO 6000, 24 hours).
   - Most mid- and late-tournament decisions are made at 10–25bb, so three tables in four average **10–25bb** and
     one in four **5–10bb** (pure push/fold), via `--depths "10,25;10,25;10,25;5,10"`.
   - Stacks spread around each table's average, as at a real table: the median table averages 12.9bb (5th to 95th
     percentile 5.2–26.5bb). By seat, 49% are under 10bb, 38% are 10–25bb and 13% are over 25bb.
   - The first version (1.5 hours at 5–15bb and 20–50bb tables) is kept as `nm10a`. It left a thin spot at
     15–20bb.
   - Recipe: NeuRD, exploiters on 25% of tables, a pull ramped from 0.1, entropy bonus 0.02, a snapshot every 4
     hours. Bubble tables are top-heavy above the bust spots.
   - Still to add: antes, and short-handed final tables (2–8 players). This needs the engine and `icm.scenarios`
     extended.
   - **Done means:** at least 85% agreement with the ICM charts at final tables and real bubbles (`nm8`: 74–81%),
     and the big blind's calling near the chart's.
4. **Pick the champions:** one cash network and one tournament network, chosen by exploitability first and real
   tables second. A 100bb-only exploit test for `nm6` makes the mixed-stack run comparable with the others.

## Phase B: search at play time (step 7 of the build order, about a week, CPUs)

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
| 2026-11-11 | `nm10` restarted at 10–25bb tables (3 in 4) and 5–10bb (1 in 4); tournament stacks are mostly 10–25bb. Cash stays at 20–50bb |
| 2026-11-11 | Advisor screenshots are capture-only and pause while a real-money client runs; no live table reading |

| Pending | Decision |
|---|---|
| Antes and short-handed final tables | Next tournament run after `nm10` |
