---
name: model-judging
description: Decide honestly whether one poker model, rule or setting is better than another — new nmXX runs vs nm25, a Kestrel rule, a search or solver variant, a heads-up blueprint vs Slumbot, a coding or vision model vs another. Use it whenever a result will be used to adopt, ship, reject or claim something ("is nm45 better?", "does the rule help?", "adopt it?", "why did the check fail?"), when a number in a doc is quoted as evidence, or when a single run reads significant. Covers pre-registration, duplicate matches, seeds and controls, intervals, and what to write down.
---

# Judging models

A training change on a plateau moves results by tenths of a bb/100, which is the same size as the noise. Two 2026-11-23
examples:
- The early-shove rule's documented +0.21 and that day's −0.50 (z −2.66, a "significant" stop) were both single
  draws. Five replicates with a control read −0.11 against −0.04: no effect.
- nm25's exploiter reference of +0.2 came from one setup. Across four seeds it reads +0.75.

Every adoption limit set from those numbers was wrong. Judging is where these mistakes get made, so judge carefully.

## Before running

1. **Pre-register.** Write the run's `research/pipeline/runs/nmXX.json` (or a `results/prereg/` file) before it starts:
   - the question;
   - every check, with its pass rule;
   - what would make you reject it.

   Don't change a check after seeing a result. A re-test of a failed run is a new registration that says so.
2. **Name the property each check measures** (strength in its own game, robustness to an exploiter, play against
   other styles, a specific leak). A check only tests the property it names.
3. **Set limits from distributions, not single readings.** A limit like "exploiter ≤ reference + 1" needs the
   reference's mean and spread over several seeds (`results/exploit_baseline/`).

## Running the comparison

- **Duplicate dealing:** the same deals with the candidate in every seat (`duplicate`, `seqtest.py match`). It removes
  most card luck.
- **Seeds and a control:**
  - `seqtest.py` fixes the deals but not the table contexts or the action sampling, so one run is one draw.
  - For anything near a boundary, run 3-5 seeds, plus a control arm (the baseline against itself) on the same seeds.
    If the control swings as much as the effect, there is no effect.
- **Same conditions:** same GPU type and wall-clock budget for timed jobs, the same game (load sets the game from the
  last-loaded config), and the same opponents.
- **Score what matters for the game:**
  - In tournaments, report **prize equity** (ICM) and also chips. A bluffer can win chips and lose equity; then the
    defender's folds are correct.
  - Heads-up: bb/100 with AIVAT or luck adjustment, and an interval.
- **Enough hands:** state the interval. 5,000 heads-up hands against Slumbot gives about ±28 bb/100, which is not a
  result.

## Reading the result

- **Report means with intervals or spreads.** Report a run of seeds as the mean and range. Don't put a "significant"
  label on one sequential stop without replicates.
- **Correct for multiple comparisons.** Over many cells (leak maps, several checks), use Holm, and say which cells
  survive. A Holm p of 0.068 is not significant.
- **Agreement isn't strength:**
  - Matching a solver, a chart or a past model is a proxy. PokerBench's 80%-accurate model lost to GPT-4.
  - Before training toward a chart, show that the chart beats the model in play.
- **Check the reference's assumptions.** For example, a push/fold chart that allows at most one caller and uses flat
  bubble payouts is not the real game.
- **Rankings from short games without duplicate dealing flip between runs.** Treat them as anecdotes.

## Writing it down

- Save every result file that a doc quotes, under `research/results/`. A number with no saved result is not evidence;
  when you find one, rerun it and save it.
- Record the verdict and the reason in the run's `.result.json` and the topic doc. Note which result supersedes which.
- Include the failed and null results. Write "not adopted: equivalent at 30,000 deals" alongside the wins.
