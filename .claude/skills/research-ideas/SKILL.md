---
name: research-ideas
description: Research and propose techniques — cutting-edge or decades old, from poker AI, game theory, RL, statistics or other fields — to solve a concrete problem in this project, and turn the best into testable proposals with a cheap first experiment. Use it whenever the user asks "how could we fix/improve X", "what does the literature say", "any new methods for...", "what else could we try", reviews papers or repos, hits a plateau, or wants ideas for training, evaluation, search, exploitation, variance reduction or Kestrel features — even if they don't say "research".
---

# Finding techniques that solve our problems

Good ideas here are usually not new to the field: AIVAT, PCFR+, safe re-solving, league training and Malmuth-Harville
ICM are all years old. They tend to fail when they were already tried in this repo, or when they can't be tested
cheaply. So the job is to match the problem to methods (old or new), check them against what we've already measured,
and hand back proposals that can be judged.

## 1. Pin down the problem

- **State the problem as a measurement,** for example "nm25's exploiter reads +0.75 bb/100 mean over 4 seeds", "the
  6-max cash hole is +5.6", or "search loses 96.5 bb/100 to calling bots". If it can't be stated as one, the first
  proposal is the measurement.
- **Name the mechanism you suspect** and the property a fix must change. Fixes aimed at the wrong property look like
  failures.

## 2. Read what we already know (before any search)

- **`research/docs/`:**
  - `2026-11-17-upgrades.md`, the plateau and seven upgrades with their results;
  - `2026-11-23-review-and-plan.md`, the current plan;
  - `2026-11-17-methods-from-other-fields.md`, `2026-11-17-model-mathematics.md` and
    `2026-11-15-what-the-models-learn.md`;
  - the literature notes (`2026-11-23-llm-poker-literature.md`).
- **`research/results/prereg/*.result.json`:** what was tried, how it was judged and the verdict.
- **`research/blueprint/model.md`:** the heads-up solvers and re-solving, with measurements.

Mark each idea you find as untried, tried (with its verdict, citing the file), or adjacent to something tried. Don't
re-propose a tried idea unless you can say what's different this time.

## 3. Search widely, read the actual sources

- **Search both directions:**
  - Recent work: arXiv (cs.GT, cs.LG, cs.AI), NeurIPS, ICML, ICLR, AAAI, the AAAI/ACPC poker workshop, and lab
    papers (DeepMind, Meta FAIR, CMU, Alberta).
  - Classic work: CFR and its variants, opponent modelling, importance sampling and control variates, sequential
    testing, curriculum and population methods, ICM models.
  - Neighbouring fields: other imperfect-information games (Stratego, Diplomacy, bridge, mahjong), finance, auction
    theory and causal inference.
- **Read the paper or the code, not the abstract or your memory.** Record for each:
  - the setting (heads-up or multiway, cash or tournament, stack depth);
  - the baseline and opponents;
  - the numbers, with sample size and interval;
  - what was ablated;
  - whether code exists and under what licence.

  Many methods are proved only heads-up, or measured against weak baselines or without intervals. Say so.
- Use web search and fetch. Delegate several fetches in parallel when there are many sources. A source that can't be
  read (paywall, 403, a browser check) is reported as unread, never summarised from memory. Don't bypass paywalls.

## 4. Turn ideas into proposals

For each idea worth keeping, write:

| Field | Content |
|---|---|
| Idea | one sentence, with its source (paper, year, link) or "classic: <name>" |
| Mechanism | why it should move *our* measured problem |
| Evidence | the source's result, and how far its setting is from ours |
| Status here | untried, or tried in <file> with its verdict |
| Cheapest test | the smallest experiment that could falsify it with our tools (seqtest, exploiter, leak map, aggressor check, resolve --test, LBR), with hands, seeds and GPU-hours |
| Pre-registration sketch | the question, checks and pass rules (model-judging) |
| Cost and risk | engineering days, compute, and what it could break (Kestrel parity, inputs Kestrel lacks) |

Rank the proposals by expected gain on the measured problem, divided by the cost of the cheapest test. Include at
least one old and one recent technique when both apply, and say plainly when the honest answer is "measure first" or
"nothing here is likely to help".

## 5. Deliver

- Write the result to `research/ideas/YYYY-MM-DD-<topic>.md`, with a sources table and the ranked proposals. Add a
  row to `research/ideas/README.md` (date, topic, top proposal, status), and commit both (publish-safety). Experiment
  write-ups stay in `research/docs/`. When a proposal is run, link its result from the idea's row and mark it tested.
- Tell the user the top two or three proposals, each with its cheapest test. Then offer to register and run the first
  one.
- Nothing from this skill changes what Kestrel plays until it passes a registered test.
- Real-money and real-time-assistance rules are never in scope as ideas.
