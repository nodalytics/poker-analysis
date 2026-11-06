# Partner pack — 2026-11-06

What we built and what it showed, one picture each. Charts are made by `scripts/partner_charts.py` from the
committed results; every chart names its source.

## Videos

| Clip | What it shows |
|---|---|
| [`../2026-11-05-solver-demo.mp4`](../2026-11-05-solver-demo.mp4) (10 s) | TexasSolver running live on the lab and getting a textbook game exactly right: bet the nuts 100%, bluff 9.9% (theory 9.4%), call 50.0% |
| [`09-solver-plays-short-stacks.gif`](09-solver-plays-short-stacks.gif) (18 s) | The newest setup at 5-20 big blinds: the solver acts itself (seats marked **S**) wherever it has an answer, the LLM decides only the rest. **P** = the same model playing alone. The three biggest pots of one session |
| [`../2026-11-05-arena-poker-round8-table1.gif`](../2026-11-05-arena-poker-round8-table1.gif) | The models' poker arena: learners, frozen twins, simulated humans, Monte Carlo and solver bots at one table |

## Pictures

| # | Milestone | One line |
|---|---|---|
| 01 | [Push/fold equilibrium](01-push-fold-10bb.png) | Our computed short-stack chart matches the published one (59% shove / 38% call at 10bb) |
| 02 | [Faster solver](02-solver-speed.png) | Same answer, at least 3.3x faster on the turn; a flop solve in 41 s on 16 threads |
| 03 | [Flop model](03-flop-model-accuracy.png) | Trained on 400 solver runs, it matches the solver's top action 81-83% on unseen flops, in milliseconds |
| 04 | [Solver advice tests](04-advice-tests.png) | Advising the LLM helped in the first run, but not reliably |
| 05 | [Why: models ignore advice](05-models-ignore-advice.png) | Told "SHOVE", most models still don't, so the solver now acts directly |
| 06 | [Tina+](06-tina-plus.png) | Fixing the simulated human's weaknesses: 1.6x the prize money over 60 tournaments |
| 07 | [Trading arena](07-trading.png) | The paper's regime gate adds nothing after costs; the model's own judgement does |
| 08 | [Self-learning ladder](08-self-learning-ladder.png) | Models that rewrite their own skill notes: one clear learner after 58 rounds |

Still running: the solver-acts-itself test at 100bb and at 5-20bb (30 sessions each); results follow.
