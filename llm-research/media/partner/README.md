# Partner pack — 2026-11-06

What we built and what it showed, one picture each. Charts are made by `scripts/partner_charts.py` from the
committed results; every chart names its source.

## Videos

| Clip | What it shows |
|---|---|
| [`../2026-11-05-solver-demo.mp4`](../2026-11-05-solver-demo.mp4) (10 s) | TexasSolver running live on the lab and getting a textbook game exactly right: bet the nuts 100%, bluff 9.9% (theory 9.4%), call 50.0% |
| [`09-solver-plays-short-stacks.gif`](09-solver-plays-short-stacks.gif) (18 s) | The newest setup at 5-20 big blinds: the solver acts itself (seats marked **S**) wherever it has an answer, the LLM decides only the rest. **P** = the same model playing alone. The three biggest pots of one session |
| [`../2026-11-05-arena-poker-round8-table1.gif`](../2026-11-05-arena-poker-round8-table1.gif) | The models' poker arena: learners, frozen twins, simulated humans, Monte Carlo and solver bots at one table |
| [`14-robust-exploitation.mp4`](14-robust-exploitation.mp4) (14 s) | Win from weak players and stay hard to beat: at p 0.25, 84% of the maximum profit at 7% of the exposure (Leduc, exact) |
| [`15-adaptive-opponent.mp4`](15-adaptive-opponent.mp4) (14 s) | A learning opponent punishes pure exploitation within ~3,000 hands; robust strategies leak slowly; the equilibrium is never beaten |
| [`16-ensembles.mp4`](16-ensembles.mp4) (13 s) | Combining strategies against unknown opponents: reading the opponent (Bayes) earns most and never loses |
| [`17-slumbot.mp4`](17-slumbot.mp4) (19 s) | The outside test against Slumbot: four runs each caught a real bug; the fifth, reading Slumbot with a model of its own play, wins +57.5 bb/100 over 3,000 hands (95% +2 to +114) |

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
| 10 | [AlphaPoker](10-alphapoker.png) | AlphaGo search and plain self-play RL stay beatable; averaging self-play (CFR) converges to 0.3 |
| 11 | [Player styles (PCA)](11-pca-styles.png) | Two components explain 80% of how 2,624 real players differ; style does not mark winners |
| 12 | [Predicting real players](12-predict-next-action.png) | About 69% of next actions predicted (63% baseline); the ESN reaches 67% from the action sequence alone |
| 13 | [Value network](13-value-network.png) | Every hand's turn value to within 2.5% of the pot in milliseconds (a solve takes 5.5 s; raw equity errs by 17.9%) |

Latest, the solver deciding every street: short stacks ahead in 56 of 88 sessions (p 0.007); 100bb, with the solver also playing pre-flop, ahead in 55 of 87 (p 0.009).
