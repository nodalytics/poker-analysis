# Neural models for the blueprint: scale, architectures and learning rules (2026-11-09)

Questions the user asked on 2026-11-09, the answers given, and the experiments started to settle them. Results are
filled in below as the runs finish.

## The questions, answered

**Can the from-scratch MLP play trillions of hands?** Yes, but not usefully yet.
- **Time:** one GPU plays about 300,000–560,000 hands a second, so a trillion hands is about 3 weeks on one GPU, or
  8–10 days on all three.
- **Volume isn't the limit:** `runmlp2` played 24 billion hands in 22 hours, and its LBR stopped improving after
  about 2 hours (+104 at 2 hours, +140 to +250 since).
- **The network is:** a copy of run 2's table into the same network (`runmlp3`) settles at LBR +140 to +165,
  against the table's own +104.

So the model has to improve before trillions of hands can pay off.

**Did the blueprint train on trillions of hands?** The newest runs do, in hand pairs:

| Run | Work in 24 hours |
|---|---|
| Runs 2, 4, 5, 6 (external-sampling CFR) | 12–23 billion iterations, each a full tree pass for one dealt hand per player |
| runvec (vector CFR, 8 threads) | 311,000 sampled boards × about a million hand pairs each ≈ **0.3 trillion hand pairs** |
| Run 7 (vector CFR, 40 threads, 1.6× faster per core) | ≈ **2.7 trillion holding-pair evaluations** by 2026-11-10 10:35 lab time |

**What the trillions count.** One vector iteration samples a full board. On it each player can hold C(47,2) = 1,081
hands, which makes 1,081 × 990 ≈ 1.07 million non-overlapping holding pairs. Run 7 does 28.9 boards a second, so
28.9 × 86,400 × 1.07 M ≈ 2.7 trillion in 24 hours. These are **evaluations, not hands played**: every pair's values
are computed in one sweep of the betting tree, and pairs the strategy never reaches are skipped. "Plays trillions of
hands" overstated it; the MLP's 27 billion are hands actually played, a different unit.

**These experiments are heads-up.** The product target is 9-max (`2026-11-09-nine-max-plan.md`). Which network and
which learning rule work best carries over to a 9-max neural trainer; the table-based runs (run 7) do not.

**Is there a better model than the MLP?** Probably: the current one is small. It has 3 layers of 256 units and
177,000 weights, standing in for a table of 55 million probabilities. Copying the table, it stalls at a
cross-entropy of 0.44 nats, against the table's own entropy of 0.37. The user asked for the candidates below to be
tested: larger networks, sequence readers (transformer, RNNs, CNNs), other network types, other learning rules
including Q-learning, and Deep CFR.

## What every network sees

In the simplified game, a decision is a (betting node, card bucket) pair. The default network gets the node's
features and an embedding of the bucket:
- the node features: street, seat, pot, price, stack depth, raises this street, and the last action;
- the bucket embedding: one table per street.

Two nodes with the same features but different histories therefore look identical to it. Several variants below
add the node's identity or its full betting history (up to 16 actions, each tagged with its street) to test whether
that is what limits it.

## Experiment 1: architecture bake-off (`scripts/arch_bakeoff.sh`)

Each network copies run 2's table from scratch for 30 minutes by distillation (`mlp_selfplay.py --distill
blueprint/run2 --minutes 0`). It is then measured three ways:
- **cross-entropy** to the table;
- **head-to-head** against the table in the simplified game (2 million hands, exact, no bet mapping);
- **LBR** in the real game (3,000 hands).

| `--arch` | Network | Weights | Seconds per training batch* |
|---|---|---|---|
| `mlp` | Current: 3 × 256 GELU layers | 0.18 M | 0.06 |
| `wide` | 3 × 1024 GELU layers | 2.3 M | 0.19 |
| `res` | Residual: 6 pre-norm blocks of 512 | 3.3 M | 0.19 |
| `nodeemb` | Residual + a learned vector per betting node (the table's memory, shared across buckets) | 21 M | 0.14 |
| `history` | Residual + a **transformer** (attention) over the betting history, 3 layers, 4 heads | 3.9 M | 0.19 |
| `rnn` | Residual + an Elman **RNN** over the history (2 layers) | 3.4 M | 0.17 |
| `lstm` | Residual + an **LSTM** over the history | 3.6 M | 0.16 |
| `gru` | Residual + a **GRU** over the history | 3.5 M | 0.16 |
| `tcn` | Residual + a temporal convolution network over the history (dilations 1, 2, 4) | 3.5 M | 0.17 |
| `cnn` | Residual + a text-**CNN** over the history (widths 2, 3, 4, max-pooled; Kim 2014) | about 3.4 M | — |
| `swiglu` | Residual with the gated SwiGLU blocks of current LLMs | 5.4 M | 0.25 |
| `moe` | Residual with mixture-of-experts blocks (4 experts, softmax gate) | 2.2 M | 0.15 |
| `kan` | Kolmogorov–Arnold network (FastKAN: learned curves per input) | 0.77 M | 0.07 |
| `ftt` | **Attention across the inputs** (FT-Transformer): each feature and the cards become tokens | 0.18 M | 1.0 (8 chunks) |
| `joint` | **One attention sequence** per decision: cards, features and the history attend to each other | 0.19 M | 0.58 (4 chunks) |

\* One batch of about 200,000 decisions on the RTX PRO 6000, from the smoke test. The networks get the same 30
minutes, so slower ones see fewer hands. That is part of what is measured.

**About the CNN.** A 2-D CNN over the cards, as in AlphaHoldem, needs the cards themselves. The simplified game
gives the network only the card bucket, so the convolutional networks read the betting history instead (`tcn`,
`cnn`). A card-image CNN belongs with the real-game models (the human clones and the opponent models) and has not
been tested.

**Queues:**
- **3090** (after `runmlp3` ends): `mlp res history rnn gru kan cnn ftt`
- **4090** (after its half of the trainability map): `wide nodeemb lstm swiglu moe tcn joint`

Results: `blueprint/bakeoff/<arch>/` on the lab.

### Results

11 of 15 networks, 30 minutes of copying each. Head-to-head: 2 million hands in the simplified game. LBR: 3,000
real-game hands, luck-adjusted (higher means **more** exploitable; run 2's own table reads about +104).

| Network | Cross-entropy (table: 0.368) | vs the table, bb/100 | LBR, bb/100 [95%] |
|---|---|---|---|
| `mlp` (current) | 0.437 | −8.1 | **+111** [+38, +185] |
| `wide` | 0.430 | −6.1 | **+102** [+22, +182] |
| `swiglu` | 0.430 | −7.2 | **+105** [+37, +173] |
| `moe` | 0.433 | −7.2 | +141 [+64, +217] |
| `kan` | 0.437 | −9.1 | +147 [+70, +224] |
| `lstm` | **0.383** | −2.0 | +224 [+122, +327] |
| `nodeemb` | **0.383** | −3.4 | +254 [+159, +348] |
| `res` | 0.437 | −7.2 | +257 [+164, +351] |
| `gru` | 0.388 | −2.7 | +355 [+243, +468] |
| `history` (transformer) | 0.388 | −2.1 | +410 [+288, +531] |
| `rnn` | 0.389 | **−0.4** | +437 [+303, +571] |
| `tcn` | **0.381** | −1.4 | +474 [+345, +603] |
| `cnn` (text-CNN) | 0.386 | −1.4 | +625 [+477, +772] |
| `ftt` (attention across inputs; only 27 M hands in 30 min) | 0.603 | −34.3 | +400 [+267, +533] |

**The two measures disagree, and that is the finding.** Networks that see the betting history or the node's
identity copy the table about four times more closely on its own path (cross-entropy gap 0.015–0.02 against
0.06–0.07, head-to-head −0.4 to −3.4 against −6 to −9). But LBR plays lines the table rarely reaches, and there
they are two to four times more exploitable than the plain networks. The plain networks match the table's own
LBR. A likely reading: with the history, a network can memorise each well-visited line and has nothing to go on
in rarely visited ones; without it, it must share what it learns between similar spots, which generalises.

Consequence for the 9-max network: **do not swap in a history reader on the strength of copying accuracy.** Opponents
at a full ring leave the trained lines constantly. A history reader must be judged by an exploitation test (LBR,
or the population and real-table evaluations), and regularised (dropout on the history, or history as an
auxiliary input the network can be trained without) before it replaces the feature-only network.

The three later history readers (`tcn`, `cnn`) repeat the pattern most sharply: the closest copies, the most
exploitable. `joint` ran out of memory on the 4090 at 4 chunks and was restarted on the 3090 at 16 on 2026-11-10.

## Experiment 2: learning rules (`scripts/learner_variants.sh`, `scripts/deep_cfr.py`)

These runs are from scratch by self-play, 30 minutes each. All use the default network and the strong magnet of
`runmlp2` (KL 0.3, magnet step 3·10⁻⁴, entropy 0.01). Each run changes one thing:

| Name | Change | Kind |
|---|---|---|
| `a2c` | None: REINFORCE with a value baseline (the current trainer) | Policy gradient |
| `ppo` | Clipped probability ratio, 4 passes per batch (Schulman et al. 2017) | Policy gradient |
| `neurd` | Neural Replicator Dynamics: the update acts on the centred logits, with thresholds (Hennes et al. 2020; used in R-NaD / DeepNash) | Policy gradient |
| `qpg` | All-actions policy gradient with a learned Q critic (Srinivasan et al. 2018) | Actor-critic |
| `qlearn` | **Q-learning**: the network outputs action values. The target is the best value at the same player's next decision, or the hand's payoff at the end. Play is the softmax over the values (temperature 1 bb) | Value-based |
| `qmc` | **Monte Carlo Q**: as `qlearn`, but the target is the hand's result | Value-based |
| `duplicate` | **Reward:** minus the result of the mirrored deal (same cards, seats swapped). Removes card luck; the baseline does not depend on the game's own actions, so the gradient stays unbiased | Reward |
| `tanh` | **Reward:** 10 · tanh(result / 10 bb). A bounded, risk-averse utility, no longer the chip-EV objective; expected to lose chips | Reward |
| `league` | 30% of each batch against one of the last 10 snapshots of itself, one every 3 minutes (fictitious play, as in AlphaStar's league) | Opponent mechanism |
| `deepcfr` | **Deep CFR**, for 120 minutes. A regret network replaces the regret table and an average network replaces the average table. Regrets come from one rollout per action, from decisions sampled with 10% exploration; training is weighted by iteration (linear CFR) | Regret-based |

Smoke tests (25–90 seconds each, against run 2's table) all ran without error. These numbers are too early to
rank anything. Deep CFR's current strategy reached −38 bb/100 in 90 seconds, which is promising.

Q-learning has no convergence guarantee in imperfect-information games. Its best-response target assumes the
opponent's play is fixed, and a deterministic policy is exploitable. It is included as a baseline and is expected
to do worse than the regularised policy methods.

**Queues:**
- **RTX PRO 6000** (after its half of the trainability map): `a2c ppo neurd qpg qlearn qmc duplicate tanh league`
- **3090** (after its bake-off queue): `deepcfr`

Results: `blueprint/learners/<name>/` on the lab.

### Results

Run on 2026-11-10 after the lab reset (4090; Deep CFR on the 3090). Head-to-head against run 2's table at 10 / 20
/ 30 minutes, then LBR on the final strategy (3,000 hands, luck-adjusted; run 2's table reads about +104):

| Rule | vs run 2 at 10 / 20 / 30 min | LBR, bb/100 [95%] |
|---|---|---|
| `a2c` (current) | −170 / −87 / −56 | +147 [+37, +257] |
| `ppo` | −163 / −102 / −71 | +97 [−17, +211] |
| **`neurd`** | −89 / −56 / −48 | **+0.4** [−101, +102] |
| `qpg` | −252 / −247 / −249 | +1,062 [+818, +1,307] (did not learn) |
| `qlearn` | −84 / −41 / −44 | +279 [+172, +385] |
| `qmc` | −59 / −47 / **−38** | +276 [+171, +381] |
| `duplicate` reward | −124 / −62 / −43 | +108 [+7, +210] |
| `tanh` reward | −329 at 10 min | (running) |
| `league` | (queued) | |
| **`deepcfr`** (120 min) | −53 at 10 min, −47 at 120 min | **+32** [−78, +141] |

- **NeuRD is the best policy rule:** faster than A2C at every point and the least exploitable of all, at the
  table's level or better (the interval is wide).
- **Deep CFR** is about as unexploitable, but it reached −47 within 10 minutes and then stopped improving for 110.
- **Q-learning** wins head-to-head (`qmc` is closest to the table at 30 minutes) but is twice as exploitable as the
  policy methods, as expected: a best-response target produces near-deterministic, readable play.
- **The duplicate reward** helps A2C a little on both measures (−43 vs −56; +108 vs +147).
- **QPG** did not learn at all with these settings; PPO is slower than A2C and somewhat less exploitable.

Consequence for 9-max: NeuRD replaces A2C in the next 9-max run (`nm4`, `--pg neurd`).

## Experiment 3: the trainability map (`scripts/trainability_map.sh`)

From the fractal note (`2026-11-09-fractal-ideas.md`, section 4): a grid of 36 runs from scratch, 8 minutes each.
The axes are learning rate {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2} × KL pull {0, 0.03, 0.1, 0.3, 1, 3}, each scored
against run 2 at 4 and 8 minutes. The low learning rates run on the 4090 and the high ones on the PRO 6000, so
each cell also records its hand count.

First cell, learning rate 1e-4 with no KL: −66.7 at 4 minutes, then −33.2 [−35.3, −31.2] at 8 minutes, over 250
million hands.

### Results

All 36 cells, bb/100 against run 2's table after 8 minutes (`blueprint/tmap/`; 95% intervals are about ±3–7):

| Learning rate ↓ / KL pull → | 0 | 0.03 | 0.1 | 0.3 | 1 | 3 |
|---|---|---|---|---|---|---|
| 1e-4 | **−33** | −72 | −134 | −204 | −247 | −255 |
| 3e-4 | −50 | −59 | −128 | −189 | −237 | −252 |
| 1e-3 | −191 (collapsed) | −99 | −144 | −145 | −234 | −246 |
| 3e-3 | −343 (collapsed) | −247 (collapsed) | −167 | −239 | −217 | −226 |
| 1e-2 | −224 (collapsed) | −531 (collapsed) | −141 | −183 | −277 | −246 |
| 3e-2 | −121 (collapsed) | −463 (collapsed) | −110 | −238 | −289 | −279 |

"Collapsed": the policy's entropy reached about 0, a deterministic strategy.

- **The KL pull slows early learning a lot.** At the two low learning rates every step up in pull costs 25–70
  bb/100 at 8 minutes. The setting all long runs use (KL 0.3, including `runmlp2` and the 9-max `nm2`) reads −189
  here against −33 with no pull. The pull is what kept `runmlp2` stable over 22 hours, so this is a short-run cost,
  not a verdict.
- **Without a pull, a learning rate of 1e-3 or more collapses** to a deterministic strategy. A small pull (0.1)
  prevents the collapse at every learning rate up to 3e-2.
- **The edge is ragged.** Above 1e-3 the results jump around without order (−110 at 3e-2 and KL 0.1, −531 at 1e-2
  and KL 0.03): the boundary between trainable and collapsing settings is irregular, as the fractal note
  predicted. Single 8-minute runs cannot say more about its shape.

**What this suggests for learning faster:** start with a weak pull at a low learning rate (1e-4 to 3e-4, KL 0 to
0.03) for speed, then raise the pull once progress against the previous checkpoint flattens, to stay stable. Not
yet tested over a long run.
