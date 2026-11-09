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

Pending.

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

Pending.

## Experiment 3: the trainability map (`scripts/trainability_map.sh`)

From the fractal note (`2026-11-09-fractal-ideas.md`, section 4): a grid of 36 runs from scratch, 8 minutes each.
The axes are learning rate {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2} × KL pull {0, 0.03, 0.1, 0.3, 1, 3}, each scored
against run 2 at 4 and 8 minutes. The low learning rates run on the 4090 and the high ones on the PRO 6000, so
each cell also records its hand count.

First cell, learning rate 1e-4 with no KL: −66.7 at 4 minutes, then −33.2 [−35.3, −31.2] at 8 minutes, over 250
million hands.

### Results

Pending.
