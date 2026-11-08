# Ideas from how LLMs are built, tested on poker — 2026-11-08

The question: do the model styles behind LLMs (attention, end-to-end learning from tokens, mixture of experts,
scaling up) help our poker bot? Which other LLM techniques are worth bringing into the blueprint?

## The test: predicting Slumbot's next action

Opponent modelling is where an attention model would plug into the blueprint: phase B node-locks the opponent's
predicted strategy into the re-solve. So the test is next-action prediction, which is how an LLM is trained.

- **Data:** 46,833 hands against Slumbot from all our runs. Slumbot always shows its cards in the API logs, so
  every one of its 129,022 decisions is labelled with the cards it held.
- **Target:** Slumbot's action class: fold, check/call, bets of ≤0.4, ≤0.8, ≤1.2 or more pot, all-in.
- **Inputs:** each decision is turned into a token sequence: Slumbot's hole cards, the board as it is dealt, every
  action with its size class, then a query token. There are also 12 hand-made features: street, hand strength
  against a random hand, price, pot, stack, raise count, position.
- **Split:** by hand. 70% train, 10% validation (early stopping), 20% test.
- **Score:** log-loss (lower is better), with a paired bootstrap over hands against the MLP.
- **Check:** the action-order parse was checked on all 267,254 actions: none was inconsistent with whose turn it
  was.
- **Scripts:** `scripts/llm_ideas_prep.py` and `scripts/llm_ideas_train.py`; results in
  `results/llm_ideas/models.json`. Run on the RTX PRO 6000 (CUDA device 1 on the lab; CUDA numbers GPUs fastest-first).

| Model | Style | Log-loss | vs MLP [95%] | Accuracy |
|---|---|---|---|---|
| Frequencies by street, facing a bet, raises | the generic model we use now | 0.510 | +0.392 [+0.385, +0.399] | — |
| Logistic regression on the features | linear | 0.523 | +0.405 | 83.9% |
| **MLP on the features** | **small feed-forward net** | **0.118** | **0** | **94.6%** |
| LSTM over tokens + features | pre-transformer sequence model | 0.141 | +0.023 [+0.018, +0.028] | 93.3% |
| Transformer over tokens + features (64 wide, 3 layers) | attention | 0.141 | +0.023 [+0.018, +0.028] | 93.4% |
| Transformer, mixture-of-experts layer (4 experts, top-1) | MoE | 0.144 | +0.026 [+0.022, +0.031] | 93.4% |
| Transformer at ¼ width | scaling | 0.144 | +0.026 | 93.3% |
| Transformer at 2× width, 4 layers | scaling | 0.139 | +0.021 [+0.017, +0.026] | 93.4% |
| **Transformer over tokens only, end to end** | **the pure LLM recipe** | **0.526** | **+0.408** | **75.8%** |

### What this shows

1. **Reading the cards is almost all of it.** Hand strength plus the situation predicts Slumbot's action 94.6% of
   the time. That cuts the log-loss by three quarters against the frequencies we use now. Slumbot plays a nearly
   pure strategy, so it is very predictable once its cards are known. This is a ceiling for opponents whose
   cards we know, not a forecast for hidden cards.
2. **Attention did not beat a small MLP.** With the same features, the transformer, the LSTM and the
   mixture-of-experts version were all slightly *worse*, with intervals excluding zero. The order of earlier
   actions adds nothing a few counts do not already capture. Slumbot's decisions depend on the current state, not
   on how the hand got there.
3. **The end-to-end LLM recipe fails at this scale.** From raw card and action tokens alone, the transformer did no
   better than the frequency table. With 90,000 examples it cannot learn hand strength from card identities.
   LLMs get there with billions of tokens; we have 10⁵ decisions per opponent. Hand evaluation must be given, not
   learned.
4. **Scaling helped a little** (¼ width 0.144 → 2× width 0.139), within the noise of the gap to the MLP. Mixture
   of experts added nothing.

**For the blueprint:** opponent models stay small feed-forward nets on hand-made features (hand strength,
price, position) for phase B. Use the MLP, not a transformer.

## Other LLM techniques, and where each fits

| LLM technique | Poker counterpart | Fit | Status |
|---|---|---|---|
| **Attention / transformers** | Sequence model of a hand or a session | Tested above: no gain for single-hand prediction. Untested: **in-context learning across hands**, a transformer reading an opponent's last N hands to adapt within a session. That is where attention could matter, against opponents who change (humans, not Slumbot) | Next, on Hand Sync / HandHQ data |
| **Next-token pre-training, then fine-tuning** | Pre-train one action model on all players, fine-tune per opponent (LoRA-style small adapters) | Good fit for phase B: a shared model plus a tiny per-opponent adapter learns from few hands | Planned with the opponent models |
| **Distillation** (big model → small model) | Train a small network to copy the blueprint's 1.1 GB table | Ships the blueprint inside the Windows advisor; also smooths over abstraction gaps (it sees real cards) | Planned with 8-bit storage |
| **Quantisation** (8-bit, 4-bit weights) | Store the average strategy in 8 bits | 4× smaller; already on the list as "compact storage" | Planned |
| **RLHF / PPO** | AlphaHoldem-style self-play RL | GPU track A on the roadmap | Planned |
| **KL-regularised RL** (RLHF's KL penalty to a reference model) | Regularised Nash dynamics (R-NaD, DeepNash for Stratego); magnetic mirror descent | Strong idea: training stays close to a reference strategy, which damps the cycling that the chaos section described. Fits the deep-RL track, or a reference-regularised re-solver | Added to the GPU track |
| **Mixture of experts** | Separate experts per street or per stack depth | Tested above: no gain for prediction. Our blueprint already splits by street | Dropped |
| **Scaling laws** | How LBR's edge falls with compute and with abstraction size | We measure it already: LBR per core-hour across runs 2–6 and the vector run | In progress |
| **Self-consistency / ensembles** (sample several answers, vote) | Average several blueprints or re-solves | Our ensemble study: averaging five networks cut exploitability by 21–26%, but reading the opponent and switching strategy beat every fixed mix | Done; worth trying on blueprint runs 4–6 |
| **Speculative decoding** (cheap model drafts, big model checks) | Blueprint answers instantly; the re-solver checks only close decisions | A good fit for live speed: re-solve only where the blueprint's top two actions are near in value | Planned with safe re-solving |
| **KV caching** (reuse earlier computation) | Cache subtree values between iterations and re-solves | The vector trainer and re-solver already reuse per-board work. Caching across decisions in a hand is the re-solver's warm start | Planned with safe re-solving |
| **Tokenisation choices** | How cards and bets are encoded | Bet sizes as classes worked; card identities as tokens did not work without a strength feature | Learned from the test |
| **Curriculum learning** | Train short stacks / later streets first, then extend | Akin to the multigrid start (coarse first); DeepStack trained river → turn → flop value nets in that order | Planned with value nets |

## Running it again

```
cd ~/llm-research
.venv/bin/python scripts/llm_ideas_prep.py results/llm_ideas/slumbot_actions.npz --workers 12
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv-gpu/bin/python scripts/llm_ideas_train.py results/llm_ideas/slumbot_actions.npz results/llm_ideas/models.json
```

## The small MLP on every other player (2026-11-08)

The same MLP, tested on the players in our self-play logs. Their cards are logged at every decision, so each
decision is labelled.

- **Data:** 1.09 million decisions from 171,529 hands; up to 40,000 decisions per player, whole hands at a time.
- **Features:** as above, extended to several players (players still in, position).
- **Models:** for each player, three, scored on the same held-out 20% of hands:
  - **freq:** the frequency table;
  - **pooled:** an MLP trained on every *other* player, the generic model that has never seen this one;
  - **own:** an MLP trained on this player alone.
- **Learning curve:** the own MLP on only 300, 1,000, 3,000 or 10,000 of the player's decisions.

Scripts: `scripts/opp_mlp_prep.py` and `scripts/opp_mlp_train.py`; results in `results/llm_ideas/players.json`.

Log-loss per decision; lower is better. "Own − pooled" has a 95% interval from a paired bootstrap over hands.

| Player | Decisions | Freq | Pooled | **Own** | Accuracy | Own − pooled | Own, 1,000 | Own, 3,000 |
|---|---|---|---|---|---|---|---|---|
| Monte (equity bot) | 62,386 | 0.597 | 0.526 | **0.117** | 95.3% | −0.41 [−0.42, −0.40] | 0.234 | 0.151 |
| Nora (nit) | 64,964 | 0.273 | 0.655 | **0.141** | 94.3% | −0.51 [−0.53, −0.50] | 0.245 | 0.181 |
| Sol (TexasSolver bot) | 59,384 | 0.609 | 0.513 | **0.194** | 92.8% | −0.32 [−0.33, −0.31] | 0.312 | 0.238 |
| Solver bot | 142,903 | 0.899 | 0.735 | **0.316** | 87.2% | −0.42 [−0.43, −0.41] | 0.526 | 0.445 |
| Tina+ | 40,000 | 0.633 | 0.864 | **0.335** | 86.1% | −0.53 [−0.56, −0.50] | 0.580 | 0.430 |
| Adaptive bot | 38,036 | 0.758 | 0.621 | **0.345** | 85.4% | −0.28 [−0.28, −0.27] | 0.478 | 0.385 |
| Big-Tiger-Gemma 27B (LLM) | 7,261 | 0.477 | 0.843 | **0.358** | 85.5% | −0.49 [−0.54, −0.43] | 0.436 | 0.367 |
| Tina | 45,558 | 0.657 | 0.721 | **0.392** | 85.2% | −0.33 [−0.34, −0.31] | 0.550 | 0.484 |
| Carl (calling station) | 82,472 | 0.732 | 1.174 | **0.468** | 77.2% | −0.71 [−0.73, −0.68] | 0.631 | 0.575 |
| Bob (recreational) | 62,494 | 0.769 | 0.863 | **0.492** | 78.5% | −0.37 [−0.39, −0.36] | 0.639 | 0.553 |
| Style-shifter 1 | 40,000 | 0.819 | 0.751 | **0.525** | 79.7% | −0.23 [−0.24, −0.21] | 0.743 | 0.606 |
| Hedge bot | 72,212 | 0.745 | 0.937 | **0.576** | 78.1% | −0.36 [−0.37, −0.35] | 0.708 | 0.629 |
| Random style 9000001 | 40,002 | 0.931 | 1.148 | **0.618** | 71.5% | −0.53 [−0.55, −0.51] | 0.783 | 0.678 |
| Max (maniac) | 79,473 | 1.066 | 1.729 | **0.646** | 68.2% | −1.08 [−1.12, −1.05] | 0.856 | 0.747 |
| Tiger-Gemma 9B (LLM) | 10,273 | 0.737 | 1.621 | **0.671** | 70.0% | −0.95 [−1.02, −0.88] | 0.753 | 0.662 |
| Qwen2.5-VL 32B (LLM) | 11,861 | 0.792 | 1.162 | **0.684** | 74.4% | −0.48 [−0.52, −0.44] | 0.755 | 0.695 |
| Style-shifter 3 | 51,781 | 1.085 | 1.317 | **0.777** | 61.9% | −0.54 [−0.56, −0.52] | 0.996 | 0.899 |
| **Blueprint** | 61,915 | 1.029 | 1.644 | **0.783** | 67.2% | −0.86 [−0.89, −0.83] | 0.930 | 0.854 |
| RL bot | 113,291 | 1.298 | 2.414 | **0.848** | 65.2% | −1.57 [−1.60, −1.53] | 1.050 | 0.972 |

### What this shows

1. **A player's own MLP wins for every one of the 19 players**, against both the frequency table and the generic
   model, with every interval far from zero.
2. **About 1,000 decisions (roughly 250–500 hands) beat the frequency table for 18 of 19 players.** The
   exception, Tiger-Gemma 9B, needs 3,000. Gains continue up to 10,000 decisions.
3. **The generic model is worse than plain frequencies for 14 of 19 players.** Players differ so much that a model
   of everyone else misreads any one of them. **Phase B must read each opponent from their own hands,** starting
   from frequencies, not from a pooled model.
4. **Predictability ranks the players.** Rule bots (Monte 95%, Nora 94%, Sol 93%) are near-deterministic.
   **The hardest to predict are the blueprint (67%), the RL bot (65%) and the style-shifter (62%).** These are the
   mixed-strategy and changing players. That is what an equilibrium strategy should look like: hard to read.
5. **LLMs read in between (70–86%).** Big-Tiger-Gemma 27B is fairly consistent; Tiger-Gemma 9B is the least
   readable LLM.

## An MLP that learns by playing itself, from scratch (2026-11-08)

`scripts/mlp_selfplay.py` trains one MLP from random weights, by self-play only.

- **The game:** the blueprint's simplified game: the same betting tree, the same 200 card buckets (`blueprint/out`),
  200bb heads-up.
- **The batch:** 65,536 hands at once on a GPU. Cards come from a C++ dealer (`bp_deal` in `blueprint/libbp3.so`).
- **Speed:** **about 300,000 hands a second on one GPU**, so a billion hands every hour (2.2 billion in the first
  two hours).
- **Network:** both seats share it. Inputs are the node (street, pot, price, stack, raises, the last action) and
  the card bucket (a learned embedding plus a strength value). Output: a probability for each action, plus a value
  head.
- **Learning:** an actor-critic with a KL pull toward a slowly moving copy of itself. This is regularised Nash
  dynamics (R-NaD, magnetic mirror descent), the same device as RLHF's pull toward a reference model.
- **Dealing:** half of every batch is fair random deals. The other half is spread evenly over the nine pairings of
  strong, medium and weak final hands (coolers, strong against weak, weak against weak). Each deal is weighted by
  p_fair / p_dealt, so rare spots are seen often while the learned strategy stays the one for fair dealing. Fair
  shares of the pairings: weak–weak 10.9%, strong–strong 3.4%.
- **MLP + blueprint:** the policy is exported in the blueprint's table layout (`avg.f32`, the same size as run 2's
  exactly). LBR, `blueprint_bot.py`, the Slumbot client and bot_sim play it with no new code.
- **Two measures every 30 minutes:**
  - LBR;
  - an exact head-to-head against run 2's table in the same game: 2 million hands, seats swapped on the same cards.

**First run** (`blueprint/runmlp`, on the 4090; KL 0.05, magnet rate 0.002):

| Time | Hands | vs run 2 (bb/100) | LBR's edge (bb/100) | Policy entropy |
|---|---|---|---|---|
| 30 min | 0.49 billion | −25.1 [−26.8, −23.4] | +165 [+77, +253] | 0.57 |
| 60 min | 1.04 billion | −33.9 [−36.3, −31.5] | +607 [+486, +728] | 0.46 |
| 90 min | 1.64 billion | −38.5 [−41.0, −35.9] | +567 [+441, +692] | 0.39 |
| 120 min | 2.25 billion | −40.0 [−42.4, −37.6] | running | 0.38 |

**What this shows:**

- **From nothing to within 25 bb/100 of the blueprint in 30 minutes.** The blueprint took 12 hours of 24 CPU
  threads.
- **Then it got worse.** The policy turns more deterministic (entropy falling), loses more to run 2, and LBR's edge
  jumps from +165 to about +600. This is the cycling of self-play in a zero-sum game: the KL pull was too weak to
  stop it.
- **Second run** (`blueprint/runmlp2`, on the PRO 6000):
  - six times the KL pull (0.3);
  - a magnet that moves seven times slower;
  - more entropy;
  - an exported average over ten times as many hands.

  It started at 12:28 lab time and is judged on the same two curves.
