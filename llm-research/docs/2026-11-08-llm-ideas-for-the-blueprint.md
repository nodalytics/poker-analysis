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
  `results/llm_ideas/models.json`. Run on the RTX 4090.

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
CUDA_VISIBLE_DEVICES=1 .venv-gpu/bin/python scripts/llm_ideas_train.py results/llm_ideas/slumbot_actions.npz results/llm_ideas/models.json
```
