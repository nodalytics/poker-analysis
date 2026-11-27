# Search with beliefs (2026-11-27)

`ninemax_belief_eval.py` on nm25: the same spots (after the flop, at most three players in, from nm25's self-play)
valued by each search with 256 rollouts, against a uniform search 64 times larger (16 x 1,024 guesses, 1,024
rollouts). The belief network (`b1/`, `ninemax_belief.py train`, 4 M deals) predicts each opponent's cards 0.73 bits
better than the uniform guess (9.32 against 10.05 bits held out).

| Search | Move-value error (bb) | Best move as the reference | Effective guesses (of 256) |
|---|---|---|---|
| v0 uniform (the old search) | 1.39 | 65.0% | 30.5 |
| v1 uniform, 8 x oversampled and resampled | 0.66 | 75.2% | 250.5 |
| v2 belief, no oversampling | 0.90 | 71.4% | 57.2 |
| **v2 belief, 8 x oversampled and resampled** | **0.48** | **80.2%** | 256 |

363 spots (`eval_b1_400.json`); a first run of 57 spots (`eval_b1.json`) ranked them the same way. Oversampling is
cheap (a guess costs a replay of the hand, its rollouts cost far more) and is now Kestrel's default
(`SearchConfig::oversample` 8). The belief proposal cuts the error by another 27% and is not in Kestrel yet: it needs
the belief network in C++.
