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

363 spots (`eval_b1_400.json`); a first run of 57 spots (`eval_b1.json`) ranked them the same way.

**A better belief network** (`b2/`, 40 M deals: 0.98 bits an opponent against b1's 0.73), the same 363 spots
(`eval_b2_400.json`; the uniform arms repeat exactly): belief x1 rests on 76.9 effective guesses against 57.2 but its
error is unchanged (0.899 bb against 0.900); belief x8's error falls from 0.483 to **0.456 bb** (agreement 79.3%
against 80.2%, within the noise of 363 spots).

**In Kestrel it is not cheap** (2026-11-27). Kestrel's search, nm25, 1,024 samples (`kestrel_search_match`, seed 7):
without oversampling +3.08 bb/100 [−1.22, +7.37] over 1,500 deals at 2.9 s a decision, with 447 of 1,024 effective deals
on average; with 8 x the first 200 deals took 2,306 s against 331 s, about 7 times slower (about 20 s a decision):
dealing, bucketing and weighting 8,192 deals costs more than the rollouts there. The match was stopped and
`SearchConfig::oversample` set back to 1 (off). Timed on 100 deals (seed 7, 1,024 samples, 16 threads):

| Oversampling | Time a decision | Effective deals (of 1,024) | Searches that fell back to the network |
|---|---|---|---|
| 1 (off) | 2.74 s | 432 | 67 of 914 |
| 2 | 3.32 s (+21%) | 600 | 38 of 888 |
| 4 | 5.31 s (+94%) | 726 | 15 of 895 |
| 8 | about 20 s (stopped) | | |

2 x is affordable (+0.6 s) for 39% more effective deals and half the fallbacks; whether it wins chips needs a paired
match (100 deals cannot separate the gains, +13 to +15 bb/100 with intervals of about ±15).

**Paired, 1,500 deals** (seed 11, both runs on the same deals, `kestrel_oversample/`: the summaries and each deal's
result without and with search; the plain arm is identical in both): search's gain over plain nm25 is +3.74
[−0.29, +7.77] bb/100 with oversampling off and +5.08 [+1.08, +9.09] with 2 x; **2 x minus off, deal by deal: +1.34
[−1.65, +4.33]**, not shown. 2 x took 4.9 s a decision against 3.9 s (the lab busy), rested on 611 effective deals
against 445, and fell back to the network 440 times against 815. The default stays off; about 4,000 more paired deals
would narrow the difference to about ±1.8.

**4,000 more paired deals** (seed 12, `kestrel_oversample/*_s12*`): 2 x minus off +1.41 [−0.45, +3.28]. **Pooled over
5,500 deals: +1.39 [−0.19, +2.97]**, still not shown, though both seeds read +1.3 to +1.4. 2 x took 4.29 s a decision
against 3.64 (+18%), rested on 614 effective deals against 445 and fell back to the network 1,071 times against 2,139.
The default stays off: a likely small gain, not established. The belief proposal cuts the error by another 27% and is not in Kestrel yet: it needs
the belief network in C++.
