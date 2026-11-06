# Ladder after round 58

## Learner vs its own frozen twin (the paired test)

| Model | Chess head-to-head (learner points / games) | p (learner better) | Poker sessions learner ahead / played | p |
|---|---|---|---|---|
| Big-Tiger-Gemma-27B-v3 | 23.5 / 50 | 0.66 | 31 / 50 | 0.06 |
| Tiger-Gemma-9B-v3 | 25.5 / 50 | 0.44 | 41 / 48 | 0.00 |
| qwen2.5vl | 28.0 / 50 | 0.24 | 30 / 50 | 0.10 |
| wizard-vicuna-uncensored | 25.0 / 50 | 0.56 | 27 / 50 | 0.34 |

p is one-sided (learner better than its twin); chess draws count half, poker ties are left out. Power: see scripts/power_sim.py.

## Chess

| Rank | Entrant | Elo |
|---|---|---|
| 1 | qwen2.5vl#learn | 1072 |
| 2 | Big-Tiger-Gemma-27B-v3#frozen | 1057 |
| 3 | qwen2.5vl#frozen | 1056 |
| 4 | Big-Tiger-Gemma-27B-v3#learn | 1005 |
| 5 | wizard-vicuna-uncensored#frozen | 1002 |
| 6 | Tiger-Gemma-9B-v3#frozen | 954 |
| 7 | Tiger-Gemma-9B-v3#learn | 948 |
| 8 | wizard-vicuna-uncensored#learn | 906 |

| Model | Learner − frozen twin |
|---|---|
| Big-Tiger-Gemma-27B-v3 | -52 |
| Tiger-Gemma-9B-v3 | -6 |
| qwen2.5vl | +16 |
| wizard-vicuna-uncensored | -96 |

## Poker

| Rank | Entrant | Elo |
|---|---|---|
| 1 | sol (TexasSolver heads-up turn/river, else Monte Carlo) | 1047 |
| 2 | tina (tight-aggressive, simulated human) | 1033 |
| 3 | Big-Tiger-Gemma-27B-v3#learn | 1022 |
| 4 | monte (Monte Carlo equity (eval7, poker-analysis toolkit)) | 1016 |
| 5 | bob (recreational, simulated human) | 1015 |
| 6 | Tiger-Gemma-9B-v3#learn | 1010 |
| 7 | qwen2.5vl#learn | 1005 |
| 8 | nora (nit, simulated human) | 1002 |
| 9 | qwen2.5vl-tools#toolkit (poker-analysis toolkit via tool calls) | 992 |
| 10 | wizard-vicuna-uncensored#learn | 979 |
| 11 | carl (calling station, simulated human) | 975 |
| 12 | wizard-vicuna-uncensored#frozen | 957 |
| 13 | qwen2.5vl#frozen | 956 |
| 14 | Big-Tiger-Gemma-27B-v3#frozen | 948 |
| 15 | max (maniac, simulated human) | 931 |
| 16 | Tiger-Gemma-9B-v3#frozen | 872 |

| Model | Learner − frozen twin |
|---|---|
| Big-Tiger-Gemma-27B-v3 | +74 |
| Tiger-Gemma-9B-v3 | +138 |
| qwen2.5vl | +49 |
| wizard-vicuna-uncensored | +22 |
