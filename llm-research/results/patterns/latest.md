# Behaviour patterns

## Chess

| Entrant | Moves | Repeats | Shuffles | No legal move | Top openings |
|---|---|---|---|---|---|
| Big-Tiger-Gemma-27B-v3#frozen | 396 | 14% | 46% | 0% | e5×4, e4×3, Nf6×1 |
| Big-Tiger-Gemma-27B-v3#learn | 441 | 6% | 60% | 0% | e4×5, Nf6×2, e5×1 |
| Tiger-Gemma-9B-v3#frozen | 431 | 9% | 25% | 3% | Nf6×5, e4×3 |
| Tiger-Gemma-9B-v3#learn | 429 | 10% | 26% | 10% | Nf6×5, e4×3 |
| qwen2.5vl#frozen | 400 | 1% | 20% | 0% | e4×7, e5×1 |
| qwen2.5vl#learn | 398 | 1% | 20% | 0% | e5×3, Nf6×2, e4×2 |
| wizard-vicuna-uncensored#frozen | 427 | 10% | 43% | 2% | Nh6×4, Nh3×3, Nc3×1 |
| wizard-vicuna-uncensored#learn | 432 | 13% | 50% | 1% | Nh6×4, Nh3×3, Nc3×1 |

Endings: adjudicated 23, fivefold_repetition 6, checkmate 3

- **qwen2.5vl#learn** wrote a rule about repetition: repeat rate 0.0 in its first round, 0.01 later; not followed.
- **wizard-vicuna-uncensored#learn** wrote a rule about repetition: repeat rate 0.05 in its first round, 0.145 later; not followed.

## Poker

| Entrant | Kind | Hands | VPIP | PFR | Aggr. | Fold to bet | All-in | Tilt VPIP (after loss / else) | Plays like |
|---|---|---|---|---|---|---|---|---|---|
| monte | Monte Carlo equity (eval7, poker-analysis toolkit) | 360 | 19% | 2% | 0.74 | 37% | 1% | 20% / 19% (n=15) | bob (recreational) |
| sol | TexasSolver heads-up turn/river, else Monte Carlo | 346 | 21% | 4% | 1.0 | 24% | 1% | 12% / 21% (n=8) | bob (recreational) |
| carl | calling station, simulated human | 52 | 69% | 2% | 3.0 | 15% | 2% | 75% / 69% (n=4) |  |
| Big-Tiger-Gemma-27B-v3#frozen | language model | 192 | 22% | 1% | 0.0 | 58% | 0% | 33% / 22% (n=6) | bob (recreational) |
| Big-Tiger-Gemma-27B-v3#learn | language model | 192 | 2% | 0% | – | – | 0% | – / 2% (n=0) | nora (nit) |
| Tiger-Gemma-9B-v3#frozen | language model | 108 | 61% | 4% | 0.14 | 10% | 2% | 83% / 60% (n=6) | carl (calling station) |
| Tiger-Gemma-9B-v3#learn | language model | 106 | 17% | 4% | 0.45 | 0% | 1% | 50% / 15% (n=6) | bob (recreational) |
| qwen2.5vl#frozen | language model | 192 | 58% | 12% | 0.45 | 32% | 1% | 33% / 61% (n=21) | carl (calling station) |
| qwen2.5vl#learn | language model | 190 | 33% | 11% | 0.54 | 38% | 0% | 0% / 34% (n=9) | bob (recreational) |
| wizard-vicuna-uncensored#frozen | language model | 192 | 11% | 0% | 0.0 | 67% | 0% | – / 11% (n=0) | bob (recreational) |
| wizard-vicuna-uncensored#learn | language model | 184 | 15% | 0% | 0.0 | 54% | 0% | – / 15% (n=0) | bob (recreational) |
| max | maniac, simulated human | 32 | 81% | 50% | 7.0 | 33% | 12% | 100% / 73% (n=10) |  |
| nora | nit, simulated human | 76 | 5% | 5% | 7.33 | 12% | 3% | 33% / 4% (n=3) |  |
| qwen2.5vl-tools#toolkit | poker-analysis toolkit via tool calls | 182 | 18% | 1% | 0.47 | 35% | 1% | – / 18% (n=0) | bob (recreational) |
| bob | recreational, simulated human | 348 | 36% | 5% | 3.77 | 30% | 1% | 33% / 36% (n=12) |  |
| tina | tight-aggressive, simulated human | 74 | 18% | 14% | 6.0 | 30% | 4% | 0% / 19% (n=6) |  |

### Betting patterns

| Entrant | Limp | C-bet | Check-raise | Median size ×pot | Overbets | Min-raises | Bluff share | Strength: big bets / small bets |
|---|---|---|---|---|---|---|---|---|
| monte | 14% | 50% (n=4) | 0% | 0.75 | 6 | 3 | 0% (n=23) | 3.09 / 2.5 |
| sol | 15% | 50% (n=6) | 0% | 0.75 | 14 | 5 | 2% (n=57) | 3.0 / 2.18 |
| carl | 59% | – (n=0) | 0% | 0.32 | 4 | 3 | 0% (n=36) | 2.5 / 1.86 |
| Big-Tiger-Gemma-27B-v3#frozen | 18% | 0% (n=1) | 0% | 3.33 | 0 | 0 | – (n=0) | – / – |
| Big-Tiger-Gemma-27B-v3#learn | 0% | 0% (n=1) | – | 3.33 | 0 | 0 | – (n=0) | – / – |
| Tiger-Gemma-9B-v3#frozen | 42% | 33% (n=3) | 0% | 4.35 | 4 | 0 | 0% (n=6) | 1.2 / 1.0 |
| Tiger-Gemma-9B-v3#learn | 7% | 67% (n=3) | 0% | 1.33 | 2 | 1 | 40% (n=5) | 1.25 / 0.0 |
| qwen2.5vl#frozen | 36% | 43% (n=14) | 0% | 2.0 | 18 | 0 | 21% (n=24) | 1.76 / 1.33 |
| qwen2.5vl#learn | 16% | 31% (n=13) | 0% | 1.43 | 4 | 0 | 57% (n=14) | 0.55 / 1.0 |
| wizard-vicuna-uncensored#frozen | 8% | – (n=0) | – | 27.07 | 0 | 0 | – (n=0) | – / – |
| wizard-vicuna-uncensored#learn | 11% | – (n=0) | 0% | 26.14 | 0 | 0 | – (n=0) | – / – |
| max | 15% | 70% (n=10) | 0% | 1.0 | 1 | 0 | 23% (n=35) | 1.06 / 1.5 |
| nora | 0% | 33% (n=3) | 0% | 0.98 | 3 | 2 | 0% (n=22) | 2.0 / 2.0 |
| qwen2.5vl-tools#toolkit | 13% | 0% (n=1) | 0% | 1.33 | 5 | 1 | 0% (n=7) | 1.67 / 1.0 |
| bob | 26% | 40% (n=5) | 0% | 0.67 | 8 | 5 | 6% (n=117) | 3.93 / 1.88 |
| tina | 3% | 38% (n=8) | 0% | 1.02 | 4 | 2 | 6% (n=18) | 1.86 / 2.18 |

Strength: 0 high card, 1 pair, 2 two pair, 3+ better, at the moment of the bet. A big-bet mean well above the small-bet mean is a sizing tell: bet size gives the hand away.

## Trading

| Entrant | Episodes | Total R | Trades/episode | Long share | Win rate | Mean R | Mean hold (h) | Exits |
|---|---|---|---|---|---|---|---|---|
| qwen2.5vl#paper | 6 | +9.97 | 3.5 | 86% | 67% | 0.475 | 6.3 | target 4, flat 14, episode_end 3 |
| qwen2.5vl#learn | 6 | +6.03 | 5.2 | 87% | 48% | 0.195 | 16.1 | flat 12, target 8, stop 8, episode_end 3 |
| Tiger-Gemma-9B-v3#frozen | 6 | +5.91 | 4.2 | 100% | 48% | 0.236 | 22.5 | stop 11, target 9, episode_end 5 |
| buy_hold | 6 | +5.85 | 4.2 | 100% | 48% | 0.234 | 22.5 | stop 11, target 9, episode_end 5 |
| Tiger-Gemma-9B-v3#learn | 6 | +5.85 | 4.2 | 100% | 48% | 0.234 | 22.5 | stop 11, target 9, episode_end 5 |
| wizard-vicuna-uncensored#frozen | 6 | +5.85 | 4.2 | 100% | 48% | 0.234 | 22.5 | stop 11, target 9, episode_end 5 |
| wizard-vicuna-uncensored#learn | 6 | +5.85 | 4.2 | 100% | 48% | 0.234 | 22.5 | stop 11, target 9, episode_end 5 |
| qwen2.5vl#frozen | 6 | +4.24 | 8.0 | 83% | 38% | 0.088 | 10.1 | flat 37, stop 2, target 6, episode_end 3 |
| paper_random_gate | 6 | +2.13 | 2.7 | 88% | 44% | 0.133 | 20.2 | stop 7, target 5, episode_end 4 |
| paper_rule | 6 | +1.23 | 2.3 | 86% | 43% | 0.088 | 29.0 | episode_end 4, reverse 1, stop 6, target 3 |
| flat | 6 | +0.00 | 0.0 | – | – | – | – |  |
| Big-Tiger-Gemma-27B-v3#frozen | 6 | -2.53 | 4.0 | 100% | 38% | -0.105 | 20.7 | stop 13, target 7, episode_end 4 |
| Big-Tiger-Gemma-27B-v3#learn | 6 | -4.25 | 2.5 | 100% | 27% | -0.284 | 15.4 | stop 6, target 2, episode_end 2, flat 5 |
| random_trader | 6 | -4.59 | 2.2 | 38% | 31% | -0.353 | 25.6 | target 2, episode_end 4, stop 7 |

The model given the paper's method entered 21 times: 0% while the method said stay out (adverse regime), 0% against the method's signal.

## Agent tests

Counts: tool errors 13, loop 10, failed but asserted a number 4, answered with no tool call 2

| File | Model | Task | Passed | Patterns |
|---|---|---|---|---|
| br_results.json | Tiger-Gemma-9B-v3 | read_price | ✗ | 7 tool errors |
| br_results.json | Tiger-Gemma-9B-v3 | click_link | ✗ | loop: same call 6x in a row; 6 tool errors |
| br_results.json | Tiger-Gemma-9B-v3 | fill_form | ✗ | 4 tool errors |
| br_results.json | Tiger-Gemma-9B-v3 | read_result | ✗ | loop: same call 8x in a row; 8 tool errors |
| br_results.json | wizard-vicuna-uncensored | read_price | ✗ | loop: same call 8x in a row; 8 tool errors |
| br_results.json | wizard-vicuna-uncensored | click_link | ✗ | loop: same call 3x in a row; 3 tool errors |
| br_results.json | wizard-vicuna-uncensored | fill_form | ✗ | loop: same call 6x in a row; 6 tool errors |
| br_results.json | wizard-vicuna-uncensored | read_result | ✗ | 4 tool errors |
| br_results.json | Big-Tiger-Gemma-27B-v3 | click_link | ✓ | 2 tool errors |
| br_results.json | Big-Tiger-Gemma-27B-v3 | fill_form | ✗ | 4 tool errors |
| br_results.json | Big-Tiger-Gemma-27B-v3 | read_result | ✗ | loop: same call 8x in a row; 8 tool errors |
| br_results.json | qwen2.5vl | fill_form | ✓ | 1 tool errors |
| dc_results.json | Tiger-Gemma-9B-v3 | script_and_run | ✗ | failed but asserted a number |
| dc_results2.json | wizard-vicuna-uncensored | write_file | ✗ | loop: same call 8x in a row |
| dc_results2.json | wizard-vicuna-uncensored | run_command | ✗ | 1 tool errors |
| dc_results2.json | wizard-vicuna-uncensored | list_and_count | ✗ | loop: same call 4x in a row |
| dc_results2.json | mistral-large | script_and_run | ✗ | failed but asserted a number |
| dc_results_tools.json | tiger-gemma-tools | run_command | ✗ | answered with no tool call |
| dc_results_tools.json | tiger-gemma-tools | script_and_run | ✗ | answered with no tool call; failed but asserted a number |
| dc_results_tools.json | qwen2.5vl-tools | run_command | ✓ | loop: same call 7x in a row |
| dc_results_tools.json | qwen2.5vl-tools | list_and_count | ✓ | loop: same call 20x in a row |
| gui_results.json | qwen2.5vl-tools | explorer_folder | ✗ | failed but asserted a number |
