# The solver stack under the LLM — 2026-11-05

The models play poker better when a solver does the heavy work, so this study builds that work out in layers and
serves it to the LLM through the poker MCP server (`../../mcp_server.py`). For our own tables and PokerTH: nothing
here reads a live real-money client.

## Does solver advice help? Yes, measurably

`scripts/solver_first_test.py`: each model sat twice at the same 9-seat table, once plain and once told before every
decision the solver's strategy for its exact hand (heads-up, turn or river) or Monte Carlo equity and pot odds
everywhere else. 30 sessions of 24 hands; one-sided sign test on the session-by-session chip difference.

| Model | Solver-first ahead | Mean chips per session | p |
|---|---|---|---|
| Qwen2.5-VL 32B | 23 of 30 | +127 | **0.003** |
| Tiger-Gemma 9B | 22 of 29 | +59 | **0.004** |
| Big-Tiger-Gemma 27B | 21 of 30 | +1 | **0.021** |

The first result in these studies to clear its test. Raw: `results/raw/solver_first_summary.json`.

## Layers

| Street | What answers it | Speed |
|---|---|---|
| Preflop, heads-up | Push/fold equilibrium, 1-25 big blinds (`push_fold_nash`) | instant (precomputed) |
| Preflop, 3-6 players | Multi-way push/fold equilibrium, every position, with or without antes | instant (precomputed) |
| Any street, any number of players | Multi-way Monte Carlo equity with ranges and dead cards (`equity_multiway`) | ~1M run-outs in 10 s on the lab |
| Flop | Flop-strategy network trained on the solver library | 0.7 ms per hand |
| Turn, river (heads-up) | Live TexasSolver solve | ~8 s |

**Preflop, heads-up** (`scripts/poker_math.py nash`): fictitious play over all 169 hand classes on a 169x169 equity
table (20,000 run-outs per pair). Shove/call shares match the published heads-up equilibrium: 10bb 59%/38%,
15bb 46%/27%, 20bb 40%/20%. Card removal between the two hands is ignored.

**Preflop, multi-way** (`nash-multiway`): 3-6 players, every position's shove range when folded to and the call ranges
against each shover; at 5, 8, 10, 12, 15 and 20 big blinds, with no ante and with a 1/8-big-blind ante.
Simplification: **at most one caller per shove**, so all-ins stay heads-up. Checks: folded to the small blind it
reproduces the heads-up solution (58% / 37%); more players behind tighten the first position (10bb: 33% three-handed,
13% six-handed); antes widen every range (six-handed 10bb first position 13% -> 19%).

**Real-time flops**: a reduced live flop solve takes 80 s at 16% exploitability, a decent one 2.4 minutes: too slow.
So flops come from a network (`scripts/ml_flop.py`) trained on the **solver library** (`scripts/solver_library.py`):
the 1,755 distinct flops sampled across textures x 2 stack depths x 2 range pairs, solved at <=1.3% exploitability,
~47 s each. First model on 3 flops (1 held out): 75% top-action agreement with the solver vs 60% for the
average-strategy baseline. A small sample; it retrains on the full 400 solves.

## Hand histories

`scripts/hand_histories.py` reads the published PHH dataset (github.com/uoftcprg/phh-dataset, MIT, commit
e47fbd581637). No poker site is scraped.

| Source | Hands | VPIP | PFR | 3-bet | Aggression |
|---|---|---|---|---|---|
| Pluribus vs professionals | 10,000 | 26% | 18% | 8.6% | 2.33 |
| HandHQ 1000NL population (2009) | 100,000 | 26% | 18% | 8.3% | 2.42 |
| HandHQ 25NL population (2009) | 100,000 | 24% | 12% | 4.6% | 3.9 |

These are the human baselines for the pattern scanner, and per-decision rows (`*.decisions.jsonl.gz`) are training
data for the learning layer next to the solver library.

## Tournaments: ten is too few to rank anyone

`scripts/tournament_env.py`, 10 multi-table tournaments, 18 entrants (each model plain and advised, two of each
personality, two Monte), top 3 paid from a 1,800 pool. Raw output in `results/tournaments/`; bootstrap 95%
intervals on mean prize in `results/tournaments/ci.json`.

| Entrant | Mean prize | 95% interval | In the money |
|---|---|---|---|
| tina#1 | 360 | 162 to 576 | 6/10 |
| max#2 | 270 | 0 to 540 | 3/10 |
| bob#2 | 198 | 0 to 414 | 3/10 |
| qwen2.5vl plain | 144 | 0 to 288 | 3/10 |
| qwen2.5vl advised | 90 | 0 to 216 | 2/10 |
| Tiger-9B advised | 36 | 0 to 108 | 1/10 |
| Tiger-9B plain, Big-Tiger plain and advised | 0 | 0 to 0 | 0/10 |

- Only tina#1's interval excludes zero, and tina#2 (the same bot) made 180 with an interval reaching 0: the spread
  between two copies of one strategy is as wide as the spread between strategies. **Ten tournaments rank nothing.**
- An earlier look at 6 of 10 said only advised LLM flows cashed. That did not hold: Qwen plain cashed 3 times to
  advised's 2. Advice did not make a measurable difference here; the paired solver-first test is the evidence for it.
- No LLM won a tournament. Big-Tiger never cashed in either flow.
- For a ranking, use many more tournaments with bots only (Tina+ vs Tina used 60), and keep LLMs to paired tests.

## Next

1. Retrain the flop network on the full library, wire it into the advice the LLM gets on the flop, and rerun the
   solver-first test with it.
2. Reinforcement learning: self-play fine-tuning in the PokerKit arena, starting from the networks, against the fixed
   players (Bob, Monte, Sol).
3. Grow the library beyond 100 flops; parse all 21.6M HandHQ hands.
