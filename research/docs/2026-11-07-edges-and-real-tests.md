# Edges, robustness and real tests — 2026-11-06/07

The question: are there strategies that win a lot without being beatable? Ideas were taken from game theory,
physics, biology and ecology, behavioural science and graph methods, and combined with RL. Each was tested,
exactly where possible (Leduc poker, small enough to score any strategy precisely), then on real hands, and
finally against Slumbot, a strong public bot. All computation ran on the lab. Units are milli-chips per hand on
Leduc and big blinds per 100 hands (bb/100) in hold'em.

## Results that hold

| Finding | Evidence | Source |
|---|---|---|
| Robust exploitation is the efficient way to win from weak players | Against the maniac at p 0.25: 84% of the maximum profit (1,949) at 7% of pure exploitation's exposure (152 vs 2,111) | `results/leduc_rnr/` |
| A learning opponent punishes pure exploitation quickly | +981 against pure exploitation within 3,000 hands; against robust p 0.25 it is still losing at 3,000 hands (−118) and reaches only +122 by 300,000; it never beats the equilibrium | `results/leduc_rl_rnr/report.json` |
| RL with a co-trained adversary keeps profit and cuts exposure about 3× | Against the maniac, p 0.5: profit 1,966 (pure exploitation: 2,006), exploitability 694 (1,994) | same |
| Tabular RL cannot reach real robustness | Floor of 630–690 exploitability, against CFR's 41 | same |
| A robust teacher speeds learning up a lot | Taught by robust exploitation, after 15,000 hands: 2,148 profit at 559 exploitability, against 1,276 at 1,193 from scratch | `teaching.json` |
| AlphaPoker networks keep the profit but are easier to beat | 85–95% of CFR's profit at 2–5× the exploitability; averaging five networks cuts that by 21–26% | `alphapoker.json`, `results/leduc_ensembles/` |
| **Reading the opponent beats every fixed strategy and every mix** | Bayes identification, then the matching robust strategy: mean 1,034, worst case +12.5, including a trap opponent. Best single strategy: 858 (worst −48). Exp3 bandit: 695. Averaging strategies is worse than mixing them | `results/leduc_ensembles/` |
| Quantal response (physics: Boltzmann play) is dominated | Converged QRE at its best: 878 profit vs the maniac at 205 exploitability; robust exploitation gets 1,653 at 107 | `results/leduc_rnr/qre.json` |
| Hand reading from betting works (exact Bayesian filter) | Pluribus hands, true hand in the top 10% after 2+ actions: 51–58% (chance 10%) | `results/range_inference/` |
| A model of one opponent reads it much better | Trained on Slumbot's revealed hands: action accuracy 81% (generic 63%), true hand in top 10% 37% (25%) | `results/slumbot_range/` |
| Real players chase losses | Down 20bb+ in a session: VPIP +1.2 points (t 8.8; shuffled control −0.06). Tilt after a 30bb+ loss: +0.7 points (t 4.0) | `results/behaviour/`, `results/regimes/` |
| River bet size is a tell; river raises are value | Nutted share rises from 11% (small bets) to 37% (overbets), while bluffs stay at 17–22%; only 8.5% of river raises are bluffs (32,829 shown river bets) | `results/behaviour/` |
| Style regimes exist | 22% of regulars switch between about 18% and 35% VPIP, in 30–40 hand stretches (5% expected by chance); table size not yet controlled | `results/regimes/` |

## Results that do not hold (nulls)

- **Past results do not predict future results:** rank correlation 0.03. Graph methods add nothing on real players:
  PageRank on money flow 0.09, the opponent-adjusted network rating 0.01, and a graph convolution over who plays whom
  0.07. Style alone reaches 0.11 [0.02, 0.22], which is only weak (393 regulars). Source: `results/graph_players/`.
- **Collusion screen:** among 1,138 pairs, there is no extreme soft-play outlier. Pairs enter pots together less
  than independence predicts, but that is ordinary crowding out.
- **Quit when ahead:** +0.2 points (t 6.5). It is real but tiny. Fatigue shows only small style drifts and no change
  in win rate.
- **RL add-ons:** belief, Boltzmann and Kelly versions of the RL agent beat nothing reliably. Their heads-up losses
  to the solver bot (−28 to −46) predate the heads-up seat fix, so they need a rerun.
- **Environment suite (36 environments):** the RL agents come last (−4 and −11, losing 19 of 36). The rule-based
  adaptive bot loses none, mean +48.7. This run predates both fixes below, and its solver settings are unverified,
  so it needs a rerun before the solver-driven bots are ranked.

## The outside test: Slumbot (heads-up, 200bb, real hands over its public API)

Each run exposed a real bug that our own simulations had hidden:

| Run | Solver bot | What it showed |
|---|---|---|
| v1 | −224 (894 hands) | The solver tree always offered an all-in; at a stack of 50 pots, a 60-iteration solve left river bluff-shoves |
| v2 | −68 (1,252) | In heads-up play, PokerKit seats the big blind at 0 and the button at 1, the reverse of our charts. The button limped and the big blind defended against the wrong range |
| v3 | −182 (3,000) | On the lab, TexasSolver ran without its hand-ranking tables. It ranked every hand equal, and ace-high shoved into a flush. Now the tables are found or the code raises |
| v4 | −64.8 [−127, +8] (3,000) | Preflop, flop and turn about even; −54 on the river, where Slumbot's bets were stronger than the generic hand reading assumed |
| v5 | **+57.5 [+2, +114] (3,000)** | Slumbot-specific hand reading, learned from its revealed hands. The river went from −54 to +56. The adaptive bot reached −22.9 [−96, +47]: its call-down rule for aggressive players backfires against a strong one |
| v5 repeat | −53.8 [−108, +1] (5,000) | The same bot on fresh hands. **Luck-adjusted** (`aivat_lite.py`: all-in equity, pre-flop and board card luck; variance 1.5–1.8× lower), v5 is +11.8 and the repeat −32.8: together about **−14 bb/100 [−45, +17], roughly even with Slumbot**. v5's raw +57.5 was mostly card luck |

v5 is opponent modelling: the model was trained on Slumbot's own past hands, as one would for a real regular. It
is not evidence of general strength.

Bugs 2 and 3 invalidate earlier heads-up results and solver-driven lab results launched without
`POKER_ANALYSIS_ROOT`. Affected runs: the heads-up cells of the environment suite, the first meta-game,
`env_suite_ens`, `hedge_long_*` and `meta_game_v2`. The Leduc work, the flop library and the value library are
unaffected.

## Finding our own leaks: local best response (LBR)

`scripts/lbr.py` is LBR (Lisý & Bowling 2017), with 1,200–3,000 hands per run at 100bb, heads-up.

- **How it plays:** LBR tracks the solver bot's range over all 1,326 holdings and, at each of its decisions, takes the
  best of fold, call, a pot-size bet or all-in, assuming the hand then checks down. Pre-flop it only calls.
- **How the bot's policy is computed:** for every holding, by the bot's own layers. Charts pre-flop, the flop model,
  TexasSolver on turns and rivers solved on its tracked range, and a strength rule elsewhere.
- **What it measures:** its winnings are a lower bound on how beatable the bot is.

| Run | LBR wins (bb/100) | Biggest source |
|---|---|---|
| v1 (3,000 hands) | +464 [+321, +608] | — |
| diagnostic (1,200) | +459 [+238, +681] | flop all-in, the bot calls: +308. The flop model only saw 2/3-pot bets and treats a shove like one |
| v2, overbet defence (call when equity against the bettor's read range beats the price) | +288 [+134, +442] | flop all-in, bot calls: +267 (half as many calls) |
| v3, plus minimum defence frequency (against B into P, defend only the top P/(P+B) of our range) | +288 [+170, +406] | the all-in leak is gone; now flop pot-size bets: bot calls +111, raises +100, folds +67 |
| v4, the same defence from 0.9x pot | *running* | |

The remaining leak is the same root cause: **the flop model has no notion of bet size**. The lasting fix is to
train it on solves with several bet sizes (pot, 2x pot, all-in) in the tree.

## Opponent modelling at scale (real players)

These runs use HandHQ: 660,000 hands and 1,181 held-out regulars. Each model is scored on the held-out players'
showdown hands, by the rank of the true holding.

| Model | Top-10% rate | Mean rank |
|---|---|---|
| Generic (trained on Pluribus's hands) | 52.6% | 0.173 |
| Trained on showdown hands, per style cluster (`opp_models.py`) | 37.0% | 0.321 |
| Trained on showdown hands, per player | 27.7% | 0.363 |
| **Generic, tilted to each player's action rates** (`opp_adjust.py`) | **53.2%** | **0.169** |
| Generic, tilted to their style cluster's rates | 53.3% | 0.170 |

- **Showdown-only training fails.** Players who reach showdown never fold, so those models cannot learn which hands
  fold.
- **The Slumbot model worked because every hand's cards were revealed.**
- **Tilting the generic model with each player's observed fold, call and raise rates** (from all their hands, cards
  or not) is unbiased and helps slightly. A player's overall frequencies say little about their specific hand.

## Outside opponents

- **Slumbot is the only strong heads-up no-limit bot available to play.** It has a free public API.
- **No strong open-source bot can be run.** Libratus, Pluribus and DeepStack were never released, and community
  rebuilds such as DeepHoldem need the retired Lua Torch stack. OpenSpiel's heads-up no-limit game runs on the lab,
  but only with a fold / call / pot / all-in action set, and nothing trained there is strong.
- **Training our own** (Deep CFR at heads-up no-limit scale) would take weeks of GPU time.
- **A smaller option:** train a sparring bot on OpenSpiel's simplified game on the lab GPU. That gives a bot of a
  different style, not a strong one.
- **Meanwhile:** Slumbot (a strong outside opponent), LBR (our worst leaks, found automatically) and our 8 bots plus
  the random-style population cover what another opponent would add.

## Ecology (meta-game, heads-up 5–20bb, rerun with both fixes)

- Two random hyper-aggressive styles take over the population (replicator dynamics: 99.9%; an open ecosystem with
  fish: 82% + 10%).
- No strategy beats everyone.
- The solver bot still loses clearly to the maniac (−36 ±13) and to the strongest style (−24 ±12), even with the
  new short-stack defence (re-shove with the push/fold calling range, priced calls). This is the main open
  weakness.

## What to build on

1. **Opponent identification, then robust exploitation.** It was strongest on Leduc, and the Slumbot model is
   the hold'em version of the same idea. Next: a hold'em classifier choosing among exploitative lines, with the
   equilibrium as the fallback.
2. **Real-player tells as priors:** chasing after losses, the size–strength relationship, and folding to river
   raises.
3. **Short-stack heads-up play against constant aggression** is unsolved: diagnose solver bot vs maniac hand by
   hand.
4. **Re-solve before an adaptive opponent learns us:** robust strategies take tens of thousands of hands to
   exploit.

Scripts: `leduc_rnr.py`, `leduc_rl_rnr.py`, `leduc_ensembles.py`, `meta_game.py`, `env_suite.py`,
`range_inference.py`, `ranges.py`, `regimes.py`, `behaviour.py`, `graph_players.py`, `slumbot_client.py`,
`slumbot_model.py`. Partner clips 14–17 in `media/partner/`.
