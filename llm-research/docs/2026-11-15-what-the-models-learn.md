# What our models learn, and where the poker programs before them fit (2026-11-15)

Two questions answered in one place: whether our networks pick up the game's "hidden patterns", and how the
well-known poker programs relate to what we build.

## Hidden patterns

There are three kinds, and our models learn only the first.

1. **The game's own structure:** which hands to bet, balance between value and bluffs, bet sizing, blockers, ICM
   pressure. This is what self-play learns, and it is measured: the poker-sense and post-flop checks
   (`ninemax_sense.py`, `ninemax_postflop.py`) and, best of all, the exploiter test (`ninemax_eval.py exploit`): a
   fresh network trains against 8 copies of the model for 30 minutes, and whatever it wins is a hole in the model.
   Against nm25 it found almost nothing: +0.2 bb/100 [−0.4, +0.7], against +1.5 for nm17 and +2.2 for nm20
   (`2026-11-11-mtt-strategy.md`, nm25).
2. **Individual opponents' habits:** the networks see the table, not the opponents' history, so they play every
   opponent the same way. Learning a particular player's tendencies is the to-do item "opponent tracking by
   username, with safe exploitation" (`kestrel/docs/design.md`, Status): statistics per username, and exploitation
   only within limits that grow with the sample, so it cannot cost much against a strong player.
3. **Patterns in the cards:** there are none to find. Our dealer passed every test on 4 million deals
   (`dealer_check.py`: positions, repeated cards, starting-hand classes, seven-card categories, independence; lowest
   p = 0.07), and real platforms shuffle properly. A model that seemed to find a card pattern would be fooling
   itself.

## The programs before ours

| Program | Year, team | Game | What it did | Where it is in Kestrel |
|---|---|---|---|---|
| **Cepheus** | 2015, Alberta | heads-up **limit** | "Essentially weakly solved" the game with CFR+; not no-limit | — |
| **DeepStack** | 2017, Alberta | heads-up no-limit | Re-solved every decision with deep value networks instead of a fixed blueprint; beat professionals; its team also devised AIVAT | our luck adjustment is AIVAT-based (`kestrel/core/src/luck.cpp`) |
| **Libratus** | 2017, CMU | heads-up no-limit | A blueprint plus safe subgame solving in later streets, plus a nightly self-improver; beat four top pros by 147 mbb/hand | our heads-up blueprint with the safe river re-solve follows it (`kestrel/core/src/resolve.cpp`) |
| **Pluribus** | 2019, CMU/Facebook | 6-player no-limit | Blueprint, then depth-limited search assuming opponents may switch between several strategies; beat elite pros | the 9-max search, in progress (below) |

Later work along the same line: **ReBeL** (2020) and **Student of Games** (2023), which combine self-play learning
with search. Our 9-max networks (NeuRD with a "magnet" pull) are closer to that self-play line than to Pluribus.
Adding Pluribus-style search on top of nm25 is the most promising next step for strength.

## Search on top of nm25 (2026-11-15)

`scripts/ninemax_search.py` on nm25, before and after the flop, 2,048 samples, greedy, 2,000 deals, the hero in every
seat against 8 copies of nm25. Version 1 rolls every sample out with the network for everyone. **Robust** (after
Pluribus) rolls each sample out four times, the opponents continuing with the network or the network biased ×5
towards folding, calling or raising, and values an action by its worst case. The opponents in the match play the
network or one of the biased versions (`--opp-bias`). Result, bb/100 in ICM equity (the plain row is nm25 without
search on the same deals, so the gain is the difference):

| Opponents play | Plain nm25 | Version 1 search | Robust search |
|---|---|---|---|
| the network | +1.9 [+0.4, +3.4] | +0.8 [−2.5, +4.2] | +3.2 [+0.8, +5.6] |
| fold-biased | −1.3 [−2.7, +0.1] | −2.0 [−5.3, +1.2] | −0.8 [−3.3, +1.6] |
| call-biased | +1.0 [−0.7, +2.6] | +3.5 [+0.2, +6.9] | +3.5 [+1.0, +6.0] |
| raise-biased | +0.6 [−1.1, +2.3] | **+6.6** [+3.6, +9.7] | +3.8 [+1.4, +6.2] |

About 17,000-18,500 searched decisions per row; version 1 took about 27 ms a decision, robust about 80 ms (shared
PRO 6000).
- **Robust is never below plain** (+1.3, +0.5, +2.5, +3.2 over the four kinds of opponent); version 1 is below plain
  against the network and against fold-biased opponents (−1.1, −0.7) and gains most against raise-biased ones (+6.0).
- The trade-off Pluribus's design aims at: robust gives up some of version 1's gain against an opponent it could
  exploit (raise-biased: +3.2 against +6.0) and never loses.
- The intervals are about ±2.5-3.5 per row and not paired, so most single differences are not significant at 2,000
  deals; version 1 against raise-biased opponents is the clearest.

**Paired, 2026-11-16** (`search3_nm25_*`, 4,000 deals per run, all runs on the same deals, the gain scored deal by
deal so the cards' luck cancels; `scripts/search_paired.py`): gain over plain nm25, bb/100

| Opponents play | Version 1 | Robust |
|---|---|---|
| the network | +2.18 [−0.32, +4.68] | +1.99 [+0.02, +3.96] |
| raise-biased | **+6.39** [+3.92, +8.86] | **+3.69** [+1.54, +5.84] |
| call-biased | **+3.89** [+1.30, +6.49] | **+3.13** [+0.98, +5.28] |
| fold-biased | +2.26 [−0.07, +4.60] | +0.82 [−1.07, +2.71] |

Both searches gain against every kind of opponent; neither loses anywhere once the deals are paired (the unpaired
run's small losses for version 1 were the cards). Version 1 gains at least as much as robust everywhere, and is three
times faster (about 27 against 80 ms a decision). Robust's promise, holding up when opponents adapt to it, is not
shown by these fixed opponents; it needs an adaptive one: an exploiter trained against each searcher.

**Then:** an exploiter test against each searcher, since robustness is what the robust version is for. Both beat the
plain network in duplicate matches, so search is ready for Kestrel; at 27-80 ms a decision it fits any platform's
time limit.

## Local best response for 9-max (2026-11-15)

`scripts/ninemax_lbr.py`: one seat is LBR (Lisý & Bowling 2017), the other eight the network. LBR plays the network
except after the flop in a pot it contests alone against one network opponent, where it keeps the opponent's range
over all 1,326 holdings (each weighted by the network's own probability of every action it took with that holding)
and plays the action with the best value assuming a check-down, over 48 sampled run-outs. Scored against the network
in the same seats on the same deals (paired), 20,000 deals each:

| | LBR's gain, bb/100 (ICM equity) | LBR decisions | 30-minute exploiter, for comparison |
|---|---|---|---|
| nm25 | +0.26 [−0.43, +0.95] | 6,941 | +0.2 |
| nm20 | +0.09 [−0.40, +0.58] | 5,550 | +2.2 |
| nm17 | −0.55 [−1.39, +0.29] | 5,966 | +1.5 |

LBR finds no hole in any of them, and does not rank them as the exploiter does (nm20 most exploitable). The two look in
different places: LBR only heads-up after the flop, with a check-down assumption; the exploiter the whole game. So the
leaks the exploiter finds are before the flop or in multiway pots, where most decisions at 10-25bb are. LBR as written
is not the exploitability test for short-stack tournaments; the exploiter stays the main one. A pre-flop LBR
(push/fold against the network's ranges, scored in ICM) would test where the leaks are.

