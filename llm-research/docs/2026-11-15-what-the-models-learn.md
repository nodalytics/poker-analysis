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

## Next: search on top of nm25

Pluribus's idea, applied to our network: at a decision, look a street or two ahead, assume each opponent may switch
between a few continuation strategies (the network's policy, and versions biased towards folding, calling and
raising), solve that small game, and play its answer instead of the network's raw policy. It only goes into Kestrel
if it beats the plain network in duplicate matches; `scripts/ninemax_search.py` holds the earlier attempt.
