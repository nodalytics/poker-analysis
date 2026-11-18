# Methods from other fields (2026-11-17)

Twelve ideas borrowed from particle physics, clinical trials, multi-agent RL, Bayesian statistics, sports ratings,
behavioural economics, speech and fraud detection: what each is, what we built, how it was tested, and what it
showed. Every detector was checked on planted effects and on data with nothing planted (`a-control-is-only-for-a-
named-property`); every family of results is corrected for looking in many places (item 1).

| # | Field | Idea | Status | Result |
|---|---|---|---|---|
| 1 | particle physics | look-elsewhere correction | built (`scripts/multitest.py`) | one past claim falls |
| 2 | particle physics | blind analysis (pre-registration) | in use | the comparison of 4-5 was fixed and pushed before its runs started |
| 3 | particle physics | earth mover's distance card buckets | tested (nm33, pre-registered) | not better; probably worse (−0.30, Holm p 0.036) |
| 4 | particle physics | seat-order-independent network (Deep Sets) | tested (nm32, pre-registered) | level with the baseline (−0.10 ± 0.27) |
| 5 | particle physics | control variates | built (`duplicate(cv=True)`) | no gain: duplicate already cancels card luck |
| 6 | clinical trials | group-sequential tests | built (`scripts/seqtest.py`) | 20-83% of deals saved, errors held |
| 7 | multi-agent RL | league training (PFSP, league exploiters) | tested (nm35 vs nm34) | hole shrunk by a third, not closed; slightly ahead of a bigger exploiter |
| 8 | Bayesian statistics | empirical-Bayes prior and forgetting profiles | built (Kestrel) | finds a player who changes, 19 hands late |
| 9 | sports ratings | Bayesian ratings, P(best) | built (`scripts/ratings_bayes.py`) | the ladder is not transitive (chi²/dof 1.65) |
| 10 | behavioural economics | tilt, suited overvaluation | built (`scripts/table_analysis.py`) | finds a planted tilter only |
| 11 | speech | learned opponent embeddings | built (`scripts/opponent_embed.py`) | 10 hands beat the six stats at 50 |
| 12 | fraud detection | collusion (soft play) | built (`scripts/table_analysis.py`) | finds a planted pair only |

## 1. The look-elsewhere effect

When several results are read together, some look significant by chance. `multitest.py` corrects each family with
Holm's step-down method (the chance of even one false claim stays at 5%) and reports Benjamini-Hochberg q-values (the
expected share of false claims). Re-reading what we reported:

- Kestrel's search at 1,024 deals: all five gains hold (Holm p 0.0002-0.024), and the three losses at 256 deals.
- The Python search run of 2026-11-16: "robust search beats the plain network" (+1.99, p 0.048 alone) **falls**
  (Holm p 0.19); the write-up is corrected.
- The version ladder: every rating differs from nm17. LBR: nothing anywhere, as reported.

## 2. Blind analysis

Physicists fix an analysis before looking at the signal. `results/prereg/2026-11-17-sets-and-emd.json` fixes the
runs, the budget (600 million hands each), the metric, 40,000 deals, the seed, the test, the Holm correction over the
two candidates and the decision rule; it was committed and pushed (`57b1a5c`) before nm31-nm33 started, so the
public commit time is the record. Not allowed afterwards: changing any of those, or re-running a comparison and
keeping the better reading.

## 3. Card buckets by earth mover's distance (nm33)

Jet physics compares events by the earth mover's distance between energy distributions (Komiske, Metodiev and Thaler
2019); Libratus and Pluribus group hands the same way, by the distance between their distributions of future strength.
`blueprint/abstraction_pa.cpp` already did this for heads-up (k-means on the CDF of next-street strength, an L2 proxy
for the 1-D EMD; OCHS on the river). A 200-bucket version (`out_pa200`) matches the baseline's bucket count, so only
the method differs. One fix first: k-means numbers its clusters arbitrarily, but the network's strength input reads
bucket / (B − 1), so the clusters were renumbered by mean strength (`renumber_buckets.py`); the correlation of bucket
number with strength went from −0.13, +0.05, +0.01 to +0.99 on each street.

## 4. A seat-order-independent network (nm32)

Particle-flow networks read a jet's particles as a set: one shared network per particle, then a sum. `NineNet` arch
`sets` reads each other seat (live, all-in, committed, acted, aggressor, chips behind, equity, bounty) with its
position relative to the player as a token, through one shared network (2 layers of 128), pooled by sum and max; the
rest of the network is unchanged. What is learnt about a seat is shared by every seat, and tables of any size read
alike. `seat_layout` was checked against `features()` in the tournament, 6-max cash and PKO modes (every token column
holds the value it claims).

## 5. Control variates

Physics Monte Carlo subtracts quantities of known mean that are correlated with the result. For duplicate matches:
each seat's card luck per street, its strength less its expectation given the previous street's bucket (mean zero
exactly, by the tower property), regressed out. Unbiased (nm25 against itself: +0.13 plain, +0.14 corrected), and the
interval shrank by **×1.00** in all three matches tried: seating the hero in every seat already cancels card luck.
What is left is the sampling of moves; removing it needs AIVAT-style terms at every decision (the value of each
alternative move), which is the next step.

## 6. Sequential tests (clinical trials)

`seqtest.py` plays a match in up to 8 stages of 5,000 deals and stops when |z| crosses the O'Brien-Fleming boundary
C / sqrt(t) (C = 2.0696 for 8 looks at 5%, computed by simulation; published tables give about 2.07), or when the 95%
interval lies within ±0.5 bb/100 (no difference that matters). On simulated matches with our duplicate spread
(0.34 bb a deal):

| True difference, bb/100 | Stops for a difference | Fixed 40,000-deal design | Deals used | Saved |
|---|---|---|---|---|
| 0 | 4.8% (the false-positive rate) | 4.9% | 31,922 | 20% |
| 0.5 | 82.7% | 83.7% | 29,580 | 26% |
| 1 | 100% | 100% | 17,128 | 57% |
| 2 | 100% | 100% | 9,978 | 75% |
| 3 | 100% | 100% | 6,805 | 83% |

On the engine, nm20 against nm17 stopped at the 4th look, 20,000 deals of 40,000, at +0.96 bb/100 (the ladder:
+1.04).

## 7. League training (multi-agent RL)

AlphaStar's league keeps main agents, exploiters of the main agent and exploiters of the whole league, and picks
opponents by how hard they are (prioritised fictitious self-play, PFSP). `ninemax_selfplay.py` already froze each
retired exploiter into the pool; new: `--pfsp p` picks pool members with weight (1 − p_win)^p, p_win the network's
running chance of beating each; `--league-share` adds a league exploiter trained against the network and every pool
member, frozen into the pool at each reset; `--max-frozen` caps the frozen exploiters. The fix found on the way: the
exploiter was seated at random among 9 seats, and on 6-max tables a third of those are always empty (wasted tables,
and the standalone exploiter diverged when a step had no decisions).

nm28 (6-max cash) left a hole of +11.0 bb/100 [+6.7, +15.2] to the 30-minute exploiter. Two fixes train from it for
12 hours with the same share of tables against exploiters (40%): **nm34** a bigger, longer-lived exploiter (40%,
reset every 2 hours), **nm35** the league (25% exploiter + 15% league exploiter, PFSP 2, 8 frozen kept).

## 8. Bayesian profiles: a population prior and forgetting (Kestrel)

- **Empirical Bayes.** Each rate's prior is fitted to the platform's own players by the method of moments: their
  average rate m, and how much they truly differ once each player's binomial noise is removed, giving a Beta prior of
  strength m(1 − m)/v_between − 1 (2 to 200 chances). On the planted lineup: VPIP strength 2 (players differ a lot),
  went-to-showdown 192 (they barely differ). Fewer than 5 players: the reference types' average with 8.
- **Forgetting.** Recent counts are multiplied by 0.986 each hand before the hand is added (a Beta-Bernoulli filter,
  the discrete cousin of a Kalman filter; half-life about 49 hands), saved with the book. The profile shows the recent
  type beside the lifetime one, and flags **changed** when they differ and a pre-flop rate has moved by z ≥ 3 (a
  label alone flipped for a player whose play varies with stack depth).
- The same filter in `table_analysis.py` found the planted switcher (tag, then a calling station from its 150th hand)
  at its 169th hand, and the tilter's bursts; nobody else.

## 9. Bayesian ratings (sports)

`ratings_bayes.py` puts a Gaussian prior on each version's rating and reads the posterior (TrueSkill's model, solved
exactly): the same ratings as the ladder, plus each version's chance of being the best and of beating each other.
Two findings: the ratings do not fit the pairs as well as their errors say (chi²/dof 1.65: styles beat each other
non-transitively, so the ladder's intervals are about 1.28× too narrow), and **by rating alone nm25 is not the likely
best**: P(best) nm20 0.38, nm26 0.30, nm27 0.29, nm25 0.035, with gaps of about 0.07 bb/100. nm25 is the champion
for its exploitability (+0.2 against nm20's +2.2), which the ladder does not measure.

## 10 and 12. Behaviour and collusion at Kestrel's tables

Kestrel's hands now record the whole table (players, starting stacks, the big blind, every action, shown cards).
`table_analysis.py` reads them; the test platform has planted bots to check it with: **switcher**, **tilter** (loose
and aggressive for 25 hands after losing 15bb or more) and an **ally** pair (never bet or raise into each other, fold
to each other's bets unless strong). 2,000 hands each, Holm over players or pairs:

| Detector | Planted lineup | Lineup with nothing planted |
|---|---|---|
| tilt | tilter (z +4.27, Holm p 0.0002), nobody else | nothing |
| change point | switcher at hand 169 (switched at 150); the tilter's bursts | nothing |
| soft play | ally pair (z −4.27, Holm p 0.0005), nobody else | nothing |
| suited overvaluation | nothing | nothing |

Each detector needed its control before it was right, and the failures are worth keeping:

- **Tilt against others' losses** (the same tournament stages): flagged the switcher, because a loose player loses
  more; the arrow runs from looseness to losses. **Against a local baseline**: still flagged it, for the same reason.
  **Before and after each own loss** keeps the player's style on both sides: only the tilter.
- **Soft play with B in the pot against B out**: flagged innocent pairs, because with B in the pot is more often
  multiway, where everyone bets less. **Stratified by players still in**: still flagged pairs, through how everyone
  treats a particular player. **Logistic main effects** (A's own aggression, everyone's aggression against B) removed
  that, but a one-way style effect remained on the null lineup (a limper folding 52% to a tag that bets only strong
  hands, against 32% expected). **Requiring both directions**, as soft play between partners is, left the ally pair
  alone, and nothing on the null lineup.
- Loss aversion near the money needs a bubble; the test platform pays all nine places, so it is not measured.

## 11. Opponent embeddings (speech)

Speaker recognition summarises a voice by pooling a network's output over many short frames (x-vectors). Here a frame
is a hand: the player's actions counted by street and kind (24), whether it showed cards and the table size. A window
of N consecutive hands of one player is classified as its type three ways: the six HUD rates Kestrel's profiles use
(logistic regression), an x-vector (a per-hand network, mean and standard deviation pooled, a classifier) and a GRU over
the hands in order. Data: 8 test-platform sessions of 2,000 hands against six bot types (tag, station, limper, odd,
min-raiser, tilter); trained on 6 sessions, tested on the other 2 (`results/ninemax/opponent_embed.json`):

| Hands | Six HUD rates | x-vector | GRU |
|---|---|---|---|
| 10 | 0.767 ± 0.023 | **0.872 ± 0.018** | 0.842 ± 0.020 |
| 20 | 0.791 ± 0.031 | **0.924 ± 0.020** | 0.918 ± 0.021 |
| 50 | 0.849 ± 0.044 | **0.981 ± 0.017** | 0.958 ± 0.025 |

The x-vector after 10 hands is more accurate than the six rates after 50. The rates fail on the odd-sizer (41% at 50
hands) and the min-raiser (49%): their style is in sizes and street-by-street patterns that six rates discard. The
bots are more regular than people, so the absolute numbers are optimistic; the comparison is the result. Next: the
embedding inside Kestrel's profiles (trained on hands labelled by the platform's own bots and the pool's rule
players), once human hands exist to check it on.

## The training comparisons

**nm31 (baseline), nm32 (Deep Sets), nm33 (EMD buckets)**, from scratch with nm25's settings, compared exactly as
pre-registered (`results/prereg/2026-11-17-sets-and-emd.result.json`): 40,000 deals, seed 2026, delta = (candidate
vs 8 × nm31 − nm31 vs 8 × candidate) / 2, Holm over the two:

| Candidate | Delta, bb/100 | 95% interval | Holm p | Verdict |
|---|---|---|---|---|
| nm32, Deep Sets | −0.10 | [−0.37, +0.17] | 0.46 | level with the baseline |
| nm33, EMD buckets | −0.30 | [−0.55, −0.05] | 0.036 | not better; probably worse |

Neither is adopted. Two deviations, recorded with the result: the compared checkpoints are at 565-585 million hands
(the trainer saved only at evaluations, so the last ~5% was lost at the hand limit; now fixed, and all three lost
about the same), and the match tool rounded to 0.1 bb/100 (nm33's z lies between about −2.0 and −2.8; not re-run, as
registered; the tool now keeps 4 places). The dashboard impression that nm33 played very well came from a handful of
eventful hands; 40,000 paired deals say otherwise, which is what a fixed plan is for.

The secondary measures, read after the primary (2026-11-18): the 30-minute exploiter finds nothing in nm31 (−0.0
[−0.6, +0.6]) but holes in nm32 (+1.9 [+1.4, +2.5]) and nm33 (+1.0 [+0.5, +1.5]), so both candidates are also more
exploitable; the copy checks read +0.4, +0.3 and −0.1. Poker sense (before / after the flop, of 16 / 12): nm31 7 / 9,
nm32 9 / 10, nm33 9 / 9, all below nm25 (11 / 12) after only 565-585 million hands from scratch.

Why the EMD buckets may not help here: at 10-25bb most decisions are before the flop or all-in on it, where future
strength distributions matter least; and the network already sees each street's bucket through its own embedding, so
it can learn draws from the bucket sequence itself. The Deep Sets network learns the same game with weights shared
across seats; at equal hands it neither gains nor loses, so its value (tables of any size alike) is untested here.

**nm34 against nm35** (the 6-max cash fixes, 12 hours each from nm28; nm35 got fewer hands on the slower GPU, about
1.9 against 2.8 billion), by the 30-minute exploiter (copy check at minute 0 in brackets):

| Model | Copy | Exploiter after 30 min |
|---|---|---|
| nm28 | −1.4 | +11.0 [+6.7, +15.2] |
| nm34, bigger exploiter | +1.6 | +8.2 [+4.4, +12.0] |
| nm35, league | −1.0 | +7.0 [+3.9, +10.0] |

Both shrank the hole by about a third, the league slightly more on fewer hands, but the intervals overlap and neither
closes it. nm35 read +2.1 at 20 minutes: the exploiter needed the full 30 to find the leak, which is why the gate is
the 30-minute reading. Next: nm35 continued with the league for another 12 hours on the faster GPU.
