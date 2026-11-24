# Slumbot NL, DyypHoldem, and a reading list (2026-11-24)

Two parts. First, Eric Jackson's *Slumbot NL: Solving Large Games with Counterfactual Regret Minimization Using Sampling
and Distributed Processing* (Computer Poker and Imperfect Information, AAAI 2013 workshop, pp. 35-38), read in full
from the AAAI proceedings PDF (`cdn.aaai.org/ocs/ws/ws0979/7044-30516-1-PB.pdf`), set against our heads-up blueprint.
Second, the papers that cite it or that it cites (Semantic Scholar lists 29 and 10), checked against what this repo
already covers. Those second-part papers have not been read in full: their summaries are the listing's one-line
descriptions, and no numbers from inside them are quoted.

## Slumbot NL (2013), read in full

| | Slumbot NL (2013) | Our heads-up blueprint (`2026-11-07-blueprint-plan.md`) |
|---|---|---|
| Game | Heads-up no-limit, blinds 50/100, 200bb stacks (about 10^164 states, Johanson 2013) | Heads-up no-limit, 200bb (the same game as Slumbot's public API) |
| Solver | CFR with a variant of Public Chance Sampling: about 50 boards an iteration (1/50,000 of the boards), not PCS's one | Vector CFR with DCFR and per-bucket updates (run 7) |
| Board order | Quasi-Monte Carlo: walk a random permutation of all ~2.5 M five-card boards, so none repeats before all are seen (training ended inside the first cycle) | Independent random deals |
| Card abstraction | Imperfect recall, hierarchical: public (board) buckets first, then hand buckets inside each, by k-means on strength and potential features; **3,904 flop, 3,602 turn, 2,173 river**, 169 preflop | 200 buckets a street |
| Betting abstraction | Pot fractions. **First bet on a street: 11 sizes** (0.25, 0.5, 0.75, 1, 1.5, 2, 4, 8, 15, 25, 50); **raise: 8** (0.5, 1, 2, 4, 8, 15, 25, 50); **3-bet: 3** (0.5, 1, 2); 4-bet and later: pot only; all-in always. Symmetric for both players | Open 2-3bb; 3-bet 3x/4x; postflop 1/3, 2/3, 1x, 2x pot; all-in |
| Off-tree bets | Ganzfried's pseudo-harmonic mapping (2013) | Action translation, plus the river and turn re-solve |
| Size | 5.7 billion information sets, 14.5 billion set-action pairs, about 6 million betting sequences | Far smaller |
| Hardware | Regrets and average strategy kept **on disk**, not in RAM; tree split by the 835 preflop sequences x 8 board groups = 6,680 river tasks over 9 machines, plus one preflop task on a central machine; each task stays on one machine, so only reach probabilities and flop values cross the network | One lab machine |
| Results | None reported in the paper | v5 vs Slumbot's API, luck-adjusted, about -14 bb/100 [-45, +17] (`2026-11-07-edges-and-real-tests.md`) |

What it means for us:

- **The paper's bot is not necessarily the bot behind today's public API.** Jackson kept developing Slumbot after
  2013. Treat these numbers as the design of the line the API comes from, not as a spec of the opponent in A4.
- **Its capacity went into bet sizes, not cards.** Jackson's stated judgement: a good card abstraction needs relatively
  few buckets, but telling bet sizes apart is vital. 11 first-bet sizes against our 4 postflop sizes is where it most
  differs from our blueprint. LBR's wins against run 7 (it folds 45-55% of bets it faces, `2026-11-07-blueprint-plan.md`)
  point the same way: our weak spot is the response to bets, and that depends on the betting tree.
- **Overbets.** Slumbot can bet 4x to 50x pot as a first bet. Our tree's largest non-all-in bet is 2x pot, so any bigger
  bet is translated or re-solved. A4 should record how often Slumbot bets more than 2x pot, and what we do then.
- **Two cheap tricks worth trying in `mccfr_vec`:** several boards an iteration (Jackson's rule of thumb: enough that
  almost every bucket gets an update each iteration), and walking a shuffled permutation of boards instead of drawing
  them at random, which removes short-run board-sampling bias (his example: too many boards containing a 5 overvalue
  every hand holding a 5).

## What we adopt

Checked against `research/blueprint/mccfr_vec.cpp` and Kestrel. Already have: one board per iteration with every hand
updated at once (Public Chance Sampling), skipping 95% of negative-regret subtrees (regret-based pruning), and the
pseudo-harmonic mapping for off-tree bets (`kestrel/core/src/headsup.cpp`).

| # | Adopt | Why | Cost | Status |
|---|---|---|---|---|
| 1 | **Shuffled flop cycle** in `mccfr_vec`: walk a permutation of all boards, split across threads, instead of each thread drawing its own (today `std::random_device` seeds per thread, so boards repeat and others wait) | Removes short-run board bias for free | Small code change | Built (`--flop-cycle`); on trial |
| 2 | **Bet sizes over card buckets**: add a small first bet (0.25 pot) and overbets (4x, maybe 8x) to the heads-up tree, before any more buckets | Run 7's finer card abstraction did not lower LBR, which matches Jackson's judgement; our leak is folding 45-55% of bets faced, a betting-tree question | Tree grows 2.9x; needs a fresh solve | Built (`BLUEPRINT_TREE=wide`: adds 0.25x and 4x pot first bets); on trial |
| 3 | Several boards per update | Jackson's reason was disk cost; in memory the gain is unclear | One more table | Built (`--batch-boards B`); on trial |
| 4 | Board-first (hierarchical) buckets | Our distribution-aware buckets did not help | | Other ways to group cards, below; a diagnostic decides first |
| 5 | Disk-based, multi-machine solving | The lab has the memory | | Skip |
| 6 | DyypHoldem code | No licence; models gone | | Code no. The idea (value networks at re-solve leaves): test plan below |
| 7 | AV-AIVAT luck correction for the 9-max checks | Could cut the 40,000-deal checks | Read first | Read next |

### Built and on trial (2026-11-24)

All three are options; with none set, `mccfr_vec` and the tree behave exactly as before.

- **`--flop-cycle [--board-seed S]`**: flops come from a shuffled cycle of all 22,100, shared by every thread
  through one counter; turn and river stay random. The cycle order is a keyed permutation (a 4-round Feistel network
  on 15 bits, walked until it lands below 22,100), so it needs no table, and each cycle uses a new key. The seed is
  kept in `OUTDIR/board_seed.txt` so a resumed run continues. Why the flop and not the whole board: there are about
  52 M ordered boards (flop, turn, river), and a run deals a few hundred thousand to a couple of million of them, so a
  cycle over whole boards would be indistinguishable from random draws. At about 0.75 boards a second per core, a
  4-core run goes through every flop in about 2 hours, so a flop cycle does change what is dealt. `--bench` checks
  that a cycle is a permutation of all 22,100 flops and that two cycles' orders differ.
- **`--batch-boards B`**: regret increments collect in a third table and are added into the regrets at a barrier every
  B boards (all threads), so the strategy is fixed across B boards, as in Slumbot NL (about 50 boards per update). The
  barrier is the one the discount and checkpoints already use; the batch is flushed before either, so neither misses
  increments. Costs one more table the size of the regrets.
- **`BLUEPRINT_TREE=wide`** (environment, read by `tree.hpp`): adds a quarter-pot and a 4x-pot first bet after the flop.
  The tree grows from 277,406 nodes and 55.5 M table entries to 795,510 nodes and 159.1 M. Every program that builds
  the tree (`mccfr_vec`, `br_vec`, `resolve`) must be run with the same setting as the run it reads, and each prints
  the tree it built; `blueprint_bot.py` reads `tree.tsv` and needs nothing.

The trials, on the lab, on the 200-bucket abstraction (`out/`), all with DCFR:

| Trial | Arms | Budget | Score |
|---|---|---|---|
| Sampling | random boards; `--flop-cycle`; `--batch-boards 50`; two runs each | 4 threads, 180 minutes each | `br_vec` (2,000 boards, 3,000 evaluation boards, purify 0.05) |
| Tree | standard; wide; one run each | 8 threads, 360 minutes each | `br_vec` inside each tree (not comparable across trees), then LBR over 30,000 hands in the real game (comparable) |

Pass rules, set before the results: a sampling option is adopted if both its runs are less exploitable than both
random-board runs; otherwise it stays an option. The wide tree is adopted if its LBR interval lies below the standard
tree's estimate at the same compute; if the intervals overlap, the result is "not shown" and the overbet count from A4
decides whether a longer wide solve is worth it.

## Other ways to group cards

Slumbot NL's board-first buckets are one option; run 7's finer distribution-aware buckets did not lower LBR. Before
building another abstraction, find out whether cards are the leak at all.

**What we have** (both imperfect recall, 169 pre-flop classes):

| | Percentile (`abstraction.cpp`, `out/`, 200 a street) | Distribution-aware (`abstraction_pa.cpp`, `out_pa/`, 500 a street) |
|---|---|---|
| River | hand strength against all opponent hands | hand strength against 8 opponent groups (OCHS) |
| Turn | mean squared strength over the rivers | 50-bin histogram of river strength |
| Flop | mean of the turn values | 50-bin histogram of turn strength |
| Grouping | equal-population cuts | k-means (squared distance between histograms, standing in for EMD) |

`vec_init` gives every street the largest street's count, so per-street counts work but cost memory at the maximum.

**A finding about LBR:** it only ever bets the pot or all-in (`scripts/lbr.py`), and both are on the blueprint's menu.
So LBR has never tested bet translation, and against a pot bet folding about 50% is the defence the price requires.
The 45-55% fold rate is therefore not by itself a leak; the question is which hands fold, which is a card question.
`LBR_SIZES` (8547f1e) lets LBR bet off-menu sizes, so translation can now be measured.

**The diagnostic (queued on the lab, starts when the sampling trial ends):** on run 2, `br_vec` inside the abstraction,
and LBR over the same 30,000 deals with on-menu sizes (pot, all-in) and with off-menu sizes (0.5, 0.75 and 1.5 pot,
all-in). Run 7's tables are no longer on the lab, so run 2 is the only blueprint measured. Rules, set before results:
- off-menu LBR beats on-menu LBR by more than 20 bb/100 with separate intervals: translation is a material leak, and
  the wide tree comes before any card work;
- `br_vec` is at least half of LBR: the solve has not converged, so work on training, not buckets;
- `br_vec` under 10 and translation not material: the gap is card resolution, and the card options below are next;
- otherwise: not shown, and card abstraction is left alone.

**Card options, ranked** (from the literature as recalled, not re-read; no numbers quoted from papers):

| # | Option | Why it might help where run 7 did not | Cost |
|---|---|---|---|
| 1 | Most buckets on the river (e.g. 200 flop, 500 turn, 3,000 river) | LBR's wins come from bets folded to, most on the river; run 7 spread buckets evenly | Small: a few lines in `vec_init` for per-street counts, one abstraction pass, one solve |
| 2 | Asymmetric abstraction: a finer best responder than the player (also a `br_vec` diagnostic of card loss) | Measures card loss inside the tree directly | Moderate |
| 3 | Board-texture features (flush, pairs, straights) added to the strength features | A soft form of Slumbot's board-first grouping in a single table | Small |
| 4 | Potential-aware flop buckets: histograms over turn buckets with real EMD (Ganzfried and Sandholm 2014) | The step beyond our histograms of a single strength number | Large |
| 5 | More OCHS opponent groups (16-32, or groups from the blueprint's own river ranges) | 8 fixed groups may not match the ranges reached | Trivial |
| 6 | Learned groupings from the networks' per-hand values | Groups hands by how they should be played, not by equity | Moderate; untested anywhere we know of |

## Value networks at the re-solve leaves (DeepStack's idea), test plan

**Today (from the code):** our safe re-solve is river only (`resolve.cpp` refuses any other street) and plays exactly
to the end of the hand, so no leaf values exist anywhere. The only turn re-solve is TexasSolver, unsafe, with one bet
and one raise size. On 200 river spots the safe re-solve lowers the opponent's best response by 161 ± 17 chips a hand
pair; LBR with it reads +105 [+42, +168], level with the blueprint (`research/blueprint/model.md` 5.4).

**The test is on the turn**, where exact answers are still affordable (48 river cards), so a network's error can be
measured against ground truth before it is trusted on the flop. A safe turn re-solve (the same gadget, with the
opponent's values from a best response to the blueprint over turn and river), with the river handled four ways:

| Arm | River | Role |
|---|---|---|
| B | TexasSolver's unsafe turn re-solve | today |
| C | every river card solved exactly | ground truth |
| D | blueprint values at the leaves | the cheap alternative |
| E | river value network at the leaves | DeepStack's idea |

**The network:** inputs are both players' ranges over river buckets, the pot relative to the stack, and board features;
outputs are counterfactual values per bucket for both players, as fractions of the pot, with DeepStack's zero-sum
correction. Data: river spots solved by `resolve.cpp` with generated ranges, one fifth taken from ranges blueprint
self-play actually reaches, boards from a shuffled cycle; 100 k, 300 k, then 1 M samples (DyypHoldem's maintainer
cites 1 M a street for the original models). The estimate is a few hundred core-hours of solving and a few GPU-hours of
training; time 1,000 river solves first to replace the estimate with a measurement.

| Stage | Work | Go on if |
|---|---|---|
| 0 | River network: data, training, held-out checks | error on blueprint-reached ranges under about 2-3% of the pot; zero-sum residual under 0.5% of the pot before the correction; the learning curve has flattened |
| 1a | Safe turn re-solve, arms C, D, E on 200 turn spots, each scored by an exact best response | E within C's standard error. If D is level with C, blueprint leaves suffice: ship D and stop the network work |
| 1b | LBR (30,000 hands), then Slumbot (20,000 hands): B against the better of C and E | LBR falls by more than about 30 bb/100 from +105 |
| 2 | Turn network, flop re-solve stopped at the turn | flop spots improve over blueprint leaves, and LBR's flop wins shrink |

**First step either way:** arm C alone, a safe turn re-solve with exact rivers. It needs no network, it is the next
step `model.md` already names, and it is the baseline every other arm is judged against.

**Expectations:** a re-solve over the same bet menu cannot fix a menu leak, so the turn re-solve should carry the wide
tree's sizes too. Network error compounds street by street, and production Kestrel has no GPU path. DeepStack never
claimed to beat Slumbot (DyypHoldem issue #4); the realistic best case is LBR lower by tens of bb/100.

## DyypHoldem (checked 2026-11-24)

`github.com/lucky72s/dyypholdem`: a Python/PyTorch port of DeepHoldem, which is the no-limit hold'em version of DeepStack
(continual re-solving with counterfactual value networks for flop, turn and river, plus a preflop auxiliary network). It
ships ACPC players and a player for Slumbot's API. Last push 2022-01-03; 30 stars.

- **It cannot be run as published.** The trained networks and the bucketing data were in Git LFS, and GitHub now answers
  HTTP 410 for them (issues #7, 2025-03, and #8, 2026-08, both open and unanswered). Running it means regenerating
  every street's training data and retraining; a maintainer's comment says the original models used 1 M samples a
  street, against a default of 100 k.
- **No measured strength.** The only result in the repo is one user's 1,803 hands against Slumbot (issue #4): -15.9
  bb/100, "baseline" -24.5 bb/100, with no interval. At heads-up variance that sample cannot tell -16 from 0 (our own
  3,000-hand Slumbot runs had intervals about 120 bb/100 wide). A reply there says the original DeepStack did not claim
  to beat Slumbot.
- **No licence.** The repository states none, so its code cannot be copied into this public repo. DeepHoldem, which it
  ports, also states none. Reading it for ideas is fine.
- **Where it touches our work:** DeepStack-style re-solving at every decision is the general form of what Kestrel's
  river and turn re-solve do with a blueprint at the leaves. DyypHoldem uses value networks at the leaves instead,
  which is how DeepStack avoids needing a full-game blueprint. It is a possible extra heads-up opponent for A4 only if
  someone retrains it, and that cost (data generation for three streets) outweighs what one more bot of unknown
  strength would tell us.

## Why it matters here

Slumbot is the opponent in A4 (run2 against Slumbot over 20,000+ hands; see `2026-11-23-review-and-plan.md`). The paper
describes how that bot's line began (table above). Knowing its abstraction tells us where its strategy is fine and where
it is coarse, and so how to read an A4 result. It does not change how the 9-max networks are trained.

## New to the repo, worth reading

| Paper | What it claims (listing's summary) | What it could change here |
|---|---|---|
| **AV-AIVAT: 74x Cheaper Agent Evaluation with Certified Anytime-Valid Stopping in Imperfect-Information Games** (Li, Chen, Huang; arXiv 2026) | AIVAT-style variance reduction combined with a sequential test that stays valid whenever it stops, plus everything a third party needs to recheck the verdict at the stopping time | The pipeline's checks (`research/scripts/pipeline.py`, `seqtest.py`) score raw chips; nm45's no-harm check ran 40,000 deals. A luck-corrected score with anytime-valid stopping could end checks much sooner. Kestrel has heads-up luck correction (`luck.cpp`); the 9-max pipeline has none. The highest-value item on this list. |
| **Approximate Exploitability: Learning a Best Response** (Timbers, Bard et al.; IJCAI 2022) | ISMCTS-BR: a search-based deep RL best response that approximates an agent's worst case | It is the same idea as our league exploiter and the exploiter check. Read it for how they keep the estimate honest (training budget against estimated exploitability) before trusting "the exploiter cannot beat it" as a bound. |
| **OpenHoldem** (Li, Xu et al.; arXiv 2020) | A standard evaluation protocol for NLHE agents, three public baselines and an online test platform | Possible extra heads-up opponents for A4 alongside Slumbot. |

## Already covered

| Paper | Where |
|---|---|
| Brown, Ganzfried, Sandholm, *Hierarchical Abstraction, Distributed Equilibrium Computation, and Post-Processing* (2015) | Purification and thresholding in `2026-11-08-blueprint-speed-accuracy.md`; Kestrel purifies at 0.05 |
| Brown, Sandholm, *Regret-Based Pruning* | `2026-11-08-blueprint-speed-accuracy.md` |
| Li, Wang, Huang, *PokerSkill* (2026); Provost et al., *GTO Wizard Benchmark* (2026) | `2026-11-23-llm-poker-literature.md` |
| Schmid et al., *Player of Games* / *Student of Games* | `2026-11-06-alphapoker.md`, `2026-11-07-blueprint-plan.md` |
| Tammelin et al., *Solving Heads-Up Limit Texas Hold'em* (Cepheus, CFR+) | Background; limit, not no-limit |
| Slumbot NL's own references: Zinkevich et al. (CFR, 2007); Lanctot et al. (MCCFR, 2009); Johanson et al. (efficient MCCFR, 2012); Gibson et al. (generalised sampling, 2012); Waugh et al. (imperfect recall, 2009); Johanson (*Measuring the Size of Large No-Limit Poker Games*, 2013); Ganzfried and Sandholm (pseudo-harmonic action translation, 2013) | The foundations of the heads-up blueprint; see `2026-11-07-blueprint-plan.md` |

## Next steps

1. Get the AV-AIVAT PDF and write up how its estimator would apply to a 9-player table (AIVAT needs a value function
   for each seat; the networks' value heads could serve) and how many deals it would have saved on nm45's checks.
2. In A4, log every Slumbot bet above 2x pot and our response to it.
3. Try multi-board sampling and a shuffled board cycle in `mccfr_vec` on a small blueprint, scored by LBR.
