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
