# Reading list from Slumbot NL's citations (2026-11-24)

A starting point, not a review: the papers that cite or are cited by Eric Jackson's *Slumbot NL: Solving Large Games
with Counterfactual Regret Minimization Using Sampling and Distributed Processing* (AAAI workshop on computer poker,
2013), checked against what this repo already covers. The source is Semantic Scholar's citation and reference lists for
that paper (29 citations, 10 references). Neither Slumbot NL nor any paper below has been read in full for this note.
Automated fetches of Semantic Scholar were refused (HTTP 403 on the page, 429 on the API), so the summaries here are the
listing's own one-line descriptions, and no numbers from inside the papers are quoted.

## Why Slumbot NL matters here

Slumbot is the opponent in A4 (run2 against Slumbot over 20,000+ hands; see `2026-11-23-review-and-plan.md`). The paper
describes how that bot was built: an abstraction of heads-up no-limit, solved with sampled CFR distributed across
machines. Knowing its abstraction (card buckets, allowed bet sizes) tells us where its strategy is coarse, and so how to
read an A4 result. It does not change how the 9-max networks are trained. To do: read the paper itself (it needs a PDF
link that can be fetched, e.g. from aaai.org) and add its abstraction and solve size here.

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
2. Read Slumbot NL for its abstraction before A4 is run.
