# AlphaPoker — the AlphaGo recipe applied to poker — 2026-11-06

AlphaGo beat the best Go players with three parts: a **policy network** that proposes moves, a **value network**
that judges positions, and **Monte Carlo tree search (MCTS)** that combines them, all improved by **self-play**.
AlphaZero dropped the human data and kept one network plus search. This study asks which of those parts carry over
to poker, tests each one exactly on a small poker game, and maps the result onto the stack we already have.

## Why AlphaGo does not carry over directly

Go and chess have **perfect information**: the state on the board is all there is, so a position has one value and
search can look ahead from it. Poker hides the opponents' cards, and that breaks three things search relies on:

- **A state has no value of its own.** What a hand is worth depends on what the opponent believes you hold, which
  depends on how you would have played every hand you could hold. Values belong to *beliefs*, not to positions.
- **Good strategies must mix.** Bluffing and slow-playing at the right frequencies is what makes a strategy hard to
  exploit. A search that picks the single best move in each spot is deterministic, so it can be read and beaten.
- **Strategy fusion.** Searching each guessed deal as if the cards were visible (perfect-information Monte Carlo)
  lets the search play differently in deals the player cannot tell apart. The result plays as if it could see.

## What worked for poker

| System | Year | Idea | What it took the AlphaGo recipe further with |
|---|---|---|---|
| DeepStack (Moravčík et al., *Science*) | 2017 | Re-solve the game at every decision, looking a few actions ahead; a deep network estimates the value of the rest | Value network over *beliefs* (ranges), not positions |
| Libratus (Brown & Sandholm, *Science*) | 2017 | A coarse blueprint strategy from counterfactual regret minimisation (CFR), refined by solving the current subgame in real time | Search as *subgame solving*, which keeps the strategy balanced |
| Pluribus (Brown & Sandholm, *Science*) | 2019 | Six-player: a blueprint from Monte Carlo CFR self-play, then depth-limited search; trained in about 8 days on one 64-core server | Self-play that *averages* strategies, plus search that assumes opponents may switch strategies |
| ReBeL (Brown et al., NeurIPS) | 2020 | Self-play reinforcement learning plus search over *public belief states*; proven to converge to an equilibrium in two-player zero-sum games | The closest thing to "AlphaZero for poker": one value/policy net, search, self-play |
| AlphaHoldem (Zhao et al., AAAI) | 2022 | End-to-end reinforcement learning with no search at play time, heads-up no-limit | Fast decisions from the network alone |
| Student of Games (Schmid et al., *Science Advances*) | 2023 | One algorithm (guided search + self-play + game-theoretic reasoning) for chess, Go and poker | Unifies the two families |

The common thread: **search and self-play survive, but both must reason about mixed strategies and beliefs.** CFR
(and its sampled version, MCCFR) supplies that; plain MCTS and plain reinforcement learning do not.

## The test: every piece, exactly, on Leduc poker

Leduc hold'em has 6 cards, one private card each, one public card and two betting rounds: 936 decision points,
small enough to compute **exact exploitability**, how many chips a perfect opponent wins per hand against a
strategy (0 = unbeatable). Every method below is turned into a full strategy and scored the same way, with
DeepMind's OpenSpiel (`scripts/alphapoker.py`, `results/alphapoker/report.json`).

Exploitability in milli-chips per hand (the ante is 1 chip; a perfect opponent wins this much from the strategy
on average; 0 = unbeatable). Run on the lab (`results/alphapoker/report.json`).

| Method | What it is | Exploitability |
|---|---|---|
| Random | Uniform over legal actions | 2,374 |
| AlphaGo search, PIMC | Sample the hidden cards, perfect-information MCTS in each world (20 x 200 simulations) | 2,018 |
| AlphaGo search, IS-MCTS | MCTS over information sets, 4,000 simulations a decision | 809 |
| Self-play RL (Q-learning) | Two learners, 1M games, no search, no averaging | 892 (another run: 1,975) |
| Fictitious self-play | Best-respond to the opponent's *average* strategy, 300 iterations | 133 |
| MCCFR (sampled self-play) | Regret matching on sampled self-play games, 1M games, 110 s | 140 |
| **CFR+** | The full solver method, 1,000 iterations | **0.3** |
| AlphaPoker v0, network 16 units | MCCFR teacher compressed into a policy network, 547 parameters | 1,785 |
| AlphaPoker v0, network 64 units | 2,179 parameters | 710 |
| AlphaPoker v0, network 128+64 | 12,419 parameters | 352 |
| AlphaPoker v0 + AlphaGo search | 64-unit network as the prior for PUCT information-set search | 527 |

Learning curves (exploitability as training grows):

| Method | Curve |
|---|---|
| Q-learning, 10k / 100k / 1M games | 2,288 / 1,904 / 892; the previous run 2,725 / 1,163 / 1,975 |
| MCCFR, 1k / 10k / 100k / 1M games | 2,065 / 1,145 / 498 / 140 |
| Fictitious play, 10 / 30 / 100 / 300 iterations | 1,174 / 544 / 250 / 133 |
| CFR+, 10 / 30 / 100 / 300 / 1,000 iterations | 610 / 82 / 13 / 2.3 / 0.3 |

What the test shows:

- **AlphaGo's search does not carry over as it is.** Guessing the cards and searching each guess (PIMC) is barely
  better than random, the strategy-fusion problem. Searching over information sets (IS-MCTS) is much better but
  still loses 0.8 chips a hand to a perfect opponent, because search that picks one best move cannot mix.
- **Self-play without averaging does not converge.** Q-learning's strategy got better and then worse again, and two
  identical runs ended at 892 and 1,975: each learner chases the other's current strategy.
- **Self-play that averages does converge,** steadily, as theory says: fictitious play and sampled CFR fall with
  every tenfold increase in games, and CFR+ reaches 0.3, effectively unbeatable. This is the learning rule behind
  Libratus and Pluribus, and the one AlphaPoker must use in place of AlphaZero's.
- **A network compresses the self-play strategy at a cost** that shrinks with size: 547 parameters keep little
  (1,785), 12,419 parameters keep most of it (352 against the teacher's 140). Hold'em is far larger than Leduc, so a
  network alone will lose more; this is why DeepStack and ReBeL re-solve the current spot at play time.
- **AlphaGo-style search on top of the network helps a little and caps it.** The 64-unit network plus PUCT search
  scored 527, better than the network alone (710) and than search alone (809), but worse than the bigger network
  and far from CFR+. Search for poker has to be CFR re-solving (as TexasSolver does on our turns and rivers), not
  MCTS.

**AlphaPoker therefore means:** CFR-family self-play (not greedy RL) to build the strategy, networks to compress it
and to value positions over beliefs, and CFR re-solving (not MCTS) at play time, with the LLM reading the table and
feeding the solver.

## AlphaPoker for hold'em: the design, mapped to what exists

| AlphaGo part | AlphaPoker version | What we have | What is missing |
|---|---|---|---|
| Policy network | Blueprint strategy over ranges | Push/fold equilibrium (fictitious play), deep pre-flop strategy, flop network trained on 400 solves (81% top-action agreement) | A network over *all* streets and multi-way pots |
| Value network | Counterfactual values of a range, DeepStack style | Not yet | Train on the solver library: inputs board + both ranges, outputs per-hand values |
| MCTS | Subgame re-solving with CFR | TexasSolver live on turn and river (4-6 s), flop in 41 s on the optimised build | Depth-limited re-solving that stops at the value network instead of the river |
| Self-play | Sampled CFR self-play, not greedy RL | RL agent (Monte Carlo control) and the bot simulations | MCCFR self-play on an abstracted hold'em game to grow the blueprint |
| — | **The LLM**: reads the table, feeds the solver, explains | The solver-acts flow (`--play`) | Screen reading (OCR) into the same flow, on play-money tables only |

Order of work this points to: the value network from the solver library first (it turns 41 s flop solves into
depth-limited solves of a second or two), then MCCFR self-play for a full-game blueprint.
