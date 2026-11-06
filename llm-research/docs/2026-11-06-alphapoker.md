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

RESULTS_TABLE

FINDINGS

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
