# A poker foundation model, after Kronos (2026-11-27)

The idea (requested 2026-11-27): what Kronos (github.com/shiyu-coder/Kronos, MIT) does for candlesticks, for poker. Kronos
turns OHLCV data into discrete tokens with a tokenizer, then pretrains a family of decoder-only transformers (4.1 M to
499.2 M parameters) on data from over 45 exchanges by next-token prediction. The poker version turns hands into
tokens and pretrains on every hand we can get: our networks' self-play (every card known), the HandHQ human hands and
the Pluribus hands.

**What it is for, and what it is not.** Predicting what players do is not playing well: every system that has beaten
professionals got its strength from game-theoretic training and search (Libratus, Pluribus, DeepStack, ReBeL, Student
of Games), with plain feed-forward networks. A pretrained model of hands is for what feeds that recipe:
- an opponent model: a player's next action from their recent hands (the basis of bounded exploitation);
- a belief model: hidden cards from the actions, against `ninemax_belief.py` (0.73 bits better than a uniform guess);
- a start for fine-tuning, and realistic training opponents.

## Stage 0: the hand language and the data

`scripts/poker_tokens.py` writes a hand (from any source) as tokens from a fixed vocabulary of 166:

```
BOS SRC_<self|human|pluribus> GAME_<tourney|cash> N_<players> ANTE_<0|1>  (POS_<p> STK_<b>) x n
[full layout: HOLE POS_<p> card card ...]
PRE (POS_<p> <action>)*  FLOP c c c (POS_<p> <action>)*  TURN c ...  RIVER c ...
[SHOW POS_<p> card card]*  [REVEAL (POS_<p> card card)*]  END (POS_<p> RES_<b>)*  EOS
```

- Positions are Kestrel's names (SB, BB, then the last seats of UTG ... BTN), so every table size reads alike.
- Stacks in 18 log-spaced buckets of big blinds (STK_U: not recorded, a third of HandHQ's hands).
- Actions: FOLD, CHECK, ALLIN, and CALL, BET or RAISE with the chips put in beyond the bet before it over the pot then,
  in 13 buckets (0.1, 0.2, 0.33, 0.5, 0.66, 0.8, 1, 1.25, 1.5, 2, 3, 5 pot).
- Results: each seat's result in big blinds, 17 signed buckets.
- Two layouts: "public" (cards only when shown; when every card is known, REVEAL lists them after the hand, so a model
  learns what the actions said about the cards) and "full" (cards first). Self-play and Pluribus hands are written in
  both; human hands, whose cards are mostly unknown, in "public".

`scripts/fm_data.py` builds shards of token ids (uint16, hands back to back, BOS to EOS, at most 50 M tokens each):
`selfplay RUN` plays the network against itself (the engine now records every action: `Tables.ev`), `phh DIRS` reads
PHH hand histories in parallel, `stats` counts and decodes examples.

First builds (lab, `~/scratch/fm/data`):

| Source | Hands read | Unreadable | Tokens a hand |
|---|---|---|---|
| Pluribus (10,000 hands, both layouts) | 20,000 sequences | 0 | 63.7 |
| nm25 self-play (20,000 hands, both layouts) | 40,000 sequences | 0 | 92.7 |
| HandHQ, 40 files | 38,098 | 1,672 (4%: straddles and posted blinds) | 47.0 |

Full build (data2): HandHQ 20,879,811 hands (1.00 B tokens, 48 a hand; 726,276 unreadable, mostly straddles),
1 M nm25 self-play hands (185 M tokens, both layouts), Pluribus 10,000 (1.5 M tokens).

**Simulations and Kestrel's tables** (added 2026-11-27). The vocabulary grows to 187 (appended, so earlier ids hold):
SRC_sim, SRC_kestrel and a player token per seat, WHO_<name>, when it is known (the simulations' personas bob, nora,
tina, tinaplus, max, carl, solverbot, sol (TexasSolver), monte, the RL bots rlbot and rlavg, generated styles; our
networks; a person at a practice table), so a model can learn styles and be checked on recognising them.
- `fm_data.py botsim` reads research's ten botsim runs (9-handed 100bb, short stacks, heads-up): about 120,000 hands
  with every player's cards (their end rows) and payoffs, 2% unreadable. Their rows record the table after each
  action, and a call that closes a street already shows the next street's empty bets, so the chips put in are read from
  each stack's fall and the pot and street bets kept by the converter.
- `fm_data.py kestrel` reads Kestrel's practice tables (the container's /data/sessions): 57,450 hands with results,
  models' and people's seats tagged.
- More self-play: 200,000 hands each from nm17, nm45 and nm46.

## Stage 1, the first model (m1, 2026-11-27)

25.6 M parameters (d 512, 8 layers, 8 heads, context 512), 2.0 B tokens of data2 (self-play 40%, human 58%, Pluribus
2%), 42 minutes on the RTX PRO 6000 (`results/foundation_model/m1_result.json`, `m1_train.log`). Held out: the first
1 M tokens of each source's first shard. Bits per action token, against a count model of the action given the actor's
position and the street:

| Held out | Model | Count model | Uncertainty removed |
|---|---|---|---|
| Human (HandHQ), 165,447 actions | **1.342** | 2.672 | 50% |
| Self-play (nm25), 96,323 | 0.468 | 1.398 | 67% |
| Pluribus against professionals, 36,694 | 0.739 | 2.284 | 68% |

On real players it halves the uncertainty about the next action: the opponent-model signal stage 1 tests for.

The cards after REVEAL (every player's, after the whole hand): 4.31 bits a card on self-play against 5.35 uniform,
4.80 against 5.45 on Pluribus. Part of that is free: at a showdown the shown cards come just before REVEAL and can be
copied, and human hands carry REVEAL only when every card was shown (0.94 bits: copying). So this is not comparable
with the belief network's 0.98 bits an opponent, which are read mid-hand at a decision; that test is stage 2's.

## Stage 1, the second model (m2): does variety help on real players?

The same model and budget (2.0 B tokens, 44 minutes) on data3: data2 plus the simulations, Kestrel's tables and
200,000 self-play hands each from nm17, nm45 and nm46 (mix: self-play 33%, human 55%, simulations 7%, Kestrel 3%,
Pluribus 2%). The question set before it ran: do they improve prediction of real players, on the same held-out hands?

| Held out, bits an action | m1 | m2 |
|---|---|---|
| **Human (HandHQ), 165,447 actions** | **1.342** | **1.342** |
| Pluribus against professionals, 36,694 | 0.739 | 0.732 |
| Simulations (personas tagged), 140,253 | — | 0.461 (count model 3.201) |
| Kestrel's practice tables, 91,241 | — | 0.640 (count model 2.051) |

**No**: the two are level on human hands to three decimals, and m2 gains a negligible 0.007 on Pluribus. The variety
did not transfer to people: human hands are already most of the data (1.0 B tokens), and our bots do not play like
2009 online players. m2 adds what the new sources hold: it predicts the simulated bots almost exactly (0.46 bits against
3.20, the player token lets it learn each persona) and reads Kestrel's tables, the partner's practice hands among them.
So m2 is the general-purpose opponent model to carry forward; better prediction of people needs more model, more
human data, or what the opponent model is for in the end: a player's own earlier hands as context.

## Ideas (2026-11-28), and which are picked

The user's list of ideas to pick from, with an assessment of each:

| Idea | Assessment | Picked |
|---|---|---|
| **Long-horizon opponent modelling**: a player's own earlier hands as context | The opponent model's real job (exploiting a player means knowing that player). HandHQ keeps each player's anonymised id across hands, so sequences of one player's consecutive hands can be built; the test is whether the last 10-40 hands lower the bits of that player's next action. Also the natural test of long-context architectures | **1st** |
| **Token arrangement**: suit-canonical cards (A♠K♠ and A♥K♥ read alike), one fused token per action (position, kind, size) | Cheap; suit isomorphism is a known sample-efficiency gain; fused tokens shorten hands about 3 times, so more hands fit in a context | **2nd** |
| **Mamba** (a state-space sequence model: linear time, constant memory), and a **hybrid** of Mamba blocks with a few attention layers | Suited to session-long histories (item 1); pure state-space models recall exact facts worse than attention, hence the hybrid for the current hand. The fast kernels (`mamba-ssm`) need a CUDA compiler the lab lacks; a pure PyTorch version runs slower. Built and compared once item 1's data exists | **3rd** |
| **Self-play RL of the transformer**: a policy head on the pretrained model, trained in the 9-max engine with NeuRD, judged by the pipeline | The largest possible gain and the largest build (the engine must write the token sequence at every decision). The small version exists: `--arch attn` (attention over the hand's actions and seats) trains at nm25's level. Done after 1-3 show which model to train | **4th** |
| **Different encoders and decoders** | Folded into 2 and 3 (what a token holds; which backbone reads it) | with 2-3 |
| **RLHF** (preference learning from human feedback) | In poker the result of the game is a stronger, unbiased signal, and self-play RL already optimises it. Human input enters better as training opponents (patterns people find), labels from reviewed hands, and rules (no open-limp). Preference learning on top would mostly pull it towards human play | not picked |

Each picked item is judged like the models above: the same held-out hands and bits an action (and, for item 1, the
focus player's actions with and without their earlier hands).

## Is opponent modelling worth money? (pre-registered 2026-11-28)

The user's reframing: instead of predicting a player's next action, ask whether knowing the player wins money, and
measure where they deviate from sound play (picked: "deviation" and "leaks"). The first test needs no network.

**The bluff test** (`scripts/opp_exploit.py`). Spots: heads-up after the flop, a player facing a single bet, and what
they did (fold, or call or raise). A bluffer with nothing betting b into P0 wins +P0 on a fold and loses -b otherwise,
so a bluff pays when the fold rate is above b / (P0 + b). Decisions scored on what the players actually did:

- *population*: the fold rate at that stake, street and bet size (5 buckets of bet over pot);
- *player*: plus the player's offset from it over their **earlier hands only** (not earlier streets of the same hand:
  a flop call informing the turn is reading the hand, not the player), shrunk with 20 pseudo-spots;
- *ceiling*: plus the player's offset over all their spots, the current one included (an upper bound, inflated for
  players with few spots);
- placebo (`--shuffle`): each spot dealt to a random player, same counts. History then holds no player, so player
  minus population must be at or below 0 there; anything above is the method, not the players.

**Rule, written before the full run** (a 25NL smoke run, with same-hand spots still in the history, read +4.4 bb per
100 spots [+3.9, +4.8], ceiling +21.6): opponent modelling is **kept** if player minus population is above 0 with its
95% interval on the six stakes, and the placebo's interval holds 0 or lies below it, after subtracting any placebo gain.
Otherwise it is a lost cause for this decision and drops down the list. The neural opponent model (the player
sequences) is held to its own rule: its 20-or-more column must be at least 0.02 bits better than the control's, with an
interval.

**Result (2026-11-28): kept.** Bb per 100 spots, 95% intervals from 1,000 bootstraps of players
(`results/foundation_model/opp_exploit/`):

| | spots | players | player minus population | after 1-9 earlier spots | 10-49 | 50 or more | ceiling minus population |
|---|---|---|---|---|---|---|---|
| six stakes | 7,260,415 | 177,342 | **+3.78 [+3.45, +4.13]** | +1.19 | +4.12 | +4.55 | +20.3 |
| six stakes, placebo | same | same | −0.63 [−0.80, −0.46] | −0.41 | −1.58 | −0.22 | +9.2 |
| PokerStars 25NL | 3,032,194 | 84,344 | +3.87 [+3.44, +4.30] | +0.91 | +3.89 | +5.17 | +21.6 |
| PokerStars 25NL, placebo | same | same | −0.39 [−0.62, −0.13] | −0.19 | −1.13 | −0.01 | +9.4 |

A player's own history is worth money, and grows with it; the placebo stays at or below 0. The size is modest: the
bluff decision is worth +164 per 100 spots from the population rate alone, and the player adds about 2% (0.04 bb a
spot). The ceiling minus its placebo (about +11) is three times what one shrunk offset captures, so a finer profile
(by street and size) or a better model has room.

**The neural opponent model, first backbone (attention, 26.4M, 300M tokens): fails its rule.** Focus-player bits an
action at 20 or more earlier hands: 1.417 on the players' own sequences, 1.366 on the control (more than the 0.02
needed, the wrong way). A caveat: control sequences can exceed the 2,048-token window and the scorer skips those, so
the control counts fewer actions (18,279 against 47,517); to fix before Mamba and the hybrid are judged. So far
counting how a player folds is worth money, and the network has not learnt to use the player's history.

**The three backbones, paired (2026-11-28).** The bins above compare different hands (a player's 20th hand in a
sequence is not a random hand; real sequences reach 20 hands only when the hands are short), so the test was redone
paired (`scripts/fm_paired.py`): the same target hand, the focus player's actions scored after L of their own earlier
hands, after L random hands from other sequences, and after none. 2,907-3,039 actions; mean ± standard error, bits an
action (`results/foundation_model/opponent_backbones/`). Data: 92,450 HandHQ players with 60 or more hands,
1,789,190 sequences of at most 2,048 tokens; each model 300M tokens on the 3090, a 50:50 mix with the control.

| backbone | parameters | own minus random, L 1 | L 5 | **L 20** | held-out bits a token | an action | a revealed card |
|---|---|---|---|---|---|---|---|
| attention (8 layers) | 26.4M | −0.003 ± 0.004 | −0.005 ± 0.005 | −0.019 ± 0.005 | 0.794 | 1.349 | 6.21 |
| Mamba (16 layers) | 27.2M | −0.004 ± 0.002 | −0.016 ± 0.003 | −0.022 ± 0.004 | 0.676 | 1.271 | 4.80 |
| **hybrid** (Mamba, attention every 4th of 13) | 26.5M | −0.010 ± 0.003 | −0.017 ± 0.004 | **−0.028 ± 0.004** | **0.624** | 1.271 | **4.71** |

(a revealed card guessed at random: 5.45 bits; the attention model had not yet learnt to read them.)

Against the rule (at least 0.02 bits at 20 hands): attention fails (0.019); Mamba passes on its estimate (0.022,
interval 0.014 to 0.031); the **hybrid passes** (0.028, interval 0.019 to 0.037). The rule was written for the bins and
judged on the paired test, which replaced them for the reason above. A player's own history carries information the
models can read, about 2% of the uncertainty of their next action at 20 hands and still rising; the state-space layers
use it from the first few hands, and the hybrid's attention adds the exact recall within a hand. **The hybrid is the
backbone for the next stages** (longer histories, the leak report, and RLGF's DPO stage).

**Then: deviation from a reference and leak reports.** Replay HandHQ spots into the 6-max cash engine and
score each human decision against nm28's policy and values: the deviation (where they differ, and by how much) and its
cost (the expected value lost against nm28's choice), summed per player by spot type as a leak report.

## Next

Stage 1: a decoder-only transformer of about 25 M parameters (Kronos-small's size) pretrained on the shards, scored on
held-out hands by next-action log-loss (against the opponent profile and the hand-reading models we have) and on the
hidden cards after the actions (the REVEAL tokens, against the belief network). Stage 2: into search if it wins
either.
