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

## Next

Stage 1: a decoder-only transformer of about 25 M parameters (Kronos-small's size) pretrained on the shards, scored on
held-out hands by next-action log-loss (against the opponent profile and the hand-reading models we have) and on the
hidden cards after the actions (the REVEAL tokens, against the belief network). Stage 2: into search if it wins
either.
