# Fullhouse engine: a bot competition's engine, and what it would take to enter (2026-11-15)

[github.com/mbatv/fullhouse-engine](https://github.com/mbatv/fullhouse-engine) (MIT, Python, last pushed 2026-05-24):
the public engine of the Fullhouse hackathon, "the UK's first quantitative poker bot competition" (London, 1-5 June
2026, £4,000, sponsored by Quadrature Capital). The event is over; the engine stays useful as a test bed and as the
template of a likely future edition. Read through the GitHub API on 2026-11-15; nothing from it was run.

## The game

| | |
|---|---|
| Variant | No-limit hold'em, **6-max** (`seat_to_act` 0-5) |
| Stakes | blinds 50/100, starting stack 10,000: **100bb**, no ante |
| Match | 400 hands; stacks carry over within the match, a busted bot sits out (no rebuys) |
| Score | chip delta per match (cash-game scoring, no ICM), summed over Swiss rounds (`engine/tournament.py`); ties broken by fewer matches played, then the best single match; the top 64 go to the final |
| Heads-up | the dealer is the small blind |

## The bot

- One file, `bot.py`, with `decide(game_state) -> {"action": ...}`; a `bot.zip` may add a read-only `data/` folder
  (at most 200 MB, read at import time only, no `.py` inside).
- The state: hole cards, board, street, pot, own stack and street bet, `amount_owed`, `can_check`, `current_bet`,
  `min_raise_to`, every seat's public state (stack, street bet, folded, all-in), and the hand's `action_log`.
- Actions: `fold`, `check`, `call`, `raise` (amount = the street total, raises below the minimum are snapped up),
  `all_in`. An invalid action, a crash or a timeout folds.
- Limits: 2 s a decision (a 30 s warm-up call before the first hand), 768 MB, half a CPU core, `--network none`,
  read-only files. Libraries: `eval7`, `numpy`, `scipy`, `treys`, `scikit-learn` and the standard library; **no
  PyTorch**. The validator rejects `eval`, `exec`, `compile`, `__import__` tricks, sockets and subprocesses.
- Reference bots: `aggressor`, `mathematician`, `shark`, `ref_bot_2`; `demo.py` plays six of them with a live web
  leaderboard (port 5001).

## What it would take to enter with our models

1. **A model for its game.** Our champion (`nm25`) plays 9-max tournaments at 10-25bb with ICM, and the heads-up
   blueprint plays 200bb heads-up; Fullhouse is 6-max cash at 100bb with chip scoring. The same model Open Poker needs
   (6-max cash, 50-250bb; `kestrel/docs/design.md`, Platforms) covers it: a 6-max, 100bb, no-ICM run of the 9-max
   trainer, checked like the tournament networks.
2. **The network without PyTorch.** The forward pass is small (36 tensors, about 4.5 MB in fp16): a NumPy port of
   `NineNet` and of `features()` fits the limits easily, and Kestrel's C++ port already shows the features and
   forward pass can be reproduced exactly from `net.bin`/`net.json` (`kestrel/core/src/net.cpp`). The self-test
   vectors (`advisor_vectors.py`) would check the NumPy port the same way.
3. **Card buckets in the data folder.** Flop and turn bucket tables plus the river cut points come to about 80 MB in
   the heads-up pack format, under the 200 MB limit; the river cut points avoid the 123 MB exact river table (7% of
   river states land one bucket off, measured for Kestrel). Bucket lookup needs the hand indexer, which would have to
   be ported to Python or replaced by a precomputed table, since a compiled extension is not on the library list.
4. **The adapter.** The same replay as Kestrel's structured codec: rebuild the hand from `action_log` into the
   network's table (positions from the button, chips in big blinds, every bet at its own size), decide, and answer
   in chips. It has to run in well under 2 s on half a core; the network itself takes milliseconds.
5. **Checks before entering:** the NumPy bot against the C++ and Python references (vectors), then matches in its own
   engine against its reference bots and against our earlier models.

## As a test bed now

The engine is a third independent rule implementation, after PokerKit and poker-pi, with its own reference bots. A
Kestrel connector for it is possible (its runner speaks newline-delimited JSON on stdin and stdout, like Kestrel's
TCP interface), but it is not needed while the test platform and poker-pi cover the same ground. It becomes worth
building with the 6-max cash model, as the place to measure it against bots written for exactly this game.

## Status

Not started: waiting on a 6-max cash model, which is also what Open Poker needs. Next edition: watch the repository
(`gh repo view mbatv/fullhouse-engine`) for new dates and rule changes; the rules above are from the 2026 edition.
