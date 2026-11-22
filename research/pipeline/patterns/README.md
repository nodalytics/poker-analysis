# Patterns that beat the bot

One file per pattern someone found: a spot and a move that wins against the networks. The pipeline tests every
new network against all of them (the `patterns` check) and can train against them (`train.opponents.patterns`), so
a pattern found once is closed in the next run and stays tested after it. Patterns mined automatically from the
practice tables are in `results/exploit_patterns/mined.json` and are used the same way.

To add one: copy a file here, give it an `id`, say who found it and how (`source`), describe it, and set the spot
(`when`) and the move (`do`, `freq`). The format is in `scripts/exploit_patterns.py`; check it with
`python3 scripts/exploit_patterns.py validate pipeline/patterns`.

| Field | Values |
|---|---|
| `when.street` | `preflop`, `flop`, `turn`, `river`, or a list |
| `when.position` | some of `SB BB UTG UTG+1 MP LJ HJ CO BTN` (omit: any) |
| `when.stack_bb` | `[low, high]`, the player's stack at the start of the hand |
| `when.facing` | pre-flop `unopened` or `raise`; after the flop `check` or `bet`; or `any` |
| `when.players` | `[low, high]` players still in the hand |
| `when.hand` | `any`, `{"top": 40}` (the best 40% of starting hands), `{"strength": [0.0, 0.3]}` (strength now, 0-1) |
| `do` | `all-in`, `raise-big`, `raise-small`, `call`, `check`, `fold` |
| `freq` | how often, when the spot comes up (0-1) |
