# 9-max tournament strategy: the principles, and the checks that hold the networks to them (2026-11-11)

The goal is a 9-max tournament (MTT) bot that plays like a strong human professional and is hard to exploit.
Self-play learns whatever wins against the opponents it trains against. If those opponents are odd (the
rule-based maniacs and stations in our pool), it learns odd habits. This page states what good tournament play
looks like, turns each principle into a measurement on the network's own policy, and records how the networks
score. Script: `scripts/ninemax_sense.py`; results: `results/ninemax/sense_*.json`.

## The game

- **Stacks are short.** Most decisions after the early levels are made with 10–25 big blinds. Under about 12bb,
  play before the flop becomes all in or fold. The training tables follow this (`nm10`, `nm12`).
- **Chips are not money (ICM).** Losing a stack costs more prize money than winning the same amount gains. So:
  - call all-ins tighter than chip odds say;
  - shove wider when others must fold to survive;
  - the effect is strongest on the bubble and at the final table, and weakest for the covering big stack.
  - Checked by `scripts/ninemax_pushfold_check.py` against `icm_pushfold.py`.
- **Position decides how wide to play.** Eight players behind UTG can wake up with a hand; one (the big blind)
  behind the small blind. Opening ranges widen from about 12–16% at UTG to 40–50% on the button.
- **Blinds and antes are worth fighting for.** With an ante, the pot before anyone acts is about 2.5bb. Stealing
  it matters more as stacks shrink.

## Before the flop

| Principle | Why | Check (20bb unless stated; chips) | Reference band |
|---|---|---|---|
| Raise strong hands, never limp them | A raise builds the pot when you are ahead and makes weak hands fold or pay to see a flop (the partner's AK vs Q9 example) | Premiums (AA, KK, QQ, AK) first in: share raised or shoved, at UTG and on the button | 95–100% |
| Fold trash early | Eight players behind; nothing to protect | UTG with 72o, 83o, 94o, T2o, J3o: share folded | 97–100% |
| Open wider as position improves | Fewer players left to wake up with a hand | Share of hands played first in, by seat | UTG 10–20%, MP 13–24%, HJ 18–30%, CO 24–38%, BTN 35–58%, SB 35–70% |
| Shove, don't min-raise, when short | At 8bb a raise commits you anyway; the shove gets folds from hands that would beat you after the flop | At 8bb, the share of played hands that go all in (UTG, CO) | 75–100% |
| Defend the big blind wide against a small open | Already 1bb in; a 2× open lays about 3.5 to 1 | Big blind vs the button's 2× open, small blind folded: share defended | 50–80% |
| No open-limping from UTG to the button | Same as raising strong hands, and it hides nothing a raise does not | A game rule since 2026-11-11 (`--no-open-limp`) | — |
| Open smaller when shorter | 2–2.2× risks less and keeps folds to a 3-bet shove cheap | The `mtt` menu (2×, 2.5×, 3× opens), from `nm10`/`nm12` on | Measured once they train |

The bands are approximate, from common tournament charts (chip value, 9-handed, no ante). They are not exact
equilibria. A WARN is a spot to look at, not a proof of a leak.

## After the flop (checks to build next)

- **Continuation bet** when the raiser has the range advantage (dry, high boards), and less on boards that hit the
  caller's range.
- **Don't over-fold to small bets.** Against a third-pot bet, folding more than about 25% of hands loses
  automatically.
- **Value-bet the river thin** against callers, and bluff with hands that block the opponent's calls.
- **Commitment.** With a stack under about 3 times the pot, top pair and good draws go all in and do not fold.
- **ICM after the flop.** Medium stacks avoid marginal all-ins against the big stack on the bubble.

## Results

`nm8` (the earlier tournament run, 20 hours, `medium` menu) and `nm10a` (1.5 hours, 5–15bb and 20–50bb tables),
100,000 hands per check (`results/ninemax/sense_nm8_nm10a.json`):

| Check | `nm8` | `nm10a` | Band |
|---|---|---|---|
| Premiums raised first in, UTG | **0%** (limps AA, KK, AK 100%) | 57% | 95–100% |
| Premiums raised first in, button | 100% | 78% | 95–100% |
| Trash folded at UTG | 100% | 96% | 97–100% |
| Played first in: UTG / MP / HJ / CO / BTN / SB | 13 / 12 / 16 / 24 / 31 / 65% | 17 / 18 / 27 / 33 / 42 / 83% | see above |
| All in, share of hands played at 8bb: UTG / CO | **9% / 30%** | 48% / 57% | 75–100% |
| Big blind defends vs button 2× open | **23%** | 32% | 50–80% |
| Within band | 5 of 12 | 5 of 12 | |

**What it shows:**
- **The limping of premiums was worse than the first look.** At 20bb `nm8` limps every AA, KK, QQ and AK at UTG,
  and opens them all on the button. It traps from early position, a habit learned against the pool's maniacs.
  The new runs cannot open-limp.
- **Too tight in the big blind** (23–32% against 50–80%). The networks give up the blinds far too easily, which an
  opponent who opens every button exploits.
- **Not shoving short.** At 8bb the networks min-raise or limp most hands they play, instead of shoving. That
  invites a reshove or a cheap flop. One cause: the training tables averaged 10–60bb, so few hands were played
  that short. `nm10` now has a quarter of its tables at 5–10bb.
- **Opening widths are roughly right** (`nm10a` is inside every band but the small blind). Position is learned
  well; the strategy within a range is the problem.

### `nm10` and `nm12` (no open-limping, `mtt` menu), 2026-11-12

Snapshots at 4 and 8 hours, 100,000 hands per check (`results/ninemax/sense_nm10_nm12.json`; log
`checks_nm10_nm12.log`). `nm10` trains at 5–15bb and 20–50bb, `nm12` at 10–25bb.

| Check | `nm10` 4h | `nm10` 8h | `nm12` 4h | `nm12` 8h | Band |
|---|---|---|---|---|---|
| Premiums raised first in, UTG / button | 99 / 99% | 98 / 99% | 100 / 100% | 99 / 99% | 95–100% |
| Trash folded at UTG | 97% | 95% | 97% | 97% | 97–100% |
| Played first in: UTG / MP / HJ / CO / BTN / SB | 10/11/16/18/22/52% | 13/16/24/28/33/50% | 12/15/20/24/30/57% | 11/15/22/25/32/54% | see above |
| All in, share of hands played at 8bb: UTG / CO | 26 / 16% | 15 / 10% | 8 / 8% | 11 / 18% | 75–100% |
| Big blind defends vs button 2× open | 34% | 39% | 23% | 31% | 50–80% |
| Within band | 3 of 12 | 7 of 12 | 6 of 12 | 8 of 12 | |

**What it shows:**
- **The partner's point is fixed.** AA, KK, QQ and AK are raised first in 98–100% of the time from every seat.
  The rule removed the limp, and the networks chose to raise rather than fold or shove them.
- **Opening is close to the charts**, slightly tight on the button (32–33% against 35–58%).
- **Still no short-stack shoving.** At 8bb they play 12–27% of hands but go all in with only 8–26% of them, and it is
  getting no better with training (`nm10` 26% → 15%). A 2× open at 8bb risks a quarter of the stack and then
  faces a reshove: against good players this loses, against the training pool it apparently does not.
- **The big blind still folds too much** (23–39% against 50–80%), though `nm10` is improving (34% → 39%).
- **ICM push/fold** (`pushfold_check_nm10_nm12_8h.json`): chip-EV agreement 90% as before; at the final table and
  on the bubble the networks shove far less than the ICM equilibrium (on the button at a final table 27–29%
  against 98%). They play ICM spots as a tight raise-or-fold game, not as push/fold.

Both open-shove and big-blind defence are things the opponents decide: a shove is good because good players fold
to it, and a wide defence is good because opens are wide. The pool's maniacs and stations call shoves light and
open everything, so neither lesson is taught. This is the case for phase C (realistic opponents) and for searching
the preflop decisions (below).

### Phase C (`nm13`, `nm14`) and the big-blind ante, 2026-11-13

`nm13` (from `nm12`) and `nm14` (fresh) trained 12 hours with the tournament regular (`reg`) in the pool.

| Check, 12 hours | `nm13` | `nm14` | Band |
|---|---|---|---|
| Premiums raised / trash folded | 100 / 97% | 99 / 98% | |
| Opening widths UTG … SB | 12–41% | 10–49% | |
| All in, share of hands played at 8bb: UTG / CO | **38 / 54%** | 19 / 21% | 75–100% |
| Big blind defends vs button min-open | 21% | 24% | 50–80% |

The regular moved `nm13`'s short-stack shoving (from 19/28% in `nm12`), not `nm14`'s, and neither big blind.
Both still beat `reg` by about +20 bb/100. Against it, min-raising at 8bb steals the blinds about as often as a
shove (it folds about 85% either way), and it risks 2bb instead of 8. **Without an ante that is a sound
strategy.** The 75–100% shove band and the 50–80% defence band came from charts for tournaments with a big-blind
ante, and our tables had none. So the checks were partly asking for answers to a different game.

**The ante is now in the game** (`RULES["ante"]`, `--ante 1`): the big blind posts one more big blind of dead
money, so 2.5bb is in the middle before anyone acts, as in almost every tournament today.
- The engine agrees with PokerKit on 13,000 random hands at 9, 6 and 3 players with mixed stacks
  (`ninemax_check.py ... 1.0`). PokerKit needs ante trimming off for a big-blind ante.
- The push/fold solver treats antes as dead money (`--bb-ante`), so a short big blind's all-in is not inflated.
  The charts with the ante (`results/ninemax/icm_pushfold_ante.json`) are much wider:

| Chip EV, 9 × 10bb | Shove by seat (UTG … SB) | Big blind calls the small blind |
|---|---|---|
| no ante | 7 8 11 14 19 22 33 59% | 37% |
| big-blind ante | 12 14 18 21 27 33 43 75% | 58% |

- `reg` uses the ante ranges when the run has one (push 12–75%, opens 16–55%, big blind defends 65% against a late
  open). The sense checks of an ante run use ante bands: opens UTG 13–22% to button 42–68%, big blind 60–85%.

`nm15` (from `nm13`) and `nm16` (fresh) train 12 hours with the ante, from 2026-11-13 14:05 lab time.

Early reading, 2.5 hours in (100,000 hands per check, ante bands):

| Check | `nm15` | `nm16` | Band |
|---|---|---|---|
| Big blind defends vs button min-open | 39% (`nm13`: 21%) | 48% (`nm14`: 24%) | 60–85% |
| All in, share of hands played at 8bb: UTG / CO | 31 / 47% | 40 / 38% | 75–100% |
| Opens first in, button | 36% | 35% | 42–68% |

The ante moved the big-blind defence at once, after five runs stuck in the 20s. Short-stack shoving has not moved
yet.

### Short-handed final tables, 2026-11-13

A final table plays down from nine to two, and short-handed poker is a different game: wider opens, wider
defence, heads-up push/fold. Until now every training hand was nine-handed.
- **Engine:** a stack of 0 is an empty seat, folded from the start. The early seats empty first (UTG, UTG+1, ...),
  so the seats left keep their names counted from the button. Heads-up the small blind is the button: first before
  the flop, last after it. Checked against PokerKit on 18,000 hands with 2–9 players seated, with and without the
  ante: 0 mismatches (`ninemax_check.py N 9 mtt mixshort 1.0`; PokerKit reverses the blinds heads-up but keeps
  its antes keyed by blind position).
- **Tournament tables:** `icm.scenarios(short=...)`: that share of final tables seats 2–8 players (uniform), with
  only places 1..n paid. ICM needs nothing new: a player with no chips takes no place while anyone has chips
  (checked against brute force). Training option `--short`.
- **Checks:** two heads-up checks in `ninemax_sense.py`. The small blind first in at 10bb should play at least the
  push/fold equilibrium's shove share (59%, 75% with the ante; `icm_pushfold.py --stacks 10,10`); the big blind
  should defend 55–85% (65–92% with the ante) against a min-raise at 20bb. The nine-handed networks fail the second
  badly: `nm13` defends 18%, `nm14` 30%.

`nm17` (from `nm15` at 2.5 hours, ante, half the final tables 2–8 handed) trains 12 hours on the 3090 from
2026-11-13 16:43 lab time. The automatic checks moved to the PRO 6000, since `nm17` fills the 3090.

**Next:**
- the checks now run by themselves on every new snapshot of a tournament run (`scripts/ninemax_autocheck.py` on the
  lab, results as `sense_<run>_<h>h.json` and `pushfold_check_<run>_<h>h.json`) and show on the watch page;
- search before the flop on the tournament networks, scored in ICM equity (`ninemax_search.py --preflop`,
  running on `nm12`);
- if the big-blind defence and short-stack shoving stay outside their bands, look at the training pool first,
  because the opponents set what is worth learning;
- add the checks after the flop.
