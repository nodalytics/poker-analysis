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

**Next:**
- run the checks on `nm10` and `nm12` at each 4-hour snapshot, and put the results on the watch page;
- if the big-blind defence and short-stack shoving stay outside their bands, look at the training pool first,
  because the opponents set what is worth learning;
- add the checks after the flop.
