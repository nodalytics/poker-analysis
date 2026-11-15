# Tournament equity beyond ICM (2026-11-15)

Every tournament network so far is trained on one equity model: ICM in its Malmuth-Harville form (`scripts/icm.py`),
over one final table, bubble or in-the-money table, with the regular prize ladder only. This note covers the
alternatives: two done today (Malmuth-Weitzman as a check, progressive knockouts in the trainer) and three written
down for later.

## Done: Malmuth-Weitzman as a check

**What it is.** Malmuth-Harville gives places from the top: the winner is drawn in proportion to chips, then second
among the rest, and so on. Malmuth-Weitzman gives them from the bottom: the next player to bust is drawn in
proportion to 1/stack. It is often argued to be more realistic for short stacks. With a single prize it is not the
chip share, unlike Malmuth-Harville.

**Code.** `icm_pushfold.py`: `icm_mw_np` (checked against brute-force enumeration at 3, 5 and 7 players, and that
every prize is paid out), and `--model mh|mw` for the push/fold solver. A busted player's 1/stack is capped at a
billionth of the table's chips, so it busts first without overflowing (the first run gave `nan` everywhere).

**The check.** The push/fold charts (fictitious play, at most one caller, big-blind ante 1bb) solved under both
models (`results/ninemax/icm_pushfold_ante.json`, `icm_pushfold_ante_mw.json`), and nm25's agreement with each
(`pushfold_nm25_vs_*.json`):

| Spot (10bb) | nm25 vs Malmuth-Harville | nm25 vs Malmuth-Weitzman |
|---|---|---|
| final table, 9 × 10bb | 60% | **85%** |
| bubble, 9 × 10bb | 53% | 52% |
| bubble, top-heavy above it | 76% | 80% |
| bubble, big stack on the button | 47% | **86%** |
| bubble, short stack in the big blind | **65%** | 57% |
| final table, mixed stacks | **86%** | 82% |
| winner-take-all ("chip EV") | 89% | 31%: under Malmuth-Weitzman a single prize is not chip EV, so this row measures the model, not nm25 |

**What it shows.**
- The equity model changes "correct" push/fold a lot near the bubble. With a big stack on the button, Malmuth-Harville
  has every position shove 100%; Malmuth-Weitzman has every position before the button fold.
- nm25, trained on Malmuth-Harville rewards, agrees better with Malmuth-Weitzman in three spots and worse in two. Its
  play is not the chart of either model: training reaches the decisions through full hands and opponents' responses,
  which the push/fold model leaves out.
- So the chart check cannot say which equity model is right, only how far the network is from each. That needs a
  model of what happens after the hand: FGS (below).

## Done: progressive knockouts (PKO) in the trainer

**The format.** A share of every buy-in (usually half) is a bounty on the player's head. Knocking a player out pays
half their bounty in cash and adds the other half to your own; the tournament winner collects their own. Bounties
change play: calling wider against shorter stacks you cover, and covering a stack is worth more.

**Code** (`ninemax_selfplay.py --tourney --bounty 0.5`):
- **Scenarios.** That share of the money at stake is bounties, one per seated player, larger for larger stacks
  (bigger stacks have usually collected more): proportional to the square root of the chip share, with log-normal
  noise; the regular prizes keep the rest. The payouts tensor carries both, `[G, 2P]`; `split_pay` separates them,
  so every other caller is unchanged.
- **Features.** Every seat's bounty, mine first (9 more inputs: 110 instead of 101).
- **Reward.** The change in equity, where equity = ICM on the regular prizes + bounty cash collected this hand + own
  bounty × chip share (the chance of winning it back). A bust's bounty goes to the hand's biggest winner, which is
  exact unless several players split the pot that busted someone.
- **Smoke test** (2 minutes on the lab): it trains and evaluates; the run itself waits for a free GPU (nm26, nm27,
  nm28 and the search experiment hold all three until about 19:00 WAT on 2026-11-15).

**Before a PKO model ships:** Kestrel's C++ features (`kestrel/core/src/net.cpp`) need the bounty inputs, its
platform adapter a `bounties` field per player, and the self-test vectors (`advisor_vectors.py`) a bounty run; a
sense check of PKO spots (calling wider when covering) like `ninemax_sense.py`.

## Later

**FGS (Future Game Simulation).** ICM ignores position and blinds: a 10bb stack is worth the same whether it pays the
big blind next hand or has the button. FGS (HoldemResources Calculator) plays the next few hands forward (an orbit of
blinds, with push/fold strategies) and applies ICM at the end. It is most accurate on bubbles and short-stacked final
tables, our 10-25bb game, and is the arbiter the Malmuth-Weitzman check above lacks. Plan: an FGS evaluator over the
push/fold model (each future hand solved as now, blinds rotating); first as a check of nm25's bubble decisions
against FGS-correct play, then possibly as the training reward, with precomputed tables or a small value network to
keep training fast.

**Satellites.** Every paid place gets the same prize (a seat in a bigger event). The most extreme ICM: near the
bubble big stacks should almost never call, and short stacks shove any two cards. `icm.scenarios` already produces
one by accident (nm7 trained on a nearly flat ladder); add it as a named scenario kind, with its own sense checks.

**Large fields.** Exact ICM costs 2^players: fine for a 9-player table, impossible with 500 left. Early and in the
middle of an MTT, either chip EV (when the money is far away) or an approximation: Monte Carlo sampling of finishing
orders, or ICM on this table with the chips' value outside it. Our scenarios cover final tables, bubbles and
in-the-money tables, not the 500-left phase, where chip EV is close enough; worth adding when Kestrel plays full MTTs
on a platform.
