# LLMs that play poker: six outside sources, read against our own results (2026-11-23)

Six sources on large language models playing (or writing bots for) no-limit hold'em, read in full where they could be
read, with their code checked where it is public. For each: what was done, the numbers with their sample sizes, what
the numbers can and cannot show, and what it means for our stack (the 9-max networks, the heads-up blueprint and
Kestrel, and the LLM-agent studies of 2026-11-05).

Units: papers report mbb/hand (thousandths of a big blind per hand); we report bb/100. **1 bb/100 = 10 mbb/hand.**

## The sources at a glance

| Source | What | Game | Strength shown | Sample, uncertainty |
|---|---|---|---|---|
| PokerSkill (Li, Wang, Huang; arXiv 2605.30094, May 2026) | frontier LLM picks among actions a hand-written rule library allows | heads-up NL, 200bb | GPT-5.5: −57 ± 21 mbb/hand against GTO Wizard's benchmark bot | ≥ 5,000 hands a run, AIVAT; "±" stated as 1 SE in one place, 95% in another |
| PokerBench (Zhuang et al.; arXiv 2501.08328, Jan 2025) | solver-labelled spots; LLMs scored on matching the solver, Llama fine-tuned on them | 6-max cash, 100bb | fine-tuned Llama-3-8B matches the solver's action 80.6%; GPT-4 65.5% | 11,000 test spots; matches 50k hands (checkpoints), 1,000 (GPT-4); no intervals |
| PokerBench arena (JoeAzar/pokerbench, GitHub, Jan-Mar 2026) | frontier LLMs at one table, 150-hand games | 6-handed and heads-up cash, 100bb | rankings that reverse between runs | 14, 50, 71 and 99 games; profit per game, no intervals |
| Husky Hold'em Bench (Kumar et al.; NeurIPS 2025 workshop poster) | LLM agents *write* bot code; the bots play a round-robin | 6-player | "Claude 4 Sonnet tops the leaderboard" | paper not readable (OpenReview browser check); abstract only |
| poker_LLM_bot_detection_ntpoker (Jay-Code-03, GitHub, Apr 2025) | screen-reading bot on a mobile poker app: range files pre-flop, GPT-4 or Claude after | heads-up | none reported | three logged sessions on one day, 61 events |
| "Can LLMs play poker? GPT-4 and Claude" (Medium, @jay_learner) | the write-up of the bot above, by its name and models | heads-up | not read | Medium returned 403; paste the text to have it added |

## PokerSkill: rules propose, the LLM chooses

**Design.** A deterministic context engine labels each decision: board texture, one of 23 hand classes, one of about
60 action-line scenarios, stack-to-pot ratio, and "pressure". A budget system (attack and defence budgets spent by
earlier aggression) turns that into a set of **viable** actions without ranking them; the LLM (GPT-5.5 at its highest
reasoning setting, Claude Opus 4.6 and 4.7 at maximum thinking, temperature 1.0) chooses one through a forced JSON
tool call, and a validator checks legality. No training, no solver at play time; the skill library and the budgets
were written and tuned by hand by the authors.

**Results** (against GTO Wizard's benchmark bot, AIVAT, at least 5,000 hands a run):

| Agent | mbb/hand | bb/100 |
|---|---|---|
| GPT-5.5 + PokerSkill | −57 ± 21 | −5.7 |
| Opus 4.6 + PokerSkill | −80 ± 29 | −8.0 |
| Opus 4.7 + PokerSkill | −87 ± 64 | −8.7 |
| the rule library alone (first viable action) | −132 ± 19 | −13.2 |
| GPT-5.5, plain prompt | −132 ± 25 | −13.2 |
| Opus 4.6, plain prompt | −204 ± 44 | −20.4 |
| Slumbot (quoted from the GTO Wizard benchmark paper, not run here) | −194 ± 41 | −19.4 |

**What holds.** The library halves the plain-prompt loss, and the LLM on top of the library beats the library alone
(−57 against −132), so the model's choice among viable actions is worth something. Every agent still **loses** to the
benchmark bot.

**What does not.** "Outperforms Slumbot" compares two different runs against GTO Wizard; Slumbot was never played. The
± convention is stated two ways (1 SE in Table 1, 95% elsewhere), and an appendix lists −61 ± 66 for the run Table 1
gives as −57 ± 21. Only one ablation (no LLM) is reported, so how much each part of the library contributes is
unknown. The released code is **compiled extensions only** (`.so` files, CC BY-NC), so the skill library cannot be
read or checked. Cost: about $0.30 a hand for GPT-5.5 and $0.07 for Opus.

**The repository also advertises** a separate commercial product giving real-time decisions at 2-9 player tables,
with the stated aim that online poker "turns to ash", and recruits professionals to test it. That is real-time
assistance against human players, which our rules exclude (the repository README: no bots, OCR or real-time assistance on real-money poker sites); nothing from it
is used here.

## PokerBench (the paper): matching a solver is not winning

**Design.** 11,000 spots (1,000 pre-flop from GTO Wizard's charts, 10,000 post-flop from a WASM port of the
post-flop solver), 6-max, 100bb, folds down-sampled because folding everything scores about 90%. Scored by action
accuracy and exact match (the right size too). Fine-tuning used 110,000 of the released 560,000 training spots.

**Results.** GPT-4 with five examples: 65.5% action accuracy. Fine-tuned Llama-3-8B: 80.6% (79.1% post-flop). In play,
heads-up, later checkpoints beat earlier ones (+24.8 to +50.9 bb/100 over 50,000 hands), **but GPT-4 beat the most
accurate fine-tuned model by 22.2 bb/100 over 1,000 hands**, mostly by donk-betting (leading into the pre-flop raiser),
a line the training data never contains. The authors note both limits themselves: the sample is small, and supervised
fine-tuning on solver-against-solver spots leaves the model with no answer to off-equilibrium play.

**For us.** The same lesson as our flop model (81-83% top-action agreement with the solver, 2026-11-05) and as our rule
"fit on the edge, not the raw score": agreement with a solver is a proxy, and play against opponents who leave the
solver's lines is the test. The released spots are a free external check for our **6-max cash** networks (nm28,
nm35, nm37): score their action against the solver label on the post-flop spots, as a cross-check, never as a target.
The repository carries no licence, so the data stays a local test set and is not redistributed.

## The PokerBench arena (GitHub): rankings that do not survive another run

**Design.** pokerkit for the rules and litellm for the models; every model gets the state and a "world-class
professional" system prompt, answers in JSON with its reasoning, and remembers the last 150 hands of each opponent.
Games are 150 hands or until one player has all the chips, 50/100 blinds, 10,000 chips (100bb). Licence: MIT with the
Commons Clause (no selling).

**Results** (average profit per game, chips):

| Run | Games | Claude Opus 4.5 | Gemini 3 Flash | Gemini 3 Pro | GPT-5.2 |
|---|---|---|---|---|---|
| "Large models", 6 players | 14 | **+2,264** (2nd) | +5,754 (1st) | −2,618 | −3,889 |
| "Small models", 4 players | 99 | **−4,014** (last) | +2,675 (1st) | | |
| Gemini, heads-up | 50 | | +3,858 | −3,858 | |

Claude moves from second to last between runs, and no interval is given for any figure. With 150-hand games, seats
shuffled once per game, and no duplicate dealing, the luck of the cards is most of each number. The arena is useful as
a harness and a replay viewer; its leaderboard is not evidence. Our own ladders paired every learner with a frozen twin
and dealt the same cards for the same reason (`docs/2026-11-05-arena-gui.md`).

## Husky Hold'em Bench: models that write bots

Agents (a five-stage refinement scaffold) write poker bots in code; the bots play a 6-player round-robin, averaged
over several trials; Claude 4 Sonnet's bot topped the table, and better models' bots played balanced or aggressive
styles. Code and tournament data are said to be open, but the paper sits behind OpenReview's browser check and no
repository could be found by name, so neither the method nor the numbers could be checked. Interesting for our
coding-model work (a coding model writing a bot is an end-to-end test of it), not for the poker engine.

## The screen-reading bot (poker_LLM_bot_detection_ntpoker)

A heads-up bot for the NT Poker mobile app ("used as a training app"), driven through an Android emulator: template
matching and Tesseract OCR read cards, stacks and buttons; pre-flop play comes from range files (SB open, BB 3-bet and
call, 4-bet and 5-bet ranges); after the flop the state, a hand analyser and a 5,000-sample Monte Carlo equity against
a range are put to GPT-4 or Claude, and the reply is tapped through adb. The logs hold three sessions on one day
(4, 51 and 6 events) and report no results; despite the name there is no bot-detection code. It is the pattern our
rules exclude on real-money sites (screen reading and automated play); the only part of interest, LLM post-flop
advice given equity and pot odds, is what our 2026-11-05 solver-stack study already measured (advice helps, but models
often ignore it, so the solver now acts directly).

## What this means for our work

1. **No source plays at our level, and none tests what we play.** Everything is heads-up or 6-max cash at 100-200bb;
   nothing touches 9-max tournaments, ICM or short stacks. The best LLM agent, −5.7 bb/100 against GTO Wizard's bot,
   is a loser against an equilibrium bot. Our heads-up blueprint plays Slumbot directly (run 2: +12.4 [−16.0, +40.7]
   bb/100 over 5,000 hands), which is not yet a win either; phase A4 of the plan (20,000+ hands) settles it.
2. **An LLM in the decision loop stays out of Kestrel.** PokerSkill's best result is a loss, at $0.07-0.30 a hand and
   seconds per decision; our networks decide in milliseconds. Where an LLM helps is choosing among a few
   pre-filtered, legal options, which matches our 2026-11-05 finding that free-form advice is ignored.
3. **Agreement with a solver is a proxy.** PokerBench's 80% model lost to GPT-4's donk bets; keep judging networks by
   play (duplicate matches, the exploiter, rule players, styles), not by chart or solver agreement. This also backs
   the plan's step A3: show a push/fold chart beats nm25 before training towards it.
4. **Variance kills LLM leaderboards.** Short games without duplicate dealing give rankings that flip between runs.
   Every comparison we publish carries an interval, and duplicate dealing stays the default.
5. **Two cheap uses, both optional:** the PokerBench spots as an external accuracy check for the 6-max cash networks,
   and GTO Wizard's benchmark (the 2026 papers' common yardstick, with AIVAT) for the heads-up blueprint, if access
   can be arranged; it would put our heads-up numbers on the same scale as these papers.
6. **Nothing here is adopted into play.** The rules stay as they are: no automation or real-time assistance on
   real-money sites, play-money and permitted APIs only.
