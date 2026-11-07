# Poker Advisor

A desktop tool that puts our solver stack to work on your own game, away from the live table.

| Tab | What it does | Status |
|---|---|---|
| **Session review** | Reads your finished hands from hand-history files and grades every decision you made. Lists your costliest mistakes and compares your frequencies with winning ranges | Working |
| **Trainer** | A table where you play our bots, with a **Hint** button before you act and a verdict after each move | Working |
| **Spot study** | Type in any situation and see the engine's mix, each opponent's likeliest holdings from hand reading, and your equity against them | Working |
| **Hand Sync** | Collects your finished hands from every site into one library, reviews each session after it ends, and keeps a trend table across sessions | Working |

It never reads the screen, watches a table or advises during a hand. Real-money sites ban real-time assistance,
and it would cheat the other players. It works on finished hands, the practice table, or a spot you type in
between sessions.

## Run it

```
cd llm-research
uv run python -m advisor.app             # the window (Windows, macOS, Linux; Tkinter ships with Python)
uv run python -m advisor.review FOLDER   # session review from the command line -> session_review.html
uv run python -m advisor.sync            # one Hand Sync pass (import new hands; review if the session is over)
uv run --with pytest python -m pytest advisor/tests
```

Reports are saved to `~/PokerReview/` and open in your browser.

## Install and package

**From source (any OS, Python 3.12+):**

```
pip install -r advisor/packaging/requirements.txt      # or: uv sync   (from llm-research/)
python -m advisor.app
```

**As a native app (no Python needed by the user):**

```
uv run --with pyinstaller==6.22.3 python advisor/packaging/build.py
```

This builds for the OS it runs on:

| OS | Output | Notes |
|---|---|---|
| Windows | `dist/PokerAdvisor.exe` | One file, no console window |
| macOS | `dist/PokerAdvisor.app` | Unsigned: right-click → Open the first time (Gatekeeper) |
| Linux | `dist/PokerAdvisor` | One file; needs Tk (`python3-tk`) on minimal systems |

PyInstaller cannot cross-compile, so each OS's app is built on that OS:

- **Locally**, with the command above.
- **Windows, from the lab's Windows 11 VM:** `python3 scripts/win_build.py build BUNDLE.zip PokerAdvisor.exe`, run on
  the lab. It installs Python 3.12 in the VM if needed, uploads the sources through Windows-MCP, builds, self-tests
  the `.exe` inside Windows and copies it back. Done on 2026-11-07: 69 MB, self-test passed, sha256 `ed8b4872…e445`.
  It is unsigned, so Windows SmartScreen asks "More info → Run anyway" the first time.
- **On CI**, with `.github/workflows/advisor-build.yml`: it builds all three, self-tests each app, and attaches
  them to the run. It runs on demand or on an `advisor-v*` tag. The nodalytics Actions account was
  billing-blocked when this was written, so the workflow needs that fixed first.

`dist/PokerAdvisor --selftest` answers two spots without opening a window, which checks that a build found its
models. The bundle includes the advisor and `scripts/` sources, the flop model, the hand-reading models and the
pre-flop charts. TexasSolver is optional and not bundled: set `TEXASSOLVER_BIN` to use it on turns and rivers.

## Hand histories it reads

| Site | Where the files are | Notes |
|---|---|---|
| ACR / Winning Poker Network | `C:\ACR Poker\handHistory\<screen name>\` (written automatically) | Pre-filled when the folder exists |
| PokerStars | `%LOCALAPPDATA%\PokerStars\HandHistory\<screen name>\` (written automatically) | Pre-filled when the folder exists |
| GGPoker | Download from PokerCraft (zip), then unzip | Opponents appear as hashed ids |
| Ignition / Bovada | Downloaded from the client | Anonymous tables |
| Winamax | `%APPDATA%\winamax\documents\accounts\<name>\history` | Euro amounts |
| PHH (open standard) | `.phh` / `.phhs` files | The format of our research datasets |

The parsers were written from each site's documented layout and tested on sample hands
(`advisor/tests/test_handhistory.py`). **The first real file from each site should be checked**: send one in, and
fix any quirk with a test. 888, partypoker and iPoker (XML) are not supported yet.

Only no-limit hold'em hands with one small and one big blind are reviewed. Hands with a straddle, a dead blind, or
lines the replay cannot follow are skipped and counted in the report.

## Spot study notation

Give the number of players, the stacks, your position and cards, then the action so far in acting order:

| Token | Meaning |
|---|---|
| `f` / `x` / `c` | fold / check / call |
| `r2.5` | bet or raise to 2.5 big blinds (this street's total) |
| `a` | all-in |
| `[Qs7h2c]` | the next street's board cards |

Example: 6-max, you in the big blind with AhKd: `f f r2.5 f f c [Qs7h2c] x r3`. UTG and HJ fold, CO opens to
2.5bb, BTN and SB fold, you call; on the flop you check and CO bets 3bb. The answer is for the player to act and
must be you. It shows the engine's mix, CO's likeliest holdings (sets, overpairs and top pair lead), and your
equity against that range.

## Hand Sync (auto mode)

- **Watches your hand-history folders.** These are the sites' default folders plus `Downloads`, for GGPoker's
  PokerCraft zips. You can add any folder.
- **Imports each hand once** into `~/PokerReview/library/<site>/<date>.txt`; a state file tracks what it has seen.
- **Reviews only when a session is over:** no new hands for 20 minutes, which you can change. It saves the report,
  opens it, and adds a row to `~/PokerReview/trends.html` with hands, bb/100, VPIP, PFR, 3-bet, fold to 3-bet,
  c-bet and mistakes per 100 decisions.
- **Auto** checks every 5 minutes while the app is open; **Sync now** checks immediately.
- **It reads your own files only.** It never reviews while hands are still arriving, and it never scrapes or
  collects other players' hands from sites. Hands of opponents you played are in your own files, so opponent
  modelling still works on them.

## How the advice is made (`engine.py`)

These are the same layers as the solver bot that beat Slumbot. The first one that covers the spot answers:

1. **Push/fold chart**: an unopened pot at 20 big blinds or less. The equilibrium was computed by us by fictitious
   play.
2. **Short-stack defence**: heads-up, 20bb or less, facing a raise. Re-shove the chart's calling range, or call
   when the price is right.
3. **Pre-flop strategy**: deeper stacks. Opening ranges by seat, and 3-bets and 4-bets from equity against the
   opponent's likely range. Separate heads-up ranges.
4. **TexasSolver**: heads-up turns and rivers, solved live. Only used when a solver binary is configured
   (`TEXASSOLVER_BIN`) and the box is ticked; it takes a few seconds per spot.
5. **Flop model**: heads-up flops. A network trained on 400 solver runs that predicts the solver's play in
   milliseconds (81–83% top-action agreement).
6. **Monte Carlo equity**: everything else. Heads-up, it computes equity against the range the opponent's betting
   implies, using hand reading from `scripts/ranges.py` and the model trained on Pluribus's hands. Multi-way, it
   uses random hands. It ignores later betting, so it can call a move "questionable" but never a "mistake".

Every answer is a **mix** (for example: fold 0%, call 30%, raise 70%) with the source and a reason. A move the
engine plays at least 35% of the time **agrees**, 10–35% is **questionable**, and under 10% is a **mistake**.

## Files

| File | Purpose |
|---|---|
| `engine.py` | `advise(state, i, ctx)`, and `Table`: builds and replays a hand in PokerKit, keeps the context the advisors need, and deals the board in live play |
| `handhistory.py` | Parsers: the PokerStars family (Stars, GG, ACR, Ignition, Bovada, Winamax) and PHH |
| `review.py` | Grades decisions, computes frequencies and behavioural patterns, writes the HTML report |
| `trainer.py` | Game logic for the practice table and the bots |
| `spot.py`, `spot_ui.py` | Spot study: the line notation, replay, analysis |
| `sync.py`, `sync_ui.py` | Hand Sync: library, de-duplication, review after the session, trends page |
| `trainer_ui.py`, `app.py` | The window (four tabs) |
| `packaging/` | `launcher.py` (the app's entry point, `--selftest`), `build.py` (PyInstaller per OS), `requirements.txt` |
| `tests/` | One sample hand per site layout; the trainer plays whole hands against every bot; spot study; Hand Sync with temporary folders and a GG zip |

The engine imports from `../scripts/` (`poker_agents`, `preflop_deep`, `ranges`, `poker_selfplay`). It loads the
committed models `mlflop/mlp.joblib`, `range_inference/action_model.joblib`, the push/fold charts in
`results/preflop/`, and optionally `slumbot_range/action_model.joblib`.

## Design notes

**Rust or Go for the advisor?** Not necessary, and not much faster:

- **The slow parts are already compiled code.** TexasSolver is C++ (we already sped it up 3.3×). Hand evaluation
  (eval7) is C. The models run in C/Fortran (scikit-learn, NumPy). Python only glues them together and parses text,
  which takes milliseconds per hand.
- **A rewrite would mean re-implementing everything:** the PokerKit rules engine, the models, hand reading and the
  charts. That is weeks of work for no visible speed-up in review, the trainer or spot study.
- **Where Rust could help later is the solver itself.** b-inary's `postflop-solver` is an open-source Rust solver
  reported to be as fast as or faster than TexasSolver (development appears to have paused). If live turn and river
  solves become the bottleneck, benchmark it against our build instead of rewriting the app.
- **Distribution does not need a rewrite:** PyInstaller packages the app as one Windows `.exe`.
