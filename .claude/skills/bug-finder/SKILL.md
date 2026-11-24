---
name: bug-finder
description: Find, verify and fix bugs in this repo's poker code — the 9-max trainer and engine (research/scripts/ninemax_*.py, icm.py, seqtest.py, pipeline*.py), the heads-up CFR solvers (research/blueprint/*.cpp) and Kestrel's C++ (kestrel/core, kestrel/connectors) — and check bug claims made by audits, reviews or PDFs. Use it whenever someone reports or suspects a bug, asks "is this a bug?", hands over an audit or code review to check, sees a result that looks wrong (a number that contradicts a doc, a check that fails or passes too easily, Python and Kestrel disagreeing), or asks to hunt for bugs before a training run. Every verdict is proved by running the real code, not by reading it.
---

# Bug finder

Reading code produces plausible bug reports; running code decides them. On 2026-11-23 an outside audit listed
14 code defects with file and line numbers. Checked here, 9 were real, 3 were deliberate design, and 5 were wrong.
The wrong ones looked as convincing as the real ones. They included the audit's top-priority fix, which gave
byte-identical output when built and run. So the method below never accepts or rejects a claim until code has run.

## The method, per claim

1. **State the claim as a testable sentence.** "X happens when Y, so Z is wrong." If a report is vague ("search is
   broken"), split it into claims you can test one by one.

2. **Find the code by symbol, not by line number.** Line numbers in reports drift. `grep -n` the function or
   variable, and read enough around it to see its inputs and every caller (`grep -rn "name(" research/scripts
   kestrel`). Python and Kestrel often implement the same thing twice: check both, and whether they agree.

3. **Check intent before calling it a bug.** Many "defects" are measured choices.
   - Read the comment beside the code, and `git log -S'<symbol>' --oneline` for the commit that introduced it.
   - Look in `research/docs/`, `research/blueprint/model.md`, `kestrel/docs/design.md` and
     `research/results/prereg/*.result.json`. They record what was tried and measured.
   - Examples from 2026-11-23: the MDF floor on re-solved spots, training without the early-shove rule, and greedy
     search were all deliberate and measured.

4. **Classify:** confirmed bug / real but deliberate (say where it is documented) / wrong claim / partly right (say
   which part). Say how big the effect is, not only whether it exists. The 2026-11-23 seating bug was real but wasted
   seats rather than corrupting learning: 8.7% of seats in tournaments, 45.7% in 6-max cash.

5. **Prove it by running code.** Pick the cheapest test that settles it:
   - **A/B build:** compile the code as it is and with the claimed fix, then run both on the same seeds. Identical
     output means the "fix" changes nothing (the re-solver's `rt` claim). Different output means measuring which is
     right (`resolve --test` measures safety).
   - **Fuzz the real function:** call the repo's own function on many random inputs aimed at the claimed failure, for
     example `Tables.payoffs` on 200,000 tables with ties, side pots, folds and antes. Use the real function, not a
     re-implementation: a copy proves only the copy.
   - **A check script that fails before the fix:** write `research/scripts/<area>_check.py` that measures the defect
     (old behaviour vs new) and exits 1 on failure. Run it before the fix (it must show the defect), then after (it
     must pass). Commit it with the fix. Example: `ninemax_seating_check.py`.
   - **Parity:** when Python and Kestrel should agree, feed both the same input and compare
     (`kestrel_selftest --vectors DIR --model M --pack P`; `kestrel_resolve_check --vectors F --run D --iters 500`).
   - **Reproduce the number:** when a doc or a report quotes a result, rerun it with the documented command and
     seed. If the result can't be found in `results/`, say so: an unsaved number is not evidence.

6. **Fix, then re-run every check that touches the code.** A fix is done when its check passes, the existing checks
   still pass (`ninemax_check.py` against PokerKit for engine changes, Kestrel's selftests for C++), and lint is clean
   (`uv run ruff check scripts` and `uv run ruff format --check scripts` from `research/`). Assert every string patch
   you apply: replacement edits fail silently when the text has moved.

## Where code runs

- **torch and CUDA live on the lab, not locally.** The trainer is CUDA-only. Run checks on the lab's CPU with
  `CUDA_VISIBLE_DEVICES=` and `.venv-gpu/bin/python`. Do this from a scratch copy (`~/claude-work/<name>/`), not
  `~/research`, while other jobs read `~/research`.
- **Lab access** is in `.secrets/lab.env`; reuse one ssh ControlMaster connection. Never print or commit the address,
  login or keys.
- **Don't take a GPU that a timed job is using.** The 30-minute exploiter is limited by wall-clock time, so sharing its
  GPU skews the reading.
- **Lab safety:** set `CUDA_DEVICE_ORDER=PCI_BUS_ID` (in that order 0 = 3090, 1 = 4090, 2 = PRO 6000). Leave other
  people's processes and containers alone. Never use `pkill -f` or a self-matching `pgrep`.
- **C++:** `research/blueprint` builds with its Makefile against `deps/libdeps.a` (on the lab). For an A/B build,
  compile copies into a scratch directory instead of editing in place. Kestrel builds locally:
  `cmake --build kestrel/build -j8`.

## Report

End with one table, then a short list of what to do:

| Claim | Verdict | Evidence (file:line, and the test run with its result) | Action |
|---|---|---|---|

- **Verdicts:** confirmed / deliberate (where documented) / wrong / partly (which part).
- **Evidence:** the command that was run and its output, such as "A/B on 300 spots: byte-identical" or "fuzz 200,000
  tables: min survivor 1.0 chip". "Read the code" alone is not evidence for a verdict.
- **Order the actions** by how much each defect affects what we train, judge or ship.
- **Record the outcome where the team will find it:** a section in the relevant `research/docs/` file, with the check
  scripts committed beside the fixes.
