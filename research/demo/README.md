# Solver demo

Short live proof that TexasSolver runs on the lab and gives correct answers.

- `solver_demo.sh`: solves a textbook river game (nuts or air against a bluff-catcher, pot-sized all-in) and prints the
  solver's strategy next to the theory: always bet the nuts, bluff 9.4% of the air, call 50%.
- `2026-11-05-solver-demo.typescript` / `.timing`: the raw terminal recording made with
  `script -q -T demo.timing -c "bash solver_demo.sh" demo.typescript`.
- `../media/2026-11-05-solver-demo.mp4`: that recording replayed with its real timing by
  `../scripts/render_terminal_video.py` (pauses over 1.5 s shortened). Result: AA bet 100%, 76 bluffed 9.9%,
  KQ calls 50.0%, exploitability 0.06%, solved in 0.05 s.
