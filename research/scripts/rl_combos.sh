#!/bin/sh
# Combined RL variants (niche ideas + RL), trained on the league, then tested on unseen opponents.
# Runs on the lab after the environment suite and meta game finish.
cd ~/research
export POKER_ANALYSIS_ROOT=$HOME/poker-analysis TEXASSOLVER_BIN=$HOME/solver-fast/build/console_solver SOLVER_THREADS=2
until grep -q ALLDONE env_suite.run.log 2>/dev/null && grep -q ALLDONE meta_game.run.log 2>/dev/null; do sleep 60; done
for v in belief boltzmann kelly belief,boltzmann,kelly; do
  d=rl_combo_$(echo $v | tr , _)
  flags=$(echo $v | sed 's/^/--/; s/,/ --/g')
  nice -n 12 .venv/bin/python -W ignore scripts/rl_agent.py $d 200 --short --league $flags --workers 14 --sessions-per-worker 4 > $d.train.log 2>&1 &
done
wait
for v in league belief boltzmann kelly belief,boltzmann,kelly; do
  if [ $v = league ]; then d=rl_league_short; mode=""; else d=rl_combo_$(echo $v | tr , _); mode=$v; fi
  export RL_Q=$HOME/research/$d/q.json RL_AVG=$HOME/research/$d/avg.json RL_MODE=$mode
  nice -n 12 .venv/bin/python -W ignore scripts/bot_sim.py eval_${d}_unseen 300 --short --workers 20 rlbot rlavg solverbot solverbot#2 sol nora > eval_$d.log 2>&1
  nice -n 12 .venv/bin/python -W ignore scripts/bot_sim.py eval_${d}_shifters 300 --short --workers 20 rlbot rlavg solverbot shift_9100001 shift_9100002 shift_9100003 >> eval_$d.log 2>&1
  nice -n 12 .venv/bin/python -W ignore scripts/bot_sim.py eval_${d}_hu 400 --short --workers 20 rlavg solverbot >> eval_$d.log 2>&1
done
echo ALLDONE > rl_combos.done
