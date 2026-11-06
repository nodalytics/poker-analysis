#!/bin/sh
# Hold'em: RL robust exploitation (rlbot vs target or co-trained adversary) and RL exploiters against our adaptive
# bots. Runs on the lab after the range-inference and regime studies free their cores.
cd ~/llm-research
export POKER_ANALYSIS_ROOT=$HOME/poker-analysis TEXASSOLVER_BIN=$HOME/solver-fast/build/console_solver SOLVER_THREADS=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
until grep -q ALLDONE range_inference.run.log 2>/dev/null && grep -q ALLDONE regimes.run.log 2>/dev/null; do sleep 60; done
for spec in "carl 1.0" "carl 0.5" "max 1.0" "max 0.5"; do
  set -- $spec
  d=rl_rnr_${1}_$(echo $2 | tr . _)
  RL_Q=$HOME/llm-research/$d/q.json nice -n 12 .venv/bin/python -W ignore scripts/rl_agent.py $d 120 --short --rnr $2 --target $1 --workers 8 --sessions-per-worker 4 > $d.train.log 2>&1 &
done
for target in adaptbot adaptbot_recent; do
  d=rl_exploit_$target
  RL_Q=$HOME/llm-research/$d/q.json nice -n 12 .venv/bin/python -W ignore scripts/rl_agent.py $d 120 --short --vs $target --workers 8 --sessions-per-worker 2 > $d.train.log 2>&1 &
done
wait
for spec in "carl 1.0" "carl 0.5" "max 1.0" "max 0.5"; do
  set -- $spec
  d=rl_rnr_${1}_$(echo $2 | tr . _)
  export RL_Q=$HOME/llm-research/$d/q.json RL_AVG=$HOME/llm-research/$d/avg.json
  nice -n 12 .venv/bin/python -W ignore scripts/bot_sim.py eval_${d}_vs_target 300 --short --workers 16 rlbot $1 > eval_$d.log 2>&1
  nice -n 12 .venv/bin/python -W ignore scripts/bot_sim.py eval_${d}_vs_solver 300 --short --workers 16 rlbot solverbot >> eval_$d.log 2>&1
done
for target in adaptbot adaptbot_recent; do
  d=rl_exploit_$target
  export RL_Q=$HOME/llm-research/$d/q.json
  nice -n 12 .venv/bin/python -W ignore scripts/bot_sim.py eval_$d 400 --short --workers 16 rlbot $target > eval_$d.log 2>&1
done
echo ALLDONE > rl_rnr_holdem.done
