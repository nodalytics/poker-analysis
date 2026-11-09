#!/bin/sh
# Learning-rule bake-off: MLP self-play from scratch, 30 minutes each, the same network and settings (the strong
# magnet of runmlp2) with one change per run: the policy update, the reward, or a league of past snapshots. Then Deep
# CFR (a regret network instead of a policy gradient) for 120 minutes. Each run is scored against run 2's table every
# 10 minutes (progress.log) and by LBR on its final export (lbr/report.json, in the background).
# Run on the lab from llm-research: sh scripts/learner_variants.sh PCI_GPU WAIT_FOR "NAME ..."
#   NAME: a2c ppo neurd qpg qlearn qmc duplicate tanh league deepcfr
GPU=$1; WAIT_FOR=$2; LIST=$3
cd ~/llm-research || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8
if [ "$WAIT_FOR" != "-" ]; then
  until [ -f "$WAIT_FOR" ]; do sleep 120; done
fi
BASE="--minutes 30 --eval-minutes 10 --kl 0.3 --magnet-tau 0.0003 --entropy 0.01 --avg-tau 0.001"
for name in $LIST; do
  out=blueprint/learners/$name
  [ -f "$out/train.log" ] && continue
  mkdir -p "$out"
  case $name in
    a2c|ppo|neurd|qpg|qlearn|qmc) cmd="scripts/mlp_selfplay.py $out $BASE --pg $name" ;;
    duplicate|tanh) cmd="scripts/mlp_selfplay.py $out $BASE --reward $name" ;;
    league) cmd="scripts/mlp_selfplay.py $out $BASE --league 0.3 --league-every 3" ;;
    deepcfr) cmd="scripts/deep_cfr.py $out --minutes 120 --eval-minutes 10" ;;
    *) echo "unknown $name"; continue ;;
  esac
  .venv-gpu/bin/python $cmd > "$out/stdout.log" 2>&1 || { echo failed >> "$out/train.log"; continue; }
  rm -f "$out/checkpoint.pt"
  (OMP_NUM_THREADS=1 LBR_BOT=blueprint BLUEPRINT_RUN=$out nice -n 10 .venv/bin/python -W ignore scripts/lbr.py \
    "$out/lbr" 3000 8 200 > "$out/lbr.log" 2>&1; rm -f "$out/avg.f32") &
done
wait
echo done > blueprint/learners/queue_$GPU.done
