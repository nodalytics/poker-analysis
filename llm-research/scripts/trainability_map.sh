#!/bin/sh
# Trainability map (docs/2026-11-09-fractal-ideas.md, section 4): short MLP self-play runs from scratch over a grid of
# learning rate x KL pull, each scored against run 2's table after 4 and 8 minutes. Sohl-Dickstein (2024) found the
# boundary between trainable and divergent hyperparameters to be fractal; this asks where our self-play sits.
# Run on the lab from llm-research: sh scripts/trainability_map.sh PCI_GPU WAIT "LRS" "KLS"
#   PCI_GPU  GPU index in PCI bus order (0 = 3090, 1 = 4090, 2 = RTX PRO 6000)
#   WAIT     1: first wait until runmlp2 and runmlp3 have stopped (their train.log says "stopped")
# Each run writes blueprint/tmap/lr<LR>_kl<KL>/progress.log; its table and checkpoint are deleted to save disk.
GPU=$1; WAIT=$2; LRS=$3; KLS=$4
cd ~/llm-research || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8
if [ "$WAIT" = 1 ]; then
  until grep -q stopped blueprint/runmlp2/train.log 2>/dev/null && grep -q stopped blueprint/runmlp3/train.log 2>/dev/null; do
    sleep 120
  done
fi
for lr in $LRS; do
  for kl in $KLS; do
    out=blueprint/tmap/lr${lr}_kl${kl}
    [ -f "$out/train.log" ] && continue
    mkdir -p "$out"
    .venv-gpu/bin/python scripts/mlp_selfplay.py "$out" --minutes 8 --eval-minutes 4 --lr "$lr" --kl "$kl" \
      --magnet-tau 0.0003 --entropy 0.01 --avg-tau 0.001 > "$out/stdout.log" 2>&1 || echo "failed" >> "$out/train.log"
    rm -f "$out/avg.f32" "$out/checkpoint.pt"
  done
done
echo done > blueprint/tmap/queue_$GPU.done
