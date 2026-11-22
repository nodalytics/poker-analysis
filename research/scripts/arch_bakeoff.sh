#!/bin/sh
# Network bake-off: each architecture copies run 2's table (distillation from scratch, DISTILL minutes), then is
# scored three ways: cross-entropy to the table and head-to-head against it in the simplified game (both in
# progress.log, every 10 minutes and at the end), and LBR in the real game (lbr/report.json, run in the background).
# Run on the lab from research: sh scripts/arch_bakeoff.sh PCI_GPU WAIT_FOR "ARCH:MICRO ..." [DISTILL]
#   PCI_GPU   GPU index in PCI bus order (0 = 3090, 1 = 4090, 2 = RTX PRO 6000)
#   WAIT_FOR  a file that must exist (or contain "stopped") before starting; "-" to start at once
#   MICRO     gradient-accumulation chunks for networks that do not fit at the full batch
GPU=$1; WAIT_FOR=$2; LIST=$3; DISTILL=${4:-30}
cd ~/research || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8
if [ "$WAIT_FOR" != "-" ]; then
  until [ -f "$WAIT_FOR" ] && { ! echo "$WAIT_FOR" | grep -q train.log || grep -q stopped "$WAIT_FOR"; }; do sleep 120; done
fi
for item in $LIST; do
  arch=${item%%:*}; micro=${item##*:}
  out=blueprint/bakeoff/$arch
  [ -f "$out/train.log" ] && continue
  mkdir -p "$out"
  .venv-gpu/bin/python scripts/mlp_selfplay.py "$out" --arch "$arch" --micro "$micro" --distill blueprint/run2 \
    --distill-minutes "$DISTILL" --minutes 0 > "$out/stdout.log" 2>&1 || { echo failed >> "$out/train.log"; continue; }
  # LBR on the exported table, in the background on the CPU; the 0.2 GB table is deleted afterwards
  (OMP_NUM_THREADS=1 LBR_BOT=blueprint BLUEPRINT_RUN=$out nice -n 10 .venv/bin/python -W ignore scripts/lbr.py \
    "$out/lbr" 3000 8 200 > "$out/lbr.log" 2>&1; rm -f "$out/avg.f32") &
done
wait
echo done > blueprint/bakeoff/queue_$GPU.done
