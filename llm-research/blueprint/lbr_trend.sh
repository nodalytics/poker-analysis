#!/bin/sh
# Luck-adjusted LBR against each new blueprint checkpoint: how fast does the blueprint get harder to beat?
# Run on the lab: sh blueprint/lbr_trend.sh RUN_DIR [HANDS] [WORKERS]  ->  RUN_DIR/lbr_trend.tsv
# Each checkpoint is copied to a snapshot first, so training can keep writing.
RUN=${1:-blueprint/run2}; HANDS=${2:-3000}; WORKERS=${3:-12}
cd ~/llm-research || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LBR_BOT=blueprint
SNAP=$RUN/snapshot
last=""
[ -f "$RUN/lbr_trend.tsv" ] || printf "iterations\tlbr_bb100\tci_lo\tci_hi\n" > "$RUN/lbr_trend.tsv"
while true; do
  it=$(cat "$RUN/iterations.txt" 2>/dev/null)
  if [ -n "$it" ] && [ "$it" != "$last" ]; then
    mkdir -p "$SNAP"
    cp "$RUN/tree.tsv" "$RUN/avg.f32" "$RUN/regret.f32" "$SNAP/"
    out=lbr_trend_$it
    BLUEPRINT_RUN=$SNAP nice -n 10 .venv/bin/python -W ignore scripts/lbr.py "$out" "$HANDS" "$WORKERS" 200 > "$out.log" 2>&1
    .venv/bin/python -c "
import json; r = json.load(open('$out/report.json'))
print('$it', r['luck_adjusted_bb100'], *r['luck_adjusted_ci95'], sep='\t')" >> "$RUN/lbr_trend.tsv"
    last=$it
  fi
  if grep -q stopped "$RUN/train.log" 2>/dev/null && [ "$it" = "$last" ]; then break; fi
  sleep 120
done
