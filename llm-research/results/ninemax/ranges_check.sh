#!/bin/bash
# size check and training signal with the players before the hero on their real ranges (--ranges), 2026-11-14
cd ~/llm-research
export PYTHONPATH=scripts CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1
R=results/ninemax
sc() { CUDA_VISIBLE_DEVICES=$1 .venv-gpu/bin/python scripts/ninemax_sizecheck.py blueprint/$2/checkpoint.pt --part net --icm --ranges --opp $3 --out $R/sizecheck_$2_live_$3_icm_ranges.json > $R/sizecheck_$2_live_$3_icm_ranges.log 2>&1; }
sg() { CUDA_VISIBLE_DEVICES=$1 .venv-gpu/bin/python scripts/ninemax_signal.py blueprint/$2/checkpoint.pt --opp $3 --out $R/signal_$2_$3.json > $R/signal_$2_$3.log 2>&1; }
(sc 0 nm16 self; sc 0 nm16 reg) &
(sc 1 nm17 self; sg 1 nm16 tag) &
(sg 2 nm16 self; sg 2 nm16 reg; sg 2 nm16 maniac) &
wait
echo ALL DONE
