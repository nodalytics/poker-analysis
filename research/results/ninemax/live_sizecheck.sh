#!/bin/bash
# size check on the live (not averaged) networks: does the live policy defend the big blind and the average lag?
cd ~/research
export PYTHONPATH=scripts CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1
run() { CUDA_VISIBLE_DEVICES=$1 .venv-gpu/bin/python scripts/ninemax_sizecheck.py blueprint/$2/checkpoint.pt --part net --icm --out results/ninemax/sizecheck_$2_live_self_icm.json > results/ninemax/sizecheck_$2_live_self_icm.log 2>&1; }
run 0 nm16 & run 0 nm17 & run 1 nm18 & run 2 nm19 & run 2 nm20 &
wait
echo ALL DONE
