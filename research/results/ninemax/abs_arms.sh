#!/bin/bash
# bucket-abstraction test, 2026-11-14: four fresh runs of the champion's recipe (nm17, no warm start), one per card
# abstraction, all on the same GPU so each gets the same hardware; 6 hours each
cd ~/research
export PYTHONPATH=scripts CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1
R="--minutes 360 --eval-minutes 30 --batch 16384 --pool-share 0.5 --pool reg,reg,tag,maniac --chunks 4 --exploit-share 0.25 --exploit-reset 60 --entropy 0.02 --pg neurd --keep-hours 2 --tourney --no-open-limp --menu mtt --depths 10,25 --ante 1 --short 0.5 --lib blueprint/libbp7.so"
launch() { mkdir -p blueprint/$1; (setsid nohup .venv-gpu/bin/python scripts/ninemax_selfplay.py blueprint/$1 $R --abs $2 > blueprint/$1/stdout.log 2>&1 &); }
launch nm21 blueprint/out               # control: 200 / 200 / 200, equal-population
launch nm22 blueprint/abs_100_200_500   # coarse to fine
launch nm23 blueprint/abs_1000_500_200  # fine to coarse
launch nm24 blueprint/abs_200g2         # 200 each, cuts denser at the strong end
echo launched
