#!/bin/sh
# Phase C launcher: wait for nm10 and nm12 to finish, export them, then train against the reg pool.
cd ~/research
until grep -q stopped blueprint/nm10/train.log 2>/dev/null && grep -q stopped blueprint/nm12/train.log 2>/dev/null; do sleep 120; done
sleep 30
export PYTHONPATH=scripts CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1
.venv-gpu/bin/python scripts/ninemax_export.py blueprint/nm10 --note "MTT, 5-15bb and 20-50bb, no open-limp, mtt menu, 24h"
.venv-gpu/bin/python scripts/ninemax_export.py blueprint/nm12 --note "MTT, 10-25bb, no open-limp, mtt menu, 24h"
COMMON="--minutes 720 --eval-minutes 30 --batch 16384 --pool-share 0.5 --pool reg,reg,tag,maniac --chunks 4 --exploit-share 0.25 --exploit-reset 60 --entropy 0.02 --pg neurd --keep-hours 4 --tourney --no-open-limp --menu mtt --depths 10,25"
mkdir -p blueprint/nm13 blueprint/nm14
CUDA_VISIBLE_DEVICES=1 setsid nohup .venv-gpu/bin/python scripts/ninemax_selfplay.py blueprint/nm13 $COMMON --init blueprint/nm12/checkpoint.pt > blueprint/nm13/stdout.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 setsid nohup .venv-gpu/bin/python scripts/ninemax_selfplay.py blueprint/nm14 $COMMON --kl-start 0.1 --kl-ramp 180 > blueprint/nm14/stdout.log 2>&1 &
echo "launched $(date)"
