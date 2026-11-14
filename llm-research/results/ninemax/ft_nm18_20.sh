#!/bin/bash
# one-factor fine-tunes from nm16's final checkpoint: control, weaker KL pull, looser NeuRD bound; then size checks
cd ~/llm-research
while kill -0 1221227 2>/dev/null || kill -0 1221229 2>/dev/null; do sleep 30; done
sleep 60
cp blueprint/nm16/checkpoint.pt blueprint/init/nm16_12h.pt
export PYTHONPATH=scripts CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=1
BASE="--minutes 120 --eval-minutes 30 --batch 16384 --pool-share 0.5 --pool reg,reg,tag,maniac --chunks 4 --exploit-share 0.25 --exploit-reset 60 --entropy 0.02 --pg neurd --keep-hours 1 --tourney --no-open-limp --menu mtt --depths 10,25 --ante 1 --init blueprint/init/nm16_12h.pt"
mkdir -p blueprint/nm18 blueprint/nm19 blueprint/nm20
CUDA_VISIBLE_DEVICES=1 .venv-gpu/bin/python scripts/ninemax_selfplay.py blueprint/nm18 $BASE --kl 0.3 > blueprint/nm18/stdout.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 .venv-gpu/bin/python scripts/ninemax_selfplay.py blueprint/nm19 $BASE --kl 0.05 > blueprint/nm19/stdout.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 .venv-gpu/bin/python scripts/ninemax_selfplay.py blueprint/nm20 $BASE --kl 0.3 --neurd-beta 6 > blueprint/nm20/stdout.log 2>&1 &
# meanwhile: the starting point's size check, on GPU 1 beside the control
CUDA_VISIBLE_DEVICES=1 .venv-gpu/bin/python scripts/ninemax_sizecheck.py blueprint/nm16/checkpoint.pt --icm --out results/ninemax/sizecheck_nm16_12h_self_icm.json > results/ninemax/sizecheck_nm16_12h_self_icm.log 2>&1
wait
for r in nm18 nm19 nm20; do
  CUDA_VISIBLE_DEVICES=1 .venv-gpu/bin/python scripts/ninemax_sizecheck.py blueprint/$r/avg_2h.pt --icm --out results/ninemax/sizecheck_${r}_2h_self_icm.json > results/ninemax/sizecheck_${r}_2h_self_icm.log 2>&1 &
done
wait
echo ALL DONE
