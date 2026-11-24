#!/bin/bash
# The ESS floor in play: the same deals (seed 4242), nm25 searching at 256 samples, with the floor (0.02) and without (0)
cd ~/research || exit 1
O=~/scratch/essmatch
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts
for opp in net call fold raise; do
  for e in 0 0.02; do
    [ -s $O/ess_${opp}_${e}.json ] && continue  # finished before the reboot
    .venv-gpu/bin/python scripts/ninemax_search.py blueprint/nm25 --hands 200000 --seed 4242 --opp-bias $opp \
      --min-ess $e --lib blueprint/libbp7.so --per-deal $O/ess_${opp}_${e}.npy --out $O/ess_${opp}_${e}.json \
      > $O/ess_${opp}_${e}.log 2>&1; echo "$opp $e rc=$?" >> $O/done.txt
  done
done
echo finished >> $O/done.txt
