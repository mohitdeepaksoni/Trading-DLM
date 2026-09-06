#!/bin/bash
# v1 pooled-training rung: pooled US+NSE, then each market alone on the same code path (CPCV 6x2, h=20, monthly dates)
cd ~/Trading-DLM && . venv/bin/activate; mkdir -p runs/p4_pooled
for M in us,nse us nse; do
  python scripts/p4_train_pooled.py --markets $M --h 20 --every 21 --steps 600 > runs/p4_pooled/train_${M//,/+}_h20.log 2>&1
done
echo "[pooled chain done]" >> runs/p4_pooled/train_nse_h20.log
