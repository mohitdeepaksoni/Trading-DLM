#!/bin/bash
cd ~/Trading-DLM && . venv/bin/activate
python scripts/p3_event_harness.py --response dlogPX --reuse-store > runs/us_sectors/harness_dlogPX_rerun.log 2>&1
python scripts/p3_event_stocks.py --root runs --top 500 > runs/us_stocks_wide/p3_event_stocks.log 2>&1
echo "[p3 stocks done]" >> runs/us_stocks_wide/p3_event_stocks.log
