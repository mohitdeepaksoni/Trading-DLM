#!/bin/bash
# rebuild NSE tables (default: results) then rerun the holdings/fundamentals ladder.  usage: run_nse_rebuild_ladder.sh [results|shareholding|results,shareholding]
ONLY=${1:-results}
cd ~/Trading-DLM && . venv/bin/activate
python scripts/build_nse_fundamentals.py --top 200 --only "$ONLY" --replace > runs/nse/rebuild_${ONLY//,/_}.log 2>&1
python scripts/p4_realdata_nse.py > runs/nse/ladder3.log 2>&1
echo "[chain done]" >> runs/nse/ladder3.log
