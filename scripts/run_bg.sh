#!/bin/bash
# Detached launcher: returns immediately, survives ssh disconnect. Usage: run_bg.sh --market nse
cd ~/Trading-DLM && . venv/bin/activate
setsid nohup python -u scripts/p0_certify_baselines.py "$@" > "runs/$2.log" 2>&1 < /dev/null &
echo "launched pid $! -> runs/$2.log"
