#!/bin/bash
cd ~/Trading-DLM && . venv/bin/activate
python -u scripts/p2_kronos_rankic.py --market nse --model NeoQuasar/Kronos-small --horizon 5 --n-dates 60 --top-n 200 > runs/nse/p2-kronos-small-h5.log 2>&1
python -u scripts/p2_kronos_rankic.py --market nse --model NeoQuasar/Kronos-base --horizon 5 --n-dates 60 --top-n 200 > runs/nse/p2-kronos-base-h5.log 2>&1
python -u scripts/p2_kronos_rankic.py --market nse --model NeoQuasar/Kronos-small --horizon 20 --n-dates 40 --step 20 --top-n 200 > runs/nse/p2-kronos-small-h20.log 2>&1
python -u scripts/p2_kronos_rankic.py --market us --model NeoQuasar/Kronos-small --horizon 20 --n-dates 40 --step 20 --top-n 22 > runs/us/p2-kronos-small-h20.log 2>&1
echo QUEUE_NSE_DONE >> runs/kronos_queue_nse.log
