#!/bin/bash
cd ~/Trading-DLM && . venv/bin/activate
python -u scripts/p2_kronos_rankic.py --market us --model NeoQuasar/Kronos-base --horizon 5 --n-dates 60 --top-n 22 > runs/us/p2-kronos-base.log 2>&1
python -u scripts/p2_kronos_rankic.py --market crypto --model NeoQuasar/Kronos-small --horizon 5 --n-dates 60 --top-n 8 > runs/crypto/p2-kronos-small.log 2>&1
python -u scripts/p2_kronos_rankic.py --market crypto --model NeoQuasar/Kronos-base --horizon 5 --n-dates 60 --top-n 8 > runs/crypto/p2-kronos-base.log 2>&1
echo QUEUE_DONE >> runs/kronos_queue.log
