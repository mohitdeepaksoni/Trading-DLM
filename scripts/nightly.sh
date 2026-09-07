#!/bin/bash
# Nightly shadow job: refresh point-in-time data, build + publish the pre-registered signal files, chain, ledger,
# OpenTimestamps, then the integrity check. Trades nothing. Run from the repo root.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/signals
python scripts/shadow_daily.py 2>&1 | tee -a runs/signals/nightly.log
python scripts/integrity_check.py 2>&1 | tee -a runs/signals/nightly.log
python scripts/report_signals.py 2>&1 | tee -a runs/signals/nightly.log
