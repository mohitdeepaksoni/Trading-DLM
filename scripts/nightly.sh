#!/bin/bash
# Nightly shadow job: refresh point-in-time data, build + publish the pre-registered signal files, chain, ledger,
# OpenTimestamps, then the integrity check. Trades nothing. Run from the repo root.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/signals
# idempotent per UTC day: the fallback cron re-fires ~75 min after the primary; if today's NSE file exists, nothing to do
if ls runs/signals/nse-$(date -u +%Y-%m-%d).json > /dev/null 2>&1; then echo "[nightly] files for $(date -u +%Y-%m-%d) already exist; skipping" | tee -a runs/signals/nightly.log; exit 0; fi
# Every step's exit code is collected and the script fails at the end if any failed. Before 2026-11-03 the steps were piped
# through tee without a check, so a crashed run still exited 0 and showed green (incident 13: 11-01..11-03 built nothing).
rc=0
python scripts/shadow_daily.py 2>&1 | tee -a runs/signals/nightly.log || rc=1
python scripts/integrity_check.py 2>&1 | tee -a runs/signals/nightly.log || rc=1
python scripts/report_signals.py 2>&1 | tee -a runs/signals/nightly.log || rc=1
python scripts/paper_trade.py --market nse 2>&1 | tee -a runs/signals/nightly.log || rc=1
python scripts/data_inventory.py --out runs/signals > /dev/null 2>&1 || echo "[nightly] data inventory FAILED" | tee -a runs/signals/nightly.log
# the run is only a success if today's files actually exist — a green tick must mean a published day
for m in nse us_stocks_wide crypto_full; do
  [ -f "runs/signals/${m}-$(date -u +%Y-%m-%d).json" ] || { echo "[nightly] MISSING runs/signals/${m}-$(date -u +%Y-%m-%d).json" | tee -a runs/signals/nightly.log; rc=1; }
done
exit $rc
