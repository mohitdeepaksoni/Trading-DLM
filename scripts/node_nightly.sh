#!/bin/bash
# Nightly job on the PERSISTENT NODE — primary runner since 2026-10-16 (incident 7).
#
# Why the node and not only GitHub Actions: the HF private-storage quota blocks parquet (LFS) pushes, so a runner's refreshed
# stores die with the runner. Sources with no history endpoint — promoter pledges, bulk/block deals, analyst consensus,
# CoinGecko supply, FII/DII flows — are snapshot-only, so a night not persisted is a night lost for good. The node's disk keeps
# them until the quota is freed, at which point the store push resumes by itself and nothing further is needed here.
#
# Schedule: 00:15 UTC (cron), i.e. after the UTC date rolls over so the files carry the new day, and well before the 05:00 UTC
# Actions fallback — which then finds the day's files in the pulled record and exits early. The two can never build the same day.
set -o pipefail
cd "$(dirname "$0")/.." || exit 1
. venv/bin/activate || exit 1
LOG=runs/node_nightly.log
{
  echo "===== node nightly $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  python scripts/shadow_sync.py pull      # safe: never overwrites a longer local record
  bash scripts/nightly.sh                 # refresh data, build + publish + stamp the files, integrity, report, paper book
  python scripts/shadow_sync.py push      # record always; stores soft-fail while the quota is full
  echo "===== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
} >> "$LOG" 2>&1
