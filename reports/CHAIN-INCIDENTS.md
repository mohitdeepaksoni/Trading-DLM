# Shadow-chain operational incidents (append-only; every gap in the live record is explained here)

| # | date (UTC) | what happened | effect on the record | fix |
|---|---|---|---|---|
| 1 | 2026-09-06 → 09-09 | the nightly job refreshed US and crypto bars but never NSE bars; NSE files carried `last_bar` 2026-09-04 for three nights | NSE books ranked on stale closes (fundamentals-driven ranks barely moved; universe edges and decision prices were stale; the stale `last_bar` is inside each file) | job refreshes NSE bars, delivery and event tables nightly from 09-10 (`scripts/shadow_daily.py`) |
| 2 | 2026-09-11 23:30 | **night 7 never ran**: no GitHub Actions run fired (state untouched since 09-11 01:38; the PAT in use cannot read the Actions log, so the exact reason — dropped or delayed schedule — is unverified) | no 2026-09-12 files at the scheduled time; the paper book had no orders due (monthly rebalance), so no fills were missed | recovery run of the same job on the box at ~07:30 UTC 09-12 (US files still pre-open; the NSE file built from data known at the 09-11 close, i.e. `last_bar` 2026-09-11, but stamped after the 09-12 NSE open — a late stamp, provably without 09-12 prices); fallback cron 00:45 UTC + concurrency group; `nightly.sh` idempotent per day; `shadow_sync push` refuses to overwrite a thicker remote state. Recovery note: the box ran an older copy of `nightly.sh` (pre-paper-engine), so the paper book was marked in a second step after the code sync; 09-12 files chained 25–28, integrity 28/28 |

Rule: an operational miss is recorded, never back-filled silently. The seven-clean-nights clock for a new file counts scheduled runs
that produced the file before its market's next open; a late-stamped file does not count as clean for that market.
