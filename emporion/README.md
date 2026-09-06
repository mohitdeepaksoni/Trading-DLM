# emporion (package) — P0 foundation

```
emporion/
  core/       registry (Agora pattern) · hashing · hash-chained Ledger · perf (@timed, budgets)
  data/       BitemporalStore (Parquet + DuckDB, as_of mandatory) · loaders: synthetic, nasdaq (US; stooq kept but JS-blocked from cloud IPs), binance (crypto), nse (India)
  features/   feature registry with declared lags; replay/live modes through one code path; content-addressed cache; P0 library
  evaluate/   metrics · costs (next-bar execution enforced) · CPCV · Deflated Sharpe · PBO (CSCV) · stationary bootstrap · certify
  baselines/  buy_and_hold · equal_weight · inverse_vol · HRP · momentum_top
```
Scripts: `p0_certify_baselines.py` (P0), `p1_vol_nulls.py` (vol nulls), `p2_regime_conformal.py` (HMM gate + conformal), `p2_kronos_rankic.py` (Kronos zero-shot RankIC), `p0_show.py`, `run_bg.sh` (detached launcher).

Run: `python -m pytest -q tests` then `python scripts/p0_certify_baselines.py --market us|crypto|nse|synthetic`.
