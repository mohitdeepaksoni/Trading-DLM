# 03 — Proving it is not luck: the evaluation toolkit

*This is the document that separates a hobby from a system. Every number in docs 01–02 came with a caveat; this is the machinery that enforces the caveats. Treat it as Agora's "same honest yardstick for every solver" — here the yardstick is the certified out-of-sample, cost-adjusted, search-deflated return distribution.*

---

## 1. The three ways backtests lie
1. **Look-ahead / leakage** — features built with information not available at decision time (restated fundamentals, survivorship-free universes, index membership as of today, adjusted prices with future splits, *LLMs that have read the future*).
2. **Selection under multiple testing** — try N strategies, report the best. With N = 100 i.i.d. null strategies the expected max Sharpe ≈ 2.5 on 5 years of daily data.
3. **Non-stationarity** — the 2010–2021 low-rate regime is not 2022–2026. A model fit on one regime is a bet on that regime persisting.

## 2. Structural leakage exclusion (must be by construction — arXiv:2608.27734)
- **Point-in-time data** for everything: prices *as quoted that day*, fundamentals *as first published*, universes *as constituted that day* (survivorship).
- **Feature registry:** every feature is a registered function `f(data_{≤t})` with a declared *availability lag*; the backtester refuses unregistered features. This alone caught a Sharpe-35 leaky oracle that passed DSR and PBO.
- **Embargo** between train and test windows ≥ maximum feature/label horizon.
- **LLM rule:** no LLM sees price/news data after the training cutoff during any evaluated run; LLM outputs are logged and hashed.

## 3. Cross-validation that respects time
- **Walk-forward** (expanding or rolling) — industry standard, single path, high variance.
- **Purged K-fold + embargo** (López de Prado 2018): drop training samples whose labels overlap the test window.
- **Combinatorial Purged CV (CPCV):** all C(N, k) train/test splits ⇒ a *distribution* of OOS paths instead of one. 2024–26 synthetic-control study (Arian, Norouzi, Seco): CPCV has the lowest PBO and best DSR statistic of the OOS methods.

## 4. Deflating for search
- **Probability of Backtest Overfitting (PBO)** via CSCV (Bailey, Borwein, López de Prado, Zhu 2014): split performance matrix into S sub-blocks, for each combination pick IS-best strategy, record its OOS rank; PBO = P(OOS rank below median). Reject if PBO > 0.1–0.2.
- **Deflated Sharpe Ratio (DSR)** (Bailey & López de Prado 2014):
```
SR*  = sqrt(V[SR_n]) · [ (1−γ) Z⁻¹(1−1/N) + γ Z⁻¹(1 − 1/(N·e)) ]     (expected max SR of N null trials)
DSR  = Z[ (SR − SR*) · sqrt(T−1) / sqrt(1 − γ₃ SR + (γ₄−1)/4 · SR²) ]
```
with γ₃, γ₄ skew and kurtosis of returns, T observations, N trials, V[SR_n] variance of trial Sharpes. **Requires logging N** — hence the *ledger*: every strategy evaluation (including hyper-parameter draws and factor candidates) is written to an append-only table with its IS/OOS stats. The search-aware evaluation in arXiv:2608.27734 is exactly this.
- **Minimum backtest length** (Bailey & López de Prado 2014): `MinBTL ≈ [ (1−γ)Z⁻¹(1−1/N) + γZ⁻¹(1−1/(Ne)) ]² / SR²` years needed for a given N and target SR.
- **Harvey–Liu haircut Sharpe** and t-stat > 3 hurdle for new factors (Harvey, Liu, Zhu 2016).

## 5. Confidence intervals that respect autocorrelation
- Stationary bootstrap (Politis–Romano) or block bootstrap of the OOS return series → CI on Sharpe, CAGR, MDD. Certify only if the CI excludes zero *after costs*.
- HAC (Newey–West) standard errors for mean-return tests.
- Report **seed robustness** (≥5 seeds) and the **spread** across CPCV paths — the 2026 E2E papers do this; adopt it.

## 6. Costs and capacity — always in the loss and in the report
- Commission + exchange fees + taxes (India: STT, stamp duty, GST on brokerage; UK: stamp duty 0.5 % on shares; US: SEC fee + FINRA TAF).
- Spread: half bid-ask per side (use actual quotes; ≥ 5 bps for mid-caps).
- Impact: √-law `Y σ √(Q/V)`; strategy **capacity** = the AUM at which impact erases the alpha.
- **Breakeven transaction cost** = the per-trade cost at which OOS Sharpe → 0; report it. Retail-realistic all-in cost for Indian cash equities ≈ 10–25 bps round-trip; US large caps ≈ 2–5 bps; crypto 10–50 bps.
- Slippage model for market orders; fill-probability model for limit orders (or assume none fill better than mid).

## 7. Compounding checks (the den Haan analogue)
Small per-step errors compound: simulate the full equity curve with the model's *own* decisions feeding the next state (position, cash, margin) — never evaluate one-step predictions in isolation. Report max drawdown, drawdown duration, Calmar, tail (p1, p0.1 daily), and *ulcer index*.

## 8. Regime-sliced reporting
Slice OOS by HMM regime / VIX tercile / rate regime / year. A strategy that earns everything in 2020–21 and nothing else is a regime bet — label it as such.

## 9. Paper-to-live drift
Track **implementation shortfall**: live P&L − backtest P&L on the same signals, decomposed into slippage, latency, partial fills, data differences. A system is "validated" only when live shortfall is within its own bootstrap CI for ≥ 3 months.

## 10. The certification checklist (gate before any capital)
```
[ ] point-in-time data; feature registry with lags; embargo ≥ horizon
[ ] CPCV with ≥ 6 blocks; ≥ 5 seeds; report distribution not point estimate
[ ] ledger: N trials recorded; DSR computed against N; PBO ≤ 0.2
[ ] costs (fees+spread+impact+tax) inside the loss and the report; breakeven cost reported
[ ] stationary-bootstrap CI on OOS Sharpe excludes 0
[ ] regime-sliced results; no single regime > 60 % of P&L
[ ] compounding simulation with own decisions; MDD & duration reported
[ ] passive benchmark (buy-and-hold / equal-weight / HRP) beaten on Sharpe AND on CI
[ ] ≥ 3 months paper trading with implementation shortfall inside CI
```
Anything that fails a line is *research*, not a strategy.

## 11. Universe construction is a leak the gate cannot see (added 2026-09-06)
A universe chosen on today's information — current market cap, current index membership, "survived to today" — makes
every backward-looking size, liquidity or "cheapness" signal look predictive: the names that were small or thinly
traded years ago and are in today's list are, by construction, the ones that rose. Measured: on a top-200-by-current-cap
US panel, "illiquidity" scored rank-IC +0.09 with t = 6.9 and certified; that is the fingerprint of selection, not
alpha. The Deflated Sharpe, PBO and bootstrap operate on the return series and are blind to it.
**Rule:** the universe at decision date t is chosen only from information known at t (e.g. top-K by trailing dollar
volume among names listed ≥ 300 days), the benchmark is the equal-weight of that same point-in-time universe, and
cross-sectional demeaning happens within it. Delisted names must be present in the pool; where a data source only
serves currently-listed names (Nasdaq screener), the residual survivorship is stated in the report.

## 12. Resolve revisions AFTER filtering by the decision date (added 2026-09-06)
A bitemporal table read "as of now" keeps only the latest revision of each fact. If a script then filters that frame by
`knowledge_time <= t`, every fact that was ever restated (a 10-K/A refiled two years later) disappears from the past —
it looks unknown until its restatement. Measured: US EPS coverage collapsed from ~80 % to ~20–40 % of the universe.
**Rule:** query the store as-of the decision date, or read `all_versions()` and resolve revisions after the date filter.
Regression test: `tests/test_store_features.py::test_all_versions_then_filter_keeps_first_prints`.

## 13. Count procedures, not evaluations (added 2026-09-06)
Re-running an unchanged procedure after a data or code fix is not a new draw from the strategy search, yet the naive
ledger count treated it as one: after a day of reruns the accruals rung faced a Deflated-Sharpe hurdle built from 78
"trials" that were mostly re-evaluations. Every trial now carries a **procedure hash** (rung name + the calling script's
code + its arguments). `Ledger.n_trials(scope, distinct=True)` counts distinct procedures — the honest search count that
the gate uses — and the strict count of all evaluations is reported alongside (`n_evaluations`, `dsr_strict`) so the
conservative number is never hidden. Changing code or parameters changes the hash and *is* a new trial.
Test: `tests/test_gate_validation.py::test_rerunning_the_same_procedure_is_not_a_new_trial`.
**Scope versioning.** Trials recorded before the procedure hash existed count one each (nothing is backfilled or
deleted). When the pipeline itself is corrected (a new universe rule, a new revision-resolution rule), the corrected
procedure family gets a new scope (`…-pit2`); the old scope stays on the ledger as the record of what was tried under the
old pipeline. The report quotes the new scope's count and names the old one.

## 14. NSE results: two sources and undeclared XBRL contexts (added 2026-09-06)
Two data defects hid inside "8,039 rows loaded" and only showed up as a degenerate SUE rung (Sharpe exactly 0) and a
value rung that started in 2022 instead of 2018:
1. **Old NSE `…_WEB.xml` instances never declare their base contexts.** Facts reference `OneD` (current quarter),
   `FourD` (year-to-date), `OneI` (instant) but only the dimensional children (`OneOperatingExpenses01D`, …) carry
   a period — and those files stamp the *quarter* period on the `Four…` children too. Taking "any context whose period
   is the quarter" therefore returned year-to-date revenue and EPS (INFY Dec-2018 EPS 24.9 instead of 8.29). Rule:
   infer only `OneD`/`OneI` from their children, never trust a `Two…/Three…/Four…/Five…/Six…` period, and read the
   quarter from `OneD` first. Test: `tests/test_nse_xbrl_contexts.py`.
2. **`corporates-financial-results` stops at the Dec-2024 quarter** (capped at 130 filings and superseded by SEBI's
   Integrated Filing from the Mar-2025 quarter). The loader now merges `integrated-filing-results` (type
   "Integrated Filing- Financials", plain XBRL in the `xbrl` field). Coverage went from 2022-03..2024-12 with 59 % null
   EPS to 2018-06..2026-06 with none, for the four symbols checked by hand.
3. A symbol whose listing call failed was silently skipped (RELIANCE was missing from the first build). Listing calls
   now retry three times and failures are printed by name.
Sanity check used before trusting the rebuild: the largest quarter-to-quarter log change in revenue per symbol
(year-to-date contamination shows as jumps > 1.0 every fourth quarter). Splits and bonuses (RELIANCE 2024, HDFCBANK
2025) still distort seasonal EPS differences for four quarters — open item: per-share figures need share-count adjustment.
Shareholding table, same pass: (a) NSE's `corporate-share-holdings-master` holds filings from **Sep-2021 only** (the
new SEBI format); `from_date/to_date` windows work but return nothing earlier, so holdings rungs run on 2021→ and the
ledger says so. (b) From the Sep-2025 filing the percentage fact is reported as a **fraction** (0.143) instead of a
percent (14.3); the loader normalises per filing (promoter + public ≤ 1.5 ⇒ ×100). Without this the
institutional-change rung saw a −14-point "sale" in every name in one quarter. (c) `shares_total` (promoter + public
shares) is now stored and the ladder restates historical EPS on the current share count before seasonal differences
(`split_adjust`, tests in `tests/test_nse_shareholding_units.py`).
