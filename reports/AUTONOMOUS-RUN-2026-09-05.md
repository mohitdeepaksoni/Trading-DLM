# Autonomous run report — 2026-09-05 (Brev `emporion`, 1×A10G)

*Four-hour unattended window. Everything below ran on the Brev box; source of truth is `Trading-DLM/` locally, mirrored with rsync. All numbers are after 10 bps one-way costs with next-bar execution, through the hash-chained ledger.*

## 1. What was built and verified today
| Piece | State | Evidence |
|---|---|---|
| P0 foundation (registry, ledger, bitemporal store, loaders, feature registry, CPCV/DSR/PBO/bootstrap, baselines) | ✅ 24/24 tests on the box | `emporion/`, `tests/` |
| Data: US ETFs (Nasdaq API, keyless, 10 y) · crypto (Binance.US fallback) · **NSE full cash market 2022→today (2.28 M bars, 3,253 symbols, complete calendar)** | ✅ | `runs/{us,crypto,nse}/store` |
| P1 vol nulls: EWMA, HAR (level form), GARCH(1,1)-t + QLIKE/MSE evaluator | ✅ tests + 3 markets | `emporion/baselines/vol.py`, `scripts/p1_vol_nulls.py` |
| P2 regime: walk-forward Gaussian HMM (filtered probs, no look-ahead) + rolling split-conformal intervals | ✅ tests + 3 markets | `emporion/baselines/regime.py`, `scripts/p2_regime_conformal.py` |
| P2 encoder decision: Kronos zero-shot RankIC vs classical signals, walk-forward, ledgered | ✅ US, crypto; NSE 200-name (see §4) | `scripts/p2_kronos_rankic.py` |
| Perf discipline | budgets as tests; every run writes `benchmarks/*.json` | `PERFORMANCE.md` |

## 2. Baseline certification (P0), per market
| Market | Panel | Equal-weight | Best baseline | Certified? |
|---|---|---|---|---|
| US ETFs | 2,513 d × 22 | SR 0.69, MDD −28.7 % | momentum 0.79 | EW certified (CI>0, DSR 0.98); nothing beats EW with CI excluding 0 |
| NSE top-200 by traded value | 1,143 d × 200 | SR 0.94, MDD −22.4 % | momentum 1.02 | CI just above 0; PBO across baselines 0.48 → not certified |
| Crypto (Binance.US) | 2,506 d × 5 | SR 1.07, MDD −75 % | — | EW CI>0 but no rung beats it |

Reading: passive equal-weight is the bar; HRP/inverse-vol *underperform* it on NSE after costs (SR 0.10–0.15) because they concentrate in low-vol names in a bull tape. CPCV paths are identical for non-fitted baselines (by construction; becomes informative at P4).

## 3. Volatility nulls (P1) — QLIKE, lower is better, walk-forward
| Market | constant | EWMA | HAR | GARCH-t | best |
|---|---|---|---|---|---|
| US | 3.09 | 1.49 | 1.56 | **1.48** | GARCH-t |
| NSE (50) | 2.44 | 2.29 | 2.39 | **2.06** | GARCH-t |
| Crypto | 2.81 | **1.98** | 2.45 | 2.12 | EWMA |
Any learned vol model must beat these numbers on the same ledger. A log-HAR on daily r² was tried first and is biased (E log r² ≠ log E r²); the level-form HAR is the honest null with daily data.

## 4. Kronos zero-shot as a cross-sectional signal (P2 encoder decision)
Protocol: 60 walk-forward dates, 5-day horizon, RankIC across assets per date, t-stat over dates; momentum/reversal/random on the same dates.
| Panel | momentum 12-1 | reversal 5d | random | **Kronos-small** | **Kronos-base** |
|---|---|---|---|---|---|
| US 22 ETFs | +0.135 (t 2.7) | −0.011 | −0.027 | −0.080 (t −1.7) | −0.085 (t −1.7) |
| Crypto 5 | +0.195 (t 2.8) | +0.025 | −0.083 | +0.003 | −0.115 (t −1.6) |
| NSE 200 | +0.046 (t 2.6) | +0.003 | +0.004 | −0.010 (t −0.7) | −0.030 (t −1.6) |
| NSE 200, 20-day horizon (24 dates) | +0.026 (t 0.9) | +0.032 | −0.014 | −0.007 (t −0.3) | — |
| US 22, 20-day horizon (40 dates) | +0.124 (t 2.5) | −0.010 | −0.002 | −0.069 (t −1.4) | — |
Speed on the A10G: 37 ms per asset-date (Kronos-small, 400-bar context, 3 samples).

**Decision:** Kronos zero-shot is **not** a usable cross-sectional signal on any of our panels: its RankIC is at or below the random control on US, crypto and NSE at 5- and 20-day horizons, while plain 12-1 momentum is significantly positive on every panel (t 2.5–2.8). The paper's headline gain (RankIC vs other TSFMs on its own benchmark) does not transfer to daily cross-sections out of the box. **P2 encoder path:** build Emporion's own per-asset encoder (K-line tokenizer + signature features + set attention, doc 07 A.3) and keep Kronos only as an optional *fine-tuned* experiment on the ablation ladder, never as the frozen backbone. The larger model was worse than the smaller one on every panel, which also argues against scaling a price-only encoder.

## 5. Regime gate and conformal intervals (P2)
| Market | time in stress | EW calm SR | EW stress SR | best gate vs EW (excess/yr CI) | conformal 90 % coverage / width |
|---|---|---|---|---|---|
| US | 30 % | 0.67 | 0.79 | thr 0.9: SR 0.78, MDD −10.8 % (vs −28.7 %), CI −5.8..+3.5 % → not certified | 0.899 / 2.55 % |
| NSE | 34 % | 0.51 | **1.61** | every gate *loses* (CI entirely negative) | 0.900 / 3.38 % |
| Crypto | 3 % | 0.46 | 2.16 | no effect | 0.898 / 12.4 % |
Reading: "flat in high vol" is not an edge — in these samples the high-vol regime carried the rebounds. Drawdown control on US came at the cost of return, and on NSE it destroyed value. This is direct evidence for doc 06 §4: abstention must key on the model's *self-consistency residual*, not on raw volatility. The conformal machinery is exact (coverage 0.90 at α = 0.10 on all three markets) and ready for conformal-Kelly sizing.

## 6. Data-source facts learned (keep)
- Yahoo → 429 and Stooq → JavaScript challenge from every cloud IP tried (OCI, AWS); Stooq also from the Mac. **Nasdaq.com historical API works keyless** (~10 y, ETFs and stocks).
- Binance.com → HTTP 451 from US AWS; **api.binance.us** serves the same klines API (shorter history for some alts).
- NSE bhavcopy: use **archives.nseindia.com** for both the legacy (`cmDDMONYYYYbhav.csv.zip`) and the new UDiFF format; it answers 404 on holidays. `nsearchives.nseindia.com` hangs on any missing path — never use it. Full 2022→today download: ~60 s with 8 threads.

## 7. Ledger state
Every trial above is a row (`runs/<market>/ledger/`), chain verified at the end of every script. Scopes: `p0-baselines-<market>-N<N>`, `p1-volnulls-…`, `p2-encoder-…-h<h>`, `p2-regime-…`. The first NSE attempt produced a broken 64-day panel (SR 3.3); those rows remain in the ledger under their own scope as a record, and the scope-by-universe rule keeps them from deflating the corrected experiment.

## 8. What is next (P3)
1. Agora bridge event harness on the dense 2025–26 US tariff events (doc 08 §3).
2. Market Accounting Matrix from NSE shareholding patterns + FII/DII flows (P3b) and the nowcaster (P3c).
3. Ablation ladder rungs 3–5 once the structural kernel exists.

---

# P3 step 1 — Agora event harness on the 2025–26 US tariff actions (added later the same day)

**Setup.** 20 dated tariff actions (CRS R48549 tables + 2026 court/Section-122/232 actions; deterministic table, no LLM labels) mapped by hand to ISIC-45 import-tariff changes on the USA anchor (partner shares × announced rates; 232 sectors exempt from reciprocal rates; China retaliation as an export-tax proxy). Agora's **exact** open-economy CGE solves baseline vs. shocked economy per event (one at a time; the batched Newton in this build stalls for E>1 while singles converge in 3–4 iterations; all 20 converged to ≤5e-10). Sector responses map to 29 US sector/industry ETFs through a hand-built exposure matrix; realised returns are beta-stripped against SPY (beta from the 250 days before the event); score = Spearman rank correlation across ETFs per event, permutation null (2,000 shuffles per window).

**Result — the market prices Agora's producer-price response on the announcement day.**
| response variable | window | n | mean rank-IC | t | perm-p | hit |
|---|---|---|---|---|---|---|
| **dlogPX (producer price)** | reaction h=1 (incl. day 0) | 20 | **+0.123** | 2.25 | **0.004** | 0.80 |
| dlogPX | reaction h=5 | 20 | +0.071 | 1.08 | 0.054 | 0.60 |
| dlogPX | reaction h=20 | 20 | +0.037 | 0.76 | 0.20 | 0.70 |
| dlogPX | tradeable h=1 / 5 / 20 (from close of day 0) | 20 | +0.033 / +0.024 / +0.017 | 0.3–0.5 | 0.2–0.35 | 0.45–0.60 |
| dlogREV (revenue) | reaction h=1 | 20 | +0.087 | 1.04 | 0.017 | 0.55 |
| dlogREV | tradeable h=20 | 20 | **−0.137** | −2.23 | 0.999 (reverse) | 0.35 |
| dlogQX (real output) | as dlogREV | | | | | |
Sector-momentum null (prior-20-day residual return) is ≤0 in every window. Hike/relief split (dlogPX):
| | hikes (n=13) | relief events (n=7) |
|---|---|---|
| reaction h=1 mean IC / hit | **+0.152 / 0.92** | +0.068 / 0.57 |
| reaction h=5 | +0.100 / 0.69 | +0.017 / 0.43 |
| tradeable h=20 | **+0.063 / 0.69** | −0.067 / 0.14 |
High-confidence events only (n=12): day-0 IC +0.113, t 2.21, p 0.017. Hikes are predicted better than relief events: the model's price channel explains who gains pricing power when a tariff goes up; on relief days the cross-section is dominated by a risk-on rebound that beta-stripping removes only partly.

**Reading.**
1. **The prior is real at the sector level**: a computable general-equilibrium *price* response, with no fitting to returns, ranks the day-0 cross-section of sector ETFs with p = 0.004 on 20 events. This is the P3 go signal — the mapping exists.
2. **It is not yet a tradeable edge**: from the close of the announcement day the price-response IC is positive but insignificant (+0.02 to +0.03), and the quantity/revenue responses *reverse* significantly over 20 days (the announcement-day move in "winner" sectors fades). With 20 events the tradeable window cannot be certified either way; the harness now accumulates events automatically.
3. **Design consequences**: use the *price* channel (dlogPX) as the structural feature, not revenue; treat the day-0 reaction as the calibration target for the exposure map (P3b learns E from holdings); the reversal suggests a second, contrarian channel worth testing with more events; relief events need the demand-pressure (flow) channel the MAM adds.

Artefacts: `runs/us_sectors/p3-event-harness-*.json` (per-event predictions and scores), ledger scope `p3-event-harness-us`, code `emporion/structure/{agora_bridge,events_us_tariffs,sector_map}.py`, `scripts/p3_event_harness.py`. Agora PROD (540M) and dense (106M) bundles are now on the box under `~/agora-models` for the fast path.
