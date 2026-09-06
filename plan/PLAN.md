# Emporion — build plan (draft v0.1, 2026-09-05; nothing started)

Goal: an open, equilibrium-anchored, honestly-evaluated DL trading engine for any market, reusing Agora (`../../Mathemetics-DLM/agora`). Phases are gated; each ends with a ledger-backed report.

| Phase | Deliverable | Reuses | Compute | Time |
|---|---|---|---|---|
| **P0 Foundations ✅ 2026-09-05** | repo skeleton (registry, device, ledger), point-in-time loaders for US (Alpaca/Tiingo) + NSE bhavcopy + one crypto venue, feature registry with lags, CPCV/DSR/PBO/bootstrap evaluator, passive baselines certified | `core/registry.py`, `core/device.py` | CPU | 2 wks |
| **P1 Nulls & baselines ✅ 2026-09-05** | GARCH-t, HAR-RV, HRP, risk-parity, BL(cap-weights), momentum/value tilts — all through the ledger; breakeven-cost report. Defines the **ablation ladder** rungs 1–2 (doc 08 §4) | — | CPU | 1 wk |
| **P2 Encoder (regime+conformal ✅; Kronos zero-shot measured 2026-09-05, see reports/)** | Kronos adapter (frozen + fine-tune), K-line tokenizer check on NSE/crypto; signature features; regime HMM + conformal sets | `nets/mlp.py`, `nets/moe.py` | 1×A10/H100 | 1–2 wks |
| **P3 Structural bridge (step 1 ✅ 2026-09-05: US tariff event harness, price channel validated day-0, tradeable TBD)** | `agora_bridge`: policy/news state → sector shock `a_t` (country-general net as a service); exposure maps E (GICS/NIC + ICIO); **historical event validation** (2018–19 tariffs, 2022 energy, 2025 tariffs) of `E a_t` vs realised sector returns | country-general 540M model, ICIO anchors | CPU + 1 GPU | 2 wks |
| **P3b Market Accounting Matrix** | calibrated investor-type holdings/flows per market (13F/N-PORT/Flow-of-Funds; NSE shareholding + FII/DII + AMFI); base-period exact clearing (Agora SAM-balancing reused) | `country_general` calibration | CPU | 1–2 wks |
| **P3c Investor-state nowcaster** | state-space filter: investor holdings observed at filing dates (13F/N-PORT/shareholding), flows daily (FII/DII, ETF create/redeem); passive types deterministic by mandate, active types latent with bounded innovation; validated separately by filing-date reconstruction error | — | CPU | 1–2 wks |
| **P4 Residual model** | structural SDF derived from MAM investors + penalised neural correction (R1, fixed + adversarial test functions, hard E[M]=1/R_f) + clearing model (R2 = `ces_exchange` subclass with investor types) trained self-supervised with RAD / self-adaptive weights / L∞; hybrid Newton polish; residual + ensemble gate | `models/ces_exchange.py`, `solvers/classical.py`, `solvers/hybrid.py`, tail-methods recipe | H100/H200 hours | 3 wks |
| **P5 Allocate + execute** | BL with conformal Ω, HRP/CVaR overlay, conformal-Kelly sizing, Almgren–Chriss + √-impact, broker adapters (Alpaca paper first, then Kite with SEBI Algo-ID), deployment-consistent `strategy.step()` | — | CPU | 2 wks |
| **P6 Simulator** | signature-kernel neural-SDE generator; Agora-shock scenario library; stress + MDD estimates | `solvers/retrieval_warmstart.py` idea | 1 GPU | 1–2 wks |
| **P7 Certification + paper trading** | **ablation ladder** rungs 3–5 (Koijen–Yogo alone → Emporion−Agora → full) on the same ledger and costs; full doc 03 checklist on US + India (+ crypto); ≥ 3 months paper with implementation shortfall; model card | — | CPU | 3 months elapsed |
| **P7b Trader surface** | hash-chained pre-registered signal log, Trade Ticket with attribution, Event Scoreboard, ledger page, risk screen, signal API, daily report (doc 07 Part B) | Agora console | CPU | 2 wks |
| **P8 Share** | Emporion-Bench (datasets, cost model, ledger spec, leaderboard), OSS engine, arXiv paper + plain-language explainer, console (reuse Agora Analyst dashboard patterns) | Agora console | — | 2–3 wks |

Critical path: P0 → P3 → P4. P3's *event validation* is the go/no-go for the whole thesis: if Agora's sector shocks do not line up with realised sector moves on known events, the structural prior adds nothing and the project becomes "honest-eval benchmark + residual SDF" only (still worth shipping).

Machine: Brev `emporion` (g5.2xlarge, 1×A10G, stoppable, $1.45/hr) — created 2026-09-05; stop when idle.

Estimated GPU cost through P6: **$200–500** (Agora-scale runs). Total elapsed to first certification: ~3 months of build + 3 months of paper trading.

## Real-time and continual-training phases (added 2026-09-05; see research/06)
Run after P5 (allocate + execute) and in parallel with P7 (certification + paper trading).

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **RT1 Bitemporal + parity** | four-clock bitemporal store (market, policy/macro, corporate, execution) with mandatory `as_of`; one feature library with `replay`/`live` modes and lag-aware live; deterministic historical event extractor (no LLM labels on history) | backtest replays byte-identically through the live path |
| **RT2 Shadow + harness** | certified model runs daily, trades nothing, logs everything; Agora event-validation harness live (prediction record per event, scored at 1/5/20 d vs zero and sector-beta nulls) | shortfall inside bootstrap CI; harness scoreboard populated |
| **RT3 Nightly re-solve + bounded refit** | Newton re-solve at fixed θ; gradient refit only on residual breach, with anchor-set rehearsal + parent regularisation; `r_self`/`r_mkt` split; conformal-calibrated alarms + CUSUM; online conformal Kelly | drift budget never silently breached; each breach triggered the slow layer |
| **RT4 Procedure certification + gates** | update procedure `U` (ensemble, thresholds, drift budget inside) certified by walk-forward replay; promotion gates: ingest → live residual → statistical shadow window → ledger → ensemble → canary → deploy; rollback to certified parent; gate evaluator with read-only model access | first model promoted to live size by the gates, not by hand |

Open decisions for you:
1. Name (Emporion is a placeholder).
2. First live market: US via Alpaca (simplest API, paper mode, no Algo-ID) vs India via Kite (home market, SEBI registration).
3. Licence for the engine (Apache-2 recommended) and whether the benchmark datasets can be redistributed (NSE bhavcopy: yes; vendor data: no — ship loaders, not data).
4. Drift-budget thresholds: accept the research/06 definition (anchor-set residual + probe disagreement) or propose another.
