# v1 channel 4 — pooled training across markets (2026-09-06)

**Question.** v0 on US large caps learned nothing beyond its value anchor (38 or 158 cross-sections, and 430+ under
CPCV). The v1 plan's remaining lever for the residual net was *more cross-sections from more markets under one law*.
Does pooling US (top-200 point-in-time, monthly dates 2017–2026) with NSE (top-200, 2022–2026) change that?

**Method.** `scripts/p4_train_pooled.py`: one feature set for both markets (value anchor from split-adjusted TTM EPS,
SUE, momentum, low-risk, log price, market token), cap-weight supply, κ from the panel estimator per date, Rf per
market (4 % / 6.5 %), law at the equilibrium price, CPCV 6×2 purged by horizon over all dates in time order, the same
code path run pooled and per market. Signal = −r_mkt; baseline = the value anchor feature alone.

| training set | test dates | Emporion −r_mkt rank-IC (t, hit) | value anchor alone (t) |
|---|---|---|---|
| US + NSE, scored on US | 105 | +0.0213 (1.28, 0.52) | +0.0219 (1.32) |
| US + NSE, scored on NSE | 40 | +0.0118 (0.43, 0.50) | +0.0126 (0.45) |
| US + NSE, pooled | 145 | +0.0187 (1.31, 0.52) | +0.0193 (1.36) |
| US only | 105 | +0.0214 (1.28) | +0.0219 (1.32) |
| NSE only | 40 | +0.0119 (0.43) | +0.0126 (0.45) |

(h = 20 trading days, 600 steps, early stopping on validation law residual, d_model 64; ledger scope `p4-v1-pooled`.)

**Reading.** Identical to the anchor in every cell, to the third decimal, pooled or not. The net satisfies the
market-clearing law without moving beliefs off the anchor — exactly what research/09 predicted: prices identify
μ and κ only up to the anchor, and 20-day payoffs are too noisy to pull beliefs away from it. Adding a second
market adds cross-sections but not identifying variation: NSE contributes 40 dates whose own anchor is weaker.
The v1 exit criterion (US rank-IC ≥ +0.045, beating the no-fit composite +0.033 with t > 2) is not met.

**Synthetic pre-training (run after all, same day).** 300 steps on `sample_panels` (64 markets × 6 dates × 40 assets,
true structural κ and φ) before each split's real fine-tune: US +0.0214 (t 1.29), NSE +0.0115, pooled +0.0187 —
unchanged to the third decimal (ledger `emporion_v1_pooled_us+nse_h20_every21_pen1.0_d64_pre300`). As expected:
pre-training fixes optimisation, and the barrier is identification.

**Decision.** The residual-net route (v0/v1 architecture, anchors as inputs, law as the objective) is **closed at
this data scale**, with and without synthetic pre-training. What remains of the deep-learning programme is a different
question: models that *measure* beliefs (consensus-revision history, now accumulating daily) and *nowcast* state
(holdings between filings) as inputs to the exact clearing solver — not nets that learn beliefs from prices.

**Product consequence.** Emporion today = (1) the pre-registered US accruals and composite signals on the shadow
chain, (2) the NSE earnings-surprise rung under watch, (3) the Agora tariff event harness as an ETF sector-rotation
tool, (4) the honesty stack (point-in-time universes, CPCV, deflated Sharpe with procedure counting, PBO,
pre-registration + OpenTimestamps). No deep model is in the product; that is the finding, ledgered.

*Ledger note.* Scope `p4-v1-pooled` also holds two smoke-test trials (NSE, `every 63`, 10–20 steps) run to check the
code path before the full runs; they are left in place — the ledger is append-only — and count toward the scope's
procedure total like any other evaluation.
