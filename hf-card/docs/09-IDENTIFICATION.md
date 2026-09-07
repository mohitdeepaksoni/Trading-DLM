# 09 — Identification: what prices and returns can and cannot tell the model

*2026-09-06. Written after the P4 v0 sweep showed that training on the pricing law degraded the model at every sample size. This note states the reason exactly, the numerical confirmation, and the reparametrisation that resolves it. It is the most important design constraint discovered so far.*

## 1. The equation
In the clearing model (doc 07 A.1–A.2), CARA investors with aggregate risk tolerance A = Σ_h 1/a_h absorb the residual supply q = (1−φ)s left by mandated index holders, and payoffs D ~ N(μ, Σ). Market clearing gives
```
R_f · p = μ − Σ q / A = μ − κ · Σ s,        κ := (1 − φ) / A .
```
The pricing kernel implied by the same investors is M ∝ exp(−q·D/A), and E[M D] = p holds with the same κ.

## 2. What is identified from prices and returns
- **Beliefs μ**: the time-series mean of realised payoffs per asset (a panel is required — the same asset observed over many dates).
- **The composite κ**: the cross-section of expected excess payoffs E[D_i] − R_f p_i = κ (Σ s)_i identifies one scalar per market by regression on (Σ s).
- **Nothing else.** A and φ enter prices, returns and the kernel only through κ. No amount of price and return data separates "investors expect less" from "investors demand more compensation" beyond this ratio.

## 3. Numerical confirmation (synthetic panels, structural estimator, no neural net)
| dates per market | corr(κ̂, κ) | corr(μ̂, μ) |
|---|---|---|
| 25 | 0.40 | 0.82 |
| 100 | 0.53 | 0.94 |
| 250 (≈1 year daily) | 0.61 | 0.98 |
| 1,000 | 0.86 | 0.99 |
| 4,000 | 0.96 | 1.00 |
A recovered from κ̂ and the true φ stays weak (0.15–0.54) because the division amplifies κ̂ noise and A is a nuisance for pricing anyway. The earlier deep-learning sweep (32,768 dates × 32 assets, i.i.d. markets rather than panels) could not identify A or φ *by construction*, and unpenalised training let the net drift along the flat direction (μ up, κ up), which is why mispricing recovery degraded from 0.95 to 0.55.

## 4. The second, deeper cause — the law was imposed at the wrong price
Even in κ-parametrisation, minimising the moment residual degraded the mispricing signal identically in-sample and
out-of-sample (0.95 → 0.11) while the *validation* residual fell. That rules out overfitting. The cause is exact:
excess returns were measured against the observed price, so the condition E[M (D/p_obs − R_f)] = 0 is literally the
statement "there is no mispricing", and minimising it teaches the model that the market is always right — the doc-06
tension between r_self and r_mkt in its purest form. Measured against the model's equilibrium price p̂, the same
condition says "beliefs and κ must be consistent with realised payoffs" and leaves η = log(p_obs/p̂) as a separate,
tradeable object. Out of sample (held-out 40 % of dates, 256 panels × 250 dates × 8 assets):
| law imposed at | training | OOS corr(r_mkt, η) init → trained | validation R1 |
|---|---|---|---|
| observed price p_obs | unpenalised | 0.95 → **0.11** | falls (wrong minimum) |
| observed price p_obs | penalised | 0.95 → 0.79 | — |
| **equilibrium price p̂** | unpenalised | 0.95 → **0.955** | wanders (needs early stopping) |
| **equilibrium price p̂** | penalised | 0.95 → 0.953 | ≈ flat |
With the analytic (noise-free) kernel the wrong-price law is even more damaging: at 32,768 × 32 the trained mispricing correlation reached **−0.50** — the model learned to call under-priced assets over-priced.
Now the defaults: `state_mode="kappa"`, `law_on="p_hat"`, validation early stopping in the trainer.

## 5. The resolution — reparametrise around what is identified
1. **State = κ**, seeded by the structural estimator on the trailing window (the returns-side nowcaster), refined by a penalised log-correction from the net. `EmporionV0(state_mode="kappa")`.
2. **Equilibrium price** computed directly as p̂ = (μ̂ − κ Σ s)/R_f. A and φ are never needed for pricing.
3. **φ from holdings** is used where it *is* needed: the demand-pressure channel (who must buy), the Market Accounting Matrix, and to back out A = (1−φ)/κ when an investor-level quantity is required. This is the precise sense in which holdings data is the identification device.
4. **Beliefs** anchored to fundamentals-level information (earnings, dividends, Agora shock-implied payoff changes); the net's correction penalised toward zero.
5. **R1 remains the law**: with κ and μ identified, the moment condition can refine both instead of drifting; it is a consistency check with teeth, not a free teacher.

## 6. Practical consequences
- Panels, not cross-sections of unrelated markets, are the training unit; κ needs years of data, so it is slow-moving state and belongs in the slow layer (doc 06 §3), never refit nightly.
- The Kronos-style price-only encoder is excluded a second time: without a fundamentals anchor there is no identification of mispricing at all (measured: corr ≈ −0.1 to −0.2).
- Tests: `tests/test_p4_identification.py` (κ and μ identified, A/φ not, κ convergence slow) and `tests/test_p4_train.py::test_law_at_equilibrium_price_preserves_mispricing_out_of_sample`.
