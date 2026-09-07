# 10 — Direction review against the 2023–2026 literature: are we heading the right way, and what to adopt

*2026-09-06. A second deep-research pass, this time targeted at the components we actually built and the results we
actually measured. Verdict first, then the evidence, then a prioritised adoption list. Sources in `REFERENCES.md`.*

## 1. Verdict
**The direction is right, and the literature has moved toward it, not away from it.** Three of our hard-won findings are
now recognisable as frontier results in their own fields: (a) the non-identification of beliefs versus risk premia from
prices alone (the subjective-expectations literature resolves it with survey data; we resolve it with holdings and
fundamentals — both are external anchors, and ours are point-in-time); (b) holdings as the primary information source
for pricing — the 2025 *Asset Embeddings* paper proves that portfolio holdings contain all pricing-relevant
information and recovers it by inverting the demand system, which is exactly the Market Accounting Matrix route we
chose; (c) the accruals edge we certified replicates in the 2023 Journal of Finance replication study across 93
countries, with the expected post-publication decay. Two adjustments follow: treat **holdings as information** (embeddings)
as a first-class rung alongside the structural clearing correction, and take the **anchor library** from the replication
study's 13 factor themes instead of inventing anchors one at a time.

## 2. Component-by-component evidence
| Our component | What the recent literature says | Implication |
|---|---|---|
| Demand-system clearing model (R2), κ from returns | Koijen–Yogo (2019, JPE) and their 2025 theory/econometrics paper: demand elasticities must be estimated with instruments (exogenous supply/flow variation); a 2024 extension allows general CRRA rather than log-linear demand; Gabaix–Koijen inelastic-markets multiplier ≈ 5 | our κ estimator is a reduced-form OLS; upgrade to the instrumented estimator once holdings panels exist (NSE first) |
| Holdings as the identification device (φ, MAM) | *Asset Embeddings* (Gabaix, Koijen, Richmond, Yogo 2025, NBER 33651): holdings contain all relevant pricing information; low-dimensional asset and investor embeddings recovered by inverting the demand system; benchmarks = relative valuations, return comovement, portfolio prediction | **adopt now**: build asset embeddings from NSE shareholding (and US 13F later) — they replace the hand-built exposure map (doc 07 TODO) and become encoder features |
| Beliefs from fundamentals, not prices (research/09) | Nagel–Xu (JFE 2023) *Dynamics of subjective risk premia*: subjective expected returns barely move with predictors; price volatility comes from time-varying **expectations of fundamentals**, not risk aversion; De la O–Myers; Cui–De la O–Myers *subjective belief factor*; analyst return expectations from price targets show a positive risk-return relation | confirms the anchor priority: beliefs should be measured, not inferred — analyst EPS/price-target expectations are the belief anchor; our daily consensus snapshot is the right (only) honest way to build that history |
| Certified accruals rung | Jensen–Kelly–Pedersen (JF 2023): accruals replicates; 85 % replication rate; returns decay post-publication (McLean–Pontiff: −26 % OOS, −58 % post-publication) but stay positive; effects hold in 93 countries | expect a live edge below the backtest; use their 13 themes as the anchor library and test on NSE where the 93-country evidence says the effects are larger |
| Structural kernel from investors (R1) | Chen–Pelger–Zhu (Management Science 2023): imposing the no-arbitrage constraint in learning improves the risk-premium signal and OOS explanation; needs an adversary and a macro-state recurrent block | keep; our derivation of M from calibrated investors is the missing structural step in their design; add their macro-state block when the nowcaster exists |
| Crypto ladder | Market + size + momentum suffice for the crypto cross-section (four-factor 2025); *Trend factor* (JFQA 2025); *Crypto pricing with hidden factors* (arXiv 2601.07664): Giglio–Xiu three-pass with latent factors plus Fear-and-Greed, altcoin-season and security-shock state variables; ML on 40 features: past alpha, illiquidity, momentum dominate, stronger in small caps | crypto rungs = size, momentum, trend, illiquidity; state variables as regime inputs; three-pass correction for premia; do **not** expect a fundamentals anchor there |
| India flows | DIIs overtook FIIs in NSE ownership in March 2025 (structural shift in the MAM); FII flows affect sectors differentially; DII inflows stabilise drawdowns in FII-outflow years | flows by category (FII/DII) as the demand-pressure state; sector-differential response is a testable rung |
| Honest evaluation | corporate-bond factor replication crisis (2026) and *Do t-statistic hurdles need to be raised?* — the field keeps finding that unadjusted results do not survive | our gate (DSR, CPCV, PBO, bootstrap, and now the validator-validation tests) is ahead of common practice; keep the trial ledger as the product |

## 3. Adoption list, in order
1. **Asset embeddings from holdings (NSE now, US 13F next).** Factorise the investor-category × asset holdings panel
   (promoter, FPI, MF, insurance, pension, retail…) into low-dimensional asset embeddings; use them (i) as the exposure
   map replacing `sector_map.py`, (ii) as encoder features, (iii) as a rung: similarity-based relative valuation.
2. **Anchor library = the 13 replication themes** (value, momentum, quality, profitability, accruals, investment,
   low-risk, size, seasonality, short-term reversal, debt issuance, profit growth, skewness). Each is a ledgered rung on
   US, NSE and (where meaningful) crypto. Stop inventing anchors; test the ones with 93-country evidence.
3. **Beliefs = measured expectations.** Consensus EPS and price targets as they accumulate; earnings-call-derived
   beliefs under the temporal quarantine later.
4. **Instrumented κ.** Replace the OLS κ with the Koijen–Yogo instrument once the holdings panel exists; report both.
5. **Crypto:** size/momentum/trend/illiquidity rungs + three-pass latent-factor premia + sentiment/altseason state.
6. **Structural kernel v1:** add the macro-state recurrent block and the adversarial test functions from CPZ once
   beliefs and κ are anchored (research/09 §5).

## 4. What we should stop or de-emphasise
- The hand-built ETF sector map and any hand-tuned "tilt" loadings (replaced by embeddings and by measured beliefs).
- Adding the clearing correction on top of an uninformative anchor (shown to subtract value); the structure earns its
  place only with anchored beliefs and data-calibrated φ.
- Price-only encoders as backbones (rejected empirically and structurally).

## 5. One-paragraph statement of direction
Emporion is a demand-system asset-pricing engine with a computable general-equilibrium prior, trained on the pricing law
at its own equilibrium price and disciplined by a ledger that cannot certify noise. The literature of the last three
years says the information is in holdings and in measured expectations, that the law improves learning when imposed
correctly, and that the anomalies worth anchoring on replicate globally with decay. That is our design, with the
identification device made explicit. The work ahead is data (holdings panels, expectations) and rungs, not a new model.
