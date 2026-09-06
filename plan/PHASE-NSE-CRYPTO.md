# Phase plan — NSE (holdings + fundamentals), full crypto, and the mathematics validation protocol

*2026-09-06. Objective restated: a deep-learning trading model whose predictions are accurate enough, and honest
enough, that people can use it to buy and sell and benefit. That objective is served by two things only: an
informative, point-in-time belief anchor per market, and a validation discipline that cannot be fooled. This plan
adds the two markets that have the data the US lacks (holdings for India, 24×7 depth for crypto) and hardens the
mathematics with a repeatable review loop.*

## 1. Mathematics validation protocol (applies to every formula, forever)
Every quantitative component must carry all four of these before it feeds a decision:
| level | what | where |
|---|---|---|
| **Identity** | an algebraic property that must hold exactly (Walras, homogeneity, E[M]=1/R_f, kernel prices the equilibrium, cash = −p·z) | `tests/test_p4_model.py` |
| **Synthetic truth** | recover a known quantity from simulated data (κ, μ, mispricing; regimes; GARCH persistence; conformal coverage) | `tests/test_p4_identification.py`, `test_p4_train.py`, `test_vol.py`, `test_regime.py` |
| **Validator validation** | the certification machinery itself: false-positive rate on noise ≈ 0, power on a real edge, DSR deflates with N, permutation p uniform under the null, bootstrap CI coverage, PBO flags selection | `tests/test_gate_validation.py` (new today) |
| **Real-data rung** | the component as a ledgered strategy with DSR/CPCV/PBO/bootstrap and a passive benchmark | `runs/*/ledger`, `scripts/p*_*.py` |
Review cadence: every new formula gets a note in `research/`, a test at each level, and a line in the report. Nothing
enters the nightly job without all four. A failed level is a result, recorded, never patched around.

## 2. NSE — the market with holdings
Data (NSE JSON API, keyless, verified 2026-09-06): quarterly financial results with broadcast timestamps (the
knowledge time), shareholding patterns per quarter with broadcast timestamps (promoter / FII / DII / MF / insurance /
retail — the Market Accounting Matrix rows), daily FII/DII cash-market flows. BSE API is blocked (HTML) — not used.
Steps: loaders → bitemporal tables (`results`, `shareholding`, `flows`) for the NSE top-200 → anchors: value (EPS
capitalised), SUE, dividend, **and the holdings state φ (non-promoter institutional share) and its quarterly change
(demand pressure)** → ladder rungs 3/3b/3c/3d and the clearing model with κ *and φ from data* (the first market where
A = (1−φ)/κ is identified) → certification.

## 3. Full crypto — the market with depth
Coinbase Exchange candles for all online USD pairs back to 2015 (402 pairs); CoinGecko daily snapshot for market cap,
circulating supply and volume (supply = the clearing model's `s`; point-in-time from today). Steps: full panel →
P0 ladder (baselines, vol nulls, regime, conformal) on the full universe → anchors available in crypto: momentum,
reversal, volume/turnover, supply growth (from snapshots), exchange-flow proxies later → the honest statement that
without a fundamentals anchor crypto mispricing is not identified (research/09), so the crypto rungs are classical
until a supply/flow state exists.

## 4. Exit criteria for this phase
- NSE: results/shareholding/flows tables in the store with knowledge times; ≥ 4 ledgered rungs; one rung where φ from
  holdings enters the clearing model; report + memory.
- Crypto: full-universe panel and ladder on the ledger; snapshot job scheduled.
- Validation: `test_gate_validation.py` green; the protocol table above referenced from the README.
