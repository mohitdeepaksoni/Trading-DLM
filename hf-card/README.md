---
license: other
license_name: research-only
library_name: pytorch
tags:
  - quantitative-finance
  - asset-pricing
  - structural-model
  - market-clearing
  - general-equilibrium
  - pre-registration
  - deflated-sharpe
  - combinatorial-purged-cv
  - point-in-time
  - opentimestamps
  - nse
  - us-equities
  - crypto
pipeline_tag: other
---

<p align="center"><img src="emporion-logo.png" alt="Emporion" width="200"></p>
<h3 align="center">EMPORION · Honest Trading Engine</h3>
<p align="center"><em>Developed by Deepak Soni — AI Architect</em></p>

![Emporion — structural, point-in-time, pre-registered](emporion-banner.png)

# Emporion — a structural deep-learning trading engine with a falsifiable record

**A market-clearing model of stock prices (beliefs × risk tolerance × supply, solved exactly), a general-equilibrium
prior from the sibling [Agora](https://huggingface.co/deepakdsoni/agora-cge-models) economy model, and — the part that
is unusual — an honesty stack that every number must pass before it may recommend a trade, ending in a public,
Bitcoin-stamped record of every signal the engine publishes, one file per market per day.**

This repository is the model card and documentation home of Emporion. **No trained weights are published yet**:
three deep-model architectures were built and evaluated, and none beat its own anchors on the pre-declared exit line
(Section 6). That result is on the ledger and in this card. Weights will be uploaded the day a model earns its place —
not before. What *is* live is the pre-registered signal chain (Section 7), which is the product until then.

> **Audience note.** Sections 1–5 give the model, the estimator and the certification mathematics for a quantitative
> reader. Section 6 gives measured results — mostly negative, all ledgered. Section 7 is the live record. Section 9
> states limitations. **If you just want to know what it does for you, read the next section first.**

---

## What it does — in plain terms

Emporion answers one question honestly: *"of the stocks I can actually trade today, which should I own for the next
month, and how sure can anyone be?"* — for US large caps, the Indian NSE, and the major crypto pairs.

It differs from a backtest in three ways you can check yourself:

1. **Every input is what was knowable that evening.** Universes are chosen by the past year's traded value, not by
   today's index; every fundamental is the version that had been filed by the decision date, restatements arrive
   when they were filed; institutional holdings are the batch that existed, not the amended one.
2. **Every signal is counted against everything that was tried.** The deflated Sharpe hurdle uses the number of
   distinct procedures on an append-only ledger; a rerun is not a new trial, a code change is. The probability of
   backtest overfitting is computed across the whole ladder of ideas, so adding ideas makes the bar higher, not lower.
3. **Every recommendation is published before the price exists.** One file per market per night, hashed, chained to
   the previous file, and stamped into Bitcoin through OpenTimestamps. Anyone can run `ots verify` on any file.

### What you get from it today

| You want… | It gives you… |
|---|---|
| A US large-cap book, monthly rebalanced | the **accruals** top-quintile and the **no-fit composite** (value, earnings surprise, accruals, momentum, equal z-weights, fixed forever) — pre-registered 2026-09-06 |
| An NSE book | the **earnings-surprise** top-quintile within the point-in-time top-200 — pre-registered 2026-09-07 |
| A same-day sector call on a tariff action | the **event rotation** book: long/short quartile of sector ETFs from Agora's producer-price response, stamped before the open |
| To know whether to trust any of it | the **live record page** rebuilt nightly from the chain alone: books, abstentions, proof status, realised returns |
| The negative results | Section 6 and `docs/EMPORION-FINDINGS.md` — what was tried and what did not survive |

---

## 1. The problem

Let a market hold N assets with payoff vector D (random, horizon h) and observed prices p. Investors are CARA with
absolute risk aversion a_k, wealth w_k, beliefs μ (shared, to be identified), covariance Σ, and a mandated
(price-insensitive) sector holding a fraction φ of supply s. The price-sensitive demand of investor k is
x_k = (a_k)^{-1} Σ^{-1}(μ − R_f p). Market clearing gives the equilibrium price

  p̂ = R_f^{-1} ( μ − κ Σ s ),  κ = (1 − φ) / A,  A = Σ_k a_k^{-1}.

The engine's object is the equilibrium price p̂ implied by beliefs and state, the signal is −r_mkt = log(p̂ / p),
and the structural stochastic discount factor is M ∝ exp(−q·D/A) with q = (1−φ)s, whose Gaussian normaliser is
analytic.

## 2. Identification (what prices can and cannot tell you)

From prices alone the pair (μ, κ) is identified only up to the anchor that fixes the level of μ: any belief shift Δμ
can be absorbed by a change in κ Σ s. Therefore (research/09):
- the state is parametrised as κ = (1−φ)/A and **seeded by a structural estimator** — the regression of realised
  excess payoffs on Σ s across assets over a trailing window;
- the clearing law is imposed **at the equilibrium price p̂, never at the observed price** (imposing it at p flips the
  sign of the mispricing signal: −0.50 correlation on synthetic markets, +0.95 when imposed at p̂);
- beliefs need an **external anchor** — filed fundamentals (earnings yield, seasonal earnings surprise, accruals),
  measured expectations (consensus), or the general-equilibrium prior — before the law can say anything.

## 3. The general-equilibrium prior

For a dated policy action (a tariff schedule by ISIC sector and partner), Agora's exact open-economy CGE for the USA
(OECD ICIO-45 anchors, Armington/CET trade, CES/KLEM technology) is solved to residual 10⁻¹⁵ and returns the
producer-price response Δlog PX per sector. Mapped to sector ETFs through a hand-built exposure table, this is a prior
on the *cross-section of day-0 reactions* that no price-derived anchor contains.

## 4. The honesty stack (the certification mathematics)

A candidate rule certifies only if all of the following hold on the point-in-time panel with costs inside returns:
- **Deflated Sharpe ratio** ≥ 0.95, where the null Sharpe distribution's variance is the robust (leave-one-out MAD)
  variance across the N *distinct procedures* in the scope — N from the ledger's procedure hashes (rung name + calling
  code + arguments), the strict evaluation count reported alongside;
- **Probability of backtest overfitting** (CSCV, 16 blocks) across all rungs in the family < 0.20;
- **Stationary-bootstrap** confidence interval of the excess return over the equal-weight point-in-time universe
  above zero; **permutation** p-value of the rank-IC below 0.05 where applicable;
- **Combinatorial purged cross-validation** (6 groups × 2 test, purged by horizon, embargoed) for anything fitted;
- and, for anything that would go live, **pre-registration**: the rule is frozen in code, the first file hashed and
  stamped, and the live prints decide.

## 5. The deep model (what was built)

Emporion v0/v1: a permutation-equivariant set encoder over per-asset features → beliefs μ_i and the investor state
(a, φ); exact CARA clearing to p̂; loss = the clearing residual at p̂ plus penalised departures of μ from its anchor;
validation early stopping on the residual; κ seeded from the estimator. Trained walk-forward, under CPCV, pooled
across US + NSE under one law, and with synthetic pre-training on sampled markets with persistent beliefs.
Emporion v2: ridge, gradient boosting and a 3-seed MLP *over the anchors* (21 features: eight price themes, value,
SUE, accruals, 13F breadth and level, Agora sector shock, market token) with a within-date rank target, judged on
identical CPCV splits against the no-fit composite.

## 6. Measured results (all on the ledger; scopes in `docs/EMPORION-FINDINGS.md`)

| question | answer |
|---|---|
| Does the structural model find mispricing when the anchor is informative? | Yes: 0.95 out-of-sample correlation with true mispricing on synthetic markets; identities and validators hold |
| Does v0/v1 beat its anchor on real data? | **No** — equal to the value anchor to the third decimal on US (430+ dates, CPCV), NSE, pooled, with and without pre-training |
| Does a network over the anchors beat equal weights (v2)? | **No** — US +0.016–0.020 vs composite +0.030 (h20); NSE, after a coverage leak was removed, +0.02–0.03 vs +0.045 |
| Which anchors rank stocks? | US accruals +0.019/+0.023 rank-IC (t 2.1/2.6), composite +0.032 (t 2.1); NSE earnings surprise +0.035/+0.036 (t 3.1/4.1) |
| Is anything certified? | **Nothing yet.** US candidates fail only PBO across the ladder (0.21–0.24 vs 0.20); NSE SUE fails only beats-EW. Hence the live chain |
| Does Agora's prior price anything? | Sector ETFs on the event day: rank-IC +0.125 (t 2.3, permutation p 0.003, n = 20), robust to dropping the metal ETFs; **not** individual stocks (+0.018, t 0.6) |
| Are holdings a signal? | Level and change: no. Breadth of ownership change: +0.036 (t 2.2) at 60 days, on the watch list |
| Is a nowcast of holdings between filings informative? | No — a Kalman filter with Z.1 drift or trailing returns equals carry-forward |
| Kronos zero-shot as an encoder? Regime gating? | Both rejected: rank-IC ≤ random; stress regimes carried the rebounds |

Three data defects were found by sanity checks on levels, not by statistics, each with a written rule and a test:
undeclared XBRL base contexts in old NSE filings (year-to-date read as quarters), NSE percentages arriving as
fractions from Sep-2025, and SEC 13F datasets carrying late filings for earlier quarters. A fourth — feature coverage
that was not point-in-time — was found by reading model coefficients after cross-validation had passed it.

## 7. The live record

Nightly on GitHub Actions (no server, no GPU): refresh point-in-time data, build the pre-registered files, hash, chain,
stamp, verify, rebuild the record page, store state. Day 1 = 2026-09-06 (attested in Bitcoin blocks 965744/965751).
Files carry the strategy text, universe rule, every score, and the target weights; a missing night is recorded as a gap,
never back-filled. Certification of a 20-day signal needs roughly 6–12 months of prints; until then the page is a
record, not a claim.

## 8. What is in this repository

- `README.md` — this card. `docs/` — the research notes 01–12 (models, state of the art, evaluation and overfitting with
  the data rules, the unique design, the practical guide, real-time training, the trader-facing design, the review
  response, identification, direction review, stock transfer, pooled result), the findings document, the status
  reports, and the pre-registration rule for events.
- **Weights: none yet.** Emporion v0/v1/v2 checkpoints exist but did not pass their exit lines; publishing them would
  imply a claim the ledger does not support. The first model that beats its anchor out of sample on the pre-declared
  line will be uploaded here with its ledger scope and CPCV record.
- The engine code, tests (56) and the nightly runner live in the private GitHub repository; the point-in-time stores
  and the chain state live in a private dataset. Ask for access.

## 9. Limitations, honestly

- Nothing is certified. The two US rules and the NSE rule are pre-registered candidates whose live record is days old.
- Survivorship: the US pool is today's top-1000 by traded value; the bias was measured on a 13F-implied survivorship-free
  quarterly panel at ≈ 0.00–0.02 rank-IC for momentum-type anchors, and not measured for the fundamental anchors.
- NSE fundamentals begin in 2018 and shareholding in Sep-2021; NSE price history begins in 2022.
- The sector map from industries to ISIC codes is hand-built (validated at rank-correlation 0.7 against real ETFs).
- The general-equilibrium prior has twenty events; the day-0 effect is significant, the tradeable-window effect is not.
- Costs are a flat 10–15 bp; no market-impact model; books are equal-weight top quintiles, not optimised portfolios.
- Crypto is rankable by risk but nothing beats equal weight after costs.

## 10. Roadmap

Measured beliefs from the accumulating consensus-revision history as an anchor the price cannot see; actual ETF
constituents for the sector map; twenty stamped live events for the harness; the write-up after the first six
months of prints. A deep model returns to this card only when it passes its exit line.

---

*Emporion is a personal, open research project. Not investment advice. Research-only licence.*
