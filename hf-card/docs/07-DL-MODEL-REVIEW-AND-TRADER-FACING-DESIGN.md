# 07 — Second review: the DL model itself, and the design that makes traders look

*2026-09-05. Two questions, answered separately. Part A is a hostile review of the Emporion model in doc 04 from a deep-learning standpoint: where it can collapse, what is under-specified, what to change. Part B is the professional, trader-facing design: what a practitioner would need to see before paying attention, and the product surface that shows it. Changes to doc 04 are marked ✚.*

---

# Part A — The DL model, reviewed hard

## A.1 The biggest risk: the pricing-kernel residual (R1) is under-identified
A free network `M_θ(I_t, ε)` can satisfy `E[M R^e] = 0` trivially (M nearly constant, or M soaking up noise so every moment is zero on the training set). Chen–Pelger–Zhu needed an adversarial test-function network and still reported instability. Doc 04 inherits that.

✚ **Fix: derive M from the clearing model instead of learning it freely.** In the market-clearing block (R2) there are H investor types with utilities `u_h` and wealth shares `ω_h`. In equilibrium the pricing kernel *is* the wealth-weighted marginal utility of the investors who clear the market:
```
M_{t+1} = Σ_h ω_h,t · β_h u'_h(c_h,t+1) / u'_h(c_h,t)
```
So R1 is not a separate free object; it is implied by R2's investors. The network's job becomes learning **investor state** (wealth shares, risk aversion, beliefs, constraint slackness) from data — a far smaller, economically meaningful latent space — and R1 becomes a *consistency check* on that state rather than an independent fit. Keep a small neural correction `M = M_struct · exp(δ_θ)` with an L2 penalty on δ so the model can express what the structural kernel cannot, but only at a price. This removes the trivial solution, ties R1 and R2 into one object (which is what "equilibrium" means), and is the single most important upgrade to the design.

✚ **Test functions:** a fixed bank (lagged returns, characteristics, macro states, Agora shocks, their products) plus a learned adversary with a small capacity budget. Report the worst-case moment violation as the tail metric, exactly as the tail-methods doc does for economies.

✚ **Normalisation:** enforce `E_t[M_{t+1}] = 1/R_f,t` as a hard constraint (a layer, not a penalty), which pins the scale.

## A.2 The second risk: the clearing model (R2) is an invented economy unless it is calibrated
Agora works because every economy is calibrated to a real Social Accounting Matrix before the net sees it. Doc 04's investor types (CRRA, prospect, mandate) have no such anchor; an uncalibrated R2 is fiction with a residual.

✚ **Fix: build the Market Accounting Matrix (MAM) — the SAM of the stock market.** Rows = investor types, columns = assets/sectors, cells = holdings; plus a flow table per period. Data exists:
- **US:** 13F institutional holdings (quarterly), mutual-fund and ETF holdings (N-PORT, monthly), Fed Flow of Funds (households/foreign), short interest, ETF creations/redemptions.
- **India:** NSE/BSE shareholding-pattern filings (quarterly: promoter/FII/DII/retail), daily FII/DII net flows, AMFI mutual-fund holdings monthly, ETF AUM.
- **Crypto:** on-chain holder cohorts, exchange balances.
Calibrate each investor type's endowment (holdings), demand elasticity, and constraints (index mandates = must hold cap-weights; leverage caps) so that the base period clears exactly — the same "balanced SAM" step Agora already implements in `country_general`.

**Ancestry, stated plainly (correction 2026-09-05):** this *is* demand-system asset pricing — Koijen & Yogo (2019, *JPE*, "A Demand System Approach to Asset Pricing") estimate investor-level characteristics-based demand from 13F holdings and clear the market by construction; Gabaix & Koijen's inelastic-markets estimate (the ≈ 0.2 elasticity used above) comes from the same family, and there is a sizeable follow-on literature (Koijen–Richmond–Yogo 2023 on which investors matter; Haddad–Huebner–Loualiche 2024 on how competitive the market is). The MAM should therefore be built *as* a Koijen–Yogo demand system — logit demand in characteristics, elasticities estimated with their instruments rather than assumed — and cited as such. **What is new is not the MAM.** It is (a) deriving the pricing kernel from it so R1 and R2 are one object (A.1), (b) coupling it to a computable general-equilibrium engine for the shock channel, and (c) wrapping the result in abstention and pre-registration. The defensible claim: *first to use a demand-system clearing model as a structural prior for a trading system, coupled to general equilibrium.* The flow table adds the demand-pressure channel (who is buying, who must buy) that price-only models cannot see.

## A.3 Architecture — make assets a set, not a vector
Assets enter and leave; markets have different N. A flat MLP over N assets is wrong.

✚ **Encoder stack (per market, shared weights):**
1. **Per-asset temporal encoder:** Kronos (fine-tuned) or own K-line tokenizer → embedding per asset over a 60–250-bar window; add signature features of the (time, price, volume) path for order-sensitive memory.
2. **Cross-sectional block:** Set Transformer / DeepSets attention over the asset set with sector and supply-chain (ICIO) edges as attention bias — permutation-invariant, variable N, and it lets a shock in sector k reach its suppliers through the graph.
3. **Structural inputs:** Agora shock distribution `a_t` (mean, spread), MAM investor state, macro state — concatenated as *set-level* context tokens.
4. **Heads:** investor-state latent (feeds M_struct), equilibrium price `p̂` per asset, per-asset conformal non-conformity score, and an **attribution head** (see A.6).
Use Fourier features on scalar inputs (kinks at constraint boundaries) and a mixture-of-experts layer keyed by regime, as `nets/moe.py` already provides. Size: 50–150 M parameters is plenty; this is not a language model.

## A.4 Training curriculum and data budget
- **States:** real history (25 years × 500–3,000 assets × 3 markets ≈ 10⁵ daily cross-sections) + signature-kernel generated paths (×10) + Agora counterfactuals (policy draws on real dates). That is millions of states — small by Agora standards; residual training converges in hours.
- **Curriculum:** (i) calm regimes, no shocks, R2 only; (ii) add R1 with the structural kernel; (iii) add shocks with growing magnitude; (iv) add the neural correction δ; (v) tail phase — RAD resampling toward high-residual states, self-adaptive weights, L∞ term. Same ladder as `dl-equilibrium-tail-methods.md`.
- **Loss balancing:** gradient-magnitude annealing across R1, R2, tail and the δ penalty; never hand-set λ's.
- **Ensemble:** K = 5 members with different seeds and anchor subsets; disagreement is an abstention input.

## A.5 What the model must report about itself (model-level evaluation, before any P&L)
| Metric | Target that would justify going on |
|---|---|
| Post-polish `r_self` p99 on held-out stress dates | < 1 % relative |
| Worst-case moment violation (adversarial g) | below the certified model's own conformal alarm |
| Event harness: Agora sector response vs realised (beta-stripped), 5-day, rank correlation | > 0.3 with CI excluding 0 on ≥ 20 events |
| RankIC of implied views at 5–20 days | 0.03–0.06 (GKX-realistic) |
| Abstention rate | 30–70 % of asset-days — a model that never abstains is broken |
| Hit rate *when acting* vs when not | must differ, or the gate is noise |

## A.6 Explainability is native — use it
Because every signal is a deviation from a structural equilibrium, the model can say *why*: decompose `p − p̂` into (a) the Agora shock channel, (b) the demand-pressure channel from the MAM flows, (c) the kernel/risk-premium channel, (d) the neural correction δ. ✚ Add an attribution head trained to reproduce this decomposition (Shapley over the four channels is cheap because there are four). This is not a nicety; it is the trader-facing product (Part B).

## A.7 Things to drop or defer
- Drop RL for allocation entirely; keep it only for execution against a LOB simulator, later.
- Defer options/deep hedging to a second release; the first release is cash equities + ETFs + futures.
- Do not fine-tune an LLM anywhere in the pipeline; LLMs remain hypothesis writers and live event classifiers under quarantine (doc 06).

---

# Part B — The design that earns a trader's attention

Traders ignore backtests, papers, and architecture diagrams. They pay attention to five things: **a timestamped record they cannot dismiss, a reason for every trade, a number for every risk, an honest failure list, and a way to use it tomorrow.** Emporion can offer all five because of how it is built.

## B.1 Pre-registered, hash-chained daily signal log (the credibility engine)
Every trading day at a fixed time, before the open, publish a hash of the full signal file (views, sizes, abstentions, attributions, `r_self`, conformal widths). Reveal the file after the close. Chain the hashes (each includes the previous) and post them somewhere public and third-party-timestamped (a public git repository, or a public ledger). This is pre-registration for trading. Nobody can accuse you of cherry-picking; the record grows one day at a time; and after six months it is the most convincing artefact you will ever have — far more than any backtest. **Start this on RT2 day one**, while trading nothing.

## B.2 The Trade Ticket — one card per decision, written for a human
```
EMPORION · 2026-09-05 · NSE:XYZ · action: BUY tilt +0.8 % (½-Kelly, conformal width 2.1 %)
Equilibrium price 412.0 vs market 396.5  →  mispricing −3.8 %  (r_mkt)   self-consistency r_self 0.4 % ✓
Why (attribution):  Agora shock channel +2.9 %  ·  demand-pressure (DII inflow) +0.6 %  ·  kernel −0.1 %  ·  correction +0.4 %
Event:  Budget 2026 customs-duty change → sector "Electrical equipment" +1.7 % output, +2.2 % price (Agora v540M-45s, calib 2025)
Horizon 5–20 d · expected cost 18 bps round-trip · capacity at this size 0.4 % of ADV
Gate: ensemble agreement 5/5 · regime P(high-vol)=0.12 · abstain=NO
Procedure U-2026-07 (certified DSR 0.93, PBO 0.11, CPCV Sharpe p5–p95 0.41–1.12) · fast-gen 12 · drift 0.31 of budget
```
Every field on that card comes from a component already in the design. No other system can print the "Why" line with an economic mechanism behind it. That line is the product.

## B.3 The Event Scoreboard (the proof the prior is real)
A public table: event, date, Agora-predicted sector response distribution, realised beta-stripped response at 1/5/20 days, score, running rank-correlation and CRPS vs the two nulls. Updated automatically by the harness (doc 06 §5). A trader who sees twenty scored events with a positive, significant rank correlation understands immediately that this is not another price-pattern model.

## B.4 Abstention as a feature, marketed honestly
"The model that tells you when not to trade." Publish the abstention rate, and the return distribution of asset-days where the model abstained vs acted. If abstained days are noisier or worse, the gate is doing its job — and that is a number a trader can verify from the signal log.

## B.5 The Ledger page (the honesty that reads as professionalism)
Every certified procedure with: trials counted, Deflated Sharpe, PBO, CPCV distribution, breakeven cost, regime slices, seed spread, and the failed candidates — **including the ones that lost**. A visible graveyard of rejected models is the strongest credibility signal in this industry, because nobody shows it.

## B.6 Risk numbers a desk expects, on one screen
Vol target and realised vol, gross/net exposure, max single-name and sector, drawdown vs backtest-MDD budget, live implementation shortfall vs CI, capacity used (% of ADV), kill-switch status, drift budget used, `r_self` trend with the conformal alarm line.

## B.7 Ways to use it tomorrow
- **Signal API** (`GET /signals?as_of=…`) returning the ticket fields as JSON — usable in any execution stack.
- **Daily one-page PDF/HTML report:** top tilts, abstentions, event scoreboard delta, risk screen.
- **Console** (reuse the Agora Analyst dashboard skeleton): Signals · Tickets · Event Scoreboard · Residual map · Ledger · Live shortfall.
- **Broker adapters:** Alpaca (paper first), Interactive Brokers, Zerodha Kite with the SEBI Algo-ID registration checklist built in.
- **Emporion-Bench** for the researchers and quants: datasets, cost model, ledger spec, leaderboard by DSR. Practitioners follow whoever owns the yardstick.

## B.8 Positioning language (for README, paper, talks)
- Name: **Emporion** (or your final choice), tagline **"Trades the distance to equilibrium."**
- Three-line pitch: *Learns the law, not the history. Says why, and says no. Publishes its record before the fact.*
- What to never claim: Sharpe without DSR, returns without costs, a track record without the hash chain.

## B.9 What "attention" realistically looks like, and the order to earn it
1. Months 0–3: hash-chained signal log + event scoreboard running, trading nothing. Share the scoreboard, not the signals.
2. Month 3–6: paper-trading shortfall inside CI; publish the paper + Emporion-Bench + open engine. Quants and academics arrive first.
3. Month 6–12: micro-live with public ledger; the trade ticket and the abstention stats become the story. Practitioners arrive when the pre-registered record is long enough to be boring — that is the point.
