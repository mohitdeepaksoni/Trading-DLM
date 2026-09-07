# 04 — The unique design: **Emporion**, an equilibrium-anchored trading engine

*Working name "Emporion" (Greek: trading post — the sibling of Agora, the marketplace). Placeholder; rename freely. This document says what nobody else has built, why it is defensible, and how it reuses the Mathemetics-DLM code.*

---

## 0. One-paragraph thesis

Every public DL trading system in doc 02 learns a map **(past prices, features) → (forecast or weights)** and is disciplined only by realised P&L. That is why they overfit and decay. Agora showed a different discipline works for economies: **learn the map from structure to equilibrium, trained self-supervised on a residual that must be zero for any valid answer** (`‖z(p)‖`), then polish with an exact solver and gate deployment on the residual. Emporion applies the same discipline to markets: the residual is the **no-arbitrage pricing-kernel condition** `E_t[M_{t+1} R^e_{i,t+1}] = 0` plus **market clearing across heterogeneous investors**, the "structure" includes **real macro-sector shocks computed by Agora's CGE**, and trading is nothing more than **sizing a bet on the certified residual** (mispricing) with conformal Kelly. Uniqueness = (i) the *only* trading engine with a computable general-equilibrium macro core feeding it, (ii) residual-trained rather than P&L-trained, (iii) the honesty ledger is part of the model, not the paper.

---

## 1. Why this is genuinely new (checked against 2025–26 literature)

| Existing | What it lacks | Emporion |
|---|---|---|
| Chen–Pelger–Zhu deep SDF (GAN-GMM) | No macro structure; adversarial training unstable; no trading/sizing layer; no equilibrium of *quantities* | Same moment residual, but trained with Agora's tail-methods (RAD sampling, self-adaptive weights, L∞ term) and *coupled* to a market-clearing residual; Newton-polished |
| ARTEMIS (neuro-symbolic constrained dynamics, 2026) | Constraints are hand-written rules, no general-equilibrium solver | Constraints come from a solved CGE (70-economy, 45/163-sector) → sector price/output/profit responses to policy & shocks |
| Kronos / TSFMs | Learn price language only; no economics | Kronos as *encoder*; Emporion adds the structural residual head |
| E2E portfolio nets | Trained on P&L ⇒ memorise regimes | Trained on residual ⇒ regime-agnostic by construction; P&L only for *evaluation* |
| Conformal Kelly | Sizing only | Sizing of a *certified* mispricing signal |
| LLM agents | Rejected by honest eval | LLMs allowed as hypothesis writers into the ledger, never as the policy |

Nobody couples an amortised CGE to a pricing-kernel net. That coupling is the moat and is also the piece that only this codebase can build today.

---

## 2. The mathematics of Emporion

> **Revision note (doc 07):** R1 is now *derived* from the calibrated investors of R2 (`M = Σ_h ω_h β_h u'_h(c_{t+1})/u'_h(c_t)` × a penalised neural correction), and R2 is calibrated to a **Market Accounting Matrix** of real holdings and flows. Read §2.2 with that change; doc 07 Part A is authoritative where they differ.

### 2.1 State and structure
- Assets i = 1..N (stocks, ETFs, futures, crypto — "any market" ⇒ per-market plug-ins as in Agora's registry).
- Information set `I_t` = (price-series embedding from Kronos/own encoder, firm features z_{i,t}, macro state m_t, **Agora sector-shock vector a_t**).
- `a_t ∈ R^S`: for the current macro-policy/news state (tariff, tax, rate, energy price, carbon), Agora's country-general net returns the equilibrium *change* in sector prices, outputs, profits, employment in one forward pass (326 k policies/s). Each asset has a sector-exposure map `E ∈ R^{N×S}` (GICS/NIC weights, supply-chain from ICIO) ⇒ asset-level structural shock `E a_t`.

### 2.2 The two residuals (the Emporion analogue of Walras)
**R1 — pricing-kernel residual (no-arbitrage):**
```
M_{t+1} = M_θ(I_t, ε_{t+1})                      (SDF net; positivity via exp)
r1_i(t) = E_t[ M_{t+1} R^e_{i,t+1} · g_φ(I_t, z_{i,t}) ]   for a bank of test functions g_φ
L_R1    = Σ_i ‖ r1_i ‖²  (+ tail term:  LogSumExp_i τ‖r1_i‖²)
```
**R2 — market-clearing residual (heterogeneous investors, Scarf-style):**
```
H investor types h with demand x_h(p; wealth_h, beliefs_h, constraints_h)   (CRRA / prospect / mandate)
z(p) = Σ_h x_h(p) − s      (s = float / supply)
L_R2 = ‖ z(p̂) ‖²          where p̂ = net's equilibrium price
```
R2 is literally `agora/models/ces_exchange.py` with assets as goods and a leverage/short constraint set — *reuse the module*. Investor "endowments" are holdings; "preferences" are risk aversion and mandates (e.g. index funds must hold cap-weights: the inelastic-market hypothesis of Gabaix–Koijen 2021 says flows move prices with multiplier ≈ 5 — this is exactly a low-elasticity CES demand).

**Mispricing signal** = `p_t − p̂_t` (observed − equilibrium) scaled by the residual-certified uncertainty. Trade toward equilibrium; size ∝ conformal-Kelly.

### 2.3 Training (self-supervised, identical recipe to Agora)
```
L = λ₁ L_R1 + λ₂ L_R2 + λ₃ L_tail + λ₄ L_cost-aware-consistency
```
- Sample *distributions of market states* — real history **and** signature-kernel neural-SDE generated paths **and** Agora-shocked counterfactuals (a tariff shock ⇒ sector shifts ⇒ what should prices be?). This is the "collocation points" trick from `dl-equilibrium-tail-methods.md`: RAD resampling toward high-residual states, self-adaptive per-state weights, gradient-magnitude balancing across R1/R2, curriculum on shock size.
- **No P&L in training.** P&L is only computed in the evaluation ledger (doc 03). This is the single most important design decision: it removes the incentive to memorise a regime.

### 2.4 Inference = warm-start + polish + gate (Agora's `HybridSolver`)
1. Net gives `p̂`, `M`, residuals in one pass.
2. Damped Newton on `z(p)=0` for a few steps (exact market clearing given investor demands) — `agora/solvers/classical.py`.
3. **Gate:** deploy signal only if post-polish residual < ε and ensemble disagreement < δ. Otherwise the asset is "no opinion" (flat). This gate is what makes the system *abstain*, which is where retail systems die (they always trade).

### 2.5 Allocation and sizing
```
views       q_i = E_t[R^e_i] implied by (M, p̂)         Ω_i = conformal interval width²
prior       π   = Black–Litterman market-equilibrium (cap-weights) — consistent with R2!
w_BL        = BL(π, P, q, Ω)   then  HRP/CVaR risk overlay, turnover penalty, box limits
size        f_i = ½ · Kelly_i · (1 / width_i)  capped at leverage L_max, regime-scaled by HMM P(high-vol)
execution   Almgren–Chriss schedule; √-impact capacity check; limit-order preference
```
Black–Litterman's prior π = λΣw_mkt *is* an equilibrium statement, so R2 and BL are the same object at two resolutions — a nice internal consistency nobody exploits.

### 2.6 The ledger is a model component
Every training run, hyper-parameter, factor candidate, and evaluation is appended to `ledger.parquet`; DSR is computed against the ledger count N automatically; the model card prints DSR, PBO, CPCV distribution, breakeven cost, regime slices. **A model without a ledger row cannot be exported.** (Same spirit as Antaḥkaraṇa release gates, but kept fully separate — this is a finance project.)

---

## 3. Where Agora's macro core adds alpha (the concrete mechanism)
1. **Policy events** (budget, tariff, rate decision, carbon price, energy shock): Agora computes the sectoral general-equilibrium response *including second-round effects* (input-output propagation via ICIO), which naive sector ETFs / analysts miss. Signal: `E a_t` vs. observed sector moves over the following days → mispricing. Literature: post-announcement drift at sector level persists for days–weeks (a slow-information channel, not a microstructure one — retail-accessible).
2. **Supply-chain contagion**: a shock to sector k propagates to suppliers/customers with lags; ICIO gives the graph; Agora gives the magnitudes; the net learns the lag structure. Cohen–Frazzini (2008) "Economic links and predictable returns" documented customer-momentum alpha; Emporion generalises it from firm links to the full IO network with equilibrium magnitudes.
3. **Cross-country**: the same shock has different sector responses in India, Spain, US (Agora has 70+ ICIO economies) ⇒ relative-value across markets — the "any market" scope becomes a breadth advantage (`IR ≈ IC·√BR`).
4. **Stress testing**: Agora-shocked scenarios are *economically coherent* stress paths for the generator, not random noise — better tail training and better MDD estimates.

Honest expectation: this is a **slow, medium-frequency (days–weeks), diversified, low-turnover** edge. It is *not* HFT. Expected certified OOS Sharpe if it works: 0.6–1.2 after costs on a 200–500-asset universe. That is what real edges look like.

---

## 4. Architecture (plug-in, registry-driven — copied from `agora/core/registry.py`)
```
emporion/
  core/        registry, market schema (assets, investors, constraints), device, ledger
  data/        point-in-time loaders: NSE/BSE bhavcopy, US (Polygon/Tiingo/Alpaca), crypto (Binance), fundamentals; feature registry with lags
  encoders/    kronos_adapter (frozen/fine-tune), own K-line tokenizer, signature features
  structure/   agora_bridge (policy/news state → sector shock a_t), exposure maps E, ICIO graph
  models/      sdf_net (R1), clearing_model (R2 = ces_exchange with asset semantics), investor_types
  solvers/     residual_trainer (RAD, self-adaptive weights, L∞), hybrid (net→Newton), gate
  allocate/    black_litterman, hrp, cvar_layer, conformal_kelly, hmm_regime
  execute/     almgren_chriss, impact, broker adapters (Kite, IBKR, Alpaca, Binance), paper
  evaluate/    cpcv, dsr, pbo, bootstrap, ledger, model_card
  sim/         neural_sde_sigkernel generator, agora_shock_scenarios
  console/     Analyst-style dashboard (reuse Agora web app patterns): signals, residual maps, ledger, live shortfall
```
Reused directly from Mathemetics-DLM: `core/registry.py`, `core/device.py`, `models/ces_exchange.py` (subclassed), `solvers/classical.py`, `solvers/hybrid.py`, `solvers/retrieval_warmstart.py`, `nets/mlp.py`, `nets/moe.py`, the country-general net as a *frozen service* (`agora_bridge`), the tail-method training loop, the console skeleton.

---

## 5. Sharing it with the world (make it a *thing*, not a repo)
- Open-source **engine + evaluation ledger + paper-trading harness** (MIT/Apache). Keep *fitted weights and live signals* private (that is the product / your own account).
- Publish a **benchmark**, not a strategy: "Emporion-Bench — equilibrium-anchored trading under honest evaluation": fixed point-in-time datasets (US + India + crypto), fixed cost model, required ledger, leaderboard by **DSR & CPCV distribution**, passive baselines pre-certified. The LLM-agents paper showed the field needs exactly this, and no one owns it yet. A benchmark is how a solo author gets cited and adopted; a strategy is how a solo author gets ignored or copied.
- Write-up: arXiv paper "Equilibrium-anchored deep trading: self-supervised pricing kernels with a computable general-equilibrium prior" + a plain-language explainer (as in Agora `analysis/PLAIN-LANGUAGE-EXPLAINER.md`).
- Model card discipline: every release ships its ledger metrics; no Sharpe without DSR.

---

## 6. Risks and the honest counter-arguments
- **Macro-to-asset mapping is noisy**: sector exposure of a conglomerate is fuzzy; second-round CGE effects are model-dependent. *Mitigation:* learn E partially; validate a_t against realised sector returns on historical events (2018–19 tariffs, 2022 energy shock, 2025 tariff rounds) *before* trading.
- **The residual may be satisfiable trivially** (M ≈ const, p̂ ≈ p). *Mitigation:* adversarial test functions g_φ (CPZ), and R2 with real investor-type heterogeneity; monitor identification metrics.
- **Edge is small and slow**: retail cost and tax may eat it in India (STT + 20 % STCG). *Mitigation:* low turnover by design (weekly rebalance), ETFs/futures where cheaper, or long-term tilts taxed at LTCG.
- **Everything might fail honest evaluation.** Then the deliverable is the benchmark + a negative result — still the most useful thing published this year on the topic. Plan for it.
