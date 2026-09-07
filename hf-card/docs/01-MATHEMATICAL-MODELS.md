# 01 — The Mathematical Models of Trading: a working catalogue

*Deep-research capture, 2026-09-05. Every formula a DL trading engine can be built on, grouped by the question it answers. Each entry says what it models, the formula, what is learnable, and where it breaks. Read this as the "physics" that the deep-learning layer (doc 02) must respect, exactly as Agora treats Walras' Law and market clearing.*

---

## 0. Map: the six questions every trading system answers

| # | Question | Classical answer | Section |
|---|---|---|---|
| 1 | How do prices move? | Stochastic processes (GBM, jumps, stochastic/rough vol) | §1 |
| 2 | What explains expected returns? | Factor models, SDF / no-arbitrage | §2 |
| 3 | What can I forecast, and how well? | Time-series econometrics (ARMA, GARCH, HAR, cointegration, HMM) | §3 |
| 4 | How much of each asset should I hold? | Portfolio theory (Markowitz → HRP → Kelly → CVaR) | §4 |
| 5 | How do I get in and out without moving the price? | Optimal execution & market microstructure | §5 |
| 6 | How do I price and hedge non-linear claims? | Options, PDEs, deep hedging | §6 |
| — | How do I know it's not luck? | Backtest statistics (doc 03) | doc 03 |

The whole of quant trading is these six blocks plus honesty. A "unique" engine is one that changes how the blocks are *coupled* (doc 04), not one that reinvents a block.

---

## 1. Price dynamics — the stochastic process layer

### 1.1 Geometric Brownian Motion (Bachelier 1900 → Samuelson 1965 → Black–Scholes 1973)
```
dS_t = μ S_t dt + σ S_t dW_t            ⇒   log S_T = log S_0 + (μ − σ²/2)T + σ W_T
```
Log-returns are i.i.d. Normal. **Learnable:** μ (nearly unlearnable — the "μ problem": σ/√T noise swamps μ at any horizon under ~10 years), σ (very learnable). **Breaks:** fat tails, vol clustering, jumps, leverage effect. Use as the null model every result must beat.

### 1.2 Merton jump-diffusion (1976) and Kou double-exponential (2002)
```
dS/S = (μ − λk) dt + σ dW + (J − 1) dN_t,     N ~ Poisson(λ),  log J ~ N(m, v²)   (Merton)
                                               log J ~ asymmetric Laplace(η₁, η₂, p) (Kou)
```
**Learnable:** jump intensity λ, jump size law. **Use:** crash-aware risk, option skew. DL analogue: a neural intensity (Hawkes) — see §5.4.

### 1.3 Stochastic volatility — Heston (1993), SABR (2002)
```
Heston:  dS = μS dt + √v S dW¹,   dv = κ(θ − v) dt + ξ √v dW²,   ⟨dW¹,dW²⟩ = ρ dt
SABR:    dF = α F^β dW¹,           dα = ν α dW²,                    ⟨dW¹,dW²⟩ = ρ dt
```
Heston has a closed-form characteristic function ⇒ FFT pricing (Carr–Madan 1999). **Learnable:** (κ,θ,ξ,ρ,v₀) by calibration to the vol surface — the classic "neural calibration" task (Horvath, Muguruza, Tomas 2021: net maps params → implied-vol surface, then invert).

### 1.4 Rough volatility — rough Bergomi (Bayer, Friz, Gatheral 2016), rough Heston (El Euch–Rosenbaum 2019)
```
log-vol driven by fractional Brownian motion with Hurst H ≈ 0.05–0.15:
v_t = ξ₀(t) · exp( η √(2H) ∫₀ᵗ (t−s)^{H−1/2} dW_s  −  (η²/2) t^{2H} )
```
Empirically the best description of realised-vol roughness and the ATM skew ∝ T^{H−1/2}. **Learnable:** H, η; but non-Markovian ⇒ simulation is expensive ⇒ prime target for neural surrogates (signatures, §7).

### 1.5 Ornstein–Uhlenbeck / mean reversion (Vasicek 1977; pairs trading)
```
dX_t = θ(μ − X_t) dt + σ dW_t      half-life = ln2/θ
discrete: X_{t+1} = a + b X_t + ε,   θ = −ln b / Δt
```
Foundation of statistical arbitrage: trade the spread X = log P_A − β log P_B when |z| = |X−μ|/σ_stat > entry. Optimal thresholds: Bertram (2010) closed-form for expected-return-per-unit-time. **Breaks:** cointegration is not stationary in time — regime-monitor it (§3.5).

### 1.6 Lévy / α-stable / Student-t tails
Empirical: daily equity returns have tail index α ≈ 3–4 (finite variance, infinite kurtosis-ish). Any loss function assuming Gaussian errors is miscalibrated in the tail — this is exactly the "tail" problem from `Mathemetics-DLM/analysis/research/dl-equilibrium-tail-methods.md`, transplanted.

---

## 2. Expected returns — factor models and the pricing kernel

### 2.1 CAPM (Sharpe 1964, Lintner 1965)
```
E[R_i] − R_f = β_i (E[R_m] − R_f),   β_i = Cov(R_i,R_m)/Var(R_m)
```
### 2.2 APT / multi-factor (Ross 1976; Fama–French 1993, 2015; Carhart 1997; Hou–Xue–Zhang q-factor 2015)
```
R_i,t − R_f = α_i + Σ_k β_ik F_k,t + ε_i,t
FF5: F = {MKT, SMB (size), HML (value), RMW (profitability), CMA (investment)} + MOM (Carhart)
```
**α is the whole game.** Everything in trading is "find α_i > 0 after costs that is not explained by known F". The "factor zoo" (Harvey, Liu, Zhu 2016: >300 published factors; most fail after t-stat > 3 hurdle) is the cautionary tale — see doc 03.

### 2.3 The stochastic discount factor (SDF) — the unifying equation
```
No-arbitrage ⇔ ∃ M_{t+1} > 0 :   E_t[ M_{t+1} R_{i,t+1} ] = 1   ∀ i
Excess-return form:               E_t[ M_{t+1} R^e_{i,t+1} ] = 0
Linear SDF:                        M_{t+1} = 1 − Σ_k b_k F^e_{k,t+1}
```
This is the finance analogue of Agora's excess-demand residual `z(p)=0`: a **conditional moment condition** that must hold for *all* assets and *all* information sets. Chen, Pelger & Zhu (2023, *Management Science*) "Deep Learning in Asset Pricing" train M as a net with an adversarial "test-asset" net that searches for information sets g(I_t) that violate it — GMM as a GAN:
```
min_M max_g   Σ_i  ‖ E[ M_{t+1} R^e_{i,t+1} g(I_t, I_{i,t}) ] ‖²
```
**This is the closest published object to a "self-supervised residual-trained" trading net** and the natural bridge to Agora's method (doc 04).

### 2.4 Consumption-based / Euler equation (Lucas 1978; Hansen–Singleton 1982)
```
M_{t+1} = β u'(c_{t+1}) / u'(c_t)      (CRRA: β (c_{t+1}/c_t)^{−γ})
```
Links asset prices to the *macro* economy — the doorway through which Agora's CGE (which produces sectoral prices, output, household consumption under policy shocks) can feed a pricing kernel.

### 2.5 Cross-sectional return prediction as a supervised problem (Gu, Kelly, Xiu 2020, *RFS*)
```
r_{i,t+1} = g*(z_{i,t}) + ε_{i,t+1},   z ∈ R^{~900} (94 firm chars × 8 macro + interactions)
```
GKX benchmark: NN3 OOS R² ≈ 0.4 %/month for all stocks, ≈ 0.7 % top-1000. Long-short decile Sharpe ≈ 1.35 (before costs). *Those are the realistic ceilings for monthly cross-section prediction.* Anyone claiming 5 % R² on monthly returns is leaking.

---

## 3. Forecasting — econometric time-series layer

### 3.1 ARMA / ARIMA (Box–Jenkins 1970)
```
φ(L)(1−L)^d y_t = θ(L) ε_t
```
Useless for returns (near-zero autocorrelation past lag 1 at daily horizon); useful for *volumes, spreads, realised vol*.

### 3.2 GARCH family (Engle 1982, Bollerslev 1986, GJR 1993, EGARCH 1991)
```
r_t = μ + σ_t ε_t,   σ²_t = ω + α ε²_{t−1} + β σ²_{t−1}          (GARCH(1,1))
GJR:  + γ ε²_{t−1} 1[ε_{t−1}<0]   (leverage asymmetry)
```
GARCH(1,1) with Student-t innovations remains a hard-to-beat 1-day vol forecaster. DL vol nets should report against it.

### 3.3 HAR-RV (Corsi 2009) — the workhorse for realised volatility
```
RV_{t+1} = c + β_d RV_t + β_w RV_t^{(w)} + β_m RV_t^{(m)} + ε
RV_t = Σ_{intraday} r²_j    (5-min sampling; use realised kernels for noise)
```
2026 result (arXiv:2607.05291): time-series foundation models forecasting realised vol only *match* HAR in zero-shot, beat it slightly after fine-tuning. HAR is the bar.

### 3.4 Cointegration & VECM (Engle–Granger 1987, Johansen 1991)
```
Δy_t = Π y_{t−1} + Σ Γ_i Δy_{t−i} + ε,   Π = αβ′ (rank r = # cointegrating vectors)
```
Basis of pairs/basket stat-arb (§1.5). Johansen test ranks the cointegrating space; rolling re-estimation with a decay test is mandatory.

### 3.5 Regime switching — Hidden Markov / Markov-switching (Hamilton 1989)
```
s_t ∈ {1..K} Markov chain, P(s_{t+1}=j | s_t=i) = A_ij
r_t | s_t ~ N(μ_{s_t}, σ²_{s_t})   (or Student-t / GARCH within regime)
```
Filtered regime probabilities `P(s_t | r_{1:t})` are the standard "should I be in risk-on" gate. Practical 2025–26 evidence: turning strategies off in the high-vol regime removes a large share of losing trades. Extension: conformal-HMM (arXiv:2210.02271) gives exactly-valid prediction sets.

### 3.6 Kalman filter / state-space (dynamic β, dynamic hedge ratio)
```
x_t = F x_{t−1} + w_t,   y_t = H x_t + v_t   ⇒ time-varying hedge ratio β_t for pairs
```

### 3.7 Signal-to-noise realities (the numbers to calibrate ambition)
| Horizon | Typical achievable OOS R² | Typical IC / RankIC |
|---|---|---|
| 1 min–1 hour (LOB mid-price direction) | accuracy 55–60 % on 3-class (LiT 59 %, FI-2010) | — |
| 1 day cross-section | 0.1–0.5 % | IC 0.02–0.06 |
| 1 month cross-section | 0.3–0.7 % (GKX) | IC 0.03–0.08 |
| Realised vol 1-day | 30–60 % (log RV) | — |

Vol is forecastable; direction barely is. This asymmetry drives the design in doc 04.

---

## 4. Portfolio construction — the allocation layer

### 4.1 Markowitz mean–variance (1952)
```
max_w  w′μ − (λ/2) w′Σw    s.t. 1′w = 1  (+ box, leverage, turnover constraints)
closed form (unconstrained): w* = (1/λ) Σ⁻¹ μ
```
**Breaks:** error-maximiser — errors in μ get amplified by Σ⁻¹. Fixes: shrinkage (Ledoit–Wolf 2004, non-linear 2020), robust/Black–Litterman, or don't estimate μ at all.

### 4.2 Black–Litterman (1992)
```
μ_BL = [ (τΣ)⁻¹ + P′Ω⁻¹P ]⁻¹ [ (τΣ)⁻¹ π + P′Ω⁻¹ q ],   π = λ Σ w_mkt   (equilibrium prior)
```
Views (P,q,Ω) blended with market equilibrium. **Where a DL model plugs in:** the net's forecasts become views q with Ω from its uncertainty (conformal/ensemble). This is the right way to use a noisy forecaster.

### 4.3 Risk parity / equal risk contribution (Maillard, Roncalli, Teïletche 2010)
```
RC_i = w_i (Σw)_i / √(w′Σw),   choose w s.t. RC_i = RC_j  ∀ i,j
```
### 4.4 Hierarchical Risk Parity (López de Prado 2016) and NCO (2019)
Tree-cluster the correlation matrix → recursive bisection allocation. No matrix inversion ⇒ robust to Σ estimation error. The default *robust* baseline for any DL allocator to beat.

### 4.5 Kelly criterion (1956) and fractional Kelly
```
binary bet:      f* = p − q/b
continuous:      f* = (μ − r)/σ²      (growth-optimal leverage; multi-asset f* = Σ⁻¹(μ − r))
growth rate:     g(f) = r + f(μ−r) − f²σ²/2   ;  half-Kelly gives 75 % of max growth at half the variance
```
Full Kelly with *estimated* μ is over-betting (μ error). Practice: ¼–½ Kelly; 2026 "Conformal Kelly" (arXiv:2608.01494) scales f by the width of a conformal interval — wider uncertainty ⇒ smaller f. Reported: 28.5 % annualised log-growth, Sharpe 1.34, MDD 27.7 % (2016–21, SPX universe, with costs) vs 15.9 % buy-and-hold; **the simplest rolling-quantile conformal variant won**; every "adaptive" tweak lost 0.7–5.3 pts/yr.

### 4.6 CVaR / Expected Shortfall optimisation (Rockafellar–Uryasev 2000)
```
CVaR_α(w) = min_ζ  ζ + (1/(1−α)) E[ (−w′R − ζ)⁺ ]     ⇒ linear program on scenarios
```
Coherent risk measure; differentiable via the ζ-trick ⇒ usable directly as a DL loss.

### 4.7 Differentiable objectives for end-to-end nets (Zhang, Zohren, Roberts 2020; arXiv:2605.28853)
```
Sharpe surrogate:  L = − mean(r_p) / std(r_p)      over a batch of consecutive periods
Sortino:           − mean(r_p) / std(min(r_p,0))
Omega:             − E[(r_p − τ)⁺] / E[(τ − r_p)⁺]
Costs:             + c · Σ_t ‖w_t − w_{t−1}‖₁      (turnover penalty — mandatory)
Combined (2026 best): Omega + CVaR + risk-parity regulariser
```
The net outputs w_t = softmax/ tanh (long-only / long-short) from features; backprop through realised P&L. Reported 2026: AttentionLSTM Omega-CVaR-RP, 50 SPX names, 2022–23 OOS Sharpe 0.29 vs SPX −0.02. *Note how small the OOS Sharpe is even for a "winner".*

### 4.8 Merton's portfolio problem (1969) — continuous-time control
```
max E[∫ e^{−ρt} u(c_t) dt],   dW = [rW + π(μ−r)W − c] dt + πσW dB
CRRA solution: π* = (μ − r)/(γ σ²)   (constant fraction — Kelly with risk-aversion γ)
```
Basis for RL formulations of trading: state (wealth, features), action (π, c), reward u(c).

### 4.9 Stochastic Portfolio Theory (Fernholz 2002)
Diversity-weighted / functionally-generated portfolios that outperform market cap weights under an "excess growth" condition — no forecasting required. A rare *model-free* edge; relevant to the "any market" ambition.

---

## 5. Execution and microstructure — the trading-cost layer

### 5.1 Almgren–Chriss optimal execution (2000)
```
minimise  E[cost] + λ Var[cost],   liquidate X shares over T
temporary impact h(v) = ε sgn(v) + η v,   permanent g(v) = γ v
solution: trajectory x_t = X sinh(κ(T−t))/sinh(κT),  κ² = λσ²/η
```
### 5.2 Square-root impact law (Torre 1997; Bouchaud et al.)
```
ΔP/P ≈ Y σ √(Q / V_daily),   Y ≈ 0.5–1
```
The single most important cost formula for anyone with real size. Costs *grow with the square root of trade size*; strategy capacity is bounded by it.

### 5.3 Market making — Avellaneda–Stoikov (2008)
```
reservation price  r = s − q γ σ² (T−t)
optimal spread     δ⁺+δ⁻ = γσ²(T−t) + (2/γ) ln(1 + γ/κ)
```
(q = inventory, κ = order-arrival decay). Guéant–Lehalle–Fernandez-Tapia (2013) extend to inventory limits.

### 5.4 Order flow — Hawkes processes (1971; Bacry et al. 2015)
```
λ(t) = μ + Σ_{t_i<t} α e^{−β(t−t_i)}
```
Self-exciting arrivals; neural Hawkes (Mei & Eisner 2017) is the DL version. Base for LOB simulators.

### 5.5 Information models — Kyle (1985), Glosten–Milgrom (1985)
Kyle's λ: price impact per unit order flow = (σ_v / σ_u)/2; informed vs noise traders. Explains *why* impact exists and why alpha decays as it is traded.

### 5.6 Order-flow imbalance (Cont, Kukanov, Stoikov 2014)
```
OFI_t = Σ (Δbid-size terms − Δask-size terms);  ΔP ≈ β · OFI  (linear, R² ~ 0.6–0.8 intraday)
```
The strongest *known* short-horizon linear predictor; every LOB DL model should beat OFI-linear.

---

## 6. Derivatives — pricing and hedging layer

### 6.1 Black–Scholes–Merton (1973)
```
∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S − rV = 0
C = S N(d₁) − K e^{−rT} N(d₂),   d₁ = [ln(S/K) + (r+σ²/2)T]/(σ√T),  d₂ = d₁ − σ√T
```
### 6.2 Greeks, implied vol, the surface
Δ, Γ, ν, Θ, ρ; implied vol σ_imp(K,T) surface; SVI parameterisation (Gatheral 2004) for arbitrage-free fits.

### 6.3 Deep hedging (Buehler, Gonon, Teichmann, Wood 2019)
```
min_θ  ρ( −Z + p + Σ_t δ_θ(t, I_t)·ΔS_t − costs )     ρ = CVaR or entropic risk
```
Replaces Δ-hedging with a net trained on simulated paths under *any* market model incl. costs. Signature-transformer variants (Sigformer) and "hedging with memory" (arXiv:2508.02759) are the 2025 frontier.

### 6.4 Neural SDE market models (Gierjatowicz, Sabate-Vidales, Šiška, Szpruch, Žurič 2020)
```
dS = μ_θ(t,S) dt + σ_θ(t,S) dW   with (μ_θ, σ_θ) nets calibrated to option prices; robust pricing bounds
```

---

## 7. Cross-cutting mathematics that makes DL work on paths

- **Path signatures** (Lyons; Chevyrev & Kormilitzin 2016): `S(X) = (1, ∫dX, ∫∫dX⊗dX, …)` — a universal, order-sensitive feature map of a path; linear functionals of the signature approximate any continuous path functional. Randomised signatures / signature kernels (arXiv:2305.16274) give non-adversarial training for market generators.
- **Optimal transport / Wasserstein** losses for distribution matching in generators.
- **Extreme value theory** (GPD tails) for tail-risk estimates beyond sample.
- **Random matrix theory** (Marchenko–Pastur) to denoise Σ: eigenvalues below λ₊ = σ²(1+√(N/T))² are noise.
- **Information coefficient algebra** (Grinold–Kahn *fundamental law*): `IR ≈ IC · √BR · TC` — Sharpe scales with skill × √(number of independent bets) × transfer coefficient. This is why breadth (many assets / many markets) matters more than a slightly better model on one asset — and why the "any market" scope is *statistically* right.

---

## 8. What the catalogue says a *practical* engine should be built from

1. **Forecast what is forecastable:** volatility, covariance, regime, liquidity/impact — not raw direction. Direction alpha enters as small, diversified, cost-aware tilts.
2. **Allocate with estimation-error-robust methods** (HRP / risk parity / BL with model views) and **size with fractional/conformal Kelly**.
3. **Respect a structural residual** (no-arbitrage SDF moments) so the model is *disciplined* the way Agora is disciplined by `z(p)=0` — this is what makes it unique (doc 04).
4. **Pay costs in the loss** (turnover, √-impact) — never as an after-thought.
5. **Prove it's not luck** with the doc 03 toolkit before a single rupee/dollar is risked.
