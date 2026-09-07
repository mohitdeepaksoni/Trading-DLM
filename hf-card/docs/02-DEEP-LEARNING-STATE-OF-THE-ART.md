# 02 — Deep learning for trading: state of the art (as of September 2026)

*What actually works, what the numbers are, and where each approach fails. Web-verified against 2025–2026 papers (links in `../REFERENCES.md`). Organised by the six questions of doc 01.*

---

## A. Foundation models for price series (the 2025–26 shift)

### A.1 Kronos — first finance-native foundation model (Tsinghua, arXiv:2508.02739, AAAI 2026, MIT licence)
- **What:** autoregressive transformer over a *learned K-line tokenizer* (OHLCV → discrete tokens preserving price + volume structure). Pre-trained on **12 B K-line records from 45 exchanges**. Models 4.1 M → 499 M params; Qlib fine-tuning pipeline; HF demo.
- **Numbers:** +93 % RankIC over the best general TSFM and +87 % over the best non-pretrained baseline on price forecasting; −9 % MAE on vol forecasting; +22 % generative fidelity for synthetic K-lines.
- **Read-across:** the "language of markets" framing = the same move as Agora's "learn the map from structure to equilibrium". *Use Kronos as a frozen encoder / fine-tunable backbone rather than training from scratch.*

### A.2 General TSFMs: Chronos-2 (Amazon, Oct 2025, MoE, multivariate + covariates), TimesFM-2.5 (Google), Moirai (Salesforce, LOTSA 27 B obs), TiRex, MOMENT
- Chronos-2 leads GIFT-Eval on WQL and MASE with no leakage; Chronos-Bolt is ~250× faster (direct quantile head).
- **Finance verdict (arXiv:2607.05291, 2026):** on realised vol, zero-shot TSFMs ≈ HAR; fine-tuned TSFMs beat HAR modestly. On returns they are not better than the classical nulls. **Use them for vol/volume/covariate forecasting and for probabilistic quantiles, not for direction.**

---

## B. Cross-sectional return prediction & alpha mining

### B.1 The GKX benchmark line (Gu–Kelly–Xiu 2020) and successors
Feed-forward nets on ~900 firm×macro features; monthly OOS R² 0.4–0.7 %; the standard "is my alpha model real" yardstick. Successors: IPCA (Kelly–Pruitt–Su 2019), autoencoder asset pricing (GKX 2021), transformer cross-sectional models (2023–25). Gains since 2020 are incremental, not step-changes.

### B.2 Formulaic alpha mining
- **AlphaGen** (Yu et al. 2023): RL policy writes factor expressions, reward = IC of the *combined* factor set.
- **AlphaForge** (AAAI 2025): generative-predictive net proposes factors, dynamic combination weights.
- **AlphaAgent** (arXiv:2502.16789, 2025): LLM proposes factors with three regularisers — complexity cap (expression-tree size), hypothesis alignment (does the formula match its stated economic rationale), AST-novelty against the existing library — explicitly to fight **alpha decay**.
- **LLM-MCTS factor search** (arXiv:2505.11122).
- **Verdict:** factor mining is an *industrial* process (thousands of candidates); its value is entirely governed by the multiple-testing correction (doc 03). Without a search-aware DSR ledger it is a leak generator.

### B.3 LLM trading agents — the honest 2026 result
arXiv:2608.27734 ("What survives honest evaluation?"): two frontier LLMs, 453 US equities + 39 ETFs, realistic costs, up to 100 candidates/run, 5 repeats, registry-validated features (no look-ahead by construction), search-aware Deflated Sharpe, CSCV-PBO, stationary bootstrap CIs. **Result: every LLM-discovered strategy rejected; only passive benchmarks certified.** A deliberately leaky oracle with Sharpe 35 *passed* conventional DSR/PBO — leakage must be excluded structurally, not statistically.
Also documented: LLM look-ahead bias via pre-training knowledge of the future. **Design consequence:** LLMs are allowed in the loop as *hypothesis proposers* and *code writers*, never as the evaluated policy, and never with access to post-cutoff price history during evaluation.

---

## C. End-to-end portfolio nets (features → weights, P&L in the loss)
- Zhang, Zohren, Roberts (2020) "Deep Learning for Portfolio Optimization": LSTM → softmax weights, Sharpe loss.
- Universal E2E (arXiv:2111.09170); Smart Predict-then-Optimize for portfolios (arXiv:2601.04062, 2026).
- **Financially Guided Deep Portfolio Optimization** (arXiv:2605.28853, May 2026): AttentionLSTM, losses = differentiable Sharpe / Omega / CVaR / risk-parity; 50 SPX names 2007–23, expanding-window walk-forward, bid-ask costs, quarterly rebalance. Best = Omega+CVaR+RP: OOS 2022–23 Sharpe **0.29**, +7.86 % vs SPX −4.52 %. Beats MVP, HRP, NCO, equal-weight.
- LSTM-vol + differentiable risk-budgeting layer: Sharpe 1.38 (2017–22), +55 % over risk parity (Sci. Reports 2025).
- Constrained E2E training with provable convergence for linear constraints (arXiv:2605.11526) — lets budget/box/turnover constraints be exact layers rather than penalties.
- **Verdict:** E2E works *when the loss carries costs and risk*; differentiable optimisation layers (cvxpylayers, OptNet) are mature. Expect OOS Sharpe 0.3–1.4 depending on universe and period — not 3.

---

## D. Reinforcement learning
- Surveys: arXiv:2411.12746 (RL in finance) and arXiv:2512.10913 (167 papers 2017–25: market making, portfolio, execution).
- **FinRL / FinRL-X** (arXiv:2603.21330, 2026): weight-centric modular stack — data → selection → allocation → timing → risk overlay → broker execution — "deployment-consistent" (same code in backtest and live). Good scaffold to fork rather than re-write.
- **HRT** (arXiv:2410.14927): bi-level — high-level PPO stock selection, low-level DDPG execution.
- **Plan Before You Trade** (arXiv:2605.12653, May 2026): inference-time optimisation — the policy re-plans against forecasts at decision time instead of being a static mapping. Analogous to Agora's "net warm-start → Newton polish".
- In-context RL from suboptimal historical data (arXiv:2601.20116, 2026): offline/decision-transformer style learning from logged trades.
- **Verdict:** RL's sample-inefficiency + non-stationarity makes pure model-free RL fragile on returns; it is strong for **execution** and **market making** (well-defined simulators). Offline RL + planning-at-inference is the current best practice for allocation.

---

## E. Limit-order-book (LOB) deep learning
- **DeepLOB** (Zhang, Zohren, Roberts 2019): CNN+Inception+LSTM on 40-level book; still the precision leader.
- **TransLOB**, **LiT** (Frontiers AI, Oct 2025): patch-based transformer; F1 58.99 %, accuracy 59.03 % on FI-2010 3-class mid-price move; robust to shift after fine-tuning.
- **LOBFrame** + "Deep LOB forecasting: a microstructural guide" (Quant. Finance 2025 / arXiv:2403.09267): model-agnostic benchmark; key lesson — predictability is concentrated in *large-tick, liquid* names and decays within seconds; forecast quality ≠ P&L once queue position and latency are modelled.
- Multi-horizon LOB on IPUs (arXiv:2105.10430).
- **Verdict:** LOB DL is real but is an infrastructure game (co-location, sub-ms). Not the retail path; relevant to Emporion (doc 04) only as a *cost/impact model* and a simulator for execution training.

---

## F. Generative market simulators (for training, stress, and hedging)
- **Neural SDEs** (Gierjatowicz et al. 2020; Šiška talks) — calibrated to option prices; robust bounds.
- **Signature-kernel scoring** for non-adversarial neural-SDE training (arXiv:2305.16274); **randomised signatures** (arXiv:2406.10214); random-convolutional-feature matching (arXiv:2606.05138, 2026).
- **Small-data simulators** (Springer 2025/26 chapter) — the honest problem: one history of one market.
- Commodity deep generators for deep hedging (arXiv:2205.13942); risk-neutral simulation (arXiv:2202.13996); FunVol (implied-vol surfaces via FPCA + neural SDE).
- **Verdict:** the generator is how you get *more than one history* to train and stress on. Signature-kernel training is the current stable recipe (no GAN instability). This directly mirrors Agora's "sample a distribution of economies" trick.

---

## G. Hedging
- Deep hedging (Buehler et al. 2019) is now standard in sell-side; **Sigformer** and "Hedging with memory" (arXiv:2508.02759, 2025) add signature features for path dependence; semi-analytic sig-vol models allow Fourier pricing.

---

## H. Uncertainty, sizing, and regimes
- **Conformal Kelly** (arXiv:2608.01494, 2026) — details in doc 01 §4.5. Simplest variant wins.
- **Conformal-HMM** (arXiv:2210.02271): exact-validity prediction sets on Markov regimes.
- Deep ensembles / MC-dropout for epistemic flags — same tooling as `dl-equilibrium-tail-methods.md` §6.

---

## I. Neuro-symbolic / economically constrained
- **ARTEMIS** (arXiv:2603.18107, 2026): neuro-symbolic framework for *economically constrained* market dynamics — the nearest neighbour to what doc 04 proposes (structural constraints as loss/architecture), but without a computable general-equilibrium core. This is the gap Agora fills.

---

## J. Summary scoreboard — what to reuse vs build

| Block | Reuse (2026 best) | Build ourselves |
|---|---|---|
| Price-series encoder | Kronos (fine-tune) / Chronos-2 quantiles | K-line tokenizer for Indian NSE/BSE + crypto if Kronos lacks them |
| Vol / covariance | HAR-RV baseline, GARCH-t, fine-tuned TSFM | denoised Σ (RMT + Ledoit–Wolf) with regime conditioning |
| Alpha | GKX-style cross-section net; AlphaAgent-style *governed* mining | SDF-residual training (doc 04) |
| Allocation | HRP / BL / differentiable CVaR layers | equilibrium-anchored allocator with cost-aware loss |
| Sizing | Conformal Kelly (simple rolling quantiles) | regime-scaled conformal Kelly |
| Execution | Almgren–Chriss + √-impact; FinRL-X broker layer | broker adapters (Kite/IBKR/Alpaca) |
| Simulator | signature-kernel neural SDE | macro-shock-conditioned generator driven by Agora |
| Honesty | DSR, CPCV, PBO, registry-validated features | the *ledger* (doc 03) as a first-class artefact |
