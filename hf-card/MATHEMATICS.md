# Emporion — the mathematics, in full

Every object below is implemented in the engine and has a test named in the last column of the table at the end.
Notation: $N$ assets, horizon $h$ sessions, payoff $D\in\mathbb{R}^N$ (gross $h$-period value per share, or gross return
in "return units"), observed price $p$, gross risk-free rate over the horizon $R_f$, supply per share $s$.

---

## 1. Investors, demand and market clearing

Four investor types clear the market (Scarf/Walras excess demand $z(p)=\sum_h x_h(p)-s$):

**Active CARA investors** ($k=1..K$), risk aversion $a_k$, beliefs $\mu$ (shared), covariance $\Sigma$. Maximising
$\mathbb{E}[-e^{-a_k W}]$ with $W = x^\top(D - R_f p)$ under $D\sim\mathcal N(\mu,\Sigma)$ gives

$$x_k(p) \;=\; \frac{1}{a_k}\,\Sigma^{-1}\big(\mu - R_f\,p\big).$$

**Leveraged investors**: the same demand with a leverage cap $\lVert x\rVert_1 \le \ell$ (projected).
**Extrapolative retail**: $\mu^{\text{ret}} = \mu + \beta\,\text{mom}$, where mom is trailing return.
**Mandated / index holders**: price-insensitive, hold a fraction $\varphi$ of supply.

Clearing $\sum_k x_k(p) + \varphi s = s$ with $A \equiv \sum_k a_k^{-1}$ gives the closed-form CARA–Gaussian equilibrium

$$\boxed{\;\hat p \;=\; R_f^{-1}\big(\mu - \kappa\,\Sigma s\big),\qquad \kappa \;=\; \frac{1-\varphi}{A}\;}$$

and, with leverage caps or extrapolators, $\hat p$ is found by damped Gauss–Newton on $z(p)$ with per-market
backtracking (the closed form is the test oracle). Walras' law holds with cash as numeraire ($z_0 = -p\cdot z$), and
the solution is scale-invariant in $(p,\mu,D)$.

The **signal** is the log mispricing $-r_{\text{mkt}} \equiv \log(\hat p / p)$: buy what trades below the model's
equilibrium.

## 2. The structural pricing kernel

The same investors imply a stochastic discount factor. With aggregate absolute risk aversion $1/A$ and the
price-sensitive sector holding $q = (1-\varphi)s$,

$$M \;=\; \frac{e^{-q^\top D/A}}{\mathbb{E}\big[e^{-q^\top D/A}\big]},\qquad
\mathbb{E}\big[e^{w^\top D}\big] = \exp\!\big(w^\top\mu + \tfrac12 w^\top\Sigma w\big)\ \ (w=-q/A),$$

so the normaliser is analytic under Gaussian payoffs (and Monte-Carlo otherwise). The kernel is projected onto
$\mathbb{E}[M] = 1/R_f$ (hard), and the moment residuals $\mathbb{E}[M R^e g]$ with fixed instruments $g$ are the GMM
test that the kernel prices the clearing equilibrium — it does, exactly, on synthetic markets.

## 3. Identification — what prices can tell you

Rearranging the clearing law, $R_f p = \mu - \kappa\Sigma s$. For any $\Delta\mu$ there is a $\Delta\kappa$ with
$\Delta\mu = \Delta\kappa\,\Sigma s$ along the direction $\Sigma s$; hence:

**Proposition (research/09).** From a panel of prices and realised payoffs, $(\mu,\kappa)$ is identified; $(A,\varphi)$
are not separately identified (only their ratio through $\kappa$); and the *level* of $\mu$ is identified only up to an
external anchor. Consequences that shaped the architecture:

1. **State** is parametrised as $\kappa=(1-\varphi)/A$ and seeded by the **panel estimator**: with the realised
   excess payoff per share $\pi_i = \overline{(D_i - R_f p_i)}$ over a trailing window and $x = \Sigma s$,
   $$\hat\kappa \;=\; \frac{\pi^\top x}{x^\top x}\ \ (\ge 0),$$
   winsorised to $[10^{-4}, 50]$; the network may only rescale it by $e^{\tfrac12\tanh(\cdot)}$.
2. **The law is imposed at $\hat p$, not at $p$.** Writing the residual at the observed price forces
   $\mu \to R_f p + \kappa\Sigma s$ — it *explains away* the mispricing and inverts the signal (correlation with true
   mispricing $-0.50$ on synthetic markets); at $\hat p$ the signal is preserved ($+0.95$).
3. **Beliefs need an anchor** the price does not contain: filed fundamentals, measured expectations, or the
   general-equilibrium prior.

## 4. Anchors (the external information)

For asset $i$ at decision date $t$, using only filings whose filing time $\le t$ (revisions resolved *after* the date
filter; research/03 §12):

- **Value anchor**: with trailing-four-quarter EPS $e_i$ (split-adjusted on the current share count) and the
  cross-sectional median earnings yield $\bar y = \operatorname{med}_j(e_j/p_j)$,
  $\ p^{\text{fund}}_i = e_i/\bar y$, $\ \text{value}_i = \log(p^{\text{fund}}_i/p_i)$ (clipped to $[-1,1]$ as a feature).
- **Earnings surprise (SUE)**: seasonal difference $d_q = e_q - e_{q-4}$ over the last 12 quarters (≥ 8),
  $\ \text{SUE}_i = d_{\text{last}} / \operatorname{sd}(d_{\text{prior}})$, clipped to $[-5,5]$.
- **Accruals** (US, XBRL): $(\Delta\text{current assets} - \Delta\text{cash} - \Delta\text{current liabilities}
  + \Delta\text{short-term debt} - \text{depreciation})/\text{average total assets}$; the signal is $-$accruals.
- **Price themes** (replication-study definitions): momentum $\log P_{t-21} - \log P_{t-252}$; short-term reversal
  $-\log(P_t/P_{t-21})$; low risk $-\sigma_{252}$; size $-\log(\text{dollar volume})$; skewness of daily returns;
  seasonality (same-month return one year earlier); illiquidity (Amihud $\overline{|r|/\$vol}$); trend
  ($P_t$ relative to its 252-day mean). All cross-sectionally z-scored within the point-in-time universe.
- **No-fit composite** (pre-registered): $\tfrac14\big(z(\text{value}) + z(\text{SUE}) + z(\text{accruals}) +
  z(\text{momentum})\big)$ — weights fixed forever.
- **Holdings state** (13F, fullest batch per quarter; research/03 §15): institutional share $\ell_i = $ inst. shares
  / shares outstanding; breadth change $\log(n^{\text{holders}}_{q}/n^{\text{holders}}_{q-1})$.
- **Asset embeddings** (Gabaix–Koijen–Richmond–Yogo): SVD of the log manager × stock holdings matrix,
  $\log(1+H) = U S V^\top$, embeddings $V_{:,1:k}$, peer-implied yield = similarity-weighted mean of peers' earnings yields.

## 5. The general-equilibrium prior (Agora)

For an economy with sectors $j\in\{1..45\}$ (OECD ICIO), each sector produces with Leontief intermediates and a
CES value-added nest (KLEM), sells domestically or exports through a CET frontier with elasticity $\omega_t$, and
buyers combine domestic and imported varieties through Armington CES with elasticity $\sigma_q$; households have LES
preferences; the government collects tariffs $\tau_j$ and taxes; the rest of the world is closed by the trade balance.
Equilibrium is the zero of the excess-demand system $F(p;\tau)=0$ (goods, factors, foreign exchange), solved by Newton
with homotopy continuation in the shock, to residual $\le 10^{-15}$ (measured $4.6\times10^{-10}$ max over 20 events).
A dated tariff action is mapped mechanically to $\Delta\tau_j$ (partner trade shares, scope, exemptions), and the
**response** is $\Delta\log PX_j$ (producer prices). Mapped to sector ETFs by an exposure table $E$ ($29\times45$,
rows sum to 1), the prediction for the event day is $E\,\Delta\log PX$; for stocks, through an industry → ISIC map
$E_{\text{ind}}$ ($127$ industries, validated at rank-correlation $0.68$ against real ETFs).

## 6. The deep model

**Encoder.** Per-asset features $f_i$ → Fourier features $\gamma(f) = [\sin(Bf), \cos(Bf)]$ → a Transformer over the
*set* of assets (self-attention with a key-padding mask), permutation-equivariant by construction. Heads:
$\mu_i = \text{anchor}_i + \delta_i$ (belief correction), and the investor state $(a,\varphi)$ from a pooled token,
with $\kappa$ rescaled from the estimator's seed as in §3.

**Clearing layer.** Exact: $\hat p = R_f^{-1}(\mu - \kappa\Sigma s)$ (or the Gauss–Newton solve with leverage caps).

**Loss** (per date, then averaged with self-adaptive per-date weights and gradient-magnitude balancing):
$$\mathcal L = \underbrace{\big\lVert R_f\hat p - (\mu - \kappa\Sigma s)\big\rVert^2}_{\text{law at }\hat p}
+ \lambda_{\text{tail}}\operatorname{LSE}_\tau(\text{residuals})
+ w_{\text{pen}}\lVert\delta\rVert^2 + w_\mu\lVert\mu - \text{anchor}\rVert^2,$$
with residual-adaptive resampling of dates and validation early stopping on the law residual (patience 4);
$w_{r2}$ (a second-moment term) initialised at 0 after it was found to dominate.

**v2 (over the anchors).** Features $X\in\mathbb{R}^{n\times 21}$ per date (themes, anchors, holdings, Agora shock,
market token, missing indicators), target $y_i = \operatorname{rank}(r_i - \bar r)/(n-1) - \tfrac12$; ridge
($\alpha=10$), histogram gradient boosting (depth 3, 300 rounds, early stopping), and an MLP (2 × 64, GELU, dropout
0.2, AdamW, weight decay $10^{-3}$, early-stopped on validation rank-IC, 3 seeds averaged).

## 7. State between filings (nowcasting)

Per asset, institutional share $x_t$ evolves as $x_t = x_{t-1} + b f_t + \eta_t$, $\eta_t\sim\mathcal N(0,q)$, with
flow proxy $f_t$ (Z.1 aggregate drift or trailing return) and loading $b$ by OLS of filing-to-filing changes on
cumulative flows; observations are exact at filing dates ($R\to 0$). The Kalman recursion gives the filtered mean and
variance; validation hides every fourth filing and compares the filter's MAE to carry-forward. (Result on real 13F:
equal — the state between filings is unforecastable from these proxies.)

## 8. Volatility, regimes and intervals (the nulls)

- **EWMA** $\sigma^2_{t+1} = \lambda\sigma^2_t + (1-\lambda)r_t^2$, $\lambda=0.94$.
- **HAR** in levels: $\text{RV}_{t+1} = c + \beta_d \text{RV}_t + \beta_w \overline{\text{RV}}_{t-4:t} + \beta_m
  \overline{\text{RV}}_{t-21:t}$ (the log form is biased in level).
- **GARCH(1,1)-t**: $\sigma^2_t = \omega + \alpha r^2_{t-1} + \beta\sigma^2_{t-1}$, Student-$t$ innovations, MLE.
- Scored by **QLIKE** $\ \log\sigma^2 + r^2/\sigma^2$ and log-variance MSE, rolling, no look-ahead.
- **Regimes**: 2-state Gaussian HMM by EM, *filtered* probabilities only. **Intervals**: rolling split-conformal
  quantile $\hat q = $ the $\lceil (n+1)(1-\alpha)\rceil$-th smallest absolute residual → coverage exact at $1-\alpha$.

## 9. Costs and allocation

Next-bar execution. Turnover measured against **drifted** weights $w^{+} = w(1+r)/(1+w^\top r)$ (long-only and
dollar-neutral), cost = bps × turnover. Baselines: equal weight, inverse volatility, HRP (single-linkage tree,
quasi-diagonalisation, recursive bisection), momentum tilt.

## 10. The honesty stack

- **Rank-IC**: Spearman correlation between the score and the forward demeaned return within the point-in-time
  universe; $t$ over dates; for overlapping horizons Newey–West with $\lceil h/\text{spacing}\rceil - 1$ lags.
- **Deflated Sharpe ratio** (Bailey & López de Prado): with $N$ *distinct procedures* in the scope,
  $$\text{SR}_0 = \sqrt{V[\widehat{\text{SR}}]}\Big((1-\gamma)\,\Phi^{-1}(1-\tfrac1N) + \gamma\,\Phi^{-1}(1-\tfrac1{Ne})\Big),\qquad
  \text{DSR} = \Phi\!\left(\frac{(\widehat{\text{SR}} - \text{SR}_0)\sqrt{T-1}}{\sqrt{1-\gamma_3\widehat{\text{SR}} + \tfrac{\gamma_4-1}{4}\widehat{\text{SR}}^2}}\right),$$
  where $V[\widehat{\text{SR}}]$ is the **leave-one-out robust** (MAD-based) variance of the Sharpe ratios of the other
  procedures in the scope (the candidate's own value never inflates its hurdle), $\gamma$ is Euler–Mascheroni, and
  $N$ counts distinct procedure hashes (rung name + calling code + arguments) — a rerun is not a trial, a code change is.
  Certify at DSR $\ge 0.95$; the strict evaluation count and its DSR are reported alongside.
- **PBO by CSCV**: the $T$ dates are cut into $S=16$ blocks; for each of the $\binom{S}{S/2}$ train/test splits the
  in-sample best rung's out-of-sample rank $\omega\in(0,1)$ among all rungs gives the logit $\lambda=\log\frac{\omega}{1-\omega}$;
  $\text{PBO} = P(\lambda < 0)$ over splits (block moments precomputed). Certify at PBO $< 0.20$ across the family.
- **Stationary bootstrap** (Politis–Romano, geometric block length) for the CI of the excess return over the
  equal-weight point-in-time universe; **permutation null** by shuffling scores across assets within each date.
- **CPCV**: $\binom{6}{2}=15$ splits, test groups purged by the horizon and embargoed by one date; every fitted
  model is scored per date across all splits, then aggregated per market.
- **Point-in-time universe**: top-$K$ by trailing 250-day median dollar volume, listed $\ge 300$ days; the benchmark
  is the equal weight of the *same* universe; feature coverage built for the union of all such universes (§16).
- **Validator validation**: the gate certifies 0/20 pure-noise strategies, has power on a Sharpe-2 edge, the permutation
  $p$ is uniform under the null, bootstrap coverage is nominal, and re-running an unchanged procedure adds no trial.

## 11. The record

Signal file $F_d$ (canonical JSON: strategy text, universe rule, all scores, target weights);
$h_d = \text{SHA256}(F_d)$; chain link $c_d = \text{SHA256}(c_{d-1}\Vert h_d)$ with $c_0 = 0^{64}$; ledger row
(append-only, hash-chained parquet); OpenTimestamps proof of $h_d$ aggregated into a Bitcoin block. Verification:
`ots verify F_d.ots` plus recomputation of the chain from the files. A missing night is a gap in the chain, never
back-filled.

## 12. Where each object lives and how it is tested

| object | module | test |
|---|---|---|
| clearing, four investor types, Gauss–Newton | `models/clearing.py` | Walras with numeraire; scale invariance; closed form = Newton |
| structural SDF, Gaussian normaliser, GMM residuals | `models/kernel.py` | kernel prices the clearing equilibrium (Monte-Carlo) |
| identification, κ estimator, law at $\hat p$ | `models/identification.py` | κ, μ recovered on panels; A, φ not; mispricing preserved OOS |
| Agora bridge, events, sector map, industry map | `structure/` | residual ≤ 5e-10 on 20 events; map validated 0.68 |
| set encoder | `nets/set_encoder.py` | equivariance / invariance |
| residual trainer | `solvers/residual_trainer.py` | structure finds mispricing; training does not destroy it |
| nowcaster | `structure/nowcaster.py` | recovers loading; hold-out vs carry-forward |
| embeddings | `structure/embeddings.py` | recovers latent clusters |
| volatility, regimes, conformal | `baselines/` | persistence recovered; exact coverage; no look-ahead |
| costs, allocation | `evaluate/costs.py`, `baselines/passive.py` | drifted-weight turnover; dollar-neutral books |
| CPCV, DSR, PBO, bootstrap, permutation | `evaluate/` | validator validation (research/03 §10) |
| anchors, NSE anchors | `anchors/` | orientation; no look-ahead; split adjustment; point-in-time filings |
| chain, ledger, OpenTimestamps | `signals/daily.py`, `core/ledger.py` | chain verification; day-1 proofs attested |
