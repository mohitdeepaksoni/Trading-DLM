# 05 — Practical guide: from a model to an account that grows

*The "financial freedom" question, answered without slogans. Financial freedom from trading is a compounding problem with three inputs — edge, risk control, and time — and one dominant failure mode: blowing up before the edge has time to compound. Everything below is about protecting the third input.*

---

## 1. The arithmetic of freedom (so the target is a number, not a feeling)
```
Wealth_T = Wealth_0 · exp( g · T ),   g = r_f + f(μ−r_f) − ½ f²σ²      (Kelly growth, doc 01 §4.5)
```
- A *certified* Sharpe 0.8 strategy at 12 % vol earns ≈ 9–10 % excess before tax; with half-Kelly leverage ≈ 1.0–1.3× that is 10–14 % a year. Compounded, capital doubles every 5–7 years. **That** is what a real, honest edge does. Anything promising 5 % a month is either leaking, levered into ruin, or lying.
- Freedom number = annual expenses ÷ safe withdrawal rate (3–4 %) ⇒ ₹1 cr of annual spend needs ₹25–33 cr; $60 k needs $1.5–2 M. Work backwards: with 12 %/yr and monthly contributions you can compute the year you cross it; with 40 %/yr *and a 60 % drawdown* you never do.
- Public "financial freedom" advice (Musk-style or otherwise) reduces to three practical statements: own productive assets, don't depend on a single income, avoid leverage you can't survive. Systematic trading is a way to add an *uncorrelated* income stream on top of index ownership — not a replacement for it.

## 2. Staged path (each stage has an exit criterion; do not skip)
| Stage | What | Capital | Exit criterion |
|---|---|---|---|
| S0 Research | Build Emporion, ledger, CPCV; publish benchmark | 0 | doc 03 checklist all green on ≥ 2 markets |
| S1 Paper | Live signals, simulated fills with real quotes, 3–6 months | 0 | implementation shortfall inside bootstrap CI |
| S2 Micro-live | Real orders, tiny size (≤ 2 % of intended) | ₹2–5 L / $5–10 k | live Sharpe within CI; no operational incidents for 3 months |
| S3 Scale | Increase to target size in 3 steps; capacity check via √-impact | as planned | drawdown < 1.5× backtest MDD |
| S4 Steady state | Monthly re-certification; quarterly retrain; ledger reviewed | — | kill-switch rules never breached |

**Kill-switch rules (hard-coded, not discretionary):** stop trading if drawdown > 1.5× backtest MDD, or 20 consecutive trading days of live shortfall outside CI, or any data-feed anomaly. Restart only after a full re-certification.

## 3. Risk rules that keep you alive
- Max portfolio vol target 10–15 % annualised; scale positions to it daily (vol targeting halves MDD in most studies).
- Max single-name 5 %, single-sector 20 %, gross leverage ≤ 1.5× (retail), no naked options.
- Regime gate: when HMM P(high-vol) > 0.7, halve exposure; when residual gate says "no opinion", be flat — abstaining is a position.
- Never override the system by hand; log every manual intervention as a ledger row (it will show you the cost of discretion).

## 4. Markets and brokers (any market ⇒ per-market plug-in)
**India (NSE/BSE)**
- Data: NSE bhavcopy (free daily OHLCV + deliverables), NSE corporate announcements, BSE filings; point-in-time fundamentals need a paid source (CMIE Prowess, Ace Equity) or careful scraping with publish timestamps.
- Brokers with APIs: Zerodha Kite Connect, Upstox, Angel One SmartAPI, Dhan, Fyers, 5paisa.
- **Regulation (SEBI circular SEBI/HO/MIRSD/MIRSD-PoD/P/2025/0000013, Feb 2025; enforced from 1 April 2026):** every algorithmic order needs an exchange-issued **Algo-ID / Strategy-ID**; API access requires **static IP**, 2FA and OAuth; strategies above the order-rate threshold must be registered through the broker; algo vendors must be exchange-empanelled; brokers are responsible for compliance. Retail algo trading remains legal — you register your strategy with your broker. Plan the registration step into S2.
- Taxes: STT on delivery and intraday, STCG 20 % (< 12 months), LTCG 12.5 % above ₹1.25 L; F&O is business income. Low turnover is a tax strategy.
**US**
- Data: Polygon, Tiingo, Alpaca (free tier), Nasdaq Data Link; point-in-time fundamentals via Compustat/CRSP (academic) or paid vendors.
- Brokers: Alpaca (API-first, paper trading), Interactive Brokers (global, futures/options), Tradier.
- Pattern-day-trader rule ($25 k minimum for > 3 day-trades/5 days); wash-sale rule.
**UK/EU**: IBKR, Trading 212 (limited API), stamp duty 0.5 % on UK shares (avoid high turnover), ISA wrapper for tax-free gains (but no margin).
**Crypto**: Binance/Coinbase/Kraken APIs; 24×7, costs 10–50 bps, liquid; good for *testing execution plumbing*; treat as its own asset class in the registry.

## 5. Data hygiene checklist (where most retail systems silently die)
- Adjusted vs unadjusted prices: train on *unadjusted as-of* with a corporate-actions table; adjust in the feature function, not in the source.
- Survivorship: keep delisted names; build the universe as of each date.
- Timestamps in exchange local time with explicit session calendars; India has muhurat sessions and holiday quirks.
- Publish lags: earnings, index rebalances, macro releases — encode in the feature registry.

## 6. Infrastructure (cheap and adequate)
- Research/training: the existing A10 box or a Brev H100/H200 for the residual-training runs (Agora-sized: hours, not days). Inference is CPU-feasible daily.
- Live: one small always-on VM (static IP for SEBI), scheduler, broker adapter, Postgres/Parquet ledger, alerting. Same "deployment-consistent" principle as FinRL-X: the backtester and the live loop call the same `strategy.step()`.
- Secrets: broker keys in a vault, never in the repo; rotate on every publish.

## 7. What to expect, truthfully
- Most of the first year is data and evaluation work, not modelling. That is normal and it is where the moat is.
- The likeliest honest outcome of any *single* idea is "not certifiable". The engine's value is that it lets you test many ideas *without* fooling yourself, and that the equilibrium prior (doc 04) is a fresh source of hypotheses.
- If Emporion certifies at Sharpe ≈ 0.8 on a diversified universe, that is a genuinely rare, publishable, and monetisable result. Run it small, let it compound, keep the day job until the account, not the backtest, says otherwise.
