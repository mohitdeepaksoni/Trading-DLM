# Emporion — what was tried, what survived, what did not (findings as of 2026-09-08)

Emporion is a deep-learning trading engine built in the open with one rule: every number that could recommend a trade
must survive a fixed honesty stack, and every trial — including the failures — stays on an append-only ledger. This
document is the record. The negative results are most of it, and they are the product's warranty.

## 1. The honesty stack (what every number below has passed or failed)
Point-in-time universes (top-K by trailing dollar volume, listed ≥ 300 days; research/03 §11); revisions of every
fundamental resolved at the decision date, not today (§12); combinatorial purged cross-validation; deflated Sharpe with
the trial count taken from distinct procedures on the ledger (§13); probability of backtest overfitting across the whole
rung ladder; stationary bootstrap and permutation nulls; costs inside returns; and pre-registration — a signal enters
the live record only as a file whose hash is chained and stamped into Bitcoin before the next price exists. Three
data defects were found by sanity checks on levels, not by statistics (§14 NSE XBRL contexts, §14 NSE percentages as
fractions, §15 SEC 13F late batches); each has a written rule and a regression test.

## 2. Markets and data (all keyless, all point-in-time)
US: Nasdaq bars (top-1000 pool), SEC XBRL company facts, SEC 13F aggregates (98 windows), Fed Z.1, consensus snapshots
(daily since 2026-09-06). India: NSE bhavcopy 2022→, quarterly Ind-AS results 2018→2026 from the legacy and the
Integrated-Filing endpoints, shareholding patterns Sep-2021→. Crypto: Coinbase daily candles (402 pairs), CoinGecko
supply. A survivorship-free quarterly US panel from 13F-implied prices measured the delisting bias on momentum-type
anchors at ≈ 0.00–0.02 rank-IC (small). Everything is backed up in a private dataset; the shadow-chain state lives there.

## 3. Baselines and nulls
| question | answer | scope |
|---|---|---|
| Does anything beat equal weight after costs? | Only EW itself certifies on US; HRP / inverse-vol lose on NSE after costs | `p0-*` |
| Which volatility model? | GARCH-t on US and NSE, EWMA on crypto (QLIKE); HAR must be in level form | `p1-*` |
| Is regime gating an edge? | No: stress regimes carried the rebounds; conformal bands are exact at 90 % | `p2-*` |
| Kronos zero-shot as an encoder? | Rejected: rank-IC ≤ random on three markets; bigger model worse | `p2-kronos` |

## 4. Anchors on real data (what actually ranks stocks)
US, point-in-time top-200, monthly dates 2017–2026, honest count (scope `p4-realdata-us-stocks-pit2`, 36 procedures):
- **Accruals** rank-IC +0.019/+0.023 (t 2.05/2.59), tilt SR 0.77, DSR 1.00, beats EW with CI above zero — fails
  only PBO across the ladder (0.21–0.23 vs 0.20). **Pre-registered** on the live chain 2026-09-06.
- **No-fit composite** (equal z-weights value/SUE/accruals/momentum, fixed forever) +0.032 (t 1.8–2.1), SR 0.81,
  DSR 0.94–0.97 — same single failure. **Pre-registered** 2026-09-06.
- Value, momentum t ≈ 1–1.5; size, illiquidity, low-risk ≈ 0 once the universe is point-in-time (they had "certified"
  with t 4.8/6.9 under a current-cap universe — the leak the gate cannot see, §11).
NSE, point-in-time top-200. **Superseded 2026-09-08** by the run on bars 2016 → 2026 with fundamentals for 387 of the 417
names ever in the universe (scope `p4-realdata-nse-N2512`, 75 SUE dates 2020 → 2026): **earnings surprise +0.016 (t 1.3) at
20 days, +0.014 (t 1.1, hit 0.69) at 60 days**; value +0.020 (t 1.0); low-risk +0.046 (t 2.2) and momentum +0.033 (t 1.9) at
60 days; clearing with SUE beliefs +0.033 (t 2.2, 52 dates). The earlier 38-date sample since 2022 had read +0.035 (t 3.1) /
+0.036 (t 4.1) — the longer sample removed most of it, and an intermediate run with 175/417 names covered read t 0.4 / 1.8:
coverage moves the number by a t-unit either way (§16). Nothing certified. SUE was pre-registered on the chain on 2026-09-07 on
the old evidence; the file stays and the live record decides. A frozen-clock replay of the live builder over 73 month-ends
reproduces the ladder (60-session excess over equal weight +0.69 %/month, t 2.1; 2022 → 2026 alone +0.54 %, t 1.6): the
pipeline is honest, the signal is modest. **Delivery percentage** (NSE MTO files, 2016 →), the first India-only state: trend over 20 vs 230 sessions ranks stocks at 60 days (+0.030, t 3.0, hit 0.61, 109 dates) but its long-only book does not beat equal weight — watch list, to be tried as a state inside the clearing model, not as a book.
Crypto (14 pairs since 2020): rankable by risk, nothing tradeable after costs.

## 5. Structure: the general-equilibrium prior (Agora)
Agora's exact open-economy solver, shocked with 20 dated US tariff actions (2025-02 → 2026-05), produces a producer-price
response per ISIC sector. It **ranks the day-0 reaction of 29 sector ETFs**: mean rank-IC +0.125, t 2.30, permutation
p 0.003, hit 0.80; unchanged without the metal ETFs (+0.126, t 2.49). It fades by day 5 and is gone by day 20; the
next-day (tradeable) window is not significant. It **does not transfer to stocks**: synthetic ETFs rebuilt from stocks
through a validated industry map carry half (+0.053, t 0.85); 500 individual stocks carry nothing (+0.018, t 0.6)
(research/11). Product surface: `event_rotation.py` — long/short quartile sector-ETF book per dated action, day-0 only;
calibration tally +0.37 %/event (t 1.43, 15/20 positive), short leg carrying it; live use under a stamp-before-the-open
rule (plan/EVENT-PREREG-RULE.md). Nothing live yet.

## 6. The deep model: mechanics sound, identification not there
Emporion v0 (set encoder → beliefs μ and investor state; exact CARA clearing; structural SDF; law imposed at the
equilibrium price) recovers mispricing on synthetic markets with an informative anchor (0.95 out of sample) and passes
its identification tests (research/09: prices identify μ and κ = (1−φ)/A only, up to the anchor). On real data it
**equals its anchor to the third decimal** in every configuration tried: US walk-forward (38 and 158 dates), CPCV over
430+ dates at h = 20 and 60, pooled US + NSE (145 dates), with and without 300 steps of synthetic pre-training
(scopes `p4-v0-us`, `p4-v1-pooled`). The pre-declared exit criterion (US rank-IC ≥ +0.045, beating the composite) was
not met. **Decision (research/12): the residual-net route is closed at this data scale.** The clearing correction
−κΣs is inert on US large caps under every supply definition tried (total shares, non-institutional float,
institutional block — the latter two re-run on corrected 13F levels under scope `pit3`, see §7). A Kalman nowcast of
per-stock institutional share between filings is no better than carrying the last filing forward (MAE 0.0296 either
way; P3c closed without a rung).

### 6b. Emporion v2 — a network over the anchors (2026-09-07)
Ridge, gradient boosting and a 3-seed MLP over 21 features (price themes, value, SUE, accruals, 13F breadth and level,
Agora sector shock, market token), within-date rank target, pooled US top-500 + NSE top-200, CPCV 6×2, judged on
identical splits against the no-fit composite (scope `p5-anchor-net`). **US: every fitted model loses to equal
weights** at every horizon and spacing (h20/21d: +0.016–0.020 vs +0.030; h20/5d: +0.003–0.008 vs +0.025; h60:
+0.007–0.022 vs +0.039). **NSE: an apparent win (+0.08–0.09, t 3–4) was a coverage leak** — fundamentals had been
fetched for today's top-200 only, so the missing-value indicator encoded "fell out of the index" (research/03 §16);
on full point-in-time coverage the fitted models fall to +0.02–0.03 while the composite rises to +0.045 (t 2.2).
Exit line not met. Closed, like v0 and v1: with ~100–600 cross-sections, fitted weights over these anchors do not
beat 1/N weights — the DeMiguel–Garlappi–Uppal result in the cross-section.

## 7. Holdings (corrected 2026-09-07)
Rerun on corrected 13F aggregates (scope `p4-realdata-us-stocks-pit3`, 103 monthly dates, 20 rungs, PBO 0.235):
- Institutional **level** +0.005 (t 0.5) and **change** +0.001 (t 0.1): nothing — now genuinely nothing, rather than
  nothing on garbage. Holdings are state, not a signal, on US large caps (as the v1 plan assumed).
- **Holders change** (quarter-on-quarter log change in the number of 13F holders, i.e. breadth of ownership):
  +0.025 (t 1.5) at 20 days, **+0.036 (t 2.23, hit 0.62) at 60 days**, tilt SR 0.53. The one holdings rung with a
  pulse; it is a known effect (breadth of ownership), it was on the ladder from the start so it costs no new trial,
  and it is NOT pre-registered — it goes on the watch list with NSE earnings surprise, nothing more.
- Supply-calibrated clearing (non-institutional float / institutional block) +0.0168 (t 1.0) at 20 days — identical
  to the corrupted run to four decimals, which is itself the finding: the clearing correction is too small to reorder
  US large caps whatever supply it is given; the value anchor sets the ranks.
- **Embedding-peer-value, re-quoted 2026-09-08 under the §15 rule for the matrices** (one window per quarter — the fullest;
  concatenating windows had added 45 % late/duplicate rows): −0.015 (t −1.2) at 20 days, −0.014 (t −1.1) at 60 days, not
  certified. Peer-implied value from 13F co-ownership carries nothing on US large caps. The caveat is closed as a negative.
Accruals (+0.0187, t 2.05; DSR 1.00 under the fresh count) and the composite (+0.0317, t 1.84; DSR 0.948) are unchanged
— they never used holdings.

## 7b. Europe and the UK — first pass (2026-09-08)
Data built in one day, all keyless: ESEF/UKSEF facts for 12 countries (12,486 filings, 2,972 issuers, FY2020→; four parser
rules in research/03 §17; Germany and Ireland are not on filings.xbrl.org), and Yahoo daily bars for 2,018 of the 3,040
issuers (2,805 lines, 5.7 M bars since 2016; a research source — no EU/UK file can join the chain until a licensed feed
exists; the Twelve Data free plan covers no European exchange). Ladder `p4-realdata-euuk` (PIT top-200 by EUR traded
value, 15/20 bp + 50 bp UK stamp, static FX, dividend-unadjusted, fundamentals from 2021 so 52 monthly dates):
- **Value** (annual EPS in the price currency, pence/pounds resolved per line): rank-IC **+0.063 (t 2.4)** at 20 d,
  **+0.080 (t 2.9, hit 0.77)** at 60 d, tilt SR 0.64, beats EW with a CI above zero (+0.5..+5.5 %/yr), fails only PBO
  (0.316 across the 10-rung ladder). Same shape as US accruals on 2026-09-06. Not pre-registered; 52 dates.
- **Accruals** flips sign (−0.052, t −2.6 at 60 d) — the US winner is a loser here with annual statements and a
  4–10-month lag; not a result, a question for the next pass. Earnings growth +0.02 (t 1.3); annual SUE too thin (16 dates).
- Momentum +0.041 (t 2.4) at 60 d; size, illiquidity, low-risk ≈ 0 or negative; composite +0.033 (t 2.0).
- P0: EW SR 0.46 (mdd −54 %), momentum-top 0.68; nothing certified.
Agora on 9 European trade events (STOXX supersector ETFs, per-economy solves): no day-0 effect (+0.05, t 0.6); the 5-day
reaction window +0.108 (t 1.1, permutation p 0.09, 9 events) — not evidence either way. Scope `p3-event-harness-eu`.

## 8. What the product is today
1. Two pre-registered US signals on a live, Bitcoin-stamped chain (day 1 = 2026-09-06), running nightly on GitHub
   Actions with no server; a trader-facing page (`REPORT.md/html`) rebuilt every night from the chain alone.
2. NSE earnings surprise under watch (joins the chain only after seven clean nights).
3. The tariff event harness as a same-day sector-ETF rotation instrument, pre-registration rule written.
4. The honesty stack itself — the part that is unusual: procedure-counted deflation, point-in-time everything, and a
   record anyone can verify with `ots verify`.
No deep model is in the product. That is a finding, not a gap: with these data the law is satisfied without moving
beliefs, and the only thing that would change it is measured beliefs (consensus revisions now accumulating daily) or
far more identifying variation than prices provide.

## 9. What would change the conclusions
- Six to twelve months of live prints (the only evidence PBO cannot deflate).
- Twenty stamped live events for the harness.
- A consensus-revision history long enough to test measured beliefs against the price-derived anchor.
- Actual ETF constituents for the sector map (the industry map is validated at rank-corr 0.7, not 1.0).
