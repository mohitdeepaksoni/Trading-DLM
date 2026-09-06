# P3 step 2 — does the Agora sector shock reach individual stocks? (2026-09-06)

**Question.** The tariff event harness showed that Agora's producer-price response (dlogPX per ICIO sector) ranks the
day-0 residual reaction of 29 sector ETFs (mean rank-IC +0.125, t 2.30, permutation p 0.003, n = 20 events). The v1
plan wanted that shock as a *per-stock* feature. Does it survive the transfer?

**Method.** `emporion/structure/industry_map.py` maps the 127 Nasdaq-screener industries to ICIO-45 sectors (hand
weights, sector fallback; 999/1000 names mapped). `scripts/p3_event_stocks.py` scores, per event, predicted =
exposure · dlogPX against residual returns (to the equal-weight point-in-time top-500 market, beta from 250 days),
rank-IC across stocks, permutation null across stocks, t over events. Diagnostics: stock-built ICIO sector portfolios,
synthetic ETFs (ETF_EXPOSURE @ sector portfolios), goods-only subsets, and the ETF harness without the four
metal/steel ETFs. Map validation (`scripts/validate_industry_map.py`): synthetic-ETF vs real-ETF daily rank
correlation.

**Results (day-0 reaction unless stated; n = 20 events).**

| unit ranked | mean rank-IC | t | perm-p | hit |
|---|---|---|---|---|
| 29 real sector ETFs (harness) | +0.125 | 2.30 | 0.003 | 0.80 |
| 25 real ETFs, metals/steel dropped | +0.126 | 2.49 | 0.005 | 0.85 |
| 29 synthetic ETFs built from stocks | +0.053 | 0.85 | 0.108 | 0.55 |
| 26 stock-built ICIO sector portfolios | −0.003 | −0.05 | 0.54 | 0.55 |
| 17 goods-sector portfolios | −0.040 | −0.59 | 0.76 | 0.45 |
| 500 stocks | +0.018 | 0.58 | 0.030 | 0.45 |
| 200 stocks | +0.013 | 0.38 | 0.20 | 0.40 |
| 413 goods-industry stocks | +0.017 | 0.43 | 0.072 | 0.55 |

Map validation: synthetic ETFs track real ETFs with rank correlation +0.68 over 671 days (+0.74 on the 20 event
days; 99 % of days above 0.3) — the industry map is not the problem.

**Reading.** The structural shock is a *sector-ETF-level* signal on the event day: it is robust to which ETFs are
included, half survives in ETFs rebuilt from stocks, and it vanishes when the unit is a stock or a broad ICIO
sector portfolio. Per-event stock-level ICs swing between −0.30 and +0.29 with no consistent sign. Two readings are
consistent with this: ETF prices react to the sector *label* on the day (flows into and out of the sector wrapper)
more than the underlying names do; and the CGE response is informative for the concentrated goods wrappers (semis,
autos, steel, copper) and not for the many services sectors that make up most of the stock universe.

**Decision.** v1 channel 1 ("Agora shock as a per-stock feature") is **not supported** at n = 20 events. The event
harness stays an ETF-level sector-rotation instrument (its own product surface); it is not fed into the per-stock
model until either (a) more events accumulate (the harness keeps ledgering every dated action) or (b) a holdings-based
exposure matrix (actual ETF constituents) shows the transfer in a way the industry map cannot. Every number above is
on the ledger (scopes `p3-event-stocks-us*`, `p3-event-harness-us-drop-*`); no stock-level trial is pre-registered.
