# Phases by market — India first, then Europe/UK, then the US, then crypto; one engine per market, combined last

*2026-09-08. Decision (user): concentrate on one market at a time, in the order the data and the evidence support, and
build each market as its own self-contained engine ("DL" = data → anchors → gates → chain file) on the shared honesty
stack. Combine later. This replaces the "all markets at once" ordering of plan/PLAN.md P3–P7 without changing any gate.*

## 0. Why this order (evidence, not preference)
| market | point-in-time data today | strongest anchor so far | live | verdict |
|---|---|---|---|---|
| **India / NSE** | full cash market bars 2022→, quarterly Ind-AS results 2018→, shareholding 2018→ (SEBI format 2021→), FII/DII flows; all keyless | **earnings surprise +0.036 (t 4.1, hit 0.76) at 60 d, SR 1.24** — fails only beats-EW | 1 file (SUE, since 2026-09-08) | **first**: strongest candidate, quarterly cadence, home market, execution path (Kite + SEBI Algo-ID) |
| **Europe / UK** | ESEF annual facts FY2020→ for 12 countries (DE/IE missing), Yahoo research bars 2016→ for 2,018 issuers | value +0.080 (t 2.9) at 60 d on 52 dates — fails PBO | none; needs a licensed feed | **second**: data exists but annual + lagged; price feed and Germany are money/sourcing questions |
| **US** | bars, XBRL, 13F, Z.1, consensus snapshots | accruals +0.023 (t 2.6); composite +0.033 (t 2.1) | 2 files since 2026-09-06 | **third**: 20 rungs, PBO 0.23, "no more US rungs" — let the chain speak; revisit with measured beliefs (consensus history) |
| **Crypto** | Coinbase 402 pairs 2015→, CoinGecko supply | rankable by risk, nothing tradeable after costs | research-track file | **fourth**: no fundamentals anchor → mispricing not identified (research/09) until a supply/flow state exists |

The deep residual model stays closed (research/12, findings §6): "an engine per market" means the anchor library, the
market's own state variables and the gates — not a neural net per market. A fitted combiner lost to 1/N three times;
the combination step below is therefore at the *portfolio* level first, and at the model level only if measured
beliefs (consensus revisions, now accumulating) ever identify something prices alone did not.

## 1. Phase IN — India engine (now → ~3 weeks of build, then the 7-clean-nights and paper-trading clocks)
**Task 0 — frozen-clock replay (user, 2026-09-08: "use old data and check our prediction with the latest"; ~1 day).**
Run the exact nightly builder (`emporion/signals/daily.py`, the code the chain executes) with the clock frozen at every
month-end from 2023-01 on, reading the bitemporal store `as_of` that date only, emit the files, then score each at 20 and
60 sessions against the equal-weight universe under the Indian cost model; ledger scope `replay-nse` marked *rehearsal*.
What it proves: research ↔ live parity (RT1) and the point-in-time machinery end to end — the replay must reproduce the
research rank-ICs; a gap is a leak. What it does not prove: the signal itself — SUE was chosen after seeing 2022–2026, so
these years stay in-sample at the level of the choice; only the live chain is selection-free evidence. Report the
distribution (months beating EW, worst run, drawdown), not the mean. **Once bars reach 2016, hold 2016–2019 out
entirely** — no one in this project has looked at NSE prices before 2022 — and replay there: the closest thing to true
out-of-sample available without waiting. Same replay later for US (2017→) and EU/UK.
**Data (all NSE, keyless, point-in-time):**
0. (done 2026-09-08) bars 2016→ (legacy archive; NSE blocks bulk downloads from cloud IPs — fetched from the Mac), fundamentals
   for 387/417 of the 2016–2026 point-in-time union (30 names have no filings on the API). On this sample **SUE falls to
   t 1.3 / 1.1 (75 dates)** from t 3.1 / 4.1 on 38 dates since 2022 — the longer sample did what a holdout should; the chain file
   stays and the live record decides.
1. **Extend bars to 2016** (bhavcopy archives, old `cm…bhav.csv.zip` format before the 2024 UDiFF change; the loader
   already reads the file — only the URL/format switch is new). Today 2022→ gives ~45 monthly decision dates; 2016→
   gives ~120. This is the single largest gain available and costs nothing. Exit: `runs/nse/store/bars` 2016→, ladder
   rerun, every NSE number re-quoted on the longer sample under a new scope version (`p4-realdata-nse-v2`).
2. **Delivery percentage** (in the bhavcopy since 2011: `DELIV_PER`) → a liquidity/conviction state no other market has.
3. **Bulk and block deals**, **insider-trading (PIT) and SAST disclosures**, **promoter pledges** — daily, keyless,
   timestamped by NSE broadcast → demand-pressure and insider-belief anchors with a knowledge time.
4. **AMFI monthly mutual-fund portfolios** → domestic-institution holdings per stock at monthly cadence (13F was quarterly
   with a 45-day lag); FII/DII flows daily (already snapshotting).
5. **Cost model**: STT 0.1 % delivery, stamp 0.015 %, exchange + SEBI charges, brokerage; slippage from delivery data.
   Replace the 15 bp assumption everywhere in the NSE ladder; the ledger scope changes with it.
6. **Corporate announcements** (NSE `corporate-announcements` feed, keyless, broadcast-timestamped): deterministic event
   CATEGORIES (results date, order win, fund raise, board meeting, pledge, rating action, …) as state variables — never text
   sentiment on history (research/06 quarantine: no language-model labels on the past; live labelling only after the stamp).
   This is the project's answer to "does the engine know about news": policy news → Agora shocks; analyst beliefs → consensus
   snapshots; company news → timestamped categories. No sentiment model (research/02: none survives honest evaluation).
7. **Corporate actions**: splits/bonuses per symbol (already handled via shareholding share counts; verify against the
   NSE corporate-actions endpoint) — the open item from research/03 §14.
**Anchors and rungs** (each a ledgered rung, no more than one new procedure per data source): SUE (pre-registered — the
chain decides, nothing changes), value, accruals (quarterly!), earnings growth, holdings change (SEBI format), MF-holdings
change (monthly), delivery-percentage change, insider net buying, pledge change, price themes, no-fit composite.
**Structure**: Agora India events — budget tariff changes, RBI actions, export bans/quotas — on NSE sectoral indices
(day-0 harness as for the US; `emporion/structure/events_in.py`, `sector_map_in.py`).
**Trader path**: 7 clean nights → NSE file is "live" on the chain (already 1); 3 months of paper via Kite paper/sandbox with
implementation shortfall measured; micro-live only after certification (research/05 staged path, kill-switch rules).
**Exit criteria**: replay reproduces the research numbers through the live path and the 2016–2019 holdout is reported; bars 2016→; ≥ 6 India-specific rungs on the ledger; cost model in every number; India event harness
with ≥ 10 dated events; the daily NSE file unchanged and green for 30 nights; a paper-trading log with shortfall.

## 2. Phase EU — Europe/UK engine (after IN; gated on two purchases)
Gate: (a) a licensed daily feed (Twelve Data Grow covers UK/DE/FR/NL; Pro adds Nordics/SIX/Milan) — without it nothing
can be pre-registered; (b) a German fundamentals source (ESEF mirror lacks DE). Until then EU stays research-only on
Yahoo bars. Work when gated open: half-year cadence handling, the accruals sign question (findings §7b), dividend-adjusted
returns, time-varying FX, UK stamp in the cost model (done), STOXX events re-checked against CRS/Official Journal;
then the same rung-by-rung ladder and the 7-clean-nights rule for one EU file (value is the candidate).

## 3. Phase US — enhanced US engine (after EU; the chain runs meanwhile)
Two files already live; no new rungs until (a) the consensus-revision history is long enough to test *measured beliefs*
against the price-derived anchor (the one thing findings §8 says could reopen the model), (b) six months of live prints
exist. Enhancement candidates, pre-declared here so they cost their trial honestly: consensus revision anchor; breadth of
ownership (holders change, on the watch list); event rotation with stamped events (≥ 20).

## 4. Phase CR — crypto engine (after US)
Classical until a state exists: supply growth and exchange-flow proxies from snapshots (accumulating daily since 2026-09-06);
the low-risk research file keeps running. Exit into "live" only with a fundamentals-like anchor that identifies mispricing.

## 5. Phase ALL — combination (last)
1. **Portfolio level first**: one book per market engine, combined by risk parity / HRP across books with a per-market
   conformal-Kelly cap (plan/PLAN.md P5), FX-hedged or not as a declared choice; the combined book is its own chain file
   with its own pre-registration. This needs no fitting and cannot leak.
2. **Model level only if identified**: a pooled model over anchors is retried only when measured beliefs exist in ≥ 2
   markets; exit line pre-declared as in research/12 (beat the composite by ≥ 0.01 rank-IC at t > 2 on pooled data).
3. **Shared structure**: Agora is already one model for every economy — the event harness becomes multi-market by
   construction (US ETFs, STOXX ETFs, NSE sector indices) and is scored on one ledger scope.

## 6. Background data accumulation — runs for every market regardless of phase (user, 2026-09-08: "data collection is hard and slow; keep gathering")
A point-in-time record exists only if it was collected while it was current: snapshots cannot be back-filled later. So
collection is decoupled from the phase order and runs unattended; the engines are built on top when their phase comes.
| source | market | cadence | where | status |
|---|---|---|---|---|
| Nasdaq bars, SEC XBRL facts, consensus snapshots (Nasdaq), CoinGecko supply, NSE FII/DII flows | US, crypto, IN | nightly | GitHub Actions shadow job (`scripts/shadow_daily.py`) | running since 2026-09-06 |
| NSE bhavcopy, results + shareholding refresh | IN | nightly / weekly (Sunday) | shadow job | running (bars 2022→; extend to 2016 once, then nightly) |
| SEC 13F structured windows (new quarterly dataset) | US | quarterly | box, `scripts/build_sec_13f.py` (§15 rule) | manual; to schedule |
| ESEF/UKSEF index + new filings → facts (incremental; slim cache) | EU/UK | weekly | box or Actions (keyless) | one-off done 2026-09-08; incremental mode to add to `build_esef_facts.py` |
| Yahoo bars for the EU/UK universe + STOXX sector ETFs (incremental) | EU/UK | weekly | box (Yahoo answers there; Actions runners untested) | to add: `build_euuk_panel.py --skip-map --since` |
| GLEIF ISIN-LEI bulk file (universe drift: new listings, delistings) | EU/UK | monthly | box | file cached; to schedule |
| Coinbase candles (all USD pairs) | crypto | nightly | shadow job | running |
| NSE delivery %, bulk/block deals, insider (PIT/SAST) disclosures, pledges, AMFI portfolios | IN | daily / monthly | box or Actions | **to build first in Phase IN** — these cannot be back-filled beyond what NSE/AMFI archive |
| Agora event tables (US tariffs, EU trade, India policy) | all | as announced | repo (deterministic tables) | US 20 events, EU 9; India to write |
Mechanics: one `scripts/data_refresh.py --sources …` entry point with per-source incremental logic and a manifest of what
was fetched (row counts, hashes, knowledge times) appended to `runs/_manifest/`; a second GitHub Actions workflow
(`ops/data-refresh.yml`, weekly, Sunday 02:00 UTC) for the keyless sources, the box for anything that needs Yahoo or >90 min;
state pushed to the private HF dataset like the shadow state (`shadow_sync.py` gains a `--paths` option); backup tarballs
monthly. Everything collected is unused until its phase — and nothing is ever re-derived from today's snapshot.

## 7. Rules that carry over unchanged
Point-in-time universes and feature coverage (research/03 §11, §16); revisions resolved per decision date (§12);
procedures counted, not evaluations (§13); every number from a ledger row; pre-registration before the next price
exists; the four-level mathematics validation protocol (plan/PHASE-NSE-CRYPTO.md §1); secular naming; compute on the
box, data state on the private HF dataset, code on GitHub; tokens pasted in chat get rotated.
