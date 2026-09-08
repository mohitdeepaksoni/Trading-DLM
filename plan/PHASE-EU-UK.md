# Phase EU/UK — Europe and the United Kingdom (planned 2026-09-07; data built 2026-09-08, first-pass ladder running)

**Why it was not first.** Every source used so far is keyless and point-in-time. For Europe and the UK the
*fundamentals* side is solved (ESEF), the *prices* side is not yet.

## Data
| need | source | status |
|---|---|---|
| Annual / half-year fundamentals, point-in-time | **ESEF XBRL via filings.xbrl.org** (JSON API, keyless): 12,777 filings of 3,040 issuers in GB, FR, NL, SE, DK, FI, IT, ES, BE, AT, PT, NO since FY2020; knowledge_time = date added to the OAM (median 155 d after period end); EPS, revenue, profit, CFO, equity, assets, shares + prior-year comparatives | **built 2026-09-08**: `loaders/esef.py` (rules in research/03 §17, tests `test_esef_facts.py`), `scripts/build_esef_facts.py`, store `runs/eu_uk/store/esef_facts`, slim gzip cache of numeric facts (~70 MB instead of ~70 GB). **DE, IE not on filings.xbrl.org** (Germany needs another source) |
| Daily bars, full market, survivorship-aware | **Yahoo (keyless, unofficial; research only)** through `loaders/yahoo.py`: ESEF issuer → ISINs (GLEIF daily bulk file, `loaders/gleif.py`) → listed line via Yahoo search → yfinance bars since 2016; primary line per issuer = most traded. Twelve Data free plan checked 2026-09-08: **no EU/UK exchange on the free tier** (LSE/Xetra/Euronext need Grow, Nordics/SIX/Milan need Pro); key stored as the `TWELVEDATA_API_KEY` secret for later. Stooq now serves a JS challenge. Yahoo answered from the Brev box (GCP) on 2026-09-08, so refreshes can run there (OCI IPs were 429'd earlier) | **built 2026-09-08**: `scripts/build_euuk_panel.py` → `universe`, `isins`, `bars`; STOXX 600 sector ETFs → `runs/eu_sectors`. A licensed feed is required before any EU/UK file joins the shadow chain |
| Holdings | no 13F equivalent; UK RNS "Holdings in Company" (TR-1) notifications (≥ 3 %) via the FCA/LSE RNS feed; EU major-holding notifications per national regulator | phase 2 |
| Macro / GE prior | Agora ICIO anchors exist for all EU-27 + UK (OECD ICIO 2018), Armington/CET calibrated | ready — events: EU CBAM, UK–EU tariff changes, energy-price shocks |
| Consensus | none keyless | skip |

## Order of work (each an ledgered rung; same honesty stack)
1. ESEF loader + point-in-time filings store for the union of PIT universes (once prices exist).
2. Prices: one keyed source, bars store, PIT universe (top-K by trailing € / £ volume, listed ≥ 300 d).
3. P0 baselines (EW / inverse-vol / HRP after costs; costs 15 bp EU, 20 bp UK incl. stamp duty 0.5 % on purchases).
4. Anchor ladder: value, SUE (half-yearly for many EU issuers — seasonal difference over 2 periods), price themes.
5. Agora events for EU/UK (CBAM phases, UK tariff schedule changes) on sector ETFs (EU sector ETFs: STOXX 600 sector
   ETFs) — day-0 harness as for the US.
6. Nothing goes live until it passes the gate or is pre-registered by the same rule.

## Cost and time
CPU only. ESEF loader ≈ 1 day; price source integration ≈ ½ day once chosen; P0 + ladder ≈ 1 day of runs.
