# Phase EU/UK — Europe and the United Kingdom (planned 2026-09-07; not started)

**Why it was not first.** Every source used so far is keyless and point-in-time. For Europe and the UK the
*fundamentals* side is solved (ESEF), the *prices* side is not yet.

## Data
| need | source | status |
|---|---|---|
| Quarterly / half-year fundamentals, point-in-time | **ESEF XBRL via filings.xbrl.org** (JSON API, keyless): every EU-regulated issuer since FY2020, UK issuers too (UKSEF); filing date = knowledge time; iXBRL with IFRS taxonomy → EPS, revenue, profit, shares | to build: `loaders/esef.py` (same parser discipline as NSE §14: contexts, duration = period, consolidated-first) |
| Daily bars, full market, survivorship-aware | no keyless daily feed found. Candidates: Euronext / Deutsche Börse official data (keyed, free tiers), Twelve Data / EODHD (keyed), exchange bulk end-of-day files (Xetra publishes daily; LSE via a licensed vendor) | **blocking item**: pick one keyed source, keep the key out of the repo (GitHub secret, like HF_TOKEN) |
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
