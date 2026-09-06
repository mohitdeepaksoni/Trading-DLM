# P4 v1 — where the deep model must earn its place (plan, 2026-09-06 evening)

## What v0 established (all ledgered)
1. Mechanics are sound: with an informative anchor the structural clearing model finds mispricing without learning
   (0.95 on synthetic); the law imposed at the equilibrium price refines without harm; identities and validators hold.
2. On real US large caps the learned corrections add nothing to the anchors — 38 or 158 training cross-sections cannot
   teach beliefs beyond a value anchor; six hand features carry what they carry (composite of published anchors: rank-IC
   +0.033, t 2.1; accruals t 2.6; nothing certified under the point-in-time rule).
3. Holdings *levels and flows* are not signals on US large caps; holdings are state (φ) and inversion material.
4. The market prices Agora's sector price response on the day (p = 0.004) — the one channel no anchor contains.

## Therefore v1 changes the question from "learn beliefs from six features" to "price what the anchors cannot"
| channel | why it is not in any anchor | how it enters v1 | data status |
|---|---|---|---|
| **Agora structural shock** (dlogPX per sector per event) — **TESTED 2026-09-06: does NOT transfer to stocks (research/11); ETF-level only** | general-equilibrium second-round effects | per-stock feature via industry map (Nasdaq screener industry → ISIC), zero between events; net learns the pass-through and decay | US: ready (screener + harness); NSE: Agora IND anchor + NSE industry field |
| **Holdings-implied demand pressure with the nowcaster** | who *must* buy (index share, flows) between filings | φ_i,t and its filtered change as state inputs; A = (1−φ)/κ per date | US 13F + Z.1 ready; NSE shareholding + FII/DII loading; nowcaster built |
| **Asset embeddings** (demand-system inversion) | latent characteristics investors trade on | encoder inputs (k = 8) replacing sector one-hots; similarity as attention bias | US matrices ready; NSE next |
| **Pooled training** | one law, many markets: 200 US + 200 NSE + 100 crypto assets × daily dates | one model, market embedding token; synthetic pre-training on `sample_panels` | US/crypto ready; NSE pending |
| **Measured beliefs** | analyst expectations, not price-derived | consensus revisions as they accumulate (snapshot history from 2026-09-06) | accumulating |

## Protocol (unchanged, enforced)
Point-in-time universe; revisions resolved at the decision date; law at p̂; κ seeded from the panel estimator; penalised
corrections with validation selection; every configuration a ledger trial in one scope per market; CPCV/DSR/PBO/bootstrap;
the pre-registered accruals rung continues on the shadow path untouched.

## Exit criterion for v1
On the held-out test years, −r_mkt from v1 must beat the no-fit composite (rank-IC ≥ +0.045 at 20 days on US, i.e.
+0.01 over the composite with t > 2) on the same ledger, or v1 is not the product — the composite plus the structural
event harness is.
