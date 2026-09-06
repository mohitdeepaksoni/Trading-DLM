# 06 — Real-time data and daily training: review of the proposal, and the DL enhancements

*2026-09-05. A collaborator proposed a streaming/retraining design for Emporion (two clocks, bitemporal store, replay/live feature parity, fast/slow training, residual-as-regime-detector, Agora in the loop, promotion gates, RT1–RT4). This document records the review verdict section by section, then the enhancements at the deep-learning level that the proposal does not yet contain. The proposal is adopted with the changes marked ✚.*

---

## 0. Verdict in one paragraph
Adopt it. The central insight — **certify the update procedure, not each nightly instance** — is the only way a daily-retrained model can coexist with a Deflated-Sharpe ledger, and it is correct. Four things need fixing before RT1: (1) the event classifier has a *temporal* leak the proposal does not name (an LLM replayed over history knows the future); (2) "residual" is doing two jobs — self-consistency of the model and mispricing of the market — and they must be separated or the abstention gate will suppress exactly the signal it is meant to certify; (3) the nightly fast layer should be a *re-solve*, not a gradient refit, most nights; (4) the drift budget needs a definition, given below. Everything else stands.

---

## 1. Two clocks and the bitemporal store — ✅ adopt, ✚ two additions

**Agreed:** market clock (daily bars suffice for a day-to-week horizon) and policy/macro clock (sparse, irregular, where Agora earns its keep) must not share a pipeline; every row carries event time and knowledge time; `as_of` is mandatory at the query API. Parquet + DuckDB with an enforced `as_of` parameter is enough at this scale; TimescaleDB if you want SQL-native time travel.

✚ **Third clock — the corporate/fundamental clock.** Earnings, restatements, filings, index rebalances, splits, dividends, delistings. It is the historical source of most leaks (restated fundamentals, today's index membership, back-adjusted prices). It has its own publish lags and *revisions*, so it is the clock that most needs bitemporality. Treat it as a separate stream with a corporate-actions table applied inside feature functions, never in the source.

✚ **Fourth clock — the execution clock.** Orders, fills, positions, cash, margin. This is state the model's own decisions create; it feeds the compounding evaluation (doc 03 §7) and the implementation-shortfall monitor. It is written by the broker adapter only, never by the model.

✚ **The LLM event classifier has a temporal leak.** The proposal correctly says an unaudited event detector is a leak and asks whether it should be subject to the ledger (answer: yes). But there is a second problem: an LLM with a 2026 training cutoff *replayed over 2018 news* knows how the 2018 tariffs played out. Its historical labels carry hindsight in the *magnitude* and *confidence* fields even if the sector and date are innocent. Fix:
- **Historical events (for the P3 harness and any training):** labels come from a *deterministic* extractor (calendars, official-release parsers, rule-based magnitude from the document's own numbers — a tariff rate is a number in the text) or from a small model trained only on pre-event text. Frontier-LLM labels for historical events are **oracle labels**: usable only to validate Agora's response magnitudes, never as an input to the trading model's training or evaluation.
- **Live events (from now on):** the LLM classifier is fine, because it cannot know the future at run time. Its output, prompt hash, model version, and confidence are logged as a ledger row, and its knowledge-time is the *run timestamp*, not the event's.
- The classifier is a model: it gets its own walk-forward evaluation (precision/recall of events, magnitude error vs. official numbers) and its own row in the ledger.

---

## 2. One feature library, two modes — ✅ adopt, ✚ two additions
**Agreed:** write features once; `replay(as_of=date)` and `live()` call the same function; a feature that cannot be computed in replay does not exist; CI proves a backtest replays byte-identically through the live path.

✚ **Lag-aware live mode.** `live()` must query `as_of = now − declared_lag(feature)`, not `now`, so live sees exactly what replay assumed. Otherwise live is *more* informed than the backtest and the shortfall monitor will misattribute the difference.

✚ **No same-bar execution.** Signals computed on the close of day *t* execute at the open of *t+1* (or a VWAP window). Encode the execution timestamp in the feature registry as a lag of its own, so replay cannot cheat.

---

## 3. Daily training that respects the loss — ✅ adopt the fast/slow split and procedure certification, ✚ change what the fast layer *is*

**Agreed and essential:** daily refits are not trials; the object under certification is the **update procedure** `U : (model_{t−1}, data_t) → model_t`, certified by walk-forward replay of `U` over history; live ledger rows carry procedure ID, procedure stats, fast-layer generation count, and cumulative drift from the certified parent. The procedure's hyper-parameters (learning rate, epochs, trigger thresholds, drift budget) are the search dimension and are counted as trials. Tuning the drift budget is itself a search — log it.

✚ **Most nights should be a re-solve, not a refit.** Emporion already has a hybrid solver: net warm-start → damped Newton on the exact clearing system. On a new day the state changes (prices, holdings, shocks) but the *law* has not. So the nightly step is:
1. Run the certified net on the new state → `p̂`, `M`, residuals.
2. Newton-polish the clearing residual at fixed parameters θ (this is inference-time optimisation, the same move as "Plan Before You Trade").
3. **Only if** the post-polish residual on today's state exceeds tolerance, take a bounded number of gradient steps on θ — the "fast refit" — under the drift budget.
Parameters therefore move only when the law demands it, which makes the drift budget meaningful and the nightly job deterministic and cheap (CPU or the A10, minutes).

✚ **Recency is the back door for P&L-chasing.** Even with a residual loss, refitting on *only the newest states* tilts the kernel toward the current regime — the same failure the law-based loss was chosen to avoid. Two guards:
- **Anchor set with rehearsal:** a frozen, stratified sample of historical states (all regimes, all Agora-shock types, the known stress dates). Every fast refit includes the anchor set in its batch, and the residual on the anchors is the primary drift metric. The law must keep holding on the past.
- **Regularise toward the certified parent:** an L2 penalty on `θ − θ_parent` (or Fisher-weighted) during fast refits. This bounds forgetting analytically and gives the drift budget a knob.
Sampling stays residual-adaptive (RAD) across old and new states, so if a new regime violates the law it gets attention *because of the residual*, not because it is recent.

✚ **Drift budget — the definition (answers open question 1).** Use three metrics in a hierarchy; the budget is breached if any primary or secondary threshold trips:
| Tier | Metric | Why |
|---|---|---|
| Primary | `Δ residual on the anchor set` (p50 and p99, R1 and R2 separately) | "does the law still hold on the past" — the only metric that speaks the model's own language |
| Secondary | prediction disagreement with the certified parent on a fixed **probe set** of states: `‖p̂ − p̂_parent‖`, rank correlation of implied views, fraction of gate decisions that flipped | behavioural drift a user would feel |
| Tertiary (log only) | parameter distance `‖θ − θ_parent‖` | cheap sanity signal; meaningless as a gate for nets |
Thresholds are set on the walk-forward replay of `U` so that the certified procedure's own historical drift stays inside them with margin; that makes the budget part of the certified object rather than a hand-tuned number.

---

## 4. Residuals as the regime detector — ✅ adopt, ✚ split the residual into two objects

**Agreed:** rising live residual = the kernel going stale before P&L shows it; one number drives both the slow-layer retrain trigger and abstention.

✚ **But "residual" is two different quantities, and they must not be conflated:**
- **Self-consistency residual** `r_self` — the residual of the *model's own* equilibrium after polish: does the net + solver produce a `p̂`, `M` that satisfies R1 and R2 on today's state? High `r_self` = the model cannot find an equilibrium = **stale or mis-specified** → *abstain and trigger retrain*.
- **Mispricing residual** `r_mkt` — the residual of the *observed* prices under the model: how far is the market from `p̂`? High `r_mkt` with low `r_self` = the model is confident **and** the market disagrees = **that is the trading signal**.
If one number is used for both, every mispricing looks like staleness and the gate suppresses the edge. The abstention gate keys on `r_self` and ensemble disagreement; the signal keys on `r_mkt` scaled by the conformal width. The regime detector watches `r_self`.

✚ **Calibrated thresholds, not fixed ones.** Set the `r_self` alarm as a conformal quantile of the certified model's own historical `r_self` distribution (controls the false-alarm rate) and run a CUSUM / Page–Hinkley change-point test on the series so a slow creep triggers as reliably as a jump. Track R1 and R2 separately: R1 drifting alone suggests the pricing kernel is stale; R2 drifting alone suggests investor-type demands (flows, mandates, leverage) have shifted.

---

## 5. Agora in the loop — ✅ adopt, ✚ interface and scoring details
**Agreed:** synchronous service `shock_vector → sector_response_distribution`; version Agora's calibration alongside the model in the ledger; the P3 event validation becomes a permanent harness (every detected event creates a prediction record; N days later the realised response is scored). This harness is a genuinely new artefact for the CGE literature regardless of the trading outcome.

✚ **Return a distribution, not a point.** Agora's response should come from its ensemble/polish path so the bridge carries uncertainty; the trading model consumes mean + spread, and the spread widens the conformal interval (smaller Kelly fraction) — the prior's uncertainty propagates to position size automatically.

✚ **Counterfactual cache.** Shock space is low-dimensional; precompute a grid for latency-free reads and call the service synchronously only for novel shocks. Cache entries are keyed by Agora calibration version so a base-year update invalidates them.

✚ **Scoring rule for the harness.** Strip market beta from realised sector returns; score at 1, 5, 20 trading-day horizons; use CRPS (distributional), sign accuracy, and cross-sector rank correlation; always against two nulls — zero response and a naive sector-beta response. Publish the running scoreboard in the console. This is the P3 go/no-go turned into a live dial.

---

## 6. Promotion gates — ✅ adopt, ✚ three additions
**Agreed order:** ingest validation → live residual within tolerance → shadow evaluation vs incumbent over a minimum window → ledger row → ensemble agreement → deploy; fast-layer models replace a parent only if the parent is certified and drift holds; kill switches live outside the model's process.

✚ **Minimum shadow window is statistical, not calendar.** Promote when the bootstrap CI of (challenger − incumbent) shortfall-adjusted return excludes a negative value, with a floor of N trading days; a fixed "30 days" will promote noise in a quiet month.
✚ **Canary sizing and rollback.** Deploy a promoted model at a fraction of target size first; rollback target is always the certified parent, which is kept warm.
✚ **Separation of duties.** The gate evaluator runs in a process with read-only access to models and write access only to the ledger; the model process cannot write its own certification.

---

## 7. Enhancements the proposal does not cover (DL level)
1. **Online conformal calibration.** The Kelly scale uses conformal intervals; they must be recalibrated daily with the *simple rolling-quantile* variant (the one that won in the 2026 study) on the live residual stream, with the non-conformity score itself versioned in the ledger.
2. **Ensemble as a first-class citizen of the update procedure.** `U` updates K members with different seeds/anchor subsets; disagreement is both an abstention input and a drift metric. Certify the ensemble procedure, not a member.
3. **Determinism.** Fixed seeds, hashed data snapshots (`as_of` + content hash), model and Agora-version hashes in every ledger row. A nightly fast step must be reproducible bit-for-bit from its inputs.
4. **Compute plan.** Nightly: re-solve + occasional bounded refit, minutes on CPU/A10. Weekly/regime-trigger: full residual training with RAD, self-adaptive weights, L∞ tail term, CPCV certification — hours on an H100. Nothing here needs continuous GPU.
5. **Test-time state re-solve for the execution layer too.** Almgren–Chriss schedules are re-solved each morning on the day's vol/volume forecast; they are part of the procedure and their parameters are versioned.

---

## 8. Answers to the two open questions
- **What does the drift budget measure?** Anchor-set residual delta (primary, R1/R2 separately), probe-set prediction disagreement with the certified parent (secondary), parameter distance logged only. Thresholds inherited from the procedure's walk-forward replay.
- **Is the event classifier subject to the ledger?** Yes — and more than that: it is *temporally quarantined*. Live LLM labels are ledgered and allowed; historical frontier-LLM labels are oracle labels restricted to validating Agora, never fed to the trading model.

---

## 9. Revised phasing (RT1–RT4, adjusted)
- **RT1** — bitemporal store (four clocks) + replay/live parity with lag-aware live mode + deterministic historical event extractor. Exit: a backtest replays byte-identically through the live path; the event extractor's historical labels contain no LLM output.
- **RT2** — live shadow of the certified model (trade nothing) **and** the event-validation harness switched on, because it needs elapsed time to accumulate evidence. Exit: shortfall inside CI; harness scoreboard populated.
- **RT3** — nightly re-solve + bounded fast refit under the drift budget; `r_self` / `r_mkt` split; conformal-calibrated alarms; online conformal Kelly. Exit: drift budget never breached silently; every breach produced a slow-layer trigger.
- **RT4** — procedure-level certification of `U` (with the ensemble and thresholds inside it) by walk-forward replay; promotion gates with statistical shadow window, canary, rollback, separation of duties. Exit: first model promoted to live with size, by the gates, not by hand.
