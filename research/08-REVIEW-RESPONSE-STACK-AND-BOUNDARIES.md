# 08 — Response to the third review: corrections, the nowcaster, the ablation ladder, the stack, and what stays out

*2026-09-05. A reviewer read doc 07 and returned four points: (1) A.2's novelty claim is false — Koijen & Yogo (2019) already clear the market from 13F holdings; (2) yes to DL, but the frequency mismatch between filings and daily signals is the real modelling problem, Agora events are too sparse to validate at budget cadence, and the ablation ladder is missing; (3) a concrete stack; (4) keep any quantum emulator out of the model and the pitch. Verdict per point below, with what was changed.*

---

## 1. The Koijen–Yogo correction — accepted, applied
The reviewer is right and the error was ours. Demand-system asset pricing (Koijen & Yogo 2019, *JPE*) estimates investor-level characteristics-based demand from 13F holdings and clears the market by construction; the inelastic-markets elasticity we quoted comes from the same family; Koijen–Richmond–Yogo (2023) and Haddad–Huebner–Loualiche (2024) extend it. Our "Market Accounting Matrix" is that object with a different name.

**Changed:** doc 07 A.2 now states the ancestry and builds the MAM *as* a Koijen–Yogo demand system (logit demand in characteristics, elasticities estimated with their instruments, not assumed). The positioning doc no longer says "nobody has"; it says law-based asset pricing exists, demand-system clearing exists, and nobody has coupled them to a computable general-equilibrium engine. The surviving claim, which is the one to defend in a paper: **first to use a demand-system clearing model as a structural prior for a trading system, coupled to general equilibrium, with the pricing kernel derived from the same investors.** References updated.

The reviewer also confirms that A.1 — the kernel implied by investor state rather than a free network — is the strongest idea and solves identification more cleanly than the adversarial approach. It is now the protected core of the design.

## 2. The frequency mismatch: the investor-state nowcaster — accepted, specified
Filings are quarterly (13F, shareholding patterns) or monthly (N-PORT, AMFI); signals are daily. If investor state between filings is a free daily latent, the identification win of A.1 evaporates because the latent absorbs noise exactly as a free kernel would. The reviewer's bound is the right one: **state changes between filings must be reconstructible from observed flows plus a small innovation term.**

✚ **Specification (new phase P3c):**
- **Accounting identity first.** Holdings at t+1 = holdings at t + net flows × allocation + price change. For *passive/mandated* types (index funds, ETFs) the allocation is known (cap-weights) so their state is **deterministic** between filings. For *active* types only the allocation of net flow is latent.
- **State-space model.** Investor state `x_t` (per type: wealth share, characteristic-demand coefficients, constraint slackness) evolves as a random walk with **bounded** innovation variance; observations are (a) exact holdings on filing dates, (b) daily net flows by type (India: NSE FII/DII cash-market net flows daily; US: ETF creations/redemptions, ICI weekly fund flows, Fed weekly data), (c) prices. Kalman filter for the linear part, particle filter if the constraint slackness makes it non-linear.
- **The net does not own the latent.** The investor-state head in the encoder is trained to reproduce the *filtered* state, not to invent one; its output is clamped to the filter's confidence band. Innovation variance is a certified hyper-parameter of the update procedure, logged like any other.
- **Separate validation.** Hold out every fourth filing date; reconstruct holdings from the previous filing plus flows; report reconstruction error by investor type. Active-type error is the honest measure of how much daily investor state we actually know. If it is large, the daily signal leans on the Agora and kernel channels and the demand-pressure channel is down-weighted automatically through the conformal width.

## 3. Sparse events — accepted, widened
Twenty scored events with a confidence interval excluding zero, at one-budget-a-year cadence, is a decade. Two fixes:
- **Validate the harness where events are dense first.** The US tariff actions of 2025–26 give dozens of dated, sector-specific shocks with published rates; the harness is validated there regardless of which market is traded first. The EU CBAM phases and China's export controls add more.
- **Agora handles more than budgets.** Energy-price shocks (continuous), monetary-policy decisions (≈ 8 per central bank per year across several banks), exchange-rate moves, commodity shocks, and scheduled macro releases are all inputs the CGE responds to. Counted this way the event stream is weekly, not annual. The harness scores each event type separately so a strong tariff score cannot hide a weak rate-decision score.

## 4. The ablation ladder — accepted, and it is the go/no-go
Same ledger, same costs, same universe, same CPCV splits, every rung counted as a trial:
1. Passive: buy-and-hold, equal-weight, HRP.
2. Price-only: momentum/value tilts; GKX-style gradient boosting on characteristics.
3. **Koijen–Yogo demand system alone** (MAM without kernel derivation, without Agora).
4. **Emporion without Agora** (structural kernel from MAM investors + abstention + conformal Kelly).
5. **Full Emporion** (adds the CGE shock channel).
Decision rule: if rung 5 does not beat rung 3 on Deflated Sharpe *and* on the CPCV distribution, the CGE coupling is a story, not an edge, and the honest product is rung 4 plus the benchmark. If rung 4 does not beat rung 3, the kernel derivation adds nothing and the product is the evaluator. Know this by month 6, not month 18. Rungs 1–2 are built in P1; rungs 3–5 in P7.

## 5. Stack — accepted with three notes
- **PyTorch 2.x, confirmed:** Agora's `nets/mlp.py` and `nets/moe.py` are PyTorch; Kronos, Set Transformer implementations and the signature-kernel libraries are PyTorch-native. No JAX.
- **Path signatures:** `signatory` is unmaintained since 2021 and does not build against current PyTorch. Use `iisignature` (with its autograd adapter) for truncated signatures, `sigkernel` for signature-kernel scores in the generator, or implement depth-3/4 truncated signatures directly (Chen's identity, a few dozen lines) — which is what we will do for the encoder features to avoid a fragile dependency.
- **Hard constraint `E_t[M] = 1/R_f`:** a projection layer, as the reviewer says.
- **Certification maths:** write CPCV/DSR/PBO ourselves (the reference implementations are paywalled; the maths is short and must be auditable anyway).
- **Conformal:** MAPIE, or a 100-line implementation for the rolling-quantile variant.
- **Data:** Polars + Arrow/Parquet; DuckDB for `as_of` replay queries; TimescaleDB or QuestDB for the live bitemporal store from RT1.
- **Tooling, staged, not all at once:** P0 = Parquet + DuckDB + git only. RT1 adds Dagster (asset lineage maps naturally onto the ledger), MLflow, DVC. Over-tooling at P0 is the most common way a solo project stalls.
- **Hash chain:** append-only Parquet, chained hashes in a public git repository, each daily hash anchored with **OpenTimestamps** (free, Bitcoin-anchored, independently verifiable) — adopted; this is exactly the third-party timestamp the pre-registration design needed.
- **Serving:** FastAPI; export via `torch.export`/ONNX (TorchScript is on its way out). Risk limits and kill switches in a separate process with **no import path to the model**. Rust only later, for LOB execution, if ever.
- **Console:** Agora Analyst skeleton.

## 6. Quantum emulator — agreed: out of the model, out of the pitch
Anything run on Qiskit Aer, PennyLane or cuQuantum is classically simulated by definition and cannot yield an edge a classical method cannot; the 30–40-qubit ceiling covers toy portfolios; QAOA/annealing for cardinality-constrained selection, amplitude estimation for pricing, and QML feature maps are unproven or slower than classical at current scale.

Two legitimate, quarantined roles:
1. **Quantum-inspired, classical, shippable:** tensor-network (MPS) generators for synthetic paths in the curriculum budget, and QUBO formulations of lot-size/cardinality-constrained allocation solved by simulated annealing in the execution layer. (A GPU QUBO annealer already exists in the author's own toolkit and can serve as the classical baseline here; it is used as an optimiser, nothing more.)
2. **A research track in Emporion-Bench:** publish the QUBO allocation instances with the simulated-annealing baseline and let others benchmark QAOA or annealing hardware against it. Owning the yardstick again.

The word "quantum" does not appear in the positioning, the README, the paper title, or the trade ticket. Part B's credibility rests on a boring, verifiable record; that word reads as the opposite to every desk we want to reach and would breach the "what to never claim" rule.

## 7. Net effect on the design
- Protected core: structural kernel from calibrated (Koijen–Yogo) investors + Agora shock channel + abstention + pre-registration.
- New module: the investor-state nowcaster (P3c), separately validated.
- New discipline: the ablation ladder decides what the product is by month 6.
- Corrected claim: coupling, not components, is the novelty.
