# Performance, tuning and speed — the Emporion engineering discipline

*The rules every module in this repository obeys. They are enforced by tests, not by review.*

1. **Measured before optimised.** Every hot function carries `@timed`; wall time, rows, throughput and peak RSS go to a perf record and to `benchmarks/*.json` with host and date. A budget breach fails the test suite the same way a wrong answer does (`tests/test_baselines_perf.py`). Numbers are always recorded with hardware and method; "it's fast" is not a result.
2. **Batched, always.** Statistics are written for a `(T, S)` matrix of strategies, never for one strategy in a loop. CPCV, PBO, bootstrap and the certification report treat all trials as one tensor. This is the same philosophy that lets Agora solve a whole distribution of economies in one forward pass.
3. **Precompute, then compose (the sparse-structure lesson).** PBO's Sharpe over any block combination is an inner product of a boolean mask with per-block sums and sums-of-squares computed once — `O(S)` per combination instead of re-slicing the returns. CPCV is boolean-mask arithmetic; no data is copied until a strategy asks for it. Where the structure is sparse, the algorithm is sparse.
4. **Replay-exact, compute-once.** The store is bitemporal and `as_of` is mandatory; features are content-addressed by (code hash, data-snapshot hash, effective knowledge time). The same input can never be computed twice, and a backtest replays byte-identically through the live path (`test_replay_live_parity_and_cache`).
5. **Columnar, lazy, out-of-core.** Parquet part files are append-only; DuckDB reads only the rows and columns a query touches; Polars for transforms. No pandas in hot paths. This is what lets the whole P0 run on a 2-CPU, 11 GB bastion.
6. **Memory is a budget, not a hope.** Peak RSS is recorded per function; anything that grows with `T × N × trials` must stream in blocks.
7. **Same code, every device.** Evaluation maths is NumPy and runs anywhere; the model layer (P4) reuses Agora's `device.py` so CPU/MPS/CUDA is a string. The bastion runs nightly work; GPUs are rented by the hour for training only.
8. **Tail-honest latency.** Reports carry p50, p99 and max, never the mean alone — the same rule the equilibrium engine applies to residuals.
9. **Determinism is a feature.** Seeds are explicit, hashes are in every ledger row, and the ledger chain must verify at the end of every run.

Current budgets (P0, synthetic 10 y × 20 assets, 5 strategies) are in `tests/test_baselines_perf.py`; measured results in `benchmarks/`.
