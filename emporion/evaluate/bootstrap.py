"""Stationary bootstrap (Politis & Romano 1994) for autocorrelated returns.
Index generation is vectorised over replicates: (B, T) in one pass."""
from __future__ import annotations

import numpy as np

from ..core.perf import timed


@timed("bootstrap.indices", rows=lambda *a, **k: int(a[0] * a[1]))
def stationary_bootstrap_indices(T: int, B: int, mean_block: float = 20.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    restart = rng.random((B, T)) < p
    restart[:, 0] = True
    starts = rng.integers(0, T, (B, T))
    idx = np.empty((B, T), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, T):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def bootstrap_ci(r, stat, B: int = 2000, mean_block: float = 20.0, alpha: float = 0.05, seed: int = 0) -> dict:
    """stat: function (T,B)-array -> (B,) statistic. Returns point, lo, hi, p(stat<=0)."""
    r = np.nan_to_num(np.asarray(r, float))
    T = r.shape[0]
    idx = stationary_bootstrap_indices(T, B, mean_block, seed)
    samp = r[idx.T]                     # (T,B)
    s = np.asarray(stat(samp))
    point = float(np.asarray(stat(r[:, None]))[0])
    return {"point": point, "lo": float(np.quantile(s, alpha / 2)), "hi": float(np.quantile(s, 1 - alpha / 2)),
            "p_le_0": float((s <= 0).mean()), "B": B}
