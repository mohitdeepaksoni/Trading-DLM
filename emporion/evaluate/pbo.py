"""Probability of Backtest Overfitting via CSCV (Bailey, Borwein, López de Prado, Zhu 2014).

Input: M (T, S) returns of S strategies. Split T into `n_blocks` blocks, take
every combination of n_blocks/2 blocks as in-sample, the rest out-of-sample;
select the IS-best strategy; record its OOS rank.  PBO = share of combinations
where the IS-best is below the OOS median.

Performance: per-block sums and sums-of-squares are precomputed once, so the
Sharpe of every strategy for any block combination is O(S) additions — no
re-slicing of the return matrix per combination."""
from __future__ import annotations

import itertools
from math import comb

import numpy as np

from ..core.perf import timed


@timed("pbo.cscv", rows=lambda *a, **k: int(np.asarray(a[0]).size))
def pbo_cscv(M, n_blocks: int = 16, max_combos: int = 20000, seed: int = 0) -> dict:
    M = np.nan_to_num(np.asarray(M, float))
    T, S = M.shape
    if S < 2:
        return {"pbo": float("nan"), "n_combos": 0, "logits": []}
    n_blocks -= n_blocks % 2
    bounds = np.linspace(0, T, n_blocks + 1).astype(int)
    s1 = np.stack([M[bounds[i]:bounds[i + 1]].sum(0) for i in range(n_blocks)])          # (B,S)
    s2 = np.stack([(M[bounds[i]:bounds[i + 1]] ** 2).sum(0) for i in range(n_blocks)])   # (B,S)
    cnt = np.diff(bounds).astype(float)                                                   # (B,)
    half = n_blocks // 2
    combos = list(itertools.combinations(range(n_blocks), half))
    if len(combos) > max_combos:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_combos, replace=False)]
    idx = np.array(combos)                                    # (C, half)
    mask = np.zeros((len(combos), n_blocks), bool)
    mask[np.arange(len(combos))[:, None], idx] = True         # (C,B)

    def _sharpe(m):
        n = (m.astype(float) @ cnt)[:, None]                  # (C,1)
        mu = (m.astype(float) @ s1) / n
        var = (m.astype(float) @ s2) / n - mu ** 2
        sd = np.sqrt(np.maximum(var, 1e-18))
        return mu / sd

    sr_is, sr_oos = _sharpe(mask), _sharpe(~mask)             # (C,S)
    best = sr_is.argmax(axis=1)
    oos_rank = (sr_oos < sr_oos[np.arange(len(combos)), best][:, None]).sum(axis=1)  # 0..S-1
    w = (oos_rank + 1) / (S + 1)
    logits = np.log(w / (1 - w))
    return {"pbo": float((logits < 0).mean()), "n_combos": len(combos), "logits": logits.tolist(),
            "mean_oos_rank_pct": float(w.mean())}
