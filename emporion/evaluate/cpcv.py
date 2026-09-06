"""Combinatorial Purged Cross-Validation (López de Prado 2018, ch. 12).

T observations are cut into `n_groups` contiguous groups; every combination of
`n_test` groups is a test set; training = the rest minus a purge window of
`horizon` observations around each test group and an `embargo` after it.
Every test group appears in C(n_groups-1, n_test-1) combinations, so the test
predictions can be stitched into  n_paths = C(n,k)*k/n  full out-of-sample paths.

Everything is index arithmetic on boolean masks — no data is copied until the
strategy factory asks for it."""
from __future__ import annotations

import itertools
from math import comb
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..core.perf import timed


def n_paths(n_groups: int, n_test: int) -> int:
    return comb(n_groups, n_test) * n_test // n_groups


def cpcv_splits(T: int, n_groups: int = 6, n_test: int = 2, horizon: int = 1, embargo: int = 0
                ) -> List[Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]]:
    """Returns list of (train_idx, test_idx, test_group_ids)."""
    bounds = np.linspace(0, T, n_groups + 1).astype(int)
    groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_groups)]
    out = []
    for combo in itertools.combinations(range(n_groups), n_test):
        test_mask = np.zeros(T, bool)
        purge = np.zeros(T, bool)
        for g in combo:
            a, b = groups[g][0], groups[g][-1]
            test_mask[a:b + 1] = True
            purge[max(0, a - horizon):min(T, b + 1 + horizon + embargo)] = True
        train_mask = ~purge
        out.append((np.flatnonzero(train_mask), np.flatnonzero(test_mask), combo))
    return out


@timed("cpcv.paths", rows=lambda *a, **k: int(a[0]))
def cpcv_paths(T: int, strategy: Callable[[np.ndarray, np.ndarray], np.ndarray],
               n_groups: int = 6, n_test: int = 2, horizon: int = 1, embargo: int = 0) -> np.ndarray:
    """strategy(train_idx, test_idx) -> out-of-sample returns over test_idx (len(test_idx),).
    Returns an array (n_paths, T) of stitched out-of-sample return paths."""
    splits = cpcv_splits(T, n_groups, n_test, horizon, embargo)
    # assign each (combo, group) test block to a path id so that every path has each group exactly once
    per_group_count = comb(n_groups - 1, n_test - 1)
    paths = np.full((per_group_count, T), np.nan)
    slot = {g: 0 for g in range(n_groups)}
    bounds = np.linspace(0, T, n_groups + 1).astype(int)
    for train_idx, test_idx, combo in splits:
        oos = np.asarray(strategy(train_idx, test_idx), float)
        assert oos.shape[0] == test_idx.shape[0], "strategy must return one return per test observation"
        for g in combo:
            a, b = bounds[g], bounds[g + 1]
            sel = (test_idx >= a) & (test_idx < b)
            paths[slot[g], a:b] = oos[sel]
            slot[g] += 1
    return paths


def path_stats(paths: np.ndarray, periods: int = 252) -> Dict[str, np.ndarray]:
    from .metrics import sharpe, max_drawdown, ann_return
    P = np.nan_to_num(paths.T, nan=0.0)   # (T, n_paths)
    return {"sharpe": sharpe(P, periods), "max_drawdown": max_drawdown(P), "ann_return": ann_return(P, periods)}
