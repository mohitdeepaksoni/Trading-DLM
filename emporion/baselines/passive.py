"""Passive and classical baselines. Each returns a weight matrix W (T,N):
weights decided at the close of t from data up to t (rolling windows only
look backwards); execution at t+1 is enforced downstream by costs.portfolio_returns."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from ..core.perf import timed
from ..core.registry import register


def _rebalance_mask(T: int, every: int) -> np.ndarray:
    m = np.zeros(T, bool)
    m[::every] = True
    return m


def _hold(W_at_rebal: np.ndarray, R: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Carry weights between rebalances, drifting with returns."""
    T, N = R.shape
    W = np.zeros((T, N))
    cur = np.zeros(N)
    for t in range(T):
        if t > 0:
            g = cur * (1 + R[t])
            cur = g / g.sum() if g.sum() != 0 else cur
        if mask[t]:
            cur = W_at_rebal[t]
        W[t] = cur
    return W


@register("baseline", "buy_and_hold")
def buy_and_hold(R: np.ndarray) -> np.ndarray:
    T, N = R.shape
    W0 = np.full((T, N), 1.0 / N)
    return _hold(W0, R, _rebalance_mask(T, T + 1) | np.eye(1, T, 0, dtype=bool)[0])


@register("baseline", "equal_weight")
def equal_weight(R: np.ndarray, every: int = 21) -> np.ndarray:
    T, N = R.shape
    return _hold(np.full((T, N), 1.0 / N), R, _rebalance_mask(T, every))


@register("baseline", "inverse_vol")
def inverse_vol(R: np.ndarray, lookback: int = 60, every: int = 21) -> np.ndarray:
    """Naive risk parity: w ∝ 1/σ on a trailing window (no look-ahead: window ends at t)."""
    T, N = R.shape
    W = np.full((T, N), 1.0 / N)
    for t in range(lookback, T):
        sd = R[t - lookback + 1:t + 1].std(axis=0, ddof=1)
        inv = 1.0 / np.where(sd > 0, sd, np.inf)
        W[t] = inv / inv.sum() if inv.sum() > 0 else 1.0 / N
    return _hold(W, R, _rebalance_mask(T, every))


def hrp_weights(cov: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity (López de Prado 2016): single-linkage on
    correlation distance → quasi-diagonalisation → recursive bisection."""
    N = cov.shape[0]
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    corr = np.clip(np.nan_to_num(corr), -1, 1)
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), "single")
    order = list(leaves_list(link))
    w = np.ones(N)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) < 2:
                continue
            a, b = c[:len(c) // 2], c[len(c) // 2:]
            va, vb = _cluster_var(cov, a), _cluster_var(cov, b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
            nxt += [a, b]
        clusters = nxt
    return w / w.sum()


def _cluster_var(cov, idx):
    sub = cov[np.ix_(idx, idx)]
    ivp = 1 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


@register("baseline", "hrp")
@timed("baseline.hrp", rows=lambda *a, **k: int(a[0].shape[0]))
def hrp(R: np.ndarray, lookback: int = 252, every: int = 21) -> np.ndarray:
    T, N = R.shape
    W = np.full((T, N), 1.0 / N)
    mask = _rebalance_mask(T, every)
    for t in range(lookback, T):
        if mask[t]:
            cov = np.cov(R[t - lookback + 1:t + 1].T, ddof=1)
            W[t] = hrp_weights(cov)
    return _hold(W, R, mask)


@register("baseline", "momentum_top")
def momentum_top(R: np.ndarray, lookback: int = 252, skip: int = 21, top_frac: float = 0.3, every: int = 21) -> np.ndarray:
    """Long-only cross-sectional momentum: equal-weight the top fraction by 12-1 return."""
    T, N = R.shape
    logp = np.cumsum(np.log1p(R), axis=0)
    W = np.full((T, N), 1.0 / N)
    k = max(1, int(round(top_frac * N)))
    for t in range(lookback, T):
        mom = logp[t - skip] - logp[t - lookback]
        top = np.argsort(mom)[-k:]
        w = np.zeros(N)
        w[top] = 1.0 / k
        W[t] = w
    return _hold(W, R, _rebalance_mask(T, every))


BASELINES = {"buy_and_hold": buy_and_hold, "equal_weight": equal_weight, "inverse_vol": inverse_vol,
             "hrp": hrp, "momentum_top": momentum_top}
