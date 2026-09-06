"""Investor-state nowcaster (doc 08 §2, P3c): holdings are observed exactly at filing dates and are latent between
them; a Kalman filter carries the state forward with a BOUNDED innovation and updates it from flows.

State per asset i:  x_t = institutional share (fraction of shares held by the tracked investor group).
Dynamics:           x_t = x_{t-1} + b * f_t + eta_t,   eta ~ N(0, q)        (f_t = observed net flow proxy, b = loading)
Observations:       filing dates: y_t = x_t exactly (R -> 0);  no observation otherwise.
Output: filtered mean and variance per day; the variance is the 'how much do we actually know' band that clamps the
encoder's investor-state head (research/09 §5). Validation = held-out filing-date reconstruction error vs. carrying the
last value forward (the naive baseline the filter must beat when flows carry information).
"""
from __future__ import annotations

import numpy as np


def kalman_nowcast(obs: np.ndarray, flows: np.ndarray, q: float = 1e-6, b: float = 0.0, r_obs: float = 1e-10):
    """obs (T,N) with NaN where unobserved; flows (T,N) drift proxies (0 if none). Returns (mean (T,N), var (T,N))."""
    T, N = obs.shape
    m = np.full((T, N), np.nan); v = np.full((T, N), np.nan)
    x = np.full(N, np.nan); P = np.full(N, np.nan)
    for t in range(T):
        # predict
        x = x + b * np.nan_to_num(flows[t]); P = P + q
        # first observation initialises
        o = obs[t]; has = np.isfinite(o)
        init = has & ~np.isfinite(x)
        x[init] = o[init]; P[init] = 0.0
        upd = has & np.isfinite(x) & ~init
        K = P[upd] / (P[upd] + r_obs)
        x[upd] = x[upd] + K * (o[upd] - x[upd]); P[upd] = (1 - K) * P[upd]
        m[t] = x; v[t] = P
    return m, v


def fit_flow_loading(obs: np.ndarray, flows: np.ndarray) -> float:
    """OLS of filing-to-filing changes in the state on cumulative flows between the same filings (the b that makes
    flows informative); zero when flows carry nothing."""
    num = den = 0.0
    T, N = obs.shape
    for i in range(N):
        idx = np.flatnonzero(np.isfinite(obs[:, i]))
        for a, c in zip(idx[:-1], idx[1:]):
            f = np.nan_to_num(flows[a + 1:c + 1, i]).sum(); d = obs[c, i] - obs[a, i]
            num += f * d; den += f * f
    return float(num / den) if den > 0 else 0.0


def holdout_error(obs: np.ndarray, flows: np.ndarray, every: int = 4, q: float = 1e-6):
    """Hide every k-th filing per asset; compare filter vs last-value carry-forward at the hidden dates."""
    T, N = obs.shape
    hidden = np.zeros_like(obs, dtype=bool)
    for i in range(N):
        idx = np.flatnonzero(np.isfinite(obs[:, i]))
        hidden[idx[every - 1::every], i] = True
    train = np.where(hidden, np.nan, obs)
    b = fit_flow_loading(train, flows)
    m, _ = kalman_nowcast(train, flows, q=q, b=b)
    m0, _ = kalman_nowcast(train, np.zeros_like(flows), q=q, b=0.0)
    e_filter = np.nanmean(np.abs(m[hidden] - obs[hidden])); e_naive = np.nanmean(np.abs(m0[hidden] - obs[hidden]))
    return {"b": b, "mae_filter": float(e_filter), "mae_last_value": float(e_naive), "n_hidden": int(hidden.sum())}
