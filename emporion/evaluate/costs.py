"""Costs inside the return, never as an afterthought.

weights W[t] are decided at the close of t and *executed at t+1* (no same-bar
execution — enforced here by the shift).  Turnover is measured against the
previous weights after they drifted with returns."""
from __future__ import annotations

import numpy as np

from ..core.perf import timed


def _drifted(w_prev: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Weights as fractions of capital after one period of returns: w_i(1+r_i) / (1 + w·r).
    Valid for long-only (reduces to g/sum(g)) AND for long-short / levered books (net exposure may be ~0)."""
    g = w_prev * (1 + r)
    cap = 1.0 + (w_prev * r).sum(axis=-1, keepdims=True)
    return np.where(np.abs(cap) > 1e-9, g / np.where(np.abs(cap) > 1e-9, cap, 1.0), 0.0)


@timed("costs.portfolio_returns", rows=lambda *a, **k: int(np.asarray(a[0]).size))
def portfolio_returns(W, R, cost_bps: float = 10.0, shift: int = 1):
    """W: (T,N) target weights decided at t; R: (T,N) simple asset returns over (t-1,t].
    Returns net portfolio returns (T,), turnover (T,).  cost_bps is per unit of
    one-way turnover (half-spread + fees)."""
    W = np.nan_to_num(np.asarray(W, float))
    R = np.nan_to_num(np.asarray(R, float))
    T, N = W.shape
    Wx = np.zeros_like(W)
    Wx[shift:] = W[:T - shift]                       # weights in force during period t
    gross = (Wx * R).sum(axis=1)
    # turnover at t = |target(t-1 decided, in force t) - drifted holdings from t-1|
    prev = np.zeros_like(W)
    prev[1:] = _drifted(Wx[:-1], R[:-1]) if T > 1 else 0
    to = np.abs(Wx - prev).sum(axis=1)
    net = gross - to * cost_bps / 1e4
    return net, to


def turnover(W, R, shift: int = 1):
    return portfolio_returns(W, R, 0.0, shift)[1]
