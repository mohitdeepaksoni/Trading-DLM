"""Vectorised performance statistics. Every function takes returns of shape
(T,) or (T, S) and works on all S strategies at once — never a loop over
strategies.  NaNs are treated as flat (0 return)."""
from __future__ import annotations

import numpy as np


def _2d(r):
    r = np.asarray(r, dtype=float)
    r = np.nan_to_num(r, nan=0.0)
    return r[:, None] if r.ndim == 1 else r


def sharpe(r, periods: int = 252):
    r = _2d(r)
    sd = r.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, r.mean(axis=0) / sd * np.sqrt(periods), 0.0)
    return out


def ann_return(r, periods: int = 252):
    r = _2d(r)
    T = r.shape[0]
    growth = np.prod(1 + r, axis=0)
    return np.power(np.maximum(growth, 1e-12), periods / max(T, 1)) - 1


def ann_vol(r, periods: int = 252):
    return _2d(r).std(axis=0, ddof=1) * np.sqrt(periods)


def max_drawdown(r):
    r = _2d(r)
    eq = np.cumprod(1 + r, axis=0)
    peak = np.maximum.accumulate(eq, axis=0)
    return (eq / peak - 1).min(axis=0)


def sortino(r, periods: int = 252):
    r = _2d(r)
    dn = np.minimum(r, 0)
    dsd = np.sqrt((dn ** 2).mean(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dsd > 0, r.mean(axis=0) / dsd * np.sqrt(periods), 0.0)


def hit_rate(r):
    r = _2d(r)
    return (r > 0).mean(axis=0)


def skew_kurt(r):
    r = _2d(r)
    m = r.mean(axis=0)
    d = r - m
    s2 = (d ** 2).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sk = np.where(s2 > 0, (d ** 3).mean(axis=0) / s2 ** 1.5, 0.0)
        ku = np.where(s2 > 0, (d ** 4).mean(axis=0) / s2 ** 2, 3.0)
    return sk, ku


def summary(r, periods: int = 252) -> dict:
    r = _2d(r)
    sk, ku = skew_kurt(r)
    return {"sharpe": sharpe(r, periods), "ann_return": ann_return(r, periods), "ann_vol": ann_vol(r, periods),
            "max_drawdown": max_drawdown(r), "sortino": sortino(r, periods), "hit_rate": hit_rate(r),
            "skew": sk, "kurtosis": ku, "T": r.shape[0]}
