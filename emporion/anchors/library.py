"""Anchor library — the replication-study factor themes (Jensen, Kelly, Pedersen 2023) as point-in-time signals.

Every theme is a function of data known at the decision date t (no look-ahead: windows end at t, features use
prices up to t and fundamentals as-of t).  Price-based themes work on every market (stocks, crypto); fundamental
themes need the bitemporal `facts`/`results` tables and are computed by the market scripts.  Signals are oriented
so that HIGHER = expected HIGHER return (the published direction), then cross-sectionally z-scored.

Price-based themes (13-theme names in brackets):
  momentum       12-1 month return                                   [momentum]
  st_reversal    -(1-month return)                                   [short-term reversal]
  low_risk       -(60-day realised vol)                              [low risk]
  size           -log(dollar volume proxy or market cap)             [size]
  skewness       -(60-day return skewness)                           [skewness]
  seasonality    same-calendar-month average return over past years  [seasonality]
  illiquidity    Amihud |r|/dollar-volume (60d)                      [liquidity risk; crypto: dominant predictor]
  trend          blend of 1/3/6/12-month trends (crypto trend factor) [trend]
"""
from __future__ import annotations

import numpy as np

THEMES = ("momentum", "st_reversal", "low_risk", "size", "skewness", "seasonality", "illiquidity", "trend")


def _z(x):
    m = np.nanmean(x); s = np.nanstd(x)
    return (x - m) / s if s > 0 else np.zeros_like(x)


def price_themes(P: np.ndarray, V: np.ndarray, t: int, periods: int = 252) -> dict:
    """P (T,N) closes, V (T,N) volume (units) — both known through index t. Returns {theme: z-scored (N,)}."""
    L = np.log(P); out = {}
    m = periods // 12
    if t >= 12 * m + 1:
        out["momentum"] = _z(L[t - m] - L[t - 12 * m])
        r = np.diff(L[t - 12 * m:t + 1], axis=0)
        out["trend"] = _z(sum(_z(L[t] - L[t - k * m]) for k in (1, 3, 6, 12)) / 4)
    if t >= m + 1:
        out["st_reversal"] = _z(-(L[t] - L[t - m]))
    if t >= 61:
        r60 = np.diff(L[t - 60:t + 1], axis=0)
        out["low_risk"] = _z(-r60.std(0))
        d = r60 - r60.mean(0); s2 = (d ** 2).mean(0)
        out["skewness"] = _z(-np.where(s2 > 0, (d ** 3).mean(0) / np.maximum(s2, 1e-12) ** 1.5, 0.0))
        dv = (P[t - 60:t + 1] * V[t - 60:t + 1]).mean(0)
        out["size"] = _z(-np.log(np.maximum(dv, 1e-9)))
        dvd = np.maximum(P[t - 59:t + 1] * V[t - 59:t + 1], 1e-9)
        out["illiquidity"] = _z(np.log((np.abs(r60) / dvd).mean(0) + 1e-18))
    if t >= 2 * periods:
        # seasonality: average return in the same calendar month (approximated by same position in the year) over past years
        seas = np.zeros(P.shape[1]); n = 0
        for y in range(1, 5):
            a, b = t - y * periods, t - y * periods + m
            if a >= 0 and b <= t:
                seas += L[b] - L[a]; n += 1
        if n:
            out["seasonality"] = _z(seas / n)
    return out
