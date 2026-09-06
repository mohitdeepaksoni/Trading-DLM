"""Volatility null models — the bar every learned vol forecaster must clear (doc 01 §3.2–3.3).

  har_rv   : Corsi (2009) HAR on log realised variance (daily, weekly, monthly means), OLS refit rolling.
  garch11  : GARCH(1,1) with Student-t innovations, MLE via scipy, refit rolling.
  ewma     : RiskMetrics λ=0.94 (the zero-parameter null).

All forecast next-day variance from returns up to t (no look-ahead); evaluated with
QLIKE and MSE on log variance against realised r_{t+1}^2 (proxy) via `vol_forecast_eval`.
Vectorised over assets where the model allows; refits are every `every` days.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from ..core.perf import timed
from ..core.registry import register


def _lagmeans(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over the last w rows (inclusive), NaN for the first w-1 rows."""
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    out = np.full_like(x, np.nan, dtype=float)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


@register("baseline", "ewma_vol")
def ewma(R: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """One-step-ahead variance forecast h[t] = forecast for period t made at t-1."""
    R = np.nan_to_num(np.asarray(R, float))
    T, N = R.shape
    h = np.zeros((T, N))
    h[0] = R[:20].var(axis=0)
    for t in range(1, T):
        h[t] = lam * h[t - 1] + (1 - lam) * R[t - 1] ** 2
    return h


@register("baseline", "har_rv")
@timed("baseline.har_rv", rows=lambda *a, **k: int(a[0].size))
def har_rv(R: np.ndarray, lookback: int = 500, every: int = 21, floor: float = 1e-8) -> np.ndarray:
    """HAR in LEVELS on daily variance proxies: v_{t+1} = c + b_d r_t^2 + b_w mean_5(r^2) + b_m mean_22(r^2) + e,
    pooled OLS across assets, refit every `every` days.  (A log-HAR on daily r^2 is biased downward by
    E[log r^2] != log E[r^2]; with only daily data the level form is the honest null.)"""
    R = np.nan_to_num(np.asarray(R, float))
    T, N = R.shape
    v = R ** 2
    X = np.stack([v, _lagmeans(v, 5), _lagmeans(v, 22)], axis=-1)              # (T,N,3), known at t
    h = np.full((T, N), np.nan)
    beta = None
    for t in range(22, T):
        if (t - 22) % every == 0 and t >= 22 + 60:
            lo = max(22, t - lookback)
            Xf = X[lo:t - 1].reshape(-1, 3); yf = v[lo + 1:t].reshape(-1)         # features at s -> target s+1 <= t-1
            ok = np.isfinite(Xf).all(1) & np.isfinite(yf)
            A = np.hstack([np.ones((ok.sum(), 1)), Xf[ok]])
            beta = np.linalg.lstsq(A, yf[ok], rcond=None)[0]
        if beta is not None and np.isfinite(X[t - 1]).all():
            h[t] = np.maximum(beta[0] + X[t - 1] @ beta[1:], floor)                  # forecast for t made at t-1
    return h


def _garch_t_nll(p, r):
    omega, alpha, beta, nu = p
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999 or nu <= 2.05:
        return 1e10
    T = r.shape[0]
    h = np.empty(T); h[0] = r.var()
    for t in range(1, T):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1]
    z2 = r ** 2 / h
    ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
          - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(z2 / (nu - 2)))
    return -ll.sum()


def fit_garch_t(r: np.ndarray):
    r = np.asarray(r, float); v = r.var()
    x0 = np.array([0.05 * v, 0.08, 0.90, 6.0])
    res = minimize(_garch_t_nll, x0, args=(r,), method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-8})
    return res.x


@register("baseline", "garch11_t")
@timed("baseline.garch11_t", rows=lambda *a, **k: int(a[0].size))
def garch11_t(R: np.ndarray, lookback: int = 750, every: int = 63) -> np.ndarray:
    """Per-asset GARCH(1,1)-t, refit every `every` days on the trailing window; filtered h between refits."""
    R = np.nan_to_num(np.asarray(R, float))
    T, N = R.shape
    h = np.full((T, N), np.nan)
    for j in range(N):
        r = R[:, j]; params = None; hj = r[:60].var()
        for t in range(1, T):
            if t >= 250 and (t - 250) % every == 0:
                params = fit_garch_t(r[max(0, t - lookback):t])
            if params is not None:
                omega, alpha, beta, _ = params
                hj = omega + alpha * r[t - 1] ** 2 + beta * hj
                h[t, j] = hj
    return h


def vol_forecast_eval(h: np.ndarray, R: np.ndarray, eps: float = 1e-10) -> dict:
    """QLIKE (Patton 2011) and MSE on log variance vs the r^2 proxy; lower is better. Pooled over assets."""
    R = np.nan_to_num(np.asarray(R, float)); h = np.asarray(h, float)
    ok = np.isfinite(h) & (h > 0)
    proxy = R ** 2 + eps
    ql = (proxy[ok] / h[ok] - np.log(proxy[ok] / h[ok]) - 1).mean()
    mse = ((np.log(h[ok]) - np.log(proxy[ok])) ** 2).mean()
    return {"qlike": float(ql), "mse_logvar": float(mse), "n": int(ok.sum()), "coverage": float(ok.mean())}


VOL_BASELINES = {"ewma_vol": ewma, "har_rv": har_rv, "garch11_t": garch11_t}
