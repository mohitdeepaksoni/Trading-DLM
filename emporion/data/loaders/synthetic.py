"""Deterministic synthetic multi-asset market: regime-switching GBM with a
common factor, fat tails and vol clustering.  Used for tests, perf budgets,
and as the fallback when a live source is unreachable."""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from ...core.registry import register
from .common import finalize_bars


@register("loader", "synthetic")
def load(n_assets: int = 20, n_days: int = 2500, seed: int = 0, start: str = "2015-01-01",
         availability_lag: dt.timedelta = dt.timedelta(days=1)) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    # two regimes: calm (vol 12%) and stress (vol 35%), Markov switching
    P = np.array([[0.995, 0.005], [0.03, 0.97]])
    s = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        s[t] = rng.choice(2, p=P[s[t - 1]])
    vol = np.where(s == 0, 0.12, 0.35) / np.sqrt(252)
    mu = np.where(s == 0, 0.08, -0.10) / 252
    f = rng.standard_t(4, size=n_days) * vol * 0.8 + mu           # common factor, fat tails
    beta = rng.uniform(0.6, 1.4, n_assets)
    idio = rng.standard_t(4, size=(n_days, n_assets)) * (vol[:, None] * 0.7)
    r = f[:, None] * beta[None, :] + idio
    px = 100 * np.exp(np.cumsum(r, axis=0))
    dates = pl.date_range(dt.date.fromisoformat(start), dt.date.fromisoformat(start) + dt.timedelta(days=n_days * 2),
                          "1d", eager=True)
    dates = [d for d in dates if d.weekday() < 5][:n_days]
    rows = []
    for j in range(n_assets):
        rows.append(pl.DataFrame({
            "symbol": [f"SYN{j:02d}"] * n_days,
            "event_time": [dt.datetime(d.year, d.month, d.day) for d in dates],
            "open": px[:, j] * (1 + rng.normal(0, 0.002, n_days)),
            "high": px[:, j] * (1 + np.abs(rng.normal(0, 0.005, n_days))),
            "low": px[:, j] * (1 - np.abs(rng.normal(0, 0.005, n_days))),
            "close": px[:, j],
            "volume": rng.lognormal(14, 0.5, n_days),
        }))
    return finalize_bars(pl.concat(rows), "synthetic", availability_lag)
