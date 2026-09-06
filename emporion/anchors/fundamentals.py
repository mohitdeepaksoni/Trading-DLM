"""Point-in-time fundamental anchors from the SEC facts table (ALL revisions; resolved at asof — research/03 §12)."""
from __future__ import annotations

import datetime as dt
import numpy as np
import polars as pl


def _known(facts: pl.DataFrame, asof: dt.datetime, concept: str) -> pl.DataFrame:
    f = facts.filter((pl.col("knowledge_time") <= asof) & (pl.col("concept") == concept))
    return f.sort("knowledge_time").group_by(["symbol", "event_time"]).last()


def _lastn(facts, asof, concept, syms, n, max_age=400):
    f = _known(facts, asof, concept).sort("event_time").group_by("symbol").agg(pl.col("value").tail(n).alias("v"), pl.col("event_time").tail(n).alias("t"))
    d = {r["symbol"]: list(r["v"]) for r in f.iter_rows(named=True) if len(r["v"]) == n and (asof - r["t"][-1]).days < max_age}
    return [d.get(s) for s in syms]


def ttm_eps(facts, asof, syms):
    return np.array([sum(v) if v else np.nan for v in _lastn(facts, asof, "eps_diluted", syms, 4)])


def shares(facts, asof, syms):
    f = facts.filter((pl.col("knowledge_time") <= asof) & (pl.col("concept") == "shares")).sort("knowledge_time").group_by("symbol").last()
    d = {r["symbol"]: r["value"] for r in f.iter_rows(named=True)}
    return np.array([d.get(s, np.nan) for s in syms])


def sue(facts, asof, syms):
    out = np.full(len(syms), np.nan)
    for i, v in enumerate(_lastn(facts, asof, "eps_diluted", syms, 12)):
        if v:
            d = np.array(v[4:]) - np.array(v[:-4]); sd = d[:-1].std(ddof=1)
            if sd > 1e-9:
                out[i] = d[-1] / sd
    return np.clip(out, -5, 5)


def accruals(facts, asof, syms):
    ni = _lastn(facts, asof, "net_income", syms, 4); cfo = _lastn(facts, asof, "cfo", syms, 4)
    a = facts.filter((pl.col("knowledge_time") <= asof) & (pl.col("concept") == "assets")).sort("knowledge_time").group_by("symbol").last()
    assets = {r["symbol"]: r["value"] for r in a.iter_rows(named=True)}
    return np.array([((sum(ni[i]) - sum(cfo[i])) / assets[s]) if (ni[i] and cfo[i] and assets.get(s)) else np.nan for i, s in enumerate(syms)])


def value_anchor(eps_ttm: np.ndarray, price: np.ndarray):
    """Fair price = EPS_ttm capitalised at the cross-sectional median earnings yield; signal = log(fair / price)."""
    ok = np.isfinite(eps_ttm) & (eps_ttm > 0) & np.isfinite(price) & (price > 0)
    out = np.full(len(price), np.nan)
    if ok.sum() >= 10:
        ey_med = np.median(eps_ttm[ok] / price[ok]); out[ok] = np.log(eps_ttm[ok] / ey_med / price[ok])
    return out
