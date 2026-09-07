"""NSE fundamentals anchors, point-in-time (research/03 §14–16): last-known quarterly values as of a decision date,
per-share figures restated on the current share count, SUE = seasonal EPS difference / its trailing std."""
from __future__ import annotations

import datetime as dt
from typing import List, Optional

import numpy as np
import polars as pl


def last_known_t(df: pl.DataFrame, asof: dt.datetime, col: str, syms: List[str], n: int = 1, days: int = 400, min_n: Optional[int] = None):
    """Last n known non-null values of `col` per symbol as of `asof` (latest revision per quarter) -> [(values, event_times) | None]."""
    min_n = n if min_n is None else min_n
    f = df.filter(pl.col("knowledge_time") <= asof).sort("knowledge_time").group_by(["symbol", "event_time"]).last()
    f = f.filter(pl.col(col).is_not_null()).sort("event_time").group_by("symbol").agg(pl.col(col).tail(n).alias("v"), pl.col("event_time").tail(n).alias("t"))
    out = {r["symbol"]: ((list(r["v"]), list(r["t"])) if len(r["v"]) >= min_n and (asof - r["t"][-1]).days < days else None) for r in f.iter_rows(named=True)}
    return [out.get(s) for s in syms]


def split_adjust(eps_vt, shares_vt):
    """Restate historical per-share figures on the CURRENT share count (splits / bonuses); ratios within 2 % are left alone."""
    if eps_vt is None:
        return None
    v, t = eps_vt
    if shares_vt is None or len(shares_vt[0]) < 2:
        return v
    sv, st = shares_vt; latest = sv[-1]
    if not latest or latest <= 0:
        return v
    out = []
    for e, q in zip(v, t):
        prior = [s for s, d in zip(sv, st) if d <= q and s and s > 0]
        ratio = (prior[-1] / latest) if prior else 1.0
        out.append(e * ratio if abs(ratio - 1.0) > 0.02 else e)
    return out


def consolidated_first(res: pl.DataFrame) -> pl.DataFrame:
    """Prefer consolidated results; fall back to standalone only for companies that never report consolidated."""
    res = res.sort(["symbol", "event_time", "consolidated", "knowledge_time"])
    res_c = res.filter(pl.col("consolidated")); cons = set(res_c["symbol"].to_list())
    return pl.concat([res_c, res.filter(~pl.col("consolidated") & ~pl.col("symbol").is_in(list(cons)))])


def nse_eps_sue_asof(res: pl.DataFrame, sh: pl.DataFrame, asof: dt.datetime, syms: List[str]):
    """(TTM EPS split-adjusted, SUE clipped ±5) per symbol as of `asof`; NaN where unknown."""
    N = len(syms)
    shs = last_known_t(sh, asof, "shares_total", syms, n=40, days=100000, min_n=1) if "shares_total" in sh.columns else [None] * N
    e4 = [split_adjust(a, b) for a, b in zip(last_known_t(res, asof, "eps_diluted", syms, n=4), shs)]
    e12 = [split_adjust(a, b) for a, b in zip(last_known_t(res, asof, "eps_diluted", syms, n=12, days=400, min_n=8), shs)]
    eps = np.array([sum(v) if v else np.nan for v in e4], dtype=float); sue = np.full(N, np.nan)
    for i, v in enumerate(e12):
        if v and len(v) >= 8:
            d = np.array(v[4:]) - np.array(v[:-4]); sd = d[:-1].std(ddof=1)
            if sd > 1e-9:
                sue[i] = np.clip(d[-1] / sd, -5, 5)
    return eps, sue
