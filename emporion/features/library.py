"""P0 feature set — deliberately classical; every one is a doc-01 formula."""
from __future__ import annotations

import polars as pl

from .registry import feature


def _last_per_symbol(df: pl.DataFrame, expr: pl.Expr, min_rows: int) -> pl.DataFrame:
    return (df.sort(["symbol", "event_time"])
              .group_by("symbol", maintain_order=True)
              .agg(pl.col("event_time").last(), expr.alias("value"), pl.len().alias("n"))
              .filter(pl.col("n") >= min_rows)
              .select(["symbol", "event_time", "value"]))


@feature("mom_12_1", lag="1d", inputs=("bars",))
def momentum_12_1(ctx) -> pl.DataFrame:
    """12-month return skipping the last month (Jegadeesh–Titman), on close."""
    bars = ctx.table("bars", columns=["close"])
    e = (pl.col("close").shift(21).last() / pl.col("close").shift(252).last() - 1.0)
    return _last_per_symbol(bars, e, 253)


@feature("vol_20", lag="1d", inputs=("bars",))
def realised_vol_20(ctx) -> pl.DataFrame:
    """20-day close-to-close realised volatility, annualised."""
    bars = ctx.table("bars", columns=["close"])
    e = ((pl.col("close") / pl.col("close").shift(1)).log().tail(20).std() * (252 ** 0.5))
    return _last_per_symbol(bars, e, 21)


@feature("rev_5", lag="1d", inputs=("bars",))
def short_term_reversal_5(ctx) -> pl.DataFrame:
    """5-day return (negative sign = reversal signal)."""
    bars = ctx.table("bars", columns=["close"])
    e = -(pl.col("close").last() / pl.col("close").shift(5).last() - 1.0)
    return _last_per_symbol(bars, e, 6)
