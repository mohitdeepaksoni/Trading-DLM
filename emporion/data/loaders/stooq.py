"""US/global daily bars from Stooq (free CSV, no key). Symbols like 'spy.us'.
Not survivorship-free — a universe table must carry listing intervals."""
from __future__ import annotations

import datetime as dt
import io

import polars as pl

from ...core.registry import register
from .common import finalize_bars, http_get

US_ETF_UNIVERSE = ["spy", "qqq", "iwm", "efa", "eem", "tlt", "ief", "lqd", "hyg", "gld", "slv",
                   "xle", "xlf", "xlk", "xlv", "xlu", "xli", "xlp", "xly", "xlb", "vnq", "dbc"]


@register("loader", "stooq")
def load(symbols=None, availability_lag: dt.timedelta = dt.timedelta(days=1)) -> pl.DataFrame:
    symbols = symbols or [f"{s}.us" for s in US_ETF_UNIVERSE]
    frames = []
    for s in symbols:
        raw = http_get(f"https://stooq.com/q/d/l/?s={s}&i=d").decode("utf-8", "replace")
        if not raw.startswith("Date"):
            continue
        df = pl.read_csv(io.StringIO(raw), try_parse_dates=True)
        df = df.rename({c: c.lower() for c in df.columns}).rename({"date": "event_time"})
        if "volume" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("volume"))
        df = df.with_columns(pl.lit(s.upper()).alias("symbol"),
                             pl.col("event_time").cast(pl.Datetime("us")))
        frames.append(df.select(["symbol", "event_time", "open", "high", "low", "close", "volume"]))
    if not frames:
        raise RuntimeError("stooq: no symbols loaded")
    return finalize_bars(pl.concat(frames), "stooq", availability_lag)
