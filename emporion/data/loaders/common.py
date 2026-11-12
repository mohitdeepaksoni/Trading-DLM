from __future__ import annotations

import datetime as dt
import io
import time
import urllib.request

import polars as pl

BAR_SCHEMA = {"symbol": pl.Utf8, "event_time": pl.Datetime("us"), "knowledge_time": pl.Datetime("us"),
              "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
              "volume": pl.Float64, "source": pl.Utf8}


def http_get(url: str, headers=None, retries: int = 3, timeout: int = 30, client: str = "urllib") -> bytes:
    """client="requests" for hosts that refuse Python's urllib: from 2026-11-11 api.nasdaq.com answers urllib with 403
    and the same request through `requests` with 200 (incident 16)."""
    hdr = {"User-Agent": "Mozilla/5.0 (Emporion research loader)"}
    if headers:
        hdr.update(headers)
    last = None
    for i in range(retries):
        try:
            if client == "requests":
                import requests  # noqa: PLC0415
                r = requests.get(url, headers=hdr, timeout=(10, timeout))
                r.raise_for_status()
                return r.content
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def finalize_bars(df: pl.DataFrame, source: str, availability_lag: dt.timedelta) -> pl.DataFrame:
    """Daily bars become *knowable* only after the session closes; we record
    knowledge_time = event_time + lag (conservative default: next day 00:00)."""
    df = df.with_columns(
        pl.col("event_time").cast(pl.Datetime("us")),
        (pl.col("event_time").cast(pl.Datetime("us")) + availability_lag).alias("knowledge_time"),
        pl.lit(source).alias("source"),
    )
    return df.select(list(BAR_SCHEMA)).cast(BAR_SCHEMA).sort(["symbol", "event_time"])
