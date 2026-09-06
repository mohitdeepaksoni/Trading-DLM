"""Crypto daily klines from Binance public API (no key). 24x7 market: a bar for
day D closes at D+1 00:00 UTC, which is exactly the knowledge_time."""
from __future__ import annotations

import datetime as dt
import json

import polars as pl

from ...core.registry import register
from .common import finalize_bars, http_get

CRYPTO_UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"]
# binance.com geo-blocks US IPs (HTTP 451); binance.us serves the same klines API for USD(T) pairs.
BASES = ["https://api.binance.com", "https://api.binance.us"]


def _pick_base() -> str:
    for b in BASES:
        try:
            http_get(f"{b}/api/v3/ping", retries=1, timeout=10)
            return b
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("binance: no reachable API base")


@register("loader", "binance")
def load(symbols=None, start: str = "2018-01-01") -> pl.DataFrame:
    symbols = symbols or CRYPTO_UNIVERSE
    base = _pick_base()
    start_ms = int(dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    frames = []
    for s in symbols:
        rows, cur = [], start_ms
        while True:
            url = f"{base}/api/v3/klines?symbol={s}&interval=1d&limit=1000&startTime={cur}"
            batch = json.loads(http_get(url))
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 1000:
                break
            cur = batch[-1][0] + 86_400_000
        if not rows:
            continue
        df = pl.DataFrame({
            "symbol": [s] * len(rows),
            "event_time": [dt.datetime.fromtimestamp(r[0] / 1000, dt.timezone.utc).replace(tzinfo=None) for r in rows],
            "open": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows],
            "volume": [float(r[5]) for r in rows],
        })
        frames.append(df)
    if not frames:
        raise RuntimeError("binance: no symbols loaded")
    return finalize_bars(pl.concat(frames), "binance" if "binance.com" in base else "binance_us", dt.timedelta(days=1))
