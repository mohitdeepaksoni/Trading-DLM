"""Full crypto: Coinbase Exchange public candles (no key) for every online USD pair, daily, paginated 300 bars
per call back to listing (BTC-USD from 2015). 24x7 market: a day's bar is known at the next UTC midnight."""
from __future__ import annotations

import datetime as dt
import json
import time
from typing import List, Optional

import polars as pl

from ...core.registry import register
from .common import http_get, finalize_bars

BASE = "https://api.exchange.coinbase.com"


def usd_pairs(min_volume_usd: float = 0.0) -> List[str]:
    p = json.loads(http_get(f"{BASE}/products", retries=2, timeout=20))
    return sorted(x["id"] for x in p if x.get("quote_currency") == "USD" and x.get("status") == "online" and not x.get("trading_disabled"))


@register("loader", "coinbase")
def load(symbols: Optional[List[str]] = None, start: str = "2015-01-01", sleep: float = 0.15) -> pl.DataFrame:
    symbols = symbols or usd_pairs()
    t0 = dt.datetime.fromisoformat(start).replace(tzinfo=dt.timezone.utc)
    end_all = dt.datetime.now(dt.timezone.utc)
    frames = []
    for s in symbols:
        rows, cur = [], t0
        while cur < end_all:
            nxt = min(cur + dt.timedelta(days=299), end_all)
            url = f"{BASE}/products/{s}/candles?granularity=86400&start={cur:%Y-%m-%dT%H:%M:%SZ}&end={nxt:%Y-%m-%dT%H:%M:%SZ}"
            try:
                batch = json.loads(http_get(url, retries=2, timeout=20))
            except Exception:  # noqa: BLE001
                batch = []
            rows.extend(batch)
            cur = nxt + dt.timedelta(days=1)
            time.sleep(sleep)
        if not rows:
            continue
        rows = sorted({r[0]: r for r in rows}.values())                       # dedupe by timestamp
        frames.append(pl.DataFrame({
            "symbol": [s] * len(rows),
            "event_time": [dt.datetime.fromtimestamp(r[0], dt.timezone.utc).replace(tzinfo=None) for r in rows],
            "low": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows], "open": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows], "volume": [float(r[5]) for r in rows],
        }))
    if not frames:
        raise RuntimeError("coinbase: nothing loaded")
    return finalize_bars(pl.concat(frames), "coinbase", dt.timedelta(days=1))
