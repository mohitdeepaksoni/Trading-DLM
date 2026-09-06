"""CoinGecko free API: daily SNAPSHOT of market cap, circulating/total supply and volume for the top coins.
No free point-in-time history, so knowledge_time = snapshot time; the store accumulates the history from today.
This is the crypto 'supply' input for the clearing model (s) and the size/liquidity state."""
from __future__ import annotations

import datetime as dt
import json
import time

import polars as pl

from ...core.registry import register
from .common import http_get


@register("loader", "coingecko_snapshot")
def snapshot(pages: int = 4, per_page: int = 250) -> pl.DataFrame:
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    rows = []
    for page in range(1, pages + 1):
        try:
            m = json.loads(http_get(f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={per_page}&page={page}",
                                    retries=2, timeout=30))
        except Exception:  # noqa: BLE001
            break
        for c in m:
            rows.append({"coin_id": c["id"], "symbol": str(c["symbol"]).upper(), "event_time": now, "knowledge_time": now,
                         "price": c.get("current_price"), "market_cap": c.get("market_cap"), "circulating_supply": c.get("circulating_supply"),
                         "total_supply": c.get("total_supply"), "volume_24h": c.get("total_volume"), "rank": c.get("market_cap_rank")})
        time.sleep(2.5)                                                         # free-tier rate limit
    if not rows:
        raise RuntimeError("coingecko: nothing loaded")
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us")),
                                           *[pl.col(c).cast(pl.Float64) for c in ("price", "market_cap", "circulating_supply", "total_supply", "volume_24h")])
