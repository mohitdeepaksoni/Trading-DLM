"""Analyst consensus snapshots from the Nasdaq API (keyless). Forward consensus has NO free point-in-time
history, so the only honest way to use it is to snapshot it daily from now on: every row is stored with
knowledge_time = snapshot time. Over months this becomes a genuine point-in-time consensus history
(the pre-registration discipline of doc 07 B.1 applied to a data source)."""
from __future__ import annotations

import datetime as dt
import json
from typing import List

import polars as pl

from ...core.registry import register
from .common import http_get
from .nasdaq import HDR


def _period_end(s: str):
    try:
        d = dt.datetime.strptime(s.strip(), "%b %Y")
        return (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
    except Exception:  # noqa: BLE001
        return None


@register("loader", "nasdaq_consensus")
def snapshot(symbols: List[str]) -> pl.DataFrame:
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    rows = []
    for s in symbols:
        try:
            doc = json.loads(http_get(f"https://api.nasdaq.com/api/quote/{s}/eps?assetclass=stocks", headers=HDR, retries=2, timeout=15, client="requests"))
        except Exception:  # noqa: BLE001
            continue
        for r in (doc.get("data") or {}).get("earningsPerShare") or []:
            pe = _period_end(r.get("period", ""))
            if pe is None or r.get("consensus") in (None, ""):
                continue
            # a forecast is an event that happens at snapshot time; the quarter it refers to is a column
            rows.append({"symbol": s, "event_time": now, "knowledge_time": now, "period_end": pe, "kind": r.get("type"),
                         "consensus_eps": float(r["consensus"]), "actual_eps": (float(r["earnings"]) if r.get("type") == "PreviousQuarter" else None)})
        try:
            d2 = (json.loads(http_get(f"https://api.nasdaq.com/api/analyst/{s}/earnings-forecast", headers=HDR, retries=1, timeout=15, client="requests")).get("data") or {})
            for r in (d2.get("yearlyForecast") or {}).get("rows") or []:
                pe = _period_end(r.get("fiscalEnd", ""))
                if pe is not None and r.get("consensusEPSForecast") not in (None, ""):
                    rows.append({"symbol": s, "event_time": now, "knowledge_time": now, "period_end": pe, "kind": "YearlyForecast",
                                 "consensus_eps": float(r["consensusEPSForecast"]), "actual_eps": None})
        except Exception:  # noqa: BLE001
            pass
    if not rows:
        raise RuntimeError("nasdaq_consensus: nothing loaded")
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us")),
                                           pl.col("period_end").cast(pl.Datetime("us")), pl.col("actual_eps").cast(pl.Float64))
