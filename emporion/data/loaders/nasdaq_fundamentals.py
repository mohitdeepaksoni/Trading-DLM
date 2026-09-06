"""Point-in-time fundamentals from the Nasdaq.com API (keyless) — the belief anchor (doc 07 A, finding 2026-09-06).

Tables (all bitemporal, knowledge_time = the date the fact became public):
  earnings   : quarterly EPS actual vs consensus; event_time = fiscal quarter end, knowledge_time = dateReported
  dividends  : cash dividends; event_time = ex-date, knowledge_time = declaration date
  universe   : screener snapshot (symbol, name, sector, industry, market cap, country); knowledge_time = as-of date
Everything is stored as first published; restatements would arrive as new rows with a later knowledge_time.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import List, Optional

import polars as pl

from ...core.registry import register
from .common import http_get
from .nasdaq import HDR


def _d(s: str) -> Optional[dt.datetime]:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%b %Y"):
        try:
            return dt.datetime.strptime(s.strip(), fmt)
        except Exception:  # noqa: BLE001
            continue
    return None


def _num(x):
    try:
        return float(str(x).replace("$", "").replace(",", "").replace("%", "").strip())
    except Exception:  # noqa: BLE001
        return None


@register("loader", "nasdaq_earnings")
def load_earnings(symbols: List[str]) -> pl.DataFrame:
    rows = []
    for s in symbols:
        try:
            doc = json.loads(http_get(f"https://api.nasdaq.com/api/company/{s}/earnings-surprise", headers=HDR, retries=2, timeout=15))
        except Exception:  # noqa: BLE001
            continue
        for r in ((doc.get("data") or {}).get("earningsSurpriseTable") or {}).get("rows") or []:
            q, rep = _d(r.get("fiscalQtrEnd", "")), _d(r.get("dateReported", ""))
            if q is None or rep is None:
                continue
            q_end = (q.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)   # month end
            rows.append({"symbol": s, "event_time": q_end, "knowledge_time": rep, "eps": _num(r.get("eps")),
                         "consensus": _num(r.get("consensusForecast")), "surprise_pct": _num(r.get("percentageSurprise"))})
    if not rows:
        raise RuntimeError("nasdaq_earnings: nothing loaded")
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us")))


@register("loader", "nasdaq_dividends")
def load_dividends(symbols: List[str], assetclass: str = "stocks") -> pl.DataFrame:
    rows = []
    for s in symbols:
        try:
            doc = json.loads(http_get(f"https://api.nasdaq.com/api/quote/{s}/dividends?assetclass={assetclass}", headers=HDR, retries=2, timeout=15))
        except Exception:  # noqa: BLE001
            continue
        for r in ((doc.get("data") or {}).get("dividends") or {}).get("rows") or []:
            ex, dec = _d(r.get("exOrEffDate", "")), _d(r.get("declarationDate", ""))
            if ex is None:
                continue
            rows.append({"symbol": s, "event_time": ex, "knowledge_time": dec or ex, "amount": _num(r.get("amount")),
                         "type": r.get("type"), "payment_date": _d(r.get("paymentDate", ""))})
    if not rows:
        raise RuntimeError("nasdaq_dividends: nothing loaded")
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us")))


@register("loader", "nasdaq_universe")
def load_universe(limit: int = 10000) -> pl.DataFrame:
    """Screener snapshot: the sector/industry map for the exposure matrix (P3b) and the universe table."""
    doc = json.loads(http_get(f"https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit={limit}&download=true", headers=HDR, retries=2, timeout=30))
    d = doc.get("data") or {}
    rows = d.get("rows") or (d.get("table") or {}).get("rows") or []
    asof = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    out = [{"symbol": r.get("symbol"), "name": r.get("name"), "sector": r.get("sector"), "industry": r.get("industry"),
            "market_cap": _num(r.get("marketCap")), "country": r.get("country"), "ipo_year": _num(r.get("ipoyear")),
            "event_time": asof, "knowledge_time": asof} for r in rows if r.get("symbol")]
    return pl.DataFrame(out).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us")))
