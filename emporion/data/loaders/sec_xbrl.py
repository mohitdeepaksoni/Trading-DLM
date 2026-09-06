"""SEC XBRL company-facts (keyless, point-in-time): quarterly EPS, net income, dividends per share, shares
outstanding — every fact carries its filing date, which becomes knowledge_time. SEC asks for a descriptive
User-Agent; requests are throttled to < 10/s.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from typing import Dict, List, Optional

import polars as pl

from ...core.registry import register
from .common import http_get

UA = {"User-Agent": "Emporion-research/0.1 (open-source trading-research engine; academic use)"}
# concept name -> ordered fallbacks (taxonomy, tag); the first tag with data wins per company
CONCEPTS = {
    "eps_diluted": [("us-gaap", "EarningsPerShareDiluted"), ("us-gaap", "EarningsPerShareBasicAndDiluted")],
    "eps_basic": [("us-gaap", "EarningsPerShareBasic"), ("us-gaap", "EarningsPerShareBasicAndDiluted")],
    "net_income": [("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss"), ("us-gaap", "NetIncomeLossAvailableToCommonStockholdersBasic")],
    "dps": [("us-gaap", "CommonStockDividendsPerShareDeclared"), ("us-gaap", "CommonStockDividendsPerShareCashPaid")],
    "shares": [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding"),
               ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding")],
    "cfo": [("us-gaap", "NetCashProvidedByUsedInOperatingActivities"), ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")],
    "assets": [("us-gaap", "Assets")],
}


def ticker_cik_map() -> Dict[str, int]:
    doc = json.loads(http_get("https://www.sec.gov/files/company_tickers.json", headers=UA, retries=2, timeout=30))
    return {v["ticker"].upper(): int(v["cik_str"]) for v in doc.values()}


def _quarter_end(frame: str) -> Optional[dt.datetime]:
    # frames look like CY2025Q1 (quarterly, duration) or CY2025Q1I (instant); we use duration quarters and instants
    if not frame or not frame.startswith("CY"):
        return None
    y = int(frame[2:6])
    if "Q" in frame:
        q = int(frame[frame.index("Q") + 1])
        m = 3 * q
        return dt.datetime(y, m, 1) + dt.timedelta(days=32)
    return None


def _month_end(d: dt.datetime) -> dt.datetime:
    return (d.replace(day=1) + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)


@register("loader", "sec_facts")
def load_facts(symbols: List[str], cik_map: Optional[Dict[str, int]] = None, sleep: float = 0.12) -> pl.DataFrame:
    """One row per (symbol, concept, quarter, filing): value, event_time = quarter end, knowledge_time = filing date.
    Quarters are derived from the fact's own period (80–100-day durations; instants for shares), every filing is
    kept (first print and later restatements are separate rows — the bitemporal store resolves them), and Q4 is
    derived from the annual 10-K value minus the three reported quarters (knowledge_time = 10-K filing)."""
    cik_map = cik_map or ticker_cik_map()
    rows = []
    for s in symbols:
        cik = cik_map.get(s.upper())
        if cik is None:
            continue
        try:
            doc = json.loads(http_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=UA, retries=2, timeout=30))
        except Exception:  # noqa: BLE001
            continue
        facts = doc.get("facts", {})
        for name, tags in CONCEPTS.items():
            units = {}
            for tax, concept in tags:
                units = facts.get(tax, {}).get(concept, {}).get("units", {})
                if units:
                    break
            q, annual, ytd = {}, {}, {}
            for unit, arr in units.items():
                for f in arr:
                    end = dt.datetime.fromisoformat(f["end"]); filed = dt.datetime.fromisoformat(f["filed"])
                    start = dt.datetime.fromisoformat(f["start"]) if f.get("start") else None
                    val = float(f["val"])
                    if start is None:                                        # instant (shares outstanding)
                        rows.append({"symbol": s.upper(), "concept": name, "event_time": end, "knowledge_time": filed, "value": val, "form": f.get("form"), "derived": False})
                        continue
                    days = (end - start).days
                    if 80 <= days <= 100:
                        rows.append({"symbol": s.upper(), "concept": name, "event_time": end, "knowledge_time": filed, "value": val, "form": f.get("form"), "derived": False})
                        q.setdefault(end.date(), []).append((filed, val))
                    elif 350 <= days <= 380:
                        annual.setdefault(end.date(), []).append((filed, val, start.date()))
                    elif 170 <= days <= 190 or 260 <= days <= 280:                       # year-to-date 6M / 9M values
                        ytd.setdefault((start.date(), end.date()), []).append((filed, val))
            # derive Q2/Q3 from year-to-date differences for filers that report cumulative quarters only
            for (st_d, en_d), lst in ytd.items():
                filed, v = min(lst)
                prev = [(e, q[e]) for e in q if st_d < e < en_d]                          # quarters inside the YTD window
                prev_ytd = [(e2, ytd[(s2, e2)]) for (s2, e2) in ytd if s2 == st_d and e2 < en_d]
                if en_d in q:
                    continue
                if (en_d - st_d).days <= 190 and len(prev) == 1:                            # 6M - Q1
                    rows.append({"symbol": s.upper(), "concept": name, "event_time": dt.datetime(en_d.year, en_d.month, en_d.day), "knowledge_time": filed,
                                 "value": v - min(prev[0][1])[1], "form": "10-Q(derived from YTD)", "derived": True}); q.setdefault(en_d, []).append((filed, v - min(prev[0][1])[1]))
                elif (en_d - st_d).days > 190 and prev_ytd:                                 # 9M - 6M
                    e6, l6 = max(prev_ytd); rows.append({"symbol": s.upper(), "concept": name, "event_time": dt.datetime(en_d.year, en_d.month, en_d.day), "knowledge_time": filed,
                                 "value": v - min(l6)[1], "form": "10-Q(derived from YTD)", "derived": True}); q.setdefault(en_d, []).append((filed, v - min(l6)[1]))
            # derive Q4 = FY - (Q1+Q2+Q3) using the FIRST filing of each piece
            for fy_end, lst in annual.items():
                filed, fy_val, fy_start = min(lst)
                qs = [d for d in q if fy_start < d <= fy_end and d != fy_end]
                if len(qs) == 3 and fy_end not in q:
                    q_vals = [min(q[d])[1] for d in qs]
                    rows.append({"symbol": s.upper(), "concept": name, "event_time": dt.datetime(fy_end.year, fy_end.month, fy_end.day),
                                 "knowledge_time": filed, "value": fy_val - sum(q_vals), "form": "10-K(derived Q4)", "derived": True})
        time.sleep(sleep)
    if not rows:
        raise RuntimeError("sec_facts: nothing loaded")
    df = (pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us")))
            .unique(subset=["symbol", "concept", "event_time", "knowledge_time"]))
    bad = df.filter(pl.col("knowledge_time") < pl.col("event_time")).height       # a fact cannot be known before its period ends
    if bad:
        print(f"[sec_facts] dropped {bad} rows with filing date before period end", file=sys.stderr, flush=True)
    return df.filter(pl.col("knowledge_time") >= pl.col("event_time")).sort(["symbol", "concept", "event_time", "knowledge_time"])
