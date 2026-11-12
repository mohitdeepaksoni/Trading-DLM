"""US daily bars from the Nasdaq.com historical-quote API (free, keyless, ~10y).
Verified reachable from OCI cloud IPs where Yahoo (429) and Stooq (JS challenge) are not."""
from __future__ import annotations

import datetime as dt
import json
import sys
import time

import polars as pl

from ...core.registry import register
from .common import finalize_bars, http_get

US_ETF_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV",
                   "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLP", "XLY", "XLB", "VNQ", "DBC"]
HDR = {"Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
       "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}


def _num(s: str) -> float:
    s = str(s).replace("$", "").replace(",", "").strip()
    return float(s) if s not in ("", "N/A", "--") else float("nan")


@register("loader", "nasdaq")
def load(symbols=None, start: str = "2010-01-01", assetclass: str = "etf",
         availability_lag: dt.timedelta = dt.timedelta(days=1)) -> pl.DataFrame:
    symbols = symbols or US_ETF_UNIVERSE
    end = dt.date.today().isoformat()
    frames = []
    misses = 0                                                     # consecutive empty symbols
    for s in symbols:
        # circuit breaker: from 2026-11-11 api.nasdaq.com stopped answering GitHub runner IPs; each symbol then burned
        # ~3 min of timeouts and the whole nightly hit the 150-min limit before any market published (incident 16)
        if misses >= 5 and not frames:
            raise RuntimeError(f"nasdaq: first {misses} symbols returned nothing; source unreachable, giving up")
        url = (f"https://api.nasdaq.com/api/quote/{s}/historical?assetclass={assetclass}"
               f"&fromdate={start}&todate={end}&limit=99999")
        rows = []
        for attempt in range(3):                                   # empty/HTML bodies happen under load: retry, then skip
            try:
                raw = http_get(url, headers=HDR, retries=2, timeout=30, client="requests")
                doc = json.loads(raw) if raw.strip() else {}
                rows = (doc.get("data") or {}).get("tradesTable", {}).get("rows") or []
                if rows:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2.0 * (attempt + 1))
        if not rows:
            print(f"[nasdaq] no data for {s}, skipped", file=sys.stderr, flush=True)
            misses += 1
            continue
        misses = 0
        frames.append(pl.DataFrame({
            "symbol": [s] * len(rows),
            "event_time": [dt.datetime.strptime(r["date"], "%m/%d/%Y") for r in rows],
            "open": [_num(r["open"]) for r in rows], "high": [_num(r["high"]) for r in rows],
            "low": [_num(r["low"]) for r in rows], "close": [_num(r["close"]) for r in rows],
            "volume": [_num(r["volume"]) for r in rows],
        }))
    if not frames:
        raise RuntimeError("nasdaq: no symbols loaded")
    return finalize_bars(pl.concat(frames).drop_nulls(), "nasdaq", availability_lag)
