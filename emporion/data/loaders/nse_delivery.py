"""NSE security-wise delivery position (the MTO file, keyless): per symbol and session, quantity traded, deliverable quantity
and the delivery percentage — a liquidity / conviction state no other market in the project has (plan/PHASES-BY-MARKET.md
Phase IN, data step 2). One file per session since 2003: https://archives.nseindia.com/archives/equities/mto/MTO_DDMMYYYY.DAT
(record type 20 rows: sr no, symbol, series, traded qty, deliverable qty, %). Series EQ only. knowledge_time = session + 1 day
(the file is published in the evening; next-day knowledge is the conservative choice used for bars too). 404 = holiday,
403 = throttling (NSE blocks bulk downloads from cloud IPs) -> retried, never treated as a holiday."""
from __future__ import annotations

import datetime as dt
import io
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import polars as pl

from ...core.registry import register

HDR = {"User-Agent": "Mozilla/5.0 (Emporion research loader)", "Accept": "*/*", "Referer": "https://www.nseindia.com/"}
HOLIDAY = "holiday"


def _url(d: dt.date) -> str:
    return f"https://archives.nseindia.com/archives/equities/mto/MTO_{d:%d%m%Y}.DAT"


def parse_mto(raw: bytes, d: dt.date) -> pl.DataFrame:
    rows = []
    for line in raw.decode("utf-8", "replace").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 7 and p[0] == "20" and p[3] == "EQ":
            try:
                rows.append((p[2] + ".NS", float(p[4]), float(p[5]), float(p[6])))
            except ValueError:
                continue
    df = pl.DataFrame(rows, schema={"symbol": pl.Utf8, "qty_traded": pl.Float64, "deliv_qty": pl.Float64, "deliv_per": pl.Float64}, orient="row")
    t = dt.datetime(d.year, d.month, d.day)
    return df.with_columns(pl.lit(t).alias("event_time"), pl.lit(t + dt.timedelta(days=1)).alias("knowledge_time"))


def _one_day(d: dt.date, retries: int = 2, timeout: int = 15):
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(_url(d), headers=HDR), timeout=timeout) as r:
                return parse_mto(r.read(), d)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return HOLIDAY
            time.sleep(3.0 + 3.0 * i)
        except Exception:  # noqa: BLE001
            time.sleep(1.0 + i)
    return None


@register("loader", "nse_delivery")
def load(start: str, end: str, workers: int = 4, log=print) -> pl.DataFrame:
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    days = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]; days = [d for d in days if d.weekday() < 5]
    frames: List[pl.DataFrame] = []; holidays = 0; failed: List[dt.date] = []; t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d, df in zip(days, ex.map(_one_day, days)):
            if df is None:
                failed.append(d)
            elif isinstance(df, str):
                holidays += 1
            else:
                frames.append(df)
    if failed:
        with ThreadPoolExecutor(max_workers=2) as ex:
            for d, df in zip(failed, ex.map(lambda x: _one_day(x, retries=4, timeout=25), failed)):
                if isinstance(df, pl.DataFrame):
                    frames.append(df)
                elif isinstance(df, str):
                    holidays += 1
    log(f"[nse_delivery] {start}..{end}: {len(frames)} sessions, {holidays} holidays, {len(failed)} retried ({time.perf_counter()-t0:.0f}s)")
    if not frames:
        return pl.DataFrame(schema={"symbol": pl.Utf8, "qty_traded": pl.Float64, "deliv_qty": pl.Float64, "deliv_per": pl.Float64, "event_time": pl.Datetime("us"), "knowledge_time": pl.Datetime("us")})
    return pl.concat(frames).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us"))).sort(["symbol", "event_time"])
