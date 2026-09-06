"""NSE India cash-market bhavcopy (official daily file, free). Series EQ only.

Two file formats exist:
  - new UDiFF format (from 2024-07-08):  archives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip
  (archives.nseindia.com answers 404 on holidays; nsearchives.nseindia.com hangs instead — never use it)
  - legacy format (before that):         archives.nseindia.com/content/historical/EQUITIES/YYYY/MON/cmDDMONYYYYbhav.csv.zip
Requests are fail-fast (one attempt, short timeout: a missing file = holiday) and
fetched with a small thread pool; progress is logged every 50 sessions.
"""
from __future__ import annotations

import datetime as dt
import io
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import polars as pl

from ...core.registry import register
from .common import finalize_bars, http_get

NSE_HDR = {"Accept": "*/*", "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/"}
NEW_FORMAT_FROM = dt.date(2024, 7, 8)


def _url(d: dt.date) -> str:
    if d >= NEW_FORMAT_FROM:
        return f"https://archives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    # legacy files are served by archives.nseindia.com; nsearchives hangs (no 404) on paths it lacks
    mon = d.strftime("%b").upper()
    return f"https://archives.nseindia.com/content/historical/EQUITIES/{d:%Y}/{mon}/cm{d:%d}{mon}{d:%Y}bhav.csv.zip"


def _parse(csv: bytes, d: dt.date) -> pl.DataFrame:
    df = pl.read_csv(io.BytesIO(csv), infer_schema_length=0)
    df = df.rename({c: c.strip() for c in df.columns})
    if "TckrSymb" in df.columns:            # new format
        df = df.filter(pl.col("SctySrs").str.strip_chars() == "EQ")
        sym, o, h, l, c, v = "TckrSymb", "OpnPric", "HghPric", "LwPric", "ClsPric", "TtlTradgVol"
    else:                                    # legacy format
        df = df.filter(pl.col("SERIES").str.strip_chars() == "EQ")
        sym, o, h, l, c, v = "SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE", "TOTTRDQTY"
    return pl.DataFrame({
        "symbol": (df[sym].str.strip_chars() + ".NS").to_list(),
        "event_time": [dt.datetime(d.year, d.month, d.day)] * df.height,
        "open": df[o].cast(pl.Float64, strict=False), "high": df[h].cast(pl.Float64, strict=False),
        "low": df[l].cast(pl.Float64, strict=False), "close": df[c].cast(pl.Float64, strict=False),
        "volume": df[v].cast(pl.Float64, strict=False),
    })


HOLIDAY = "holiday"


def _one_day(d: dt.date, retries: int = 1, timeout: int = 12):
    """Returns a DataFrame; HOLIDAY for an HTTP 404 (no session file); None for a
    transient failure (timeout / 5xx / throttling) that deserves a retry pass."""
    import urllib.error
    import urllib.request
    hdr = {"User-Agent": "Mozilla/5.0 (Emporion research loader)", **NSE_HDR}
    raw = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(_url(d), headers=hdr), timeout=timeout) as r:
                raw = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return HOLIDAY
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0 + i)
    if raw is None:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            csv = z.read(z.namelist()[0])
        return _parse(csv, d)
    except Exception:  # noqa: BLE001
        return None


@register("loader", "nse")
def load(start: str, end: str, symbols=None, workers: int = 8, min_sessions: int = 10) -> pl.DataFrame:
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    days = []
    d = d0
    while d <= d1:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    frames, t0, holidays, failed = [], time.perf_counter(), 0, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (d, df) in enumerate(zip(days, ex.map(_one_day, days)), 1):
            if df is None:
                failed.append(d)
            elif isinstance(df, str):
                holidays += 1
            else:
                frames.append(df)
            if i % 100 == 0 or i == len(days):
                print(f"[nse] {i}/{len(days)} sessions  ok={len(frames)} holidays={holidays} failed={len(failed)}  "
                      f"{time.perf_counter()-t0:.0f}s", file=sys.stderr, flush=True)
    # second pass: transient failures, gently (2 workers, retries) — holidays are not retried
    if failed:
        with ThreadPoolExecutor(max_workers=2) as ex:
            for d, df in zip(failed, ex.map(lambda x: _one_day(x, retries=3, timeout=20), failed)):
                if isinstance(df, pl.DataFrame):
                    frames.append(df)
                elif isinstance(df, str):
                    holidays += 1
        print(f"[nse] retry pass: {len(failed)} failed -> ok={len(frames)} holidays={holidays}  "
              f"{time.perf_counter()-t0:.0f}s", file=sys.stderr, flush=True)
    if len(frames) < min_sessions:
        raise RuntimeError(f"nse: only {len(frames)} session files loaded")
    df = pl.concat(frames)
    if symbols:
        df = df.filter(pl.col("symbol").is_in(symbols))
    return finalize_bars(df.drop_nulls(), "nse_bhavcopy", dt.timedelta(days=1))
