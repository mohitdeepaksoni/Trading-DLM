"""Yahoo Finance (keyless, unofficial): ISIN -> listed symbol via the search endpoint, daily bars via yfinance.

Research-grade source for Europe/UK (plan/PHASE-EU-UK.md): the only keyless daily feed that covers LSE, Euronext,
Xetra, SIX, Nasdaq Nordic, Borsa Italiana, BME and Oslo since well before 2016. Not a live-chain source (terms of use,
no SLA): any EU/UK file that ever reaches the shadow chain needs a licensed feed (Twelve Data Grow or similar).
`close`/`volume` are Yahoo's split-adjusted series (not dividend-adjusted): per-share prices stay comparable to EPS.
Not survivorship-free by itself: the universe comes from the ESEF issuer list (all regulated-market issuers since FY2020),
so names that delisted after 2021 keep their history; names gone before 2021 are absent — stated in every report.
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import polars as pl

from ...core.registry import register
from .common import BAR_SCHEMA, finalize_bars, http_get

# ESEF country -> Yahoo exchange code of the home regulated market
HOME_EXCHANGE = {"GB": "LSE", "FR": "PAR", "NL": "AMS", "SE": "STO", "DK": "CPH", "FI": "HEL", "IT": "MIL", "ES": "MCE", "BE": "BRU",
                 "AT": "VIE", "PT": "LIS", "NO": "OSL", "DE": "GER", "IE": "ISE", "CH": "EBS", "LU": "LUX", "PL": "WSE", "IS": "ICE", "GR": "ATH"}
EURO_EXCHANGES = set(HOME_EXCHANGE.values())
# ISIN prefixes worth trying for an issuer of a given country (holding-company domiciles included); bonds (XS, US 144A) skipped
ISIN_PREFIXES = {"GB": ("GB", "JE", "GG", "IM", "BM", "KY", "VG", "IE", "NL", "LU"), "NL": ("NL", "GB", "LU", "BM", "KY"), "IE": ("IE", "GB", "JE"),
                 "LU": ("LU", "NL", "GB")}
UA = {"User-Agent": "Mozilla/5.0"}


def search(q: str, retries: int = 2) -> List[dict]:
    """Yahoo search; `requests` with (connect, read) timeouts — urllib sockets hung the thread pool on 2026-09-08."""
    import requests  # noqa: PLC0415
    url = f"https://query2.finance.yahoo.com/v1/finance/search?{urllib.parse.urlencode({'q': q, 'quotesCount': 10, 'newsCount': 0})}"
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=(5, 10))
            if r.status_code == 429:
                time.sleep(5.0 * (i + 1)); continue
            r.raise_for_status(); return r.json().get("quotes", [])
        except Exception:  # noqa: BLE001
            time.sleep(1.0 * (i + 1))
    return []


_SUFFIX = {"plc", "ag", "sa", "se", "nv", "n.v.", "s.a.", "ab", "asa", "oyj", "abp", "spa", "s.p.a.", "a/s", "as", "oy", "bv", "b.v.", "ltd", "limited",
           "group", "holdings", "holding", "company", "co", "the", "publ", "(publ)", "societe", "anonyme", "aktiengesellschaft", "kgaa", "&"}


def _tokens(name: str) -> set:
    import re  # noqa: PLC0415
    return {t for t in re.sub(r"[^a-z0-9 ]", " ", (name or "").lower()).split() if t not in _SUFFIX and len(t) > 1}


def candidates(isins: List[str], country: str, name: str = "", max_hits: int = 4, max_isins: int = 15) -> List[Dict[str, str]]:
    """EQUITY quotes for an issuer: first a search by NAME (one call; banks carry thousands of bond ISINs), then by its
    country-prefixed ISINs (capped). Home-exchange hits first, then other European exchanges. Several lines (ordinary vs
    preference, A vs B) come back as separate candidates; the panel build keeps the most traded one."""
    pref = ISIN_PREFIXES.get(country, (country,)); home = HOME_EXCHANGE.get(country)
    hits: List[Dict[str, str]] = []; seen = set()
    def add(q, isin):
        if q.get("quoteType") != "EQUITY" or not q.get("symbol") or q["symbol"] in seen:
            return
        ex = q.get("exchange", "")
        if ex != home and ex not in EURO_EXCHANGES:
            return
        seen.add(q["symbol"]); hits.append({"isin": isin, "symbol": q["symbol"], "exchange": ex, "yahoo_name": q.get("longname") or q.get("shortname") or ""})
    if name:
        nt = _tokens(name)
        for q in search(name):
            qt = _tokens(q.get("longname") or q.get("shortname") or "")
            if nt and qt and len(nt & qt) / max(1, min(len(nt), len(qt))) >= 0.6:
                add(q, "")
    if sum(h["exchange"] == home for h in hits) < 1:
        cands = sorted((s for s in isins if s[:2] in pref), key=lambda s: s[:2] != country)[:max_isins]
        for isin in cands:
            for q in search(isin):
                add(q, isin)
            if sum(h["exchange"] == home for h in hits) >= 2:
                break
    hits.sort(key=lambda h: h["exchange"] != home)
    return hits[:max_hits]


def pick_symbol(isins: List[str], country: str) -> Optional[Dict[str, str]]:
    c = candidates(isins, country, max_hits=1); return c[0] if c else None


@register("loader", "yahoo_symbols")
def symbols(issuers: pl.DataFrame, isins: pl.DataFrame, workers: int = 4, log=print) -> pl.DataFrame:
    """issuers: (lei, name, country); isins: (lei, isin) -> one row per (issuer, candidate listing); rank 0 = first home-exchange hit."""
    by = {r["lei"]: r["isin"] for r in isins.group_by("lei").agg(pl.col("isin")).iter_rows(named=True)}
    recs = list(issuers.iter_rows(named=True)); rows = []; t0 = time.perf_counter(); n_done = 0
    from concurrent.futures import as_completed  # noqa: PLC0415
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(candidates, by.get(r["lei"], []), r["country"], r["name"]): r for r in recs}
        for fu in as_completed(futs):
            r = futs[fu]; n_done += 1
            try:
                hits = fu.result()
            except Exception as e:  # noqa: BLE001
                log(f"[yahoo_symbols] {r['lei']}: {str(e)[:60]}"); hits = []
            for k, hit in enumerate(hits):
                rows.append({"lei": r["lei"], "name": r["name"], "country": r["country"], **hit, "rank": k})
            if n_done % 250 == 0:
                log(f"[yahoo_symbols] {n_done}/{len(recs)} issuers, {len(rows)} candidate lines ({time.perf_counter()-t0:.0f}s)")
    log(f"[yahoo_symbols] {len(recs)} issuers -> {len({r['lei'] for r in rows})} with a listed equity, {len(rows)} candidate lines ({time.perf_counter()-t0:.0f}s)")
    return pl.DataFrame(rows, schema={"lei": pl.Utf8, "name": pl.Utf8, "country": pl.Utf8, "isin": pl.Utf8, "symbol": pl.Utf8, "exchange": pl.Utf8, "yahoo_name": pl.Utf8, "rank": pl.Int32})


@register("loader", "yahoo")
def load(symbols: List[str], start: str = "2016-01-01", batch: int = 200, availability_lag: dt.timedelta = dt.timedelta(days=1), log=print) -> pl.DataFrame:
    """Daily bars for many symbols through yfinance (batched, threaded). Rows with a null close are dropped."""
    import yfinance as yf  # noqa: PLC0415  (optional dependency: research path only)
    frames = []; t0 = time.perf_counter()
    for b in range(0, len(symbols), batch):
        chunk = symbols[b:b + batch]
        for attempt in range(3):
            try:
                df = yf.download(chunk, start=start, auto_adjust=False, actions=False, threads=True, progress=False, group_by="ticker"); break
            except Exception as e:  # noqa: BLE001
                log(f"[yahoo] batch {b} attempt {attempt+1} failed: {str(e)[:80]}"); time.sleep(10 * (attempt + 1)); df = None
        if df is None or df.empty:
            continue
        for s in chunk:
            try:
                d = df[s] if len(chunk) > 1 else df
            except KeyError:
                continue
            d = d.dropna(subset=["Close"])
            if d.empty:
                continue
            frames.append(pl.DataFrame({"symbol": [s] * len(d), "event_time": [x.to_pydatetime().replace(tzinfo=None) for x in d.index],
                                        "open": d["Open"].astype(float).to_list(), "high": d["High"].astype(float).to_list(), "low": d["Low"].astype(float).to_list(),
                                        "close": d["Close"].astype(float).to_list(), "volume": d["Volume"].astype(float).to_list()}))
        log(f"[yahoo] {min(b + batch, len(symbols))}/{len(symbols)} symbols, {sum(f.height for f in frames):,} bars ({time.perf_counter()-t0:.0f}s)")
        time.sleep(1.0)
    if not frames:
        raise RuntimeError("yahoo: no symbols loaded")
    return finalize_bars(pl.concat(frames), "yahoo", availability_lag).select(list(BAR_SCHEMA))
