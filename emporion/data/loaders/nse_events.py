"""NSE corporate-event feeds (keyless JSON, broadcast-timestamped) — the India-only information sources of Phase IN
(plan/PHASES-BY-MARKET.md §1 data steps 3, 6): every row carries the NSE broadcast time as knowledge_time.
  sast          Reg 29 (SAST) acquisition / sale disclosures, history 2016 →  (date-range endpoint)
  announcements corporate announcements with NSE's category (`desc`), history 2016 → (date-range endpoint; ~100k rows / year)
  insider       PIT (insider trading) disclosures, per symbol (the date-range form of the endpoint returns nothing)
  pledge        promoter pledge snapshot (current only — accumulate nightly; company NAME, no symbol in the feed)
  largedeals    today's bulk + block deals snapshot (the historical endpoint answers 503 — accumulate nightly)
Probed from the box on 2026-09-09: SAST 712 rows in Q1-2016, announcements 12k rows in Q1-2016, pledge/largedeals current only.
NSE needs a browser-like session: the loader warms the relevant page first and re-warms on 401/403/503."""
from __future__ import annotations

import datetime as dt
import time
from typing import Dict, List, Optional

import polars as pl

from ...core.registry import register

HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
       "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9", "X-Requested-With": "XMLHttpRequest",
       "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-insider-trading"}
WARM = ["https://www.nseindia.com/companies-listing/corporate-filings-insider-trading", "https://www.nseindia.com/report-detail/display-bulk-and-block-deals"]


class NseSession:
    def __init__(self, sleep: float = 1.2):
        import requests  # noqa: PLC0415
        self.s = requests.Session(); self.s.headers.update(HDR); self.sleep = sleep; self.warm()

    def warm(self):
        for u in WARM:
            try:
                self.s.get(u, timeout=20)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.8)

    def get(self, url: str, retries: int = 4):
        for i in range(retries):
            try:
                r = self.s.get(url, timeout=45)
                if r.status_code == 200 and r.text.strip():
                    time.sleep(self.sleep); return r.json()
                if r.status_code in (401, 403, 503):
                    time.sleep(4.0 * (i + 1)); self.warm(); continue
                time.sleep(2.0 * (i + 1))
            except Exception:  # noqa: BLE001
                time.sleep(3.0 * (i + 1))
        raise RuntimeError(f"NSE: no answer for {url[:90]}")


def _rows(j) -> List[dict]:
    return j.get("data", j) if isinstance(j, dict) else (j or [])


def _ts(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    s = s.strip().replace(",", "")
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y %H-%M", "%d-%b-%Y", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S", "%d%m%Y%H%M%S"):
        try:
            return dt.datetime.strptime(s[:len(dt.datetime.now().strftime(fmt))], fmt)
        except ValueError:
            continue
    return None


def _num(x) -> Optional[float]:
    try:
        return float(str(x).replace(",", "").strip()) if x not in (None, "", "-", "NA") else None
    except ValueError:
        return None


def _months(start: str, end: str):
    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    while a <= b:
        m = a.month + 1; nxt = dt.date(a.year + (m > 12), 1 if m > 12 else m, 1); c = min(b, nxt - dt.timedelta(days=1))
        yield a, c; a = c + dt.timedelta(days=1)


def _d(x: dt.date) -> str:
    return x.strftime("%d-%m-%Y")


@register("loader", "nse_sast")
def load_sast(start: str, end: str, session: Optional[NseSession] = None, log=print) -> pl.DataFrame:
    s = session or NseSession(); rows = []
    for a, b in _months(start, end):
        for r in _rows(s.get(f"https://www.nseindia.com/api/corporate-sast-reg29?index=equities&from_date={_d(a)}&to_date={_d(b)}")):
            kt = _ts(r.get("timestamp") or r.get("sysTime")); ad = (r.get("acquirerDate") or "").split(" to ")[-1].strip(); et = _ts(ad) or kt
            if not kt or not r.get("symbol"):
                continue
            rows.append({"symbol": r["symbol"] + ".NS", "application_no": str(r.get("application_no")), "event_time": min(et, kt), "knowledge_time": kt,
                         "acq_sale": r.get("acqSaleType"), "acq_type": r.get("acqType"), "mode": r.get("acquisitionMode"), "acquirer": r.get("acquirerName"),
                         "promoter": r.get("promoterType") == "Y", "shares_acq": _num(r.get("noOfShareAcq")), "shares_sale": _num(r.get("noOfShareSale")),
                         "shares_after": _num(r.get("noOfShareAft")), "pct_acq": _num(r.get("totAcqShare")), "reg": r.get("regType")})
        log(f"[nse_sast] {a}..{b}: {len(rows):,} rows so far")
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us"))) if rows else pl.DataFrame()


@register("loader", "nse_announcements")
def load_announcements(start: str, end: str, session: Optional[NseSession] = None, log=print) -> pl.DataFrame:
    s = session or NseSession(); rows = []
    for a, b in _months(start, end):
        for r in _rows(s.get(f"https://www.nseindia.com/api/corporate-announcements?index=equities&from_date={_d(a)}&to_date={_d(b)}")):
            kt = _ts(r.get("an_dt")) or _ts(r.get("sort_date")) or _ts(r.get("dt"))
            if not kt or not r.get("symbol"):
                continue
            rows.append({"symbol": r["symbol"] + ".NS", "seq_id": str(r.get("seq_id")), "event_time": kt, "knowledge_time": kt, "category": r.get("desc"),
                         "industry": r.get("smIndustry"), "has_xbrl": bool(r.get("hasXbrl")), "text": (r.get("attchmntText") or "")[:300]})
        log(f"[nse_announcements] {a}..{b}: {len(rows):,} rows so far")
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us"))) if rows else pl.DataFrame()


@register("loader", "nse_insider")
def load_insider(symbols: List[str], session: Optional[NseSession] = None, log=print) -> pl.DataFrame:
    """PIT disclosures per symbol (the endpoint's date-range form returns nothing): transaction window -> event_time, intimation broadcast -> knowledge_time."""
    s = session or NseSession(); rows = []; n0 = 0
    for i, sym in enumerate(symbols):
        base = sym.replace(".NS", "")
        try:
            data = _rows(s.get(f"https://www.nseindia.com/api/corporates-pit?index=equities&symbol={base}"))
        except RuntimeError as e:
            log(f"[nse_insider] {base}: {str(e)[:60]}"); continue
        for r in data:
            kt = _ts(r.get("date")) or _ts(r.get("intimDt")); et = _ts(r.get("acqtoDt")) or _ts(r.get("acqfromDt")) or kt
            if not kt:
                continue
            rows.append({"symbol": base + ".NS", "did": str(r.get("did")), "event_time": min(et, kt), "knowledge_time": kt, "person_category": r.get("personCategory"),
                         "acquirer": r.get("acqName"), "mode": r.get("acqMode"), "sec_type": r.get("secType"), "transaction": r.get("tdpTransactionType"),
                         "buy_qty": _num(r.get("buyQuantity")), "buy_value": _num(r.get("buyValue")), "sell_qty": _num(r.get("sellquantity")), "sell_value": _num(r.get("sellValue")),
                         "shares_before": _num(r.get("befAcqSharesNo")), "shares_after": _num(r.get("afterAcqSharesNo")), "pct_after": _num(r.get("afterAcqSharesPer"))})
        if (i + 1) % 50 == 0:
            log(f"[nse_insider] {i+1}/{len(symbols)} symbols, {len(rows):,} rows (+{len(rows)-n0})"); n0 = len(rows)
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us"))) if rows else pl.DataFrame()


@register("loader", "nse_pledge")
def load_pledge(session: Optional[NseSession] = None, log=print) -> pl.DataFrame:
    """Current promoter-pledge snapshot (no history on the endpoint): event_time = shareholding quarter, knowledge_time = broadcast."""
    s = session or NseSession(); rows = []
    for r in _rows(s.get("https://www.nseindia.com/api/corporate-pledgedata?index=equities")):
        kt = _ts(r.get("broadcastDt")); et = _ts(r.get("shp")) or kt
        if not kt:
            continue
        rows.append({"company": r.get("comName"), "event_time": min(et, kt), "knowledge_time": kt, "pct_promoter_holding": _num(r.get("percPromoterHolding")),
                     "pct_promoter_pledged": _num(r.get("percPromoterShares")), "pct_total_pledged": _num(r.get("percTotShares")), "shares_pledged": _num(r.get("numSharesPledged")),
                     "promoter_shares": _num(r.get("totPromoterHolding")), "issued_shares": _num(r.get("totIssuedShares"))})
    log(f"[nse_pledge] snapshot {len(rows):,} companies")
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us"))) if rows else pl.DataFrame()


@register("loader", "nse_largedeals")
def load_largedeals(session: Optional[NseSession] = None, log=print) -> pl.DataFrame:
    """Today's bulk + block deals (published after the close; knowledge_time = session + 1 day, conservative)."""
    s = session or NseSession(); j = s.get("https://www.nseindia.com/api/snapshot-capital-market-largedeal"); rows = []
    for kind, key in (("bulk", "BULK_DEALS_DATA"), ("block", "BLOCK_DEALS_DATA")):
        for r in (j.get(key) or []) if isinstance(j, dict) else []:
            et = _ts(r.get("date"))
            if not et or not r.get("symbol"):
                continue
            rows.append({"symbol": r["symbol"] + ".NS", "kind": kind, "client": r.get("clientName"), "side": r.get("buySell"), "event_time": et,
                         "knowledge_time": et + dt.timedelta(days=1), "qty": _num(r.get("qty")), "price": _num(r.get("watp"))})
    log(f"[nse_largedeals] {len(rows)} deals as on {j.get('as_on_date') if isinstance(j, dict) else '?'}")
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us"))) if rows else pl.DataFrame()
