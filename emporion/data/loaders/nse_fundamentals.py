"""NSE India point-in-time fundamentals, holdings and flows (keyless NSE JSON API + the linked XBRL filings).

  results      : quarterly Ind-AS results — revenue, profit for the period, diluted EPS; event_time = quarter end,
                 knowledge_time = broadcast timestamp of the filing (first print; revisions keep their own rows)
  shareholding : quarterly shareholding pattern — % held by promoter group, FPI/FII, mutual funds, insurance,
                 other institutions, non-institutions (retail) — the Market Accounting Matrix rows;
                 knowledge_time = broadcast timestamp
  flows        : daily FII/FPI and DII cash-market buy/sell/net (INR crore); knowledge_time = next day 00:00
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

XBRL_CACHE = os.environ.get("EMPORION_XBRL_CACHE", "runs/nse/xbrl_cache")


def _fetch_xbrl(url: str) -> Optional[str]:
    """Download an XBRL document once; later runs read it from the cache (item 5: never refetch).
    A 404 is cached as a miss (empty marker) — old filings have no XBRL and retrying them cost 4.5 s each."""
    import urllib.error
    import urllib.request
    os.makedirs(XBRL_CACHE, exist_ok=True)
    path = os.path.join(XBRL_CACHE, hashlib.sha1(url.encode()).hexdigest() + ".xml")
    if os.path.exists(path):
        txt = open(path, encoding="utf-8", errors="replace").read()
        return txt if txt else None
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=XUA), timeout=20) as r:
            xml = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            open(path, "w").close()                                      # negative cache
            return None
        raise
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    return xml

import polars as pl

from ...core.registry import register
from .common import http_get

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
      "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/"}
XUA = {"User-Agent": UA["User-Agent"], "Accept": "*/*", "Referer": "https://www.nseindia.com/"}


def _dt(s: str) -> Optional[dt.datetime]:
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y", "%d-%B-%Y"):
        try:
            return dt.datetime.strptime(s.strip().title() if "-" in s else s, fmt)
        except Exception:  # noqa: BLE001
            continue
    return None


def _xbrl_facts(xml: str) -> Dict[str, Dict[str, float]]:
    """{tag_localname: {contextRef: value}} for numeric facts."""
    out: Dict[str, Dict[str, float]] = {}
    for m in re.finditer(r"<in-[a-z\-]+:([A-Za-z]+)[^>]*contextRef=\"([^\"]+)\"[^>]*>([^<]*)<", xml):
        tag, ctx, val = m.group(1), m.group(2), m.group(3).strip().replace(",", "")
        try:
            out.setdefault(tag, {})[ctx] = float(val)
        except ValueError:
            continue
    return out


def _contexts(xml: str) -> Dict[str, dict]:
    """contextRef -> {start, end, members:[...]} from xbrli:context blocks."""
    ctxs = {}
    for m in re.finditer(r'<xbrli:context id="([^"]+)">(.*?)</xbrli:context>', xml, flags=re.S):
        cid, body = m.group(1), m.group(2)
        sd = re.search(r"<xbrli:startDate>([^<]+)<", body); ed = re.search(r"<xbrli:endDate>([^<]+)<", body)
        inst = re.search(r"<xbrli:instant>([^<]+)<", body)
        members = re.findall(r'>([A-Za-z:\-]+Member)<', body)
        ctxs[cid] = {"start": sd.group(1) if sd else None, "end": (ed.group(1) if ed else (inst.group(1) if inst else None)), "members": members}
    # Older NSE "WEB.xml" instances reference the base contexts of the BSE results taxonomy without declaring them:
    #   OneD = current quarter, TwoD = preceding quarter, ThreeD = same quarter last year,
    #   FourD = year-to-date, FiveD = prior year-to-date, SixD = previous full year, OneI = instant at quarter end.
    # Only the dimensional children (OneOperatingExpenses01D, ...) carry a period, and those files stamp the
    # current-quarter period on ALL children (FourOperatingExpenses01D too) — so the declared periods cannot be
    # trusted for anything but "One*". We infer OneD/OneI from their children and leave the YTD contexts undefined.
    for ref in ("OneD", "OneI"):
        if ref in ctxs or f'contextRef="{ref}"' not in xml:
            continue
        kids = [v for c, v in ctxs.items() if c.startswith("One") and c.endswith(ref[-1]) and c != ref]
        if kids and len({(k["start"], k["end"]) for k in kids}) == 1:
            ctxs[ref] = {"start": kids[0]["start"], "end": kids[0]["end"], "members": []}
    return ctxs


def _safe_fetch(url: str):
    try:
        return _fetch_xbrl(url)
    except Exception:  # noqa: BLE001
        return None


def _results_listing(s: str) -> List[dict]:
    """Two NSE sources, normalised to {toDate, broadCastDate, consolidated, audited, xbrl}:
      corporates-financial-results  — legacy quarterly results (capped at 130 filings; stops at the Dec-2024 quarter)
      integrated-filing-results     — SEBI Integrated Filing (Financials), mandatory from the Mar-2025 quarter
    """
    out = []
    lst = json.loads(http_get(f"https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={s}&period=Quarterly", headers=UA, retries=2, timeout=20))
    for r in lst:
        out.append({"toDate": r.get("toDate", ""), "broadCastDate": r.get("broadCastDate", ""), "consolidated": r.get("consolidated"), "audited": r.get("audited"), "xbrl": r.get("xbrl")})
    url = f"https://www.nseindia.com/api/integrated-filing-results?index=equities&symbol={s}&period=Quarterly&type=Integrated%20Filing-%20Financials"
    j = json.loads(http_get(url, headers=UA, retries=2, timeout=20))
    data = j.get("data", []) if isinstance(j, dict) else []
    if isinstance(j, dict) and j.get("totalCount", 0) > len(data):
        j = json.loads(http_get(url + f"&size={int(j['totalCount'])}", headers=UA, retries=2, timeout=20)); data = j.get("data", data)
    for d in data:
        if d.get("type_Sub") not in (None, "Original", "Revised"):
            continue
        out.append({"toDate": d.get("qe_Date", ""), "broadCastDate": d.get("broadcast_Date", ""),
                    "consolidated": "Consolidated" if d.get("consolidated") == "Consolidated" else "Non-Consolidated",
                    "audited": d.get("audited"), "xbrl": d.get("xbrl")})
    return out


@register("loader", "nse_results")
def load_results(symbols: List[str], sleep: float = 0.6, workers: int = 3) -> pl.DataFrame:
    rows = []; failed = []
    for s in symbols:
        lst = None
        for attempt in range(3):
            try:
                lst = _results_listing(s); break
            except Exception as e:  # noqa: BLE001
                time.sleep(2.0 * (attempt + 1)); err = e
        if lst is None:
            failed.append(s); print(f"[nse_results] {s}: listing failed after 3 attempts: {err}", flush=True); continue
        # prefer consolidated; fall back to standalone when a company reports only standalone
        todo = [r for r in lst if r.get("xbrl") and _dt(r.get("toDate", "")) and _dt(r.get("broadCastDate", "")) and _dt(r["toDate"]) >= dt.datetime(2018, 1, 1)]
        with ThreadPoolExecutor(max_workers=workers) as ex:               # documents from the archive host: parallel + cached
            docs = list(ex.map(lambda r: _safe_fetch(r["xbrl"]), todo))
        for r, xml in zip(todo, docs):
            if xml is None:
                continue
            q_end = _dt(r.get("toDate", "")); bc = _dt(r.get("broadCastDate", ""))
            facts, ctx = _xbrl_facts(xml), _contexts(xml)
            # current-quarter duration context = the one whose end == quarter end and length ~ 3 months, no members
            cur = [c for c, v in ctx.items() if v["end"] == q_end.strftime("%Y-%m-%d") and v["start"] and not v["members"]
                   and 80 <= (dt.date.fromisoformat(v["end"]) - dt.date.fromisoformat(v["start"])).days <= 100
                   and not c.startswith(("Two", "Three", "Four", "Five", "Six"))]   # never a YTD / prior-period base context
            cur.sort(key=lambda c: c != "OneD")                                   # canonical current-quarter context first
            if not cur:
                continue
            def pick(*tags):                                             # first tag found in ANY current-quarter context
                for t in tags:
                    for c0 in cur:
                        if t in facts and c0 in facts[t]:
                            return facts[t][c0]
                return None
            rows.append({"symbol": f"{s}.NS", "event_time": q_end, "knowledge_time": bc,
                         "consolidated": r.get("consolidated") == "Consolidated", "audited": r.get("audited"),
                         "revenue": pick("RevenueFromOperations", "Income"), "profit": pick("ProfitLossForPeriod", "ProfitLossForPeriodFromContinuingOperations"),
                         "eps_diluted": pick("DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations", "DilutedEarningsLossPerShareFromContinuingOperations",
                                             "DilutedEarningsPerShareAfterExtraordinaryItems", "DilutedEarningsLossPerShare"),
                         "eps_basic": pick("BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations", "BasicEarningsLossPerShareFromContinuingOperations",
                                           "BasicEarningsPerShareAfterExtraordinaryItems", "BasicEarningsLossPerShare")})
        time.sleep(sleep)                                                   # polite pause between API list calls only
    if failed:
        print(f"[nse_results] FAILED symbols ({len(failed)}): {failed}", flush=True)
    if not rows:
        raise RuntimeError("nse_results: nothing loaded")
    schema = {"symbol": pl.Utf8, "event_time": pl.Datetime("us"), "knowledge_time": pl.Datetime("us"), "consolidated": pl.Boolean, "audited": pl.Utf8,
              "revenue": pl.Float64, "profit": pl.Float64, "eps_diluted": pl.Float64, "eps_basic": pl.Float64}
    return (pl.DataFrame(rows, schema=schema).unique(["symbol", "event_time", "knowledge_time", "consolidated"]).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us")),
                                            *[pl.col(c).cast(pl.Float64) for c in ("revenue", "profit", "eps_diluted", "eps_basic")])
            .filter(pl.col("knowledge_time") >= pl.col("event_time")).sort(["symbol", "event_time", "knowledge_time"]))


# XBRL category members observed in NSE shareholding filings (in-bse-shp taxonomy)
_SHP_MAP = {"ShareholdingOfPromoterAndPromoterGroup": "promoter", "PublicShareholding": "public",
            "InstitutionsDomestic": "institutions_domestic", "InstitutionsForeign": "institutions_foreign",
            "InstitutionsForeignPortfolioInvestorCategoryOne": "fpi_cat1", "MutualFundsOrUTI": "mutual_funds",
            "InsuranceCompanies": "insurance", "ProvidentFundsOrPensionFunds": "pension", "NonInstitutions": "non_institutions",
            "ResidentIndividualShareholdersHoldingNominalShareCapitalUpToRsTwoLakh": "retail_small",
            "BodiesCorporate": "bodies_corporate", "AlternativeInvestmentFunds": "aif", "CentralGovernmentStateGovernment": "government"}


@register("loader", "nse_shareholding")
def load_shareholding(symbols: List[str], sleep: float = 0.6, workers: int = 3) -> pl.DataFrame:
    rows = []
    for s in symbols:
        try:
            lst = json.loads(http_get(f"https://www.nseindia.com/api/corporate-share-holdings-master?index=equities&symbol={s}", headers=UA, retries=2, timeout=20))
        except Exception:  # noqa: BLE001
            continue
        todo = [r for r in lst if r.get("xbrl") and _dt(r.get("date", "")) and _dt(r.get("broadcastDate", "")) and _dt(r["date"]) >= dt.datetime(2018, 1, 1)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            docs = list(ex.map(lambda r: _safe_fetch(r["xbrl"]), todo))
        for r, xml in zip(todo, docs):
            if xml is None:
                continue
            q_end = _dt(r.get("date", "")); bc = _dt(r.get("broadcastDate", ""))
            facts, ctx = _xbrl_facts(xml), _contexts(xml)
            pct = facts.get("ShareholdingAsAPercentageOfTotalNumberOfShares", {})
            nsh = facts.get("NumberOfShares", {})
            rec = {"symbol": f"{s}.NS", "event_time": q_end, "knowledge_time": bc}
            for cid, v in pct.items():
                mem = [m.split(":")[-1].replace("Member", "") for m in ctx.get(cid, {}).get("members", [])]
                for key, col in _SHP_MAP.items():
                    if len(mem) == 1 and mem[0] == key and f"pct_{col}" not in rec:
                        rec[f"pct_{col}"] = v
                        if cid in nsh:
                            rec[f"shares_{col}"] = nsh[cid]
            if len(rec) > 3:
                rows.append(rec)
        time.sleep(sleep)
    if not rows:
        raise RuntimeError("nse_shareholding: nothing loaded")
    df = pl.DataFrame(rows, infer_schema_length=None)
    # Filings from Sep-2025 on report ShareholdingAsAPercentageOfTotalNumberOfShares as a FRACTION (0.143) instead of a
    # percentage (14.3). Normalise per filing: if promoter + public <= 1.5 the filing is in fractions -> x100.
    pct_cols = [c for c in df.columns if c.startswith("pct_")]
    tot = pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in ("pct_promoter", "pct_public") if c in df.columns])
    scale = pl.when(tot <= 1.5).then(100.0).otherwise(1.0)
    df = df.with_columns([(pl.col(c) * scale).alias(c) for c in pct_cols])
    df = df.with_columns(pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in ("shares_promoter", "shares_public") if c in df.columns]).alias("shares_total"))
    return (df.with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us")))
              .filter(pl.col("knowledge_time") >= pl.col("event_time")).sort(["symbol", "event_time", "knowledge_time"]))


@register("loader", "nse_flows")
def load_flows() -> pl.DataFrame:
    """Latest day's FII/DII cash-market flows (INR crore). Daily snapshot job accumulates the history."""
    lst = json.loads(http_get("https://www.nseindia.com/api/fiidiiTradeReact", headers=UA, retries=2, timeout=20))
    rows = []
    for r in lst:
        d = _dt(r["date"])
        rows.append({"category": r["category"], "event_time": d, "knowledge_time": d + dt.timedelta(days=1),
                     "buy": float(r["buyValue"]), "sell": float(r["sellValue"]), "net": float(r["netValue"])})
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us")))
