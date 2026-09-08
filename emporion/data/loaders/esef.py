"""ESEF / UKSEF filings via filings.xbrl.org (keyless): the point-in-time fundamentals source for Europe and the UK.
  index : one row per filing — LEI, name, country, period_end (event_time), date_added (knowledge_time), json_url, sha256
  facts : xBRL-JSON facts of one filing -> the REPORTING period's EPS / revenue / profit / equity / assets / shares
Annual (and half-year where filed) IFRS statements since FY2020; ~25k filings across EU/EEA + UK + others.

Parser discipline (research/03 §14, applied to xBRL-JSON — rules learnt on 2026-09-08):
  * a filing carries the current period AND its comparatives; the reporting period is the LATEST period end among the
    core concepts (profit / revenue / EPS), and the longest duration (<= 400 d) ending there. Taking the first or the
    earliest context returns last year's numbers (CANAL+ FY2025 file: EPS -0.15 = FY2024 instead of -0.05).
  * xBRL-JSON (OIM) period ends and instants are EXCLUSIVE midnight stamps: ".../2026-01-01T00:00:00" is the year that
    ends on 2025-12-31. Stored event_time is the inclusive date.
  * only facts whose dimensions are exactly {concept, entity, period, unit} are consolidated totals; anything with an
    explicit axis (ComponentsOfEquityAxis, segments, ...) is a breakdown and ignored.
  * the index's `period_end` is not trusted (9 filings carry 2026-12-31 .. 2032-12-31); the period comes from the facts.
  * per-share sanity: implied_eps = profit_owners / shares is stored beside the reported EPS. UK filers often state EPS
    in pence with a GBP unit; the ratio reported/implied ≈ 100 flags it (eps_ratio column) — never silently rescaled.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import polars as pl

from ...core.registry import register

BASE = "https://filings.xbrl.org"
UA = {"User-Agent": "Mozilla/5.0 (research; emporion)", "Accept": "application/vnd.api+json"}


FILE_UA = {"User-Agent": UA["User-Agent"], "Accept": "*/*"}          # the file server answers 406 to the JSON:API Accept header


def _get(url: str, retries: int = 3, timeout: int = 60, headers: Optional[dict] = None) -> bytes:
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers or UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            time.sleep(2.0 * (i + 1))
    raise RuntimeError("unreachable")


def _dt(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s[:26], fmt)
        except ValueError:
            continue
    return None


@register("loader", "esef_index")
def load_index(countries: Optional[List[str]] = None, page_size: int = 200, sleep: float = 0.3, log=print) -> pl.DataFrame:
    """Crawl the whole filings index (JSON:API pagination), keep filings with a period_end and a json_url.
    knowledge_time = date_added (the moment the filing became public on the OAM mirror); period_end_api kept for reference."""
    rows = []; url = f"{BASE}/api/filings?{urllib.parse.urlencode({'page[size]': page_size, 'sort': '-date_added', 'include': 'entity'})}"
    page = 0
    while url:
        j = json.loads(_get(url)); page += 1
        ent = {i["id"]: i["attributes"] for i in j.get("included", []) if i["type"] == "entity"}
        for f in j["data"]:
            a = f["attributes"]; e = ent.get((f.get("relationships", {}).get("entity", {}).get("data") or {}).get("id"), {})
            if countries and a.get("country") not in countries:
                continue
            pe, da = _dt(a.get("period_end")), _dt(a.get("date_added"))
            if pe is None or da is None or not a.get("json_url"):
                continue
            rows.append({"lei": e.get("identifier"), "name": e.get("name"), "country": a.get("country"), "event_time": pe, "knowledge_time": max(da, pe),
                         "json_url": a["json_url"], "sha256": a.get("sha256"), "errors": a.get("error_count"), "fxo_id": a.get("fxo_id")})
        nxt = j.get("links", {}).get("next"); url = nxt if nxt and nxt != j["links"].get("self") else None
        if page % 20 == 0:
            log(f"[esef_index] page {page}, {len(rows):,} filings so far")
        time.sleep(sleep)
    df = pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us")))
    return df.unique(["lei", "event_time", "sha256"]).sort(["country", "lei", "event_time"])


# ----- facts -----
EPS_TAGS = ("ifrs-full:DilutedEarningsLossPerShare", "ifrs-full:BasicEarningsLossPerShare")
EPS_BASIC_TAGS = ("ifrs-full:BasicEarningsLossPerShare",)
REV_TAGS = ("ifrs-full:Revenue", "ifrs-full:RevenueFromContractsWithCustomers", "ifrs-full:RevenueFromInterest", "ifrs-full:InterestRevenueCalculatedUsingEffectiveInterestMethod")
PROFIT_OWNERS_TAGS = ("ifrs-full:ProfitLossAttributableToOwnersOfParent",)
PROFIT_TAGS = ("ifrs-full:ProfitLoss",)
CFO_TAGS = ("ifrs-full:CashFlowsFromUsedInOperatingActivities",)
DIV_TAGS = ("ifrs-full:DividendsPaidClassifiedAsFinancingActivities", "ifrs-full:DividendsPaid", "ifrs-full:DividendsRecognisedAsDistributionsToOwnersOfParent")
WSHARES_TAGS = ("ifrs-full:WeightedAverageShares", "ifrs-full:AdjustedWeightedAverageShares")
EQUITY_OWNERS_TAGS = ("ifrs-full:EquityAttributableToOwnersOfParent",)
EQUITY_TAGS = ("ifrs-full:Equity",)
ASSETS_TAGS = ("ifrs-full:Assets",)
LIAB_TAGS = ("ifrs-full:Liabilities",)
SHARES_TAGS = ("ifrs-full:NumberOfSharesOutstanding", "ifrs-full:NumberOfSharesIssued")
CORE_TAGS = PROFIT_OWNERS_TAGS + PROFIT_TAGS + REV_TAGS + EPS_TAGS
ALLOWED_DIMS = {"concept", "entity", "period", "unit", "language", "noteId"}
FACT_COLUMNS = ["eps_diluted", "eps_basic", "revenue", "profit_owners", "profit", "cfo", "dividends_paid", "weighted_shares", "equity_owners",
                "equity", "assets", "liabilities", "shares_outstanding", "implied_eps", "eps_ratio", "currency", "eps_unit", "period_start",
                "period_end", "duration_days", "period_type", "eps_diluted_prior", "revenue_prior", "profit_owners_prior", "n_numeric"]


def _iso(s: str) -> Tuple[dt.datetime, bool]:
    """xBRL-JSON date/dateTime -> (datetime, is_midnight_stamp)."""
    s = s.strip()
    if "T" in s:
        return dt.datetime.fromisoformat(s[:19]), s[11:19] in ("00:00:00", "")
    return dt.datetime.fromisoformat(s[:10]), False


def _inclusive(end: dt.datetime, midnight: bool) -> dt.datetime:
    return end - dt.timedelta(days=1) if midnight else end


def _is_numeric(v) -> bool:
    x = v.get("value")
    if isinstance(x, bool) or x is None:
        return False
    if isinstance(x, (int, float)):
        return True
    try:
        float(str(x)); return True
    except ValueError:
        return False


def slim(doc: dict) -> dict:
    """Numeric facts only (text blocks are >90 % of the bytes); this is what the cache keeps."""
    return {"documentInfo": {k: doc.get("documentInfo", {}).get(k) for k in ("documentType", "taxonomy")},
            "facts": {k: v for k, v in doc.get("facts", {}).items() if _is_numeric(v)}}


def _period_type(days: int) -> str:
    if 330 <= days <= 400:
        return "annual"
    if 150 <= days <= 200:
        return "half"
    if 75 <= days <= 110:
        return "quarter"
    return "other"


def parse_facts(doc: dict) -> Optional[Dict[str, object]]:
    """Reporting-period values of one xBRL-JSON document (see module docstring for the rules). None if no core concept
    carries a duration context (e.g. an empty or non-IFRS filing)."""
    facts = doc.get("facts", {})
    dur: Dict[str, List[tuple]] = {}; inst: Dict[str, List[tuple]] = {}; n_num = 0
    for v in facts.values():
        d = v.get("dimensions", {}); c = d.get("concept"); per = d.get("period")
        if not c or not per or not _is_numeric(v):
            continue
        n_num += 1
        if any(k not in ALLOWED_DIMS for k in d):
            continue
        val = float(v["value"]); unit = d.get("unit")
        try:
            if "/" in per:
                a, b = per.split("/"); s, _ = _iso(a); e, mid = _iso(b)
                dur.setdefault(c, []).append((s, _inclusive(e, mid), val, unit))
            else:
                at, mid = _iso(per); inst.setdefault(c, []).append((_inclusive(at, mid), val, unit))
        except ValueError:
            continue
    core = [(s, e) for c in CORE_TAGS for (s, e, _, _) in dur.get(c, []) if 60 <= (e - s).days + 1 <= 400]
    if not core:
        return None
    end = max(e for _, e in core)
    length = max((e - s).days + 1 for s, e in core if e == end)
    start = end - dt.timedelta(days=length - 1)

    def dpick(tags, s=start, e=end, tol=5):
        for t in tags:
            for (a, b, val, u) in dur.get(t, []):
                if b == e and abs((a - s).days) <= tol:
                    return val, u
        return None, None

    def ipick(tags, at=end, tol=1):
        for t in tags:
            for (a, val, u) in inst.get(t, []):
                if abs((a - at).days) <= tol:
                    return val
        return None

    eps, eps_unit = dpick(EPS_TAGS); eps_b, _ = dpick(EPS_BASIC_TAGS); rev, rev_u = dpick(REV_TAGS)
    po, po_u = dpick(PROFIT_OWNERS_TAGS); pt, pt_u = dpick(PROFIT_TAGS); cfo, _ = dpick(CFO_TAGS); dv, _ = dpick(DIV_TAGS); wsh, _ = dpick(WSHARES_TAGS)
    eq_o = ipick(EQUITY_OWNERS_TAGS); eq = ipick(EQUITY_TAGS); assets = ipick(ASSETS_TAGS); liab = ipick(LIAB_TAGS); sh = ipick(SHARES_TAGS)
    profit_for_eps = po if po is not None else pt
    shares_for_eps = wsh if wsh else sh
    implied = (profit_for_eps / shares_for_eps) if (profit_for_eps is not None and shares_for_eps) else None
    ratio = (eps / implied) if (eps is not None and implied not in (None, 0.0) and abs(implied) > 1e-12) else None
    # comparatives: the same concepts for the SAME period one year earlier (seasonal; within a 20-day tolerance) —
    # for half-years the preceding half is not the comparative, the prior year's same half is
    prev_end = end - dt.timedelta(days=365)
    def ppick(tags):
        for t in tags:
            for (a, b, val, u) in dur.get(t, []):
                if abs((b - prev_end).days) <= 20 and abs(((b - a).days + 1) - length) <= 20:
                    return val
        return None
    cur = (po_u or pt_u or rev_u or "")
    return {"eps_diluted": eps, "eps_basic": eps_b, "revenue": rev, "profit_owners": po, "profit": pt, "cfo": cfo, "dividends_paid": dv,
            "weighted_shares": wsh, "equity_owners": eq_o, "equity": eq, "assets": assets, "liabilities": liab, "shares_outstanding": sh,
            "implied_eps": implied, "eps_ratio": ratio, "currency": cur.replace("iso4217:", "") or None, "eps_unit": eps_unit,
            "period_start": start, "period_end": end, "duration_days": length, "period_type": _period_type(length),
            "eps_diluted_prior": ppick(EPS_TAGS), "revenue_prior": ppick(REV_TAGS), "profit_owners_prior": ppick(PROFIT_OWNERS_TAGS) or ppick(PROFIT_TAGS),
            "n_numeric": n_num}


def _cache_path(cache_dir: str, json_url: str) -> str:
    return os.path.join(cache_dir, hashlib.sha1(json_url.encode()).hexdigest() + ".json.gz")


def fetch_slim(json_url: str, cache_dir: str, sleep: float = 0.0) -> dict:
    """Slim (numeric-only) document for one filing; downloaded once, cached gzipped (~5–50 KB instead of ~6 MB)."""
    path = _cache_path(cache_dir, json_url)
    if os.path.exists(path):
        with gzip.open(path, "rb") as f:
            return json.loads(f.read())
    raw = _get(BASE + urllib.parse.quote(json_url, safe="/"), headers=FILE_UA); s = slim(json.loads(raw))   # some paths carry spaces/umlauts
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb") as f:
        f.write(json.dumps(s, separators=(",", ":")).encode())
    os.replace(tmp, path)
    if sleep:
        time.sleep(sleep)
    return s


@register("loader", "esef_facts")
def load_facts(index: pl.DataFrame, cache_dir: str, workers: int = 6, sleep: float = 0.05, log=print) -> pl.DataFrame:
    """Download (cached) and parse the xBRL-JSON of every filing in `index`; one row per filing.
    event_time = reporting period end from the facts (inclusive date); knowledge_time = the index's date_added.
    Rows whose period end lies after the date added are impossible and dropped (logged)."""
    os.makedirs(cache_dir, exist_ok=True)
    recs = list(index.iter_rows(named=True)); rows = []; failed = 0; empty = 0; impossible = 0; t0 = time.perf_counter()

    def work(r):
        try:
            return r, parse_facts(fetch_slim(r["json_url"], cache_dir, sleep)), None
        except Exception as e:  # noqa: BLE001
            return r, None, str(e)[:80]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (r, f, err) in enumerate(ex.map(work, recs)):
            if err is not None:
                failed += 1
                if failed <= 20:
                    log(f"[esef_facts] {r['lei']} {r['json_url'][-40:]}: failed {err}")
            elif f is None:
                empty += 1
            elif f["period_end"] > r["knowledge_time"]:
                impossible += 1
            else:
                rows.append({"lei": r["lei"], "name": r["name"], "country": r["country"], "event_time": f["period_end"], "knowledge_time": r["knowledge_time"],
                             "sha256": r["sha256"], "json_url": r["json_url"], **{k: f[k] for k in FACT_COLUMNS}})
            if (i + 1) % 500 == 0:
                log(f"[esef_facts] {i+1:,}/{len(recs):,} parsed={len(rows):,} empty={empty} failed={failed} impossible={impossible} ({time.perf_counter()-t0:.0f}s)")
    log(f"[esef_facts] done: {len(rows):,} rows from {len(recs):,} filings; empty={empty} failed={failed} impossible(period_end>date_added)={impossible} ({time.perf_counter()-t0:.0f}s)")
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    return df.with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us")),
                           pl.col("period_start").cast(pl.Datetime("us")), pl.col("period_end").cast(pl.Datetime("us"))).sort(["country", "lei", "event_time", "knowledge_time"])
