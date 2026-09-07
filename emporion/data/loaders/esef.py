"""ESEF / UKSEF filings via filings.xbrl.org (keyless): the point-in-time fundamentals source for Europe and the UK.
  index : one row per filing — LEI, name, country, period_end (event_time), date_added (knowledge_time), json_url, sha256
  facts : xBRL-JSON facts of one filing -> EPS (diluted/basic), revenue, profit for the reporting period
Annual (and half-year where filed) IFRS statements since FY2020; ~26k filings across EU/EEA + UK + others."""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

import polars as pl

from ...core.registry import register

BASE = "https://filings.xbrl.org"
UA = {"User-Agent": "Mozilla/5.0 (research; emporion)", "Accept": "application/vnd.api+json"}


def _get(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
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
    """Crawl the whole filings index (JSON:API pagination), keep filings with a period_end and a json_url."""
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


EPS_TAGS = ("ifrs-full:DilutedEarningsLossPerShare", "ifrs-full:BasicEarningsLossPerShare")
REV_TAGS = ("ifrs-full:Revenue", "ifrs-full:RevenueFromContractsWithCustomers")
PROFIT_TAGS = ("ifrs-full:ProfitLoss", "ifrs-full:ProfitLossAttributableToOwnersOfParent")


def parse_facts(doc: dict) -> Dict[str, Optional[float]]:
    """From an xBRL-JSON document: values for the LONGEST duration context ending at the report period end, no
    dimensions other than period/entity/unit (i.e. the consolidated total, not a segment)."""
    facts = doc.get("facts", {}); best: Dict[str, tuple] = {}
    for v in facts.values():
        d = v.get("dimensions", {}); c = d.get("concept"); per = d.get("period", "")
        if not c or "/" not in per or any(k not in ("concept", "entity", "period", "unit", "language") for k in d):
            continue
        try:
            a, b = per.split("/"); start, end = dt.datetime.fromisoformat(a[:19]), dt.datetime.fromisoformat(b[:19]); val = float(v.get("value"))
        except Exception:  # noqa: BLE001
            continue
        length = (end - start).days
        key = (end, -length)
        if c not in best or key < best[c][0]:
            best[c] = (key, val, start, end)
    def pick(tags):
        for t in tags:
            if t in best:
                return best[t][1], best[t][2], best[t][3]
        return None, None, None
    eps, s, e = pick(EPS_TAGS); rev = pick(REV_TAGS)[0]; prof = pick(PROFIT_TAGS)[0]
    return {"eps": eps, "revenue": rev, "profit": prof, "period_start": s, "period_end": e, "basic_eps": pick(EPS_TAGS[1:])[0]}


@register("loader", "esef_facts")
def load_facts(index: pl.DataFrame, cache_dir: str, sleep: float = 0.2, log=print) -> pl.DataFrame:
    """Download (cached) and parse the xBRL-JSON of every filing in `index`; one row per filing."""
    import hashlib, os
    os.makedirs(cache_dir, exist_ok=True); rows = []
    for i, r in enumerate(index.iter_rows(named=True)):
        path = os.path.join(cache_dir, hashlib.sha1(r["json_url"].encode()).hexdigest() + ".json")
        if os.path.exists(path):
            raw = open(path, "rb").read()
        else:
            try:
                raw = _get(BASE + r["json_url"]); open(path, "wb").write(raw); time.sleep(sleep)
            except Exception as e:  # noqa: BLE001
                log(f"[esef_facts] {r['lei']} {r['event_time']:%Y-%m-%d}: download failed {str(e)[:60]}"); continue
        try:
            f = parse_facts(json.loads(raw))
        except Exception as e:  # noqa: BLE001
            log(f"[esef_facts] {r['lei']}: parse failed {str(e)[:60]}"); continue
        rows.append({"lei": r["lei"], "name": r["name"], "country": r["country"], "event_time": r["event_time"], "knowledge_time": r["knowledge_time"], **f})
        if (i + 1) % 200 == 0:
            log(f"[esef_facts] {i+1}/{index.height}")
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us")))
