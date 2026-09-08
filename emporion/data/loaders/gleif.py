"""GLEIF (keyless): LEI -> ISINs.
Preferred route: the daily ISIN-LEI golden-copy file (https://mapping.gleif.org/api/v2/isin-lei, ~33 MB zip, one CSV LEI,ISIN):
one download covers every LEI. Fallback per LEI: https://api.gleif.org/api/v1/lei-records/{lei}/isins (~0.3 s each; the API
stalled after ~750 calls on 2026-09-08, hence the bulk route)."""
from __future__ import annotations

import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import polars as pl

from ...core.registry import register
from .common import http_get


def bulk_mapping(cache_path: Optional[str] = None, log=print) -> pl.DataFrame:
    """The latest ISIN-LEI file as a (lei, isin) frame (about 9 million rows)."""
    import os
    if cache_path and os.path.exists(cache_path):
        raw = open(cache_path, "rb").read()
    else:
        j = json.loads(http_get("https://mapping.gleif.org/api/v2/isin-lei", timeout=60)); a = j["data"][0]["attributes"]
        t0 = time.perf_counter(); raw = http_get(a["downloadLink"], timeout=600); log(f"[gleif] {a['fileName']} {len(raw)/1e6:.1f} MB ({time.perf_counter()-t0:.0f}s)")
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True); open(cache_path, "wb").write(raw)
    z = zipfile.ZipFile(io.BytesIO(raw)); name = z.namelist()[0]
    df = pl.read_csv(z.open(name).read(), schema_overrides={"LEI": pl.Utf8, "ISIN": pl.Utf8}).rename({"LEI": "lei", "ISIN": "isin"})
    return df


def isins_of(lei: str, retries: int = 3) -> List[str]:
    out = []; url = f"https://api.gleif.org/api/v1/lei-records/{lei}/isins?page[size]=100"
    while url:
        for i in range(retries):
            try:
                j = json.loads(http_get(url, headers={"Accept": "application/vnd.api+json"}, retries=1, timeout=30)); break
            except Exception:  # noqa: BLE001
                if i == retries - 1:
                    return out
                time.sleep(2.0 * (i + 1))
        out += [d["attributes"]["isin"] for d in j.get("data", [])]
        nxt = j.get("links", {}).get("next"); url = nxt if nxt and nxt != url else None
    return out


@register("loader", "gleif_isins")
def load(leis: List[str], workers: int = 4, bulk: bool = True, cache_path: Optional[str] = None, log=print) -> pl.DataFrame:
    """One row per (lei, isin) for the requested LEIs. LEIs without any ISIN are absent (logged as a count)."""
    t0 = time.perf_counter()
    if bulk:
        try:
            m = bulk_mapping(cache_path, log=log).filter(pl.col("lei").is_in(list(leis)))
            log(f"[gleif] bulk file: {len(leis)} LEIs -> {m.height} ISINs; {len(set(leis)) - m['lei'].n_unique()} LEIs without any ISIN ({time.perf_counter()-t0:.0f}s)")
            return m.sort(["lei", "isin"])
        except Exception as e:  # noqa: BLE001
            log(f"[gleif] bulk route failed ({str(e)[:80]}); falling back to per-LEI calls")
    rows = []; none = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (lei, li) in enumerate(zip(leis, ex.map(isins_of, leis))):
            if not li:
                none += 1
            rows += [{"lei": lei, "isin": s} for s in li]
            if (i + 1) % 250 == 0:
                log(f"[gleif] {i+1}/{len(leis)} LEIs, {len(rows)} ISINs, {none} without ({time.perf_counter()-t0:.0f}s)")
    log(f"[gleif] {len(leis)} LEIs -> {len(rows)} ISINs; {none} LEIs without any ISIN ({time.perf_counter()-t0:.0f}s)")
    return pl.DataFrame(rows, schema={"lei": pl.Utf8, "isin": pl.Utf8})
