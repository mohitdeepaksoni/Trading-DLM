"""SEC Form 13F structured data sets (keyless): per-stock institutional ownership, point-in-time, plus the
manager × stock holdings matrix for asset embeddings (Gabaix–Koijen–Richmond–Yogo).

Each quarterly window zip has INFOTABLE.tsv (holdings, CUSIP, shares) and SUBMISSION.tsv (filing date, CIK,
period). Amendments: the latest filing per (CIK, period) wins. knowledge_time per (CUSIP, period) = the LATEST
filing date among the holders of that CUSIP for that period (conservative: nothing is known before all filers have
reported; the 13F deadline is 45 days after quarter end). CUSIP -> ticker from the SEC fails-to-deliver files.
"""
from __future__ import annotations

import datetime as dt
import io
import os
import re
import zipfile
from typing import Dict, List, Tuple

import polars as pl

from ...core.registry import register
from .common import http_get

UA = {"User-Agent": "Emporion-research/0.1 (open-source trading-research engine; academic use)"}
INDEX = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"


def window_urls() -> List[str]:
    html = http_get(INDEX, headers=UA, retries=2, timeout=60).decode("utf-8", "replace")
    rel = re.findall(r'href="(/files/structureddata/data/form-13f-data-sets/[^"]+\.zip)"', html)
    return ["https://www.sec.gov" + r for r in rel]                          # newest first as listed


def cusip_symbol_map(months: int = 6) -> Dict[str, str]:
    """Union of CUSIP->SYMBOL from the last `months` fails-to-deliver files (two per month)."""
    out: Dict[str, str] = {}
    d = dt.date.today().replace(day=1)
    for _ in range(months):
        for half in ("a", "b"):
            try:
                raw = http_get(f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{d:%Y%m}{half}.zip", headers=UA, retries=1, timeout=60)
                with zipfile.ZipFile(io.BytesIO(raw)) as z:
                    txt = z.read(z.namelist()[0]).decode("latin-1")
                for line in txt.splitlines()[1:]:
                    p = line.split("|")
                    if len(p) >= 3 and p[1] and p[2]:
                        out.setdefault(p[1].strip(), p[2].strip())
            except Exception:  # noqa: BLE001
                continue
        d = (d - dt.timedelta(days=1)).replace(day=1)
    return out


def _date(s: str) -> dt.datetime:
    return dt.datetime.strptime(s.strip(), "%d-%b-%Y")


@register("loader", "sec_13f_window")
def load_window(url: str, workdir: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (aggregate per CUSIP x period, manager x CUSIP matrix rows) for one dataset window."""
    os.makedirs(workdir, exist_ok=True)
    zpath = os.path.join(workdir, os.path.basename(url))
    if not os.path.exists(zpath):
        with open(zpath, "wb") as f:
            f.write(http_get(url, headers=UA, retries=2, timeout=600))
    with zipfile.ZipFile(zpath) as z:
        names = {os.path.basename(n).upper(): n for n in z.namelist()}          # some windows nest or lowercase the members
        sub = pl.read_csv(io.BytesIO(z.read(names["SUBMISSION.TSV"])), separator="\t", infer_schema_length=0)
        info = pl.read_csv(io.BytesIO(z.read(names["INFOTABLE.TSV"])), separator="\t", infer_schema_length=0,
                           columns=["ACCESSION_NUMBER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"])
    sub = (sub.filter(pl.col("SUBMISSIONTYPE").is_in(["13F-HR", "13F-HR/A"]))
              .with_columns(pl.col("FILING_DATE").map_elements(_date, return_dtype=pl.Datetime("us")).alias("filed"),
                            pl.col("PERIODOFREPORT").map_elements(_date, return_dtype=pl.Datetime("us")).alias("period")))
    # latest filing per (CIK, period): amendments supersede
    sub = sub.sort("filed").group_by(["CIK", "period"]).last().select(["ACCESSION_NUMBER", "CIK", "filed", "period"])
    info = (info.filter((pl.col("SSHPRNAMTTYPE") == "SH") & pl.col("PUTCALL").is_null())
                .with_columns(pl.col("SSHPRNAMT").cast(pl.Float64, strict=False), pl.col("VALUE").cast(pl.Float64, strict=False))
                .join(sub, on="ACCESSION_NUMBER", how="inner"))
    matrix = info.group_by(["CIK", "CUSIP", "period"]).agg(pl.col("SSHPRNAMT").sum().alias("shares"), pl.col("VALUE").sum().alias("value"), pl.col("filed").max().alias("filed"))
    agg = (matrix.group_by(["CUSIP", "period"]).agg(pl.col("shares").sum().alias("inst_shares"), pl.col("value").sum().alias("inst_value"),
                                                    pl.len().alias("n_holders"), pl.col("filed").max().alias("knowledge_time"))
                 .rename({"period": "event_time"}))
    return agg, matrix
