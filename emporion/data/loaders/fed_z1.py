"""Federal Reserve Z.1 (Financial Accounts of the United States) — corporate-equity holdings by sector, from the
data-download SDMX package (keyless zip, ~36 MB; parsed with iterparse from the 590 MB XML).

Series FL<sector>3064105.Q = market value of corporate equities held (USD millions), quarterly. Sectors used:
  15 households & nonprofits   26 rest of world          65 mutual funds        56 ETFs           55 closed-end funds
  57 private pension funds     59 federal govt retirement 22 state/local retirement (3064145)   54 life insurers
  51 P&C insurers              66 broker-dealers          89 all sectors (total market value)
knowledge_time = quarter end + 80 days (the Z.1 release lag is ~10–11 weeks; conservative), so nothing is known early.
Output columns: sector, sector_name, event_time (quarter end), knowledge_time, value_usd_mn.
"""
from __future__ import annotations

import datetime as dt
import xml.etree.ElementTree as ET
from typing import Dict

import polars as pl

from ...core.registry import register

SECTORS: Dict[str, str] = {
    "FL153064105.Q": "households", "FL263064105.Q": "rest_of_world", "FL653064100.Q": "mutual_funds",
    "FL563064100.Q": "etfs", "FL553064103.Q": "closed_end_funds", "FL573064105.Q": "private_pensions",
    "FL593064105.Q": "federal_retirement", "FL223064145.Q": "state_local_retirement", "FL543064105.Q": "life_insurers",
    "FL513064105.Q": "pc_insurers", "FL663064105.Q": "broker_dealers", "FL893064105.Q": "total",
}
NS = {"kf": "http://www.federalreserve.gov/structure/compact/Z1_Z1", "frb": "http://www.federalreserve.gov/structure/compact/common"}
RELEASE_LAG_DAYS = 80


def _quarter_end(tp: str) -> dt.datetime:
    return dt.datetime.fromisoformat(tp[:10])          # Z.1 TIME_PERIOD is already the quarter-end date (e.g. 1945-12-31)


@register("loader", "fed_z1")
def load(xml_path: str) -> pl.DataFrame:
    rows = []
    for _, el in ET.iterparse(xml_path, events=("end",)):
        if el.tag.endswith("Series"):
            name = el.attrib.get("SERIES_NAME")
            if name in SECTORS:
                for obs in el.iter():
                    if obs.tag.endswith("Obs"):
                        tp, v = obs.attrib.get("TIME_PERIOD"), obs.attrib.get("OBS_VALUE")
                        if tp and v not in (None, "", "ND", "NA"):
                            qe = _quarter_end(tp)
                            rows.append({"sector": name, "sector_name": SECTORS[name], "event_time": qe,
                                         "knowledge_time": qe + dt.timedelta(days=RELEASE_LAG_DAYS), "value_usd_mn": float(v)})
            el.clear()
    if not rows:
        raise RuntimeError("fed_z1: no series parsed")
    return pl.DataFrame(rows).with_columns(pl.col("event_time").cast(pl.Datetime("us")), pl.col("knowledge_time").cast(pl.Datetime("us")))


def mandated_share(df: pl.DataFrame) -> pl.DataFrame:
    """phi_t = (mutual funds + ETFs + closed-end + pensions + insurers) / total — the institutional 'mandated' share of
    US corporate equity; households + rest of world + broker-dealers form the residual (active/other) float."""
    w = df.pivot(index="event_time", on="sector_name", values="value_usd_mn").sort("event_time")
    inst = [c for c in ("mutual_funds", "etfs", "closed_end_funds", "private_pensions", "federal_retirement", "state_local_retirement", "life_insurers", "pc_insurers") if c in w.columns]
    return w.with_columns((sum(pl.col(c).fill_null(0) for c in inst) / pl.col("total")).alias("phi_mandated"),
                          ((pl.col("etfs").fill_null(0) + pl.col("mutual_funds").fill_null(0)) / pl.col("total")).alias("phi_funds"),
                          (pl.col("households") / pl.col("total")).alias("share_households"),
                          (pl.col("rest_of_world") / pl.col("total")).alias("share_foreign"))
