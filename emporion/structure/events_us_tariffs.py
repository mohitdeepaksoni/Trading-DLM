"""Dated US tariff events 2025–2026 as Agora shocks — the deterministic event table (doc 06 §1).

Source of dates/rates: CRS R48549 "Presidential 2025 Tariff Actions: Timeline and Status"
(updated 2026-01-12; tables 2–3), Federal Register proclamations, and for 2026 the Supreme
Court IEEPA ruling (2026-02-20), the Section 122 proclamations, and the April/June 2026 Section
232 metals proclamations. NO language-model labels: every number below is an announced rate
times a documented partner import share, mapped to ISIC sectors by hand.

`date` = first trading session on which the market could react (announcements after the close
or on weekends roll to the next session). `shock` = delta import tariff by ISIC sector code for
the USA economy; `te` = delta export tax proxy for foreign retaliation on US exports.
Partner shares of US goods imports (Census 2024): CHN .134, CAN .126, MEX .155, EU .185,
JPN .045, KOR .040, IND .027, BRA .013, GBR .020, VNM .042. `confidence` reflects how
well the action maps to a sector-level tariff change (1 = clean, 0.5 = rough approximation).
"""
from __future__ import annotations

GOODS_A = ["01T02", "03"]
GOODS_I = ["05T06", "07T08", "09", "10T12", "13T15", "16", "17T18", "19", "20", "21", "22", "23", "24", "25",
           "26", "27", "28", "29", "30", "31T33"]
GOODS = GOODS_A + GOODS_I
SEC232_EXEMPT = {"24", "29"}       # steel/alu/autos excluded from reciprocal tariffs
SHARE = {"CHN": .134, "CAN": .126, "MEX": .155, "EU": .185, "JPN": .045, "KOR": .040, "IND": .027, "BRA": .013,
         "GBR": .020, "VNM": .042}


def _goods(rate: float, exempt=SEC232_EXEMPT, only=None) -> dict:
    return {c: rate for c in (only or GOODS) if c not in exempt}


def _add(*dicts) -> dict:
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0.0) + v
    return out


EVENTS = [
    dict(date="2025-02-03", name="IEEPA: Canada 25% (energy 10%), Mexico 25%, China 10% announced (EO 14193/94/95, Sat Feb 1)",
         shock=_add(_goods(0.25 * SHARE["CAN"], exempt=set(), only=[c for c in GOODS if c not in ("05T06", "19")]),
                    {"05T06": 0.10 * SHARE["CAN"], "19": 0.10 * SHARE["CAN"]},
                    _goods(0.25 * SHARE["MEX"], exempt=set()), _goods(0.10 * SHARE["CHN"], exempt=set())),
         confidence=0.8, authority="IEEPA"),
    dict(date="2025-02-04", name="Canada/Mexico tariffs paused one month (Feb 3 evening)",
         shock=_add(_goods(-0.25 * SHARE["CAN"], exempt=set(), only=[c for c in GOODS if c not in ("05T06", "19")]),
                    {"05T06": -0.10 * SHARE["CAN"], "19": -0.10 * SHARE["CAN"]}, _goods(-0.25 * SHARE["MEX"], exempt=set())),
         confidence=0.8, authority="IEEPA"),
    dict(date="2025-02-10", name="Section 232: steel & aluminium 25% global (Procl. 10895/10896)",
         shock={"24": 0.25, "25": 0.10}, confidence=1.0, authority="232"),
    dict(date="2025-03-04", name="Canada/Mexico 25% take effect; China raised to 20% (Mar 3)",
         shock=_add(_goods(0.25 * SHARE["CAN"], exempt=set(), only=[c for c in GOODS if c not in ("05T06", "19")]),
                    {"05T06": 0.10 * SHARE["CAN"], "19": 0.10 * SHARE["CAN"]},
                    _goods(0.25 * SHARE["MEX"], exempt=set()), _goods(0.10 * SHARE["CHN"], exempt=set())),
         confidence=0.8, authority="IEEPA"),
    dict(date="2025-03-06", name="USMCA-compliant goods exempted from Canada/Mexico tariffs (~half of flows)",
         shock=_add(_goods(-0.125 * SHARE["CAN"], exempt=set()), _goods(-0.125 * SHARE["MEX"], exempt=set())),
         confidence=0.6, authority="IEEPA"),
    dict(date="2025-03-27", name="Section 232: autos & parts 25% (Procl. 10908, Mar 26 evening)",
         shock={"29": 0.25}, confidence=1.0, authority="232"),
    dict(date="2025-04-03", name="'Liberation Day' reciprocal tariffs: 10% baseline + country rates (EO 14257, Apr 2 after close)",
         shock=_goods(0.22), confidence=0.9, authority="IEEPA"),
    dict(date="2025-04-09", name="China to 125%; 90-day pause of country-specific rates to 10% (intraday Apr 9)",
         shock=_add(_goods(-0.12 * (1 - SHARE["CHN"])), _goods(0.91 * SHARE["CHN"])), confidence=0.7, authority="IEEPA"),
    dict(date="2025-04-11", name="Smartphones/computers/semiconductors exempted from reciprocal tariffs",
         shock={"26": -0.12}, confidence=0.7, authority="IEEPA"),
    dict(date="2025-05-12", name="Geneva truce: China reciprocal 125% -> 10% for 90 days",
         shock=_goods(-1.15 * SHARE["CHN"]), confidence=0.9, authority="IEEPA"),
    dict(date="2025-06-02", name="Steel & aluminium 25% -> 50% (announced May 30 evening; effective Jun 4)",
         shock={"24": 0.25, "25": 0.10}, confidence=1.0, authority="232"),
    dict(date="2025-07-31", name="Copper 50% on semi-finished products (Procl. 10962, Jul 30); Brazil 40% (EO 14323)",
         shock=_add({"24": 0.125}, _goods(0.40 * SHARE["BRA"])), confidence=0.7, authority="232/IEEPA"),
    dict(date="2025-08-01", name="Revised reciprocal rates 10–41% (EO 14326, Jul 31 evening; effective Aug 7)",
         shock=_goods(0.07), confidence=0.7, authority="IEEPA"),
    dict(date="2025-08-07", name="India +25% for Russian oil (EO 14329, Aug 6)",
         shock=_goods(0.25 * SHARE["IND"]), confidence=0.9, authority="IEEPA"),
    dict(date="2025-09-30", name="Section 232: softwood lumber 10%, upholstered furniture/cabinets 25% (Procl. 10976, Sep 29)",
         shock={"16": 0.10, "31T33": 0.125}, confidence=0.9, authority="232"),
    dict(date="2025-10-20", name="Section 232: medium/heavy trucks 25%, buses 10% (Procl. 10984, Oct 17 evening)",
         shock={"29": 0.075}, confidence=0.8, authority="232"),
    dict(date="2025-11-05", name="China fentanyl tariff 20% -> 10%; truce extended one year (EO 14357/14358, Nov 4)",
         shock=_goods(-0.10 * SHARE["CHN"], exempt=set()), confidence=0.9, authority="IEEPA"),
    dict(date="2026-02-20", name="Supreme Court strikes IEEPA tariffs; Section 122 10% global imposed same day (15% announced Feb 21)",
         shock=_add(_goods(-0.17), _goods(-0.10 * SHARE["CHN"], exempt=set()),
                    _goods(-0.35 * SHARE["CAN"] * 0.5, exempt=set()), _goods(-0.25 * SHARE["MEX"] * 0.5, exempt=set()),
                    _goods(0.15, exempt=set())),
         confidence=0.6, authority="IEEPA/122"),
    dict(date="2026-04-06", name="Section 232 metals: rates raised, duty base expanded to derivatives (Procl. Apr 2; effective Apr 6)",
         shock={"24": 0.10, "25": 0.15}, confidence=0.5, authority="232"),
    dict(date="2026-05-07", name="Court of International Trade strikes Section 122 15% global tariff",
         shock=_goods(-0.15, exempt=set()), confidence=0.5, authority="122"),
]

# Foreign retaliation on US exports (export-tax proxy: rate x partner share of US goods exports; CHN share ~.07)
RETALIATION = {
    "2025-03-04": {c: 0.15 * 0.07 for c in GOODS_A},                       # China 10–15% on US farm goods (Mar 4)
    "2025-04-04": {c: 0.34 * 0.07 for c in GOODS},                         # China 34% on all US goods (Apr 4)
    "2025-04-09": {c: 0.50 * 0.07 for c in GOODS},                         # China to 84% then 125% (Apr 9–11)
    "2025-05-12": {c: -1.15 * 0.07 for c in GOODS},                        # Geneva: China 125% -> 10%
}


def events():
    out = []
    for e in EVENTS:
        e = dict(e); e["te"] = RETALIATION.get(e["date"], {}); out.append(e)
    return out
