"""Dated trade-policy events for Europe/UK 2025–2026 as Agora shocks — deterministic table, FIRST PASS (2026-09-08).

Convention (as in events_us_tariffs.py): `date` = first Xetra session on which the market could react; per economy code,
`tm` = delta import tariff by ISIC sector applied BY that economy (CBAM), `te` = delta export-tax proxy faced by that economy's
exporters (US tariffs on European goods). Rates are announced ad-valorem changes times a documented destination share of the
economy's goods exports (Eurostat 2024: US = 0.20 of extra-EU goods exports; ONS 2024: US = 0.16 of UK goods exports; autos to
the US ≈ 0.20 of EU and 0.27 of UK vehicle exports; steel to the US ≈ 0.10 of EU steel exports). `confidence` < 1 marks a rough
mapping. Every entry must be re-checked against the CRS timeline (R48549) and the EU Official Journal before any live use;
the numbers here are hand-mapped from the announcements, no language-model labels.
"""
from __future__ import annotations

from .sector_map_eu import EU_WEIGHTS

GOODS = ["01T02", "03", "05T06", "07T08", "09", "10T12", "13T15", "16", "17T18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31T33"]
SEC232 = {"24", "29"}
US_SHARE = {"EU": 0.20, "GBR": 0.16}
AUTO_SHARE = {"EU": 0.20, "GBR": 0.27}
STEEL_SHARE = {"EU": 0.10, "GBR": 0.05}
EU = {k: v for k, v in EU_WEIGHTS.items() if k != "GBR"}


def _goods(rate, exempt=SEC232):
    return {c: rate for c in GOODS if c not in exempt}


def _eu(te=None, tm=None):
    return {c: {"w": w, "te": te or {}, "tm": tm or {}} for c, w in EU.items()}


def _gbr(te=None, tm=None):
    return {"GBR": {"w": EU_WEIGHTS["GBR"], "te": te or {}, "tm": tm or {}}}


EVENTS = [
    dict(date="2025-02-11", name="US Section 232: steel & aluminium 25% global (Procl. Feb 10, after Xetra close)",
         economies={**_eu(te={"24": 0.25 * STEEL_SHARE["EU"], "25": 0.10 * STEEL_SHARE["EU"]}), **_gbr(te={"24": 0.25 * STEEL_SHARE["GBR"], "25": 0.10 * STEEL_SHARE["GBR"]})}, confidence=0.7),
    dict(date="2025-03-27", name="US Section 232: autos 25% (Procl. Mar 26, after Xetra close)",
         economies={**_eu(te={"29": 0.25 * AUTO_SHARE["EU"]}), **_gbr(te={"29": 0.25 * AUTO_SHARE["GBR"]})}, confidence=0.8),
    dict(date="2025-04-03", name="US reciprocal tariffs: EU 20%, UK 10% (EO 14257, Apr 2 after close)",
         economies={**_eu(te=_goods(0.20 * US_SHARE["EU"])), **_gbr(te=_goods(0.10 * US_SHARE["GBR"]))}, confidence=0.8),
    dict(date="2025-04-10", name="US 90-day pause: reciprocal to 10% for all but China (Apr 9, US midday)",
         economies=_eu(te=_goods(-0.10 * US_SHARE["EU"])), confidence=0.8),
    dict(date="2025-05-09", name="US–UK Economic Prosperity Deal: autos quota at 10%, steel/alu to 0% (May 8)",
         economies=_gbr(te={"29": -0.15 * AUTO_SHARE["GBR"], "24": -0.25 * STEEL_SHARE["GBR"]}), confidence=0.6),
    dict(date="2025-06-02", name="US steel & aluminium to 50% (announced Fri May 30 evening; Procl. Jun 3)",
         economies={**_eu(te={"24": 0.25 * STEEL_SHARE["EU"], "25": 0.10 * STEEL_SHARE["EU"]}), **_gbr(te={"25": 0.10 * STEEL_SHARE["GBR"]})}, confidence=0.6),
    dict(date="2025-07-28", name="EU–US Turnberry framework: 15% all-in incl. autos (from 27.5%), goods from 10% (Sun Jul 27)",
         economies=_eu(te={**_goods(0.05 * US_SHARE["EU"], exempt={"24", "29"}), "29": (0.15 - 0.275) * AUTO_SHARE["EU"]}), confidence=0.8),
    dict(date="2026-01-02", name="EU CBAM definitive phase: certificates due on steel, aluminium, cement, fertilisers, electricity imports (Jan 1)",
         economies=_eu(tm={"24": 0.04, "23": 0.03, "20": 0.02, "35": 0.01}), confidence=0.5),
    dict(date="2026-02-23", name="US Supreme Court strikes IEEPA tariffs: reciprocal 15% (EU) / 10% (UK) void (Fri Feb 20, US hours)",
         economies={**_eu(te=_goods(-0.15 * US_SHARE["EU"])), **_gbr(te=_goods(-0.10 * US_SHARE["GBR"]))}, confidence=0.6),
]


def events():
    return list(EVENTS)
