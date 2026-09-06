"""Nasdaq-screener industry -> OECD ICIO (ISIC rev.4) sector exposure, weights sum to 1.
Hand-built, coarse by design (same discipline as sector_map.ETF_EXPOSURE). Unknown industries fall back to the
screener sector. Every weight is a judgement; the test only checks coverage, validity and normalisation."""
from __future__ import annotations

import re
from typing import Dict, Iterable, List

import numpy as np

ICIO_CODES = ['01T02', '03', '05T06', '07T08', '09', '10T12', '13T15', '16', '17T18', '19', '20', '21', '22', '23', '24',
              '25', '26', '27', '28', '29', '30', '31T33', '35', '36T39', '41T43', '45T47', '49', '50', '51', '52', '53',
              '55T56', '58T60', '61', '62T63', '64T66', '68', '69T75', '77T82', '84', '85', '86T88', '90T93', '94T96', '97T98']


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


INDUSTRY_TO_ICIO: Dict[str, Dict[str, float]] = {
    # basic materials / chemicals / metals
    "metal mining": {"07T08": 1.0}, "precious metals": {"07T08": 1.0}, "major chemicals": {"20": 1.0},
    "specialty chemicals": {"20": 1.0}, "agricultural chemicals": {"20": 1.0}, "paper": {"17T18": 1.0},
    "aluminum": {"24": 1.0}, "steel/iron ore": {"24": 1.0}, "paints/coatings": {"20": 1.0},
    "mining & quarrying of nonmetallic minerals (no fuels)": {"23": 0.5, "07T08": 0.5},
    "plastic products": {"22": 1.0}, "containers/packaging": {"22": 0.5, "17T18": 0.5},
    "building products": {"23": 0.5, "16": 0.5}, "metal fabrications": {"25": 1.0}, "tools/hardware": {"25": 1.0},
    # consumer discretionary
    "business services": {"77T82": 1.0}, "diversified commercial services": {"77T82": 1.0},
    "rental/leasing companies": {"77T82": 1.0}, "misc corporate leasing services": {"77T82": 1.0},
    "professional services": {"69T75": 1.0}, "advertising": {"69T75": 1.0},
    "restaurants": {"55T56": 1.0}, "hotels/resorts": {"55T56": 1.0},
    "department/specialty retail stores": {"45T47": 1.0}, "retail-auto dealers and gas stations": {"45T47": 1.0},
    "retail: building materials": {"45T47": 1.0}, "clothing/shoe/accessory stores": {"45T47": 1.0},
    "catalog/specialty distribution": {"45T47": 1.0}, "food distributors": {"45T47": 1.0},
    "other specialty stores": {"45T47": 1.0}, "auto & home supply stores": {"45T47": 1.0},
    "consumer electronics/video chains": {"45T47": 1.0}, "durable goods": {"45T47": 1.0},
    "electronics distribution": {"45T47": 1.0}, "food chains": {"45T47": 1.0},
    "retail-drug stores and proprietary stores": {"45T47": 1.0},
    "retail: computer software & peripheral equipment": {"45T47": 1.0},
    "services-misc. amusement & recreation": {"90T93": 1.0}, "other consumer services": {"94T96": 1.0},
    "package goods/cosmetics": {"20": 1.0}, "air freight/delivery services": {"53": 0.5, "51": 0.5},
    "marine transportation": {"50": 1.0}, "homebuilding": {"41T43": 1.0},
    "telecommunications equipment": {"26": 1.0}, "auto parts:o.e.m.": {"29": 1.0}, "motor vehicles": {"29": 1.0},
    "automotive aftermarket": {"29": 0.5, "45T47": 0.5}, "auto manufacturing": {"29": 1.0},
    "transportation services": {"52": 1.0}, "industrial machinery/components": {"28": 1.0},
    "home furnishings": {"31T33": 1.0}, "recreational games/products/toys": {"31T33": 1.0},
    "oil and gas field machinery": {"28": 0.5, "09": 0.5}, "newspapers/magazines": {"58T60": 1.0},
    "broadcasting": {"58T60": 1.0}, "publishing": {"58T60": 1.0},
    "consumer electronics/appliances": {"26": 0.5, "27": 0.5}, "apparel": {"13T15": 1.0},
    "shoe manufacturing": {"13T15": 1.0}, "garments and clothing": {"13T15": 1.0},
    "military/government/technical": {"30": 0.5, "69T75": 0.5}, "industrial specialties": {"31T33": 0.5, "28": 0.5},
    # staples
    "packaged foods": {"10T12": 1.0}, "beverages (production/distribution)": {"10T12": 1.0},
    "meat/poultry/fish": {"10T12": 1.0}, "specialty foods": {"10T12": 1.0},
    "farming/seeds/milling": {"01T02": 0.5, "10T12": 0.5},
    # energy / utilities
    "oil & gas production": {"05T06": 1.0}, "integrated oil companies": {"05T06": 0.5, "19": 0.5},
    "oilfield services/equipment": {"09": 1.0}, "natural gas distribution": {"35": 1.0},
    "electrical products": {"27": 1.0}, "electric utilities: central": {"35": 1.0}, "power generation": {"35": 1.0},
    "oil/gas transmission": {"49": 1.0}, "water supply": {"36T39": 1.0}, "environmental services": {"36T39": 1.0},
    # finance / real estate
    "major banks": {"64T66": 1.0}, "finance: consumer services": {"64T66": 1.0}, "investment managers": {"64T66": 1.0},
    "investment bankers/brokers/service": {"64T66": 1.0}, "property-casualty insurers": {"64T66": 1.0},
    "life insurance": {"64T66": 1.0}, "specialty insurers": {"64T66": 1.0}, "accident &health insurance": {"64T66": 1.0},
    "finance/investors services": {"64T66": 1.0}, "real estate": {"68": 1.0}, "real estate investment trusts": {"68": 1.0},
    # health care
    "biotechnology: pharmaceutical preparations": {"21": 1.0},
    "biotechnology: biological products (no diagnostic substances)": {"21": 1.0}, "other pharmaceuticals": {"21": 1.0},
    "biotechnology: in vitro & in vivo diagnostic substances": {"21": 1.0},
    "medicinal chemicals and botanical products": {"21": 1.0},
    "medical/dental instruments": {"31T33": 0.6, "26": 0.4}, "medical specialities": {"31T33": 1.0},
    "ophthalmic goods": {"31T33": 0.5, "26": 0.5}, "hospital/nursing management": {"86T88": 1.0},
    "medical/nursing services": {"86T88": 1.0}, "misc health and biotechnology services": {"86T88": 1.0},
    "biotechnology: commercial physical & biological resarch": {"69T75": 1.0},
    "biotechnology: electromedical & electrotherapeutic apparatus": {"26": 1.0}, "precision instruments": {"26": 1.0},
    "medical electronics": {"26": 1.0}, "biotechnology: laboratory analytical instruments": {"26": 1.0},
    # industrials
    "aerospace": {"30": 1.0}, "ordnance and accessories": {"25": 0.5, "30": 0.5},
    "trucking freight/courier services": {"49": 1.0}, "railroads": {"49": 1.0},
    "integrated freight & logistics": {"52": 1.0}, "fluid controls": {"28": 1.0},
    "construction/ag equipment/trucks": {"28": 1.0}, "pollution control equipment": {"28": 1.0},
    "engineering & construction": {"41T43": 1.0},
    "water sewer pipeline comm & power line construction": {"41T43": 1.0},
    "office equipment/supplies/services": {"26": 0.5, "77T82": 0.5},
    "multi-sector companies": {"28": 0.4, "27": 0.2, "30": 0.2, "64T66": 0.2},
    # technology / telecom
    "computer software: prepackaged software": {"62T63": 0.7, "58T60": 0.3}, "semiconductors": {"26": 1.0},
    "edp services": {"62T63": 1.0}, "computer software: programming data processing": {"62T63": 1.0},
    "electronic components": {"26": 1.0}, "computer manufacturing": {"26": 1.0},
    "radio and television broadcasting and communications equipment": {"26": 1.0},
    "computer peripheral equipment": {"26": 1.0}, "computer communications equipment": {"26": 1.0},
    "cable & other pay television services": {"61": 1.0},
}

SECTOR_FALLBACK: Dict[str, Dict[str, float]] = {
    "health care": {"21": 1.0}, "technology": {"26": 0.5, "62T63": 0.5}, "finance": {"64T66": 1.0},
    "utilities": {"35": 1.0}, "energy": {"05T06": 1.0}, "industrials": {"28": 1.0},
    "consumer discretionary": {"45T47": 1.0}, "consumer staples": {"10T12": 1.0}, "basic materials": {"20": 1.0},
    "real estate": {"68": 1.0}, "telecommunications": {"61": 1.0},
}


def exposure(industry: str, sector: str = "") -> Dict[str, float]:
    """ICIO weights for one screener row; empty dict when neither industry nor sector is known."""
    w = INDUSTRY_TO_ICIO.get(_norm(industry)) or SECTOR_FALLBACK.get(_norm(sector)) or {}
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()} if tot > 0 else {}


def exposure_matrix(industries: Iterable[str], sectors: Iterable[str], codes: List[str] = ICIO_CODES) -> np.ndarray:
    """(N, S) row-stochastic exposure matrix; all-zero rows for unmapped names."""
    ci = {c: i for i, c in enumerate(codes)}
    rows = []
    for ind, sec in zip(industries, sectors):
        r = np.zeros(len(codes))
        for c, w in exposure(ind, sec).items():
            r[ci[c]] = w
        rows.append(r)
    return np.array(rows) if rows else np.zeros((0, len(codes)))
