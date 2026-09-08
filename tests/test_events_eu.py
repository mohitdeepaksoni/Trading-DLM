"""EU/UK event table and STOXX sector map: weights sum to one, codes are ICIO sectors, dates are weekdays, shocks are finite."""
import datetime as dt
import numpy as np
from emporion.structure.events_eu import events, GOODS
from emporion.structure.sector_map_eu import ETF_EXPOSURE_EU, EU_WEIGHTS

ICIO = set(GOODS) | {"35", "36T39", "41T43", "45T47", "49", "50", "51", "52", "53", "55T56", "58T60", "61", "62T63", "64T66", "68", "69T75", "77T82", "84", "85", "86T88", "90T93", "94T96", "97T98"}


def test_sector_map_weights_and_codes():
    for etf, w in ETF_EXPOSURE_EU.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, etf
        assert set(w) <= ICIO, etf
    assert abs(sum(EU_WEIGHTS.values()) - 1.0) < 1e-9


def test_events_are_dated_weekdays_with_finite_shocks():
    ev = events(); assert len(ev) >= 8 and len({e["date"] for e in ev}) == len(ev)
    for e in ev:
        assert dt.date.fromisoformat(e["date"]).weekday() < 5
        assert 0 < e["confidence"] <= 1 and e["economies"]
        for c, spec in e["economies"].items():
            assert c in EU_WEIGHTS and spec["w"] > 0
            for d in (spec["tm"], spec["te"]):
                assert set(d) <= ICIO and all(np.isfinite(v) and abs(v) < 0.5 for v in d.values())
        assert any(spec["tm"] or spec["te"] for spec in e["economies"].values())
