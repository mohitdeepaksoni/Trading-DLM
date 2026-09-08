"""ESEF xBRL-JSON parser rules (research/03 §14 discipline applied to Europe/UK, 2026-09-08)."""
import datetime as dt

from emporion.data.loaders.esef import parse_facts, slim

E = "http://x/e"


def _f(concept, period, value, unit="iso4217:EUR", **axes):
    d = {"concept": concept, "entity": E, "period": period, "unit": unit}; d.update(axes)
    return {"value": value, "dimensions": d}


def _doc():
    facts = {
        # current year FY2025 (exclusive midnight stamps) and comparatives FY2024
        "f1": _f("ifrs-full:ProfitLossAttributableToOwnersOfParent", "2025-01-01T00:00:00/2026-01-01T00:00:00", "-47000000"),
        "f2": _f("ifrs-full:ProfitLossAttributableToOwnersOfParent", "2024-01-01T00:00:00/2025-01-01T00:00:00", "-147000000"),
        "f3": _f("ifrs-full:DilutedEarningsLossPerShare", "2025-01-01T00:00:00/2026-01-01T00:00:00", "-0.05", unit="iso4217:EUR/xbrli:shares"),
        "f4": _f("ifrs-full:DilutedEarningsLossPerShare", "2024-01-01T00:00:00/2025-01-01T00:00:00", "-0.15", unit="iso4217:EUR/xbrli:shares"),
        "f5": _f("ifrs-full:Revenue", "2025-01-01T00:00:00/2026-01-01T00:00:00", "6500000000"),
        "f6": _f("ifrs-full:Revenue", "2024-01-01T00:00:00/2025-01-01T00:00:00", "6418000000"),
        # a segment breakdown of revenue (explicit axis) larger than the total: must be ignored
        "f7": _f("ifrs-full:Revenue", "2025-01-01T00:00:00/2026-01-01T00:00:00", "9000000000", **{"ifrs-full:SegmentsAxis": "x:Europe"}),
        # instants: assets / equity / shares at both balance-sheet dates
        "f8": _f("ifrs-full:Assets", "2026-01-01T00:00:00", "12294000000"), "f9": _f("ifrs-full:Assets", "2025-01-01T00:00:00", "9370000000"),
        "f10": _f("ifrs-full:EquityAttributableToOwnersOfParent", "2026-01-01T00:00:00", "3650000000"),
        "f11": _f("ifrs-full:NumberOfSharesOutstanding", "2026-01-01T00:00:00", "991959494", unit="xbrli:shares"),
        # a text block (non-numeric) must not count
        "f12": {"value": "<p>notes</p>", "dimensions": {"concept": "ifrs-full:DisclosureOfNotesExplanatory", "entity": E, "period": "2025-01-01T00:00:00/2026-01-01T00:00:00", "language": "en"}},
    }
    return {"documentInfo": {"documentType": "https://xbrl.org/2021/xbrl-json"}, "facts": facts}


def test_reporting_period_is_the_latest_and_ends_are_inclusive():
    f = parse_facts(_doc())
    assert f["period_end"] == dt.datetime(2025, 12, 31) and f["period_start"] == dt.datetime(2025, 1, 1)
    assert f["period_type"] == "annual" and f["duration_days"] == 365
    assert f["eps_diluted"] == -0.05 and f["profit_owners"] == -47e6 and f["revenue"] == 6.5e9      # current year, not the comparative
    assert f["eps_diluted_prior"] == -0.15 and f["revenue_prior"] == 6.418e9 and f["profit_owners_prior"] == -147e6
    assert f["assets"] == 12.294e9 and f["equity_owners"] == 3.65e9 and f["shares_outstanding"] == 991959494
    assert f["currency"] == "EUR" and f["n_numeric"] == 11


def test_segment_breakdowns_are_ignored_and_implied_eps_checks_units():
    f = parse_facts(_doc())
    assert f["revenue"] == 6.5e9                                             # not the 9e9 segment fact
    assert abs(f["implied_eps"] - (-47e6 / 991959494)) < 1e-9 and abs(f["eps_ratio"] - 1.055) < 0.01
    # pence-stated EPS with a GBP unit shows up as a ratio near 100, never rescaled silently
    d = _doc(); d["facts"]["f3"]["value"] = "-4.7"
    g = parse_facts(d); assert 90 < g["eps_ratio"] < 110 and g["eps_diluted"] == -4.7


def test_half_year_and_date_only_periods():
    d = _doc()
    d["facts"] = {"h1": _f("ifrs-full:ProfitLoss", "2025-01-01/2025-06-30", "10"), "h2": _f("ifrs-full:ProfitLoss", "2024-01-01/2024-06-30", "8"),
                  "a1": _f("ifrs-full:ProfitLoss", "2024-01-01/2024-12-31", "20")}
    f = parse_facts(d)
    assert f["period_end"] == dt.datetime(2025, 6, 30) and f["period_type"] == "half" and f["profit"] == 10.0 and f["profit_owners_prior"] == 8.0


def test_empty_or_non_ifrs_document_returns_none_and_slim_keeps_numeric_only():
    assert parse_facts({"facts": {}}) is None
    s = slim(_doc()); assert len(s["facts"]) == 11 and "f12" not in s["facts"]
    assert parse_facts(s) == parse_facts(_doc())                              # the slim cache is parse-equivalent
