"""Validator-validation for the NSE XBRL parser: undeclared base contexts and YTD exclusion (research/03 §13)."""
import datetime as dt
from emporion.data.loaders import nse_fundamentals as nf

OLD = """<?xml version="1.0"?><xbrli:xbrl xmlns:in-bse-fin="x">
<xbrli:context id="OneOperatingExpenses01D"><xbrli:period><xbrli:startDate>2018-10-01</xbrli:startDate><xbrli:endDate>2018-12-31</xbrli:endDate></xbrli:period><xbrldi:explicitMember>in-bse-fin:AMember</xbrldi:explicitMember></xbrli:context>
<xbrli:context id="FourOperatingExpenses01D"><xbrli:period><xbrli:startDate>2018-10-01</xbrli:startDate><xbrli:endDate>2018-12-31</xbrli:endDate></xbrli:period><xbrldi:explicitMember>in-bse-fin:AMember</xbrldi:explicitMember></xbrli:context>
<in-bse-fin:RevenueFromOperations contextRef="OneD" unitRef="INR">214</in-bse-fin:RevenueFromOperations>
<in-bse-fin:RevenueFromOperations contextRef="FourD" unitRef="INR">611</in-bse-fin:RevenueFromOperations>
<in-bse-fin:DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations contextRef="OneD" unitRef="INRPerShare">8.29</in-bse-fin:DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations>
<in-bse-fin:DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations contextRef="FourD" unitRef="INRPerShare">24.9</in-bse-fin:DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations>
</xbrli:xbrl>"""


def _current_quarter_contexts(ctx, q_end):
    cur = [c for c, v in ctx.items() if v["end"] == q_end and v["start"] and not v["members"]
           and 80 <= (dt.date.fromisoformat(v["end"]) - dt.date.fromisoformat(v["start"])).days <= 100
           and not c.startswith(("Two", "Three", "Four", "Five", "Six"))]
    cur.sort(key=lambda c: c != "OneD"); return cur


def test_oned_inferred_fourd_left_undefined():
    ctx = nf._contexts(OLD)
    assert ctx["OneD"] == {"start": "2018-10-01", "end": "2018-12-31", "members": []}
    assert "FourD" not in ctx                      # YTD base context must NOT inherit the quarter period
    assert ctx["OneOperatingExpenses01D"]["members"]


def test_quarter_value_not_ytd():
    ctx = nf._contexts(OLD); facts = nf._xbrl_facts(OLD)
    cur = _current_quarter_contexts(ctx, "2018-12-31")
    assert cur == ["OneD"]
    assert facts["RevenueFromOperations"][cur[0]] == 214.0        # quarter, not the 611 year-to-date figure
    assert facts["DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations"][cur[0]] == 8.29
