import datetime as dt
import numpy as np, polars as pl
from emporion.anchors.nse import nse_eps_sue_asof, consolidated_first


def _res(sym, eps, cons=True):
    q = [dt.datetime(2022, 3, 31) + dt.timedelta(days=91 * i) for i in range(len(eps))]
    return pl.DataFrame({"symbol": [sym] * len(eps), "event_time": q, "knowledge_time": [d + dt.timedelta(days=40) for d in q],
                         "consolidated": [cons] * len(eps), "eps_diluted": [float(e) for e in eps]})


def test_sue_and_ttm_from_quarterly_eps():
    eps = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]                       # seasonal difference +1 every year -> last diff 1, std of prior diffs 0 -> no SUE (degenerate)
    res = consolidated_first(_res("A", eps)); asof = res["knowledge_time"].max() + dt.timedelta(days=1)
    ttm, sue = nse_eps_sue_asof(res, pl.DataFrame({"symbol": [], "event_time": [], "knowledge_time": []}), asof, ["A", "B"])
    assert ttm[0] == 12.0 and np.isnan(ttm[1]) and np.isnan(sue[0])   # constant seasonal differences -> zero std -> undefined SUE (not a number, never a fake 0)
    eps = [1, 2, 1, 2, 2, 3, 2, 3, 3, 4, 3, 9]                       # last quarter jumps: SUE large and positive
    res = consolidated_first(_res("A", eps)); asof = res["knowledge_time"].max() + dt.timedelta(days=1)
    ttm, sue = nse_eps_sue_asof(res, pl.DataFrame({"symbol": [], "event_time": [], "knowledge_time": []}), asof, ["A"])
    assert ttm[0] == 19.0 and sue[0] == 5.0                              # clipped at +5


def test_point_in_time_ignores_unknown_filings():
    res = consolidated_first(_res("A", [1, 1, 1, 1, 5, 5, 5, 5, 6]))
    asof = res["knowledge_time"][3] + dt.timedelta(days=1)             # only 4 quarters known
    ttm, sue = nse_eps_sue_asof(res, pl.DataFrame({"symbol": [], "event_time": [], "knowledge_time": []}), asof, ["A"])
    assert ttm[0] == 4.0 and np.isnan(sue[0])
