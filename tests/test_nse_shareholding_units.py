"""Shareholding % units (fraction vs percent) and split adjustment of per-share figures."""
import datetime as dt, sys, os
import polars as pl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from p4_realdata_nse import split_adjust


def test_split_adjust_restates_pre_split_eps():
    q = [dt.datetime(2024, 3, 31), dt.datetime(2024, 6, 30), dt.datetime(2024, 12, 31)]
    eps = [20.0, 21.0, 11.0]                                   # 1:1 bonus in Oct-2024 halves reported EPS
    shares = ([100.0, 100.0, 200.0], [dt.datetime(2024, 3, 31), dt.datetime(2024, 6, 30), dt.datetime(2024, 12, 31)])
    assert split_adjust((eps, q), shares) == [10.0, 10.5, 11.0]


def test_split_adjust_leaves_small_buybacks_alone():
    q = [dt.datetime(2024, 3, 31), dt.datetime(2024, 6, 30)]
    shares = ([101.0, 100.0], q)
    assert split_adjust(([5.0, 5.0], q), shares) == [5.0, 5.0]
    assert split_adjust(([5.0], [q[0]]), None) == [5.0]


def test_fraction_filings_are_scaled_to_percent():
    df = pl.DataFrame({"pct_promoter": [14.6, 0.146], "pct_public": [85.4, 0.854], "shares_promoter": [1.0, 1.0], "shares_public": [6.0, 6.0]})
    tot = pl.sum_horizontal([pl.col("pct_promoter"), pl.col("pct_public")]); scale = pl.when(tot <= 1.5).then(100.0).otherwise(1.0)
    out = df.with_columns([(pl.col(c) * scale).alias(c) for c in ("pct_promoter", "pct_public")])
    assert out["pct_promoter"].to_list() == [14.6, 14.6]
