import datetime as dt
import numpy as np
import polars as pl
import pytest

from emporion.core.hashing import frame_hash
from emporion.data.store import BitemporalStore
from emporion.data.loaders import synthetic
from emporion.features import compute, available_features, feature_spec


def _store(tmp_path, n_assets=5, n_days=400):
    S = BitemporalStore(str(tmp_path / "store"))
    S.register_table("bars", ["symbol"])
    S.append("bars", synthetic.load(n_assets=n_assets, n_days=n_days, seed=1))
    return S


def test_as_of_excludes_future_knowledge(tmp_path):
    S = _store(tmp_path)
    d = dt.datetime(2015, 6, 1)
    df = S.as_of("bars", d)
    assert df.height > 0
    assert (df["knowledge_time"] <= d).all()
    assert (df["event_time"] < d).all()     # daily bars only known next day


def test_revisions_resolve_to_latest_known(tmp_path):
    S = BitemporalStore(str(tmp_path / "s"))
    S.register_table("fund", ["symbol"])
    e = dt.datetime(2020, 3, 31)
    first = pl.DataFrame({"symbol": ["A"], "event_time": [e], "knowledge_time": [dt.datetime(2020, 5, 1)], "eps": [1.0]})
    restated = pl.DataFrame({"symbol": ["A"], "event_time": [e], "knowledge_time": [dt.datetime(2020, 8, 1)], "eps": [0.7]})
    S.append("fund", first); S.append("fund", restated)
    assert S.as_of("fund", dt.datetime(2020, 4, 15)).height == 0          # not yet published
    assert S.as_of("fund", dt.datetime(2020, 6, 1))["eps"][0] == 1.0       # first print
    assert S.as_of("fund", dt.datetime(2020, 9, 1))["eps"][0] == 0.7       # restatement, one row only
    assert S.as_of("fund", dt.datetime(2020, 9, 1)).height == 1


def test_store_rejects_impossible_rows(tmp_path):
    S = BitemporalStore(str(tmp_path / "s"))
    S.register_table("t", ["symbol"])
    bad = pl.DataFrame({"symbol": ["A"], "event_time": [dt.datetime(2020, 1, 2)], "knowledge_time": [dt.datetime(2020, 1, 1)]})
    with pytest.raises(ValueError):
        S.append("t", bad)


def test_unregistered_feature_does_not_exist(tmp_path):
    S = _store(tmp_path)
    with pytest.raises(KeyError):
        compute("definitely_not_registered", S, dt.datetime(2016, 1, 4))


def test_replay_live_parity_and_cache(tmp_path):
    S = _store(tmp_path)
    d = dt.datetime(2016, 3, 1)
    cache = str(tmp_path / "cache")
    for f in available_features():
        a = compute(f, S, d, mode="replay", cache_dir=cache)
        b = compute(f, S, d, mode="live", now=d, cache_dir=cache)   # clock frozen at d
        assert frame_hash(a) == frame_hash(b), f
        assert a.height == 5 and a["value"].is_not_nan().all()
    # cache populated: one file per feature
    assert len(list((tmp_path / "cache").glob("*.parquet"))) == len(available_features())


def test_feature_lag_shifts_knowledge(tmp_path):
    S = _store(tmp_path)
    spec = feature_spec("vol_20")
    d = dt.datetime(2016, 3, 1)
    out = compute("vol_20", S, d, mode="replay")
    assert out["as_of"][0] == d - spec.lag


def test_all_versions_then_filter_keeps_first_prints(tmp_path):
    """Resolving revisions BEFORE filtering by decision date hides restated quarters; all_versions + filter does not."""
    S = BitemporalStore(str(tmp_path / "s")); S.register_table("f", ["symbol"])
    e = dt.datetime(2022, 3, 31)
    S.append("f", pl.DataFrame({"symbol": ["A"], "event_time": [e], "knowledge_time": [dt.datetime(2022, 4, 26)], "v": [1.86]}))
    S.append("f", pl.DataFrame({"symbol": ["A"], "event_time": [e], "knowledge_time": [dt.datetime(2024, 11, 18)], "v": [1.86]}))
    asof = dt.datetime(2024, 6, 1)
    wrong = S.as_of("f", dt.datetime(2030, 1, 1)).filter(pl.col("knowledge_time") <= asof)
    right = S.all_versions("f").filter(pl.col("knowledge_time") <= asof)
    assert wrong.height == 0 and right.height == 1
    assert S.as_of("f", asof).height == 1                      # the store's own as-of is correct by construction
