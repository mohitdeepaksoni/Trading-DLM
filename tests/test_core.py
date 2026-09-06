import json
import numpy as np
import polars as pl
import pytest

from emporion.core.ledger import Ledger, GENESIS
from emporion.core.hashing import frame_hash, content_hash


def test_ledger_chain_and_tamper(tmp_path):
    L = Ledger(str(tmp_path / "ledger"))
    r0 = L.append("trial", "a", {"sharpe_is": 1.0}, scope="s")
    r1 = L.append("trial", "b", {"sharpe_is": 0.5}, scope="s")
    L.append("note", "n", {"x": 1}, scope="other")
    assert r0["prev_hash"] == GENESIS and r1["prev_hash"] == r0["row_hash"]
    assert L.n_trials("s") == 2 and L.n_trials() == 2
    assert sorted(L.trial_sharpes("s")) == [0.5, 1.0]
    assert L.verify_chain()
    # tamper: rewrite part 1 with a different payload
    p = sorted((tmp_path / "ledger").glob("part-*.parquet"))[1]
    df = pl.read_parquet(p).with_columns(pl.lit(json.dumps({"sharpe_is": 9.0})).alias("payload"))
    df.write_parquet(p)
    assert not L.verify_chain()


def test_ledger_rejects_unknown_kind(tmp_path):
    L = Ledger(str(tmp_path / "l"))
    with pytest.raises(ValueError):
        L.append("bogus", "x", {})


def test_frame_hash_is_order_invariant():
    a = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    b = pl.DataFrame({"x": [3, 1, 2], "y": ["c", "a", "b"]})
    assert frame_hash(a) == frame_hash(b)
    assert frame_hash(a) != frame_hash(a.with_columns(pl.col("x") * 2))
    assert content_hash({"b": 1, "a": 2}) == content_hash({"a": 2, "b": 1})
