"""One feature library, two modes.

A feature is a registered function f(ctx) -> DataFrame[symbol, event_time, value]
with a declared availability lag and version.  `compute()` runs it in
  replay mode: knowledge as_of = decision_time - lag  (historical replay)
  live   mode: knowledge as_of = now - lag             (production)
through the *same* code path.  Unregistered features cannot be computed;
results are content-addressed (feature code hash + data snapshot hash + as_of)
so nothing is ever computed twice.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import os
from typing import Callable, Dict, List, Optional

import polars as pl

from ..core.hashing import code_hash, content_hash
from ..core.perf import timed
from ..data.store import BitemporalStore

_FEATURES: Dict[str, "FeatureSpec"] = {}


@dataclasses.dataclass
class FeatureSpec:
    name: str
    fn: Callable
    lag: dt.timedelta
    inputs: List[str]
    version: int
    code: str


def feature(name: str, lag: str = "1d", inputs=("bars",), version: int = 1):
    """Register a feature with its declared availability lag ('1d', '36h', '0')."""
    def deco(fn):
        _FEATURES[name] = FeatureSpec(name, fn, _parse_lag(lag), list(inputs), version, code_hash(fn))
        return fn
    return deco


def _parse_lag(s: str) -> dt.timedelta:
    if s in ("0", "", None):
        return dt.timedelta(0)
    unit = s[-1]
    n = float(s[:-1])
    return {"d": dt.timedelta(days=n), "h": dt.timedelta(hours=n), "m": dt.timedelta(minutes=n)}[unit]


def available_features() -> List[str]:
    return sorted(_FEATURES)


def feature_spec(name: str) -> FeatureSpec:
    if name not in _FEATURES:
        raise KeyError(f"feature {name!r} is not registered; a feature that is not registered does not exist. "
                       f"Registered: {available_features()}")
    return _FEATURES[name]


class FeatureContext:
    """What a feature is allowed to see: tables as-of the effective knowledge time, nothing else."""

    def __init__(self, store: BitemporalStore, as_of: dt.datetime, mode: str, lookback: dt.timedelta):
        self.store, self.as_of, self.mode, self.lookback = store, as_of, mode, lookback

    def table(self, name: str, **kw) -> pl.DataFrame:
        return self.store.as_of(name, self.as_of, event_start=self.as_of - self.lookback, **kw)


@timed("features.compute")
def compute(name: str, store: BitemporalStore, decision_time: dt.datetime, mode: str = "replay",
            now: Optional[dt.datetime] = None, lookback: dt.timedelta = dt.timedelta(days=400),
            cache_dir: Optional[str] = None) -> pl.DataFrame:
    spec = feature_spec(name)
    if mode == "replay":
        as_of = decision_time - spec.lag
    elif mode == "live":
        now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        as_of = now - spec.lag
    else:
        raise ValueError("mode must be 'replay' or 'live'")
    key = None
    if cache_dir:
        snap = [store.snapshot_hash(t, as_of) for t in spec.inputs]
        key = content_hash({"f": name, "v": spec.version, "code": spec.code, "as_of": as_of, "snap": snap,
                            "lookback": lookback})
        path = os.path.join(cache_dir, f"{name}-{key[:16]}.parquet")
        if os.path.exists(path):
            return pl.read_parquet(path)
    ctx = FeatureContext(store, as_of, mode, lookback)
    out = spec.fn(ctx)
    if not {"symbol", "event_time", "value"} <= set(out.columns):
        raise ValueError(f"feature {name} must return columns symbol,event_time,value")
    out = out.with_columns(pl.lit(name).alias("feature"), pl.lit(as_of).alias("as_of"))
    if key:
        os.makedirs(cache_dir, exist_ok=True)
        out.write_parquet(path)
    return out
