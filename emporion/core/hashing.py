"""Content addressing. Everything that can be cached or audited has a hash:
data snapshots, feature code, ledger rows. Deterministic across machines."""
from __future__ import annotations

import hashlib
import inspect
import json
from typing import Any

import polars as pl


def content_hash(obj: Any) -> str:
    """sha256 of a canonical JSON encoding (sorted keys, no whitespace)."""
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(s.encode()).hexdigest()


def frame_hash(df: pl.DataFrame) -> str:
    """sha256 of the frame's *content*: schema + canonical-sorted rows serialised
    as CSV (shortest round-trip float repr).  Independent of Arrow chunking or
    buffer layout, so a Parquet round-trip does not change the hash."""
    h = hashlib.sha256(json.dumps([[c, str(t)] for c, t in df.schema.items()]).encode())
    if df.height:
        canon = df.sort(df.columns).rechunk()
        h.update(canon.write_csv().encode())
    return h.hexdigest()


def code_hash(fn: Any) -> str:
    """sha256 of a function's source — a feature whose code changed is a new feature."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(fn)
    return hashlib.sha256(src.encode()).hexdigest()
