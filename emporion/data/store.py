"""Bitemporal store — every row carries event_time and knowledge_time; nothing
can be read without an `as_of` knowledge time.  Parquet part files (append-only)
queried through DuckDB so a replay never loads more than it needs.

Revisions: the same (keys, event_time) may be appended again later with a newer
knowledge_time (restated fundamentals, corrected prints).  `as_of` returns, for
each key/event_time, the latest revision whose knowledge_time <= as_of.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
from typing import Dict, List, Optional, Sequence

import duckdb
import polars as pl

from ..core.hashing import frame_hash
from ..core.perf import timed


def _ts(x) -> dt.datetime:
    if isinstance(x, dt.datetime):
        return x
    if isinstance(x, dt.date):
        return dt.datetime(x.year, x.month, x.day)
    return dt.datetime.fromisoformat(str(x))


class BitemporalStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._keys: Dict[str, List[str]] = {}
        self._con = duckdb.connect()

    # ----- schema -----
    def register_table(self, table: str, keys: Sequence[str]) -> None:
        """Declare the business keys that identify a row (e.g. ["symbol"])."""
        self._keys[table] = list(keys)
        os.makedirs(os.path.join(self.root, table), exist_ok=True)

    def keys(self, table: str) -> List[str]:
        if table not in self._keys:
            meta = os.path.join(self.root, table, "_keys.txt")
            if os.path.exists(meta):
                self._keys[table] = open(meta).read().split()
            else:
                raise KeyError(f"table {table!r} not registered")
        return self._keys[table]

    def _parts(self, table: str) -> List[str]:
        return sorted(glob.glob(os.path.join(self.root, table, "part-*.parquet")))

    # ----- write -----
    @timed("store.append", rows=lambda *a, **k: a[2].height)
    def append(self, table: str, df: pl.DataFrame) -> str:
        keys = self.keys(table)
        need = set(keys) | {"event_time", "knowledge_time"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"{table}: missing columns {sorted(missing)}")
        if (df["knowledge_time"] < df["event_time"]).any():
            raise ValueError("knowledge_time before event_time — impossible row")
        df = df.with_columns(pl.col("event_time").cast(pl.Datetime("us")),
                             pl.col("knowledge_time").cast(pl.Datetime("us")))
        os.makedirs(os.path.join(self.root, table), exist_ok=True)
        with open(os.path.join(self.root, table, "_keys.txt"), "w") as f:
            f.write(" ".join(keys))
        n = len(self._parts(table))
        h = frame_hash(df)
        df.write_parquet(os.path.join(self.root, table, f"part-{n:06d}-{h[:8]}.parquet"))
        return h

    # ----- read (as_of is mandatory) -----
    @timed("store.as_of")
    def as_of(self, table: str, as_of, event_start=None, event_end=None,
              columns: Optional[Sequence[str]] = None, where: Optional[str] = None) -> pl.DataFrame:
        parts = self._parts(table)
        keys = self.keys(table)
        if not parts:
            return pl.DataFrame()
        k = ", ".join(keys)
        cols = "*" if columns is None else ", ".join(sorted(set(columns) | set(keys) | {"event_time", "knowledge_time"}))
        conds = [f"knowledge_time <= TIMESTAMP '{_ts(as_of).isoformat(sep=' ')}'"]
        if event_start is not None:
            conds.append(f"event_time >= TIMESTAMP '{_ts(event_start).isoformat(sep=' ')}'")
        if event_end is not None:
            conds.append(f"event_time <= TIMESTAMP '{_ts(event_end).isoformat(sep=' ')}'")
        if where:
            conds.append(f"({where})")
        files = "[" + ", ".join(f"'{p}'" for p in parts) + "]"
        sql = f"""
            SELECT {cols} FROM read_parquet({files})
            WHERE {' AND '.join(conds)}
            QUALIFY row_number() OVER (PARTITION BY {k}, event_time ORDER BY knowledge_time DESC) = 1
            ORDER BY event_time, {k}
        """
        return self._con.execute(sql).pl()

    def all_versions(self, table: str, columns: Optional[Sequence[str]] = None) -> pl.DataFrame:
        """Every revision of every row (no as-of resolution). Use when a script will resolve revisions itself at
        many decision dates: filter knowledge_time <= t, THEN take the latest revision per key/event_time.
        Resolving first (as_of(now)) and filtering afterwards hides every quarter that was later restated."""
        parts = self._parts(table)
        if not parts:
            return pl.DataFrame()
        keys = self.keys(table)
        cols = "*" if columns is None else ", ".join(sorted(set(columns) | set(keys) | {"event_time", "knowledge_time"}))
        files = "[" + ", ".join(f"'{p}'" for p in parts) + "]"
        return self._con.execute(f"SELECT {cols} FROM read_parquet({files}) ORDER BY event_time, knowledge_time").pl()

    def live(self, table: str, lag: dt.timedelta = dt.timedelta(0), now: Optional[dt.datetime] = None,
             **kw) -> pl.DataFrame:
        """Live mode = as_of(now - lag). `now` is injectable so tests can freeze the clock."""
        now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        return self.as_of(table, now - lag, **kw)

    def snapshot_hash(self, table: str, as_of) -> str:
        return frame_hash(self.as_of(table, as_of))

    def tables(self) -> List[str]:
        return sorted(d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d)))
