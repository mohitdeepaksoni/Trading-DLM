"""Append-only, hash-chained ledger — a model component, not an appendix.

Every trial (strategy evaluation, hyper-parameter draw, factor candidate),
procedure certification, event prediction, daily signal, and perf record is a
row.  Rows are Parquet part files that are never rewritten; each row's hash
includes the previous row's hash, so the chain is tamper-evident and the trial
count that feeds the Deflated Sharpe ratio cannot be quietly reduced.
"""
from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import os
from typing import Any, Dict, Optional

import polars as pl

GENESIS = "0" * 64
SCHEMA = {
    "seq": pl.Int64, "ts": pl.Datetime("us"), "kind": pl.Utf8, "name": pl.Utf8,
    "scope": pl.Utf8, "payload": pl.Utf8, "inputs_hash": pl.Utf8,
    "prev_hash": pl.Utf8, "row_hash": pl.Utf8,
}
KINDS = ("trial", "procedure", "event", "signal", "perf", "note", "certification")


def _row_hash(seq: int, ts: str, kind: str, name: str, scope: str, payload: str,
              inputs_hash: str, prev_hash: str) -> str:
    s = "|".join([str(seq), ts, kind, name, scope, payload, inputs_hash, prev_hash])
    return hashlib.sha256(s.encode()).hexdigest()


class Ledger:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._cache: Optional[pl.DataFrame] = None

    # ----- read -----
    def _parts(self):
        return sorted(glob.glob(os.path.join(self.root, "part-*.parquet")))

    def frame(self) -> pl.DataFrame:
        parts = self._parts()
        if not parts:
            return pl.DataFrame(schema=SCHEMA)
        return pl.concat([pl.read_parquet(p) for p in parts]).sort("seq")

    def last(self) -> Dict[str, Any]:
        parts = self._parts()
        if not parts:
            return {"seq": -1, "row_hash": GENESIS}
        row = pl.read_parquet(parts[-1]).row(0, named=True)
        return row

    # ----- write -----
    def append(self, kind: str, name: str, payload: Dict[str, Any], scope: str = "default",
               inputs_hash: str = "") -> Dict[str, Any]:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        last = self.last()
        seq = int(last["seq"]) + 1
        ts = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        ts_s = ts.isoformat(timespec="microseconds")
        payload_s = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        rh = _row_hash(seq, ts_s, kind, name, scope, payload_s, inputs_hash, last["row_hash"])
        row = {"seq": seq, "ts": ts, "kind": kind, "name": name, "scope": scope,
               "payload": payload_s, "inputs_hash": inputs_hash,
               "prev_hash": last["row_hash"], "row_hash": rh}
        pl.DataFrame([row], schema=SCHEMA).write_parquet(
            os.path.join(self.root, f"part-{seq:09d}.parquet"))
        return row

    # ----- queries -----
    def _trials(self, scope: Optional[str] = None):
        f = self.frame().filter(pl.col("kind") == "trial")
        if scope is not None:
            f = f.filter(pl.col("scope") == scope)
        return [json.loads(p) for p in f["payload"].to_list()]

    def n_trials(self, scope: Optional[str] = None, distinct: bool = False) -> int:
        """Number of recorded trials — the N that deflates every Sharpe ratio.
        distinct=True counts DISTINCT PROCEDURES (payload 'procedure' hash): re-evaluating an unchanged procedure is
        not a new draw from the search (doc 06 §3). Trials without a procedure hash count individually."""
        rows = self._trials(scope)
        if not distinct:
            return len(rows)
        seen, n = set(), 0
        for d in rows:
            h = d.get("procedure")
            if h is None:
                n += 1
            elif h not in seen:
                seen.add(h); n += 1
        return n

    def trial_sharpes(self, scope: Optional[str] = None, distinct: bool = False) -> list:
        rows = self._trials(scope)
        if not distinct:
            return [float(d["sharpe_is"]) for d in rows if "sharpe_is" in d]
        latest = {}
        for i, d in enumerate(rows):
            if "sharpe_is" not in d:
                continue
            latest[d.get("procedure") or f"_row{i}"] = float(d["sharpe_is"])    # latest evaluation of each procedure
        return list(latest.values())

    def verify_chain(self) -> bool:
        """Recompute every hash; True iff the chain is intact and contiguous."""
        f = self.frame()
        prev = GENESIS
        for i, r in enumerate(f.iter_rows(named=True)):
            if r["seq"] != i or r["prev_hash"] != prev:
                return False
            ts_s = r["ts"].isoformat(timespec="microseconds")
            rh = _row_hash(r["seq"], ts_s, r["kind"], r["name"], r["scope"], r["payload"],
                           r["inputs_hash"], r["prev_hash"])
            if rh != r["row_hash"]:
                return False
            prev = rh
        return True
