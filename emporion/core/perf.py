"""Measured before optimised.

Every hot function is wrapped with @timed; it records wall time, rows processed,
throughput and peak RSS.  Tests import `budget()` to turn those numbers into
pass/fail floors, so a performance regression fails CI exactly like a wrong
answer would.  Reports are saved as JSON under benchmarks/ with date + host.
"""
from __future__ import annotations

import functools
import json
import os
import platform
import resource
import socket
import time
from typing import Any, Callable, Dict, List, Optional

_RECORDS: List[Dict[str, Any]] = []


def _peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if platform.system() == "Darwin" else r / 1e3  # bytes on mac, KB on linux


def timed(name: Optional[str] = None, rows: Optional[Callable[..., int]] = None):
    """Decorator. `rows(*args, **kw)` returns the unit count for throughput."""
    def deco(fn):
        nm = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kw):
            t0 = time.perf_counter()
            out = fn(*args, **kw)
            wall = time.perf_counter() - t0
            n = int(rows(*args, **kw)) if rows else None
            _RECORDS.append({"name": nm, "wall_s": wall, "rows": n,
                             "rows_per_s": (n / wall if (n and wall > 0) else None),
                             "peak_rss_mb": _peak_rss_mb()})
            return out
        return wrapper
    return deco


def records() -> List[Dict[str, Any]]:
    return list(_RECORDS)


def clear() -> None:
    _RECORDS.clear()


def report() -> Dict[str, Dict[str, float]]:
    """Per-name summary: count, p50, p99, max wall; best throughput."""
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in _RECORDS:
        by.setdefault(r["name"], []).append(r)
    out = {}
    for k, rs in by.items():
        w = sorted(x["wall_s"] for x in rs)
        n = len(w)
        tp = [x["rows_per_s"] for x in rs if x["rows_per_s"]]
        out[k] = {"n": n, "p50_s": w[n // 2], "p99_s": w[min(n - 1, int(0.99 * n))],
                  "max_s": w[-1], "best_rows_per_s": max(tp) if tp else None,
                  "peak_rss_mb": max(x["peak_rss_mb"] for x in rs)}
    return out


def save_report(path: str, extra: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    doc = {"host": socket.gethostname(), "platform": platform.platform(),
           "cpu_count": os.cpu_count(), "summary": report(), "records": records()}
    if extra:
        doc.update(extra)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    return path


def budget(name: str, max_p50_s: float) -> None:
    """Assert a recorded function's median wall time is within budget."""
    rep = report()
    if name not in rep:
        raise AssertionError(f"no perf record for {name!r}; recorded: {sorted(rep)}")
    got = rep[name]["p50_s"]
    if got > max_p50_s:
        raise AssertionError(f"perf budget breached: {name} p50={got:.4f}s > {max_p50_s}s")
