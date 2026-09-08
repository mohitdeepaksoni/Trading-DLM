"""Extend the NSE bars store backwards (plan/PHASES-BY-MARKET.md Phase IN, data step 1): legacy bhavcopy archive files
(`cm<DD><MON><YYYY>bhav.csv.zip`, series EQ) for a date range, appended to runs/nse/store/bars with the same schema and
knowledge_time = session + 1 day. Idempotent per range: sessions already in the store are skipped.
    python scripts/extend_nse_bars.py --start 2016-01-01 --end 2021-12-31 [--workers 8]
Caveats stated with every number produced on the longer sample: symbol renames are not mapped (a renamed stock appears
as two symbols), and the point-in-time universe rule (top-K by trailing traded value, listed >= 300 d) handles listings and
delistings by construction.
"""
import argparse, datetime as dt, os, sys, time
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.registry import get
from emporion.data.store import BitemporalStore
import emporion.data.loaders  # noqa: F401

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs/nse"); ap.add_argument("--start", default="2016-01-01"); ap.add_argument("--end", default="2021-12-31")
ap.add_argument("--workers", type=int, default=8); ap.add_argument("--chunk-months", type=int, default=6); args = ap.parse_args()
store = BitemporalStore(os.path.join(args.root, "store")); store.register_table("bars", ["symbol"])
now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
have = store.as_of("bars", now, columns=["close"], event_start=dt.datetime.fromisoformat(args.start), event_end=dt.datetime.fromisoformat(args.end) + dt.timedelta(days=1))
have_days = set(have["event_time"].dt.date().unique().to_list()) if have.height else set()
print(f"[store] {len(have_days)} sessions already present in {args.start}..{args.end}", flush=True)
d0 = dt.date.fromisoformat(args.start); d1 = dt.date.fromisoformat(args.end); t0 = time.perf_counter(); total = 0
while d0 <= d1:
    m = d0.month - 1 + args.chunk_months; c1 = min(d1, dt.date(d0.year + m // 12, m % 12 + 1, 1) - dt.timedelta(days=1))
    try:
        bars = get("loader", "nse")(start=d0.isoformat(), end=c1.isoformat(), workers=args.workers, min_sessions=1)
    except Exception as e:  # noqa: BLE001
        print(f"[chunk] {d0}..{c1}: failed {str(e)[:100]}", flush=True); d0 = c1 + dt.timedelta(days=1); continue
    bars = bars.filter(~pl.col("event_time").dt.date().is_in(list(have_days))) if have_days else bars
    if bars.height:
        h = store.append("bars", bars); total += bars.height
        print(f"[chunk] {d0}..{c1}: {bars.height:,} rows, {bars['symbol'].n_unique()} symbols, {bars['event_time'].n_unique()} sessions hash={h[:10]} ({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
        print(f"[chunk] {d0}..{c1}: nothing new", flush=True)
    d0 = c1 + dt.timedelta(days=1)
allb = store.as_of("bars", now, columns=["close"])
print(f"[done] +{total:,} rows; store now {allb.height:,} rows, {allb['symbol'].n_unique()} symbols, {allb['event_time'].min():%Y-%m-%d}..{allb['event_time'].max():%Y-%m-%d}, {allb['event_time'].n_unique()} sessions", flush=True)
