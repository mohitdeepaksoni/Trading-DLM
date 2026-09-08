"""NSE delivery position (MTO files) -> runs/nse/store/delivery, chunked and idempotent per session (plan Phase IN, data step 2).
    python scripts/build_nse_delivery.py --start 2016-01-01 --end 2026-09-08 [--workers 4]
NSE blocks bulk archive downloads from cloud IPs (403 after ~170 files): run the backfill from a residential IP; the nightly
single-file refresh is unaffected."""
import argparse, datetime as dt, os, sys, time
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.registry import get
from emporion.data.store import BitemporalStore
import emporion.data.loaders  # noqa: F401

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs/nse"); ap.add_argument("--start", default="2016-01-01"); ap.add_argument("--end", default=dt.date.today().isoformat())
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--chunk-months", type=int, default=6); args = ap.parse_args()
store = BitemporalStore(os.path.join(args.root, "store")); store.register_table("delivery", ["symbol"]); now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
have = store.as_of("delivery", now, columns=["deliv_per"], event_start=dt.datetime.fromisoformat(args.start), event_end=dt.datetime.fromisoformat(args.end) + dt.timedelta(days=1)) if "delivery" in store.tables() else pl.DataFrame()
have_days = set(have["event_time"].dt.date().unique().to_list()) if have.height else set()
print(f"[store] {len(have_days)} delivery sessions already present in {args.start}..{args.end}", flush=True)
d0 = dt.date.fromisoformat(args.start); d1 = dt.date.fromisoformat(args.end); t0 = time.perf_counter(); total = 0
while d0 <= d1:
    m = d0.month - 1 + args.chunk_months; c1 = min(d1, dt.date(d0.year + m // 12, m % 12 + 1, 1) - dt.timedelta(days=1))
    df = get("loader", "nse_delivery")(d0.isoformat(), c1.isoformat(), workers=args.workers, log=lambda s: print(s, flush=True))
    if have_days and df.height:
        df = df.filter(~pl.col("event_time").dt.date().is_in(list(have_days)))
    if df.height:
        h = store.append("delivery", df); total += df.height
        print(f"[chunk] {d0}..{c1}: {df.height:,} rows, {df['symbol'].n_unique()} symbols, {df['event_time'].n_unique()} sessions, median deliv% {df['deliv_per'].median():.1f} hash={h[:10]} ({time.perf_counter()-t0:.0f}s)", flush=True)
    d0 = c1 + dt.timedelta(days=1)
alld = store.as_of("delivery", now, columns=["deliv_per"])
print(f"[done] +{total:,} rows; delivery table {alld.height:,} rows, {alld['symbol'].n_unique()} symbols, {alld['event_time'].min():%Y-%m-%d}..{alld['event_time'].max():%Y-%m-%d}, {alld['event_time'].n_unique()} sessions", flush=True)
