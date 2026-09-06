"""NSE point-in-time fundamentals + holdings + flows for the top-N symbols by traded value in the NSE bars store.
    python scripts/build_nse_fundamentals.py --top 200
"""
import argparse, datetime as dt, os, sys, time
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.registry import get
from emporion.data.store import BitemporalStore
import emporion.data.loaders  # noqa: F401

ap = argparse.ArgumentParser(); ap.add_argument("--top", type=int, default=200); ap.add_argument("--root", default="runs/nse"); ap.add_argument("--only", default="", help="comma list: results,shareholding,flows"); ap.add_argument("--replace", action="store_true", help="drop existing parts of each rebuilt table first"); args = ap.parse_args()
only = set(args.only.split(",")) if args.only else {"results", "shareholding", "flows"}
import glob, shutil
def _drop(table):
    if args.replace:
        bak = os.path.join(args.root, "store", f"_{table}.bak-{dt.datetime.now():%Y%m%d%H%M}")
        if os.path.isdir(os.path.join(args.root, "store", table)): shutil.move(os.path.join(args.root, "store", table), bak); print(f"[replace] {table} -> {bak}", flush=True)
store = BitemporalStore(os.path.join(args.root, "store")); now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
bars = store.as_of("bars", now, columns=["close", "volume"], event_start=now - dt.timedelta(days=365))
tv = bars.group_by("symbol").agg((pl.col("close") * pl.col("volume")).median().alias("tv"), pl.len().alias("n")).filter(pl.col("n") > 200).sort("tv", descending=True).head(args.top)
syms = [s.replace(".NS", "") for s in tv["symbol"].to_list()]
print(f"[universe] {len(syms)} NSE symbols by traded value: {syms[:8]}", flush=True)
t0 = time.perf_counter()
if "results" in only:
    res = get("loader", "nse_results")(syms, sleep=0.4); _drop("results")
    store.register_table("results", ["symbol", "consolidated"]); h = store.append("results", res)
    print(f"[results] {res.height:,} rows, {res['symbol'].n_unique()} symbols, {res['event_time'].min():%Y-%m-%d}..{res['event_time'].max():%Y-%m-%d} hash={h[:10]} ({time.perf_counter()-t0:.0f}s)", flush=True)
if "shareholding" in only:
    sh = get("loader", "nse_shareholding")(syms, sleep=0.4); _drop("shareholding")
    store.register_table("shareholding", ["symbol"]); h = store.append("shareholding", sh)
    print(f"[shareholding] {sh.height:,} rows, {sh['symbol'].n_unique()} symbols, cols={[c for c in sh.columns if c.startswith('pct_')]} hash={h[:10]} ({time.perf_counter()-t0:.0f}s)", flush=True)
if "flows" in only:
    fl = get("loader", "nse_flows")(); store.register_table("flows", ["category"]); store.append("flows", fl)
    print(f"[flows] {fl.height} rows (daily snapshot job from here)", flush=True)
print("[done]")
