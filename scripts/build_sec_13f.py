"""Build per-stock institutional ownership (point-in-time) + manager x stock matrices from the last N 13F windows.
    python scripts/build_sec_13f.py --windows 12
"""
import argparse, os, sys, time
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.data.store import BitemporalStore
from emporion.data.loaders.sec_13f import window_urls, cusip_symbol_map, load_window
ap = argparse.ArgumentParser(); ap.add_argument("--windows", type=int, default=12); ap.add_argument("--root", default="runs/us_stocks_wide"); args = ap.parse_args()
t0 = time.perf_counter(); work = "runs/sec_13f"; os.makedirs(os.path.join(work, "matrix"), exist_ok=True)
cmap = cusip_symbol_map(months=8); print(f"[cusip] {len(cmap):,} CUSIP->symbol pairs from fails-to-deliver files ({time.perf_counter()-t0:.0f}s)", flush=True)
urls = window_urls()[:args.windows]; print(f"[13f] {len(urls)} windows: {os.path.basename(urls[-1])} .. {os.path.basename(urls[0])}", flush=True)
store = BitemporalStore(os.path.join(args.root, "store"))
if "inst_holdings" not in store.tables(): store.register_table("inst_holdings", ["cusip"])
else: store.keys("inst_holdings")
for u in reversed(urls):
    name = os.path.basename(u).replace("_form13f.zip", "")
    if os.path.exists(os.path.join(work, "matrix", f"{name}.parquet")):
        print(f"[13f] {name}: already loaded, skipped", flush=True); continue
    try:
        agg, matrix = load_window(u, work)
    except Exception as e:  # noqa: BLE001
        print(f"[13f] {name}: FAILED {str(e)[:80]}", flush=True); continue
    agg = agg.with_columns(pl.col("CUSIP").map_elements(lambda c: cmap.get(c), return_dtype=pl.Utf8).alias("symbol")).rename({"CUSIP": "cusip"})
    store.append("inst_holdings", agg.select(["cusip", "symbol", "event_time", "knowledge_time", "inst_shares", "inst_value", "n_holders"]))
    matrix.write_parquet(os.path.join(work, "matrix", f"{name}.parquet"))
    print(f"[13f] {name}: {agg.height:,} (cusip,period) rows, mapped {int(agg['symbol'].is_not_null().sum()):,}; matrix {matrix.height:,} rows; periods {agg['event_time'].min():%Y-%m-%d}..{agg['event_time'].max():%Y-%m-%d} ({time.perf_counter()-t0:.0f}s)", flush=True)
    try:
        os.remove(os.path.join(work, os.path.basename(u)))
    except OSError:
        pass
print("[done]")
