"""Build the US single-stock panel: top-N by market cap from the Nasdaq screener (US-listed common stocks),
10 years of daily bars (Nasdaq API), SEC XBRL facts (EPS/net income/DPS/shares) — all into runs/us_stocks/store.
    python scripts/build_us_stocks_panel.py --top 200 [--skip-facts]
"""
from __future__ import annotations

import argparse, datetime as dt, os, sys, time
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.registry import get
from emporion.data.store import BitemporalStore
import emporion.data.loaders  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=1000)
    ap.add_argument("--root", default="runs/us_stocks")
    ap.add_argument("--skip-bars", action="store_true")
    ap.add_argument("--skip-facts", action="store_true")
    args = ap.parse_args()
    store = BitemporalStore(os.path.join(args.root, "store"))
    t0 = time.perf_counter()
    uni = get("loader", "nasdaq_universe")()
    uni = (uni.filter(pl.col("market_cap").is_not_null() & (pl.col("country") == "United States")
                      & ~pl.col("symbol").str.contains(r"[\^/\.]") & ~pl.col("name").str.to_lowercase().str.contains("etf|fund|trust|acquisition"))
              .sort("market_cap", descending=True).head(args.top))
    store.register_table("universe", ["symbol"]); store.append("universe", uni)
    syms = uni["symbol"].to_list()
    print(f"[universe] {len(syms)} US stocks by market cap; top: {syms[:8]}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    if not args.skip_bars:
        bars = get("loader", "nasdaq")(symbols=syms, start="2014-01-01", assetclass="stocks")
        store.register_table("bars", ["symbol"]); h = store.append("bars", bars)
        print(f"[bars] {bars.height:,} rows, {bars['symbol'].n_unique()} symbols, {bars['event_time'].min():%Y-%m-%d}..{bars['event_time'].max():%Y-%m-%d} hash={h[:10]} ({time.perf_counter()-t0:.0f}s)", flush=True)
    if not args.skip_facts:
        facts = get("loader", "sec_facts")(syms)
        store.register_table("facts", ["symbol", "concept"]); h = store.append("facts", facts)
        print(f"[facts] {facts.height:,} rows, {facts['symbol'].n_unique()} symbols with XBRL facts ({time.perf_counter()-t0:.0f}s)", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()
