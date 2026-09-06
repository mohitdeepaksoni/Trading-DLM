"""Survivorship-free QUARTERLY US panel from 13F filings (item 3): for every CUSIP held by any 13F filer, the
holdings-implied price = median over managers of value/shares at each quarter end (2013→2026), including names that
were later delisted. CUSIP -> ticker from ten years of fails-to-deliver files (which list delisted symbols too).
Output: runs/us_survfree/store tables `qprices` (symbol, cusip, event_time=quarter end, knowledge_time=quarter end
+ 46 days [13F deadline], price, n_managers) and the CUSIP map. The SEC facts already cover delisted filers."""
import datetime as dt, glob, os, sys, time
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.data.store import BitemporalStore
from emporion.data.loaders.sec_13f import cusip_symbol_map
t0 = time.perf_counter()
cmap = cusip_symbol_map(months=130); print(f"[cusip] {len(cmap):,} CUSIP->symbol pairs from ~11 years of fails-to-deliver files ({time.perf_counter()-t0:.0f}s)", flush=True)
frames = []
for f in sorted(glob.glob("runs/sec_13f/matrix/*.parquet")):
    m = pl.scan_parquet(f).filter((pl.col("shares") > 0) & (pl.col("value") > 0)).with_columns((pl.col("value") * 1000.0 / pl.col("shares")).alias("px")).collect()
    # 13F VALUE is in thousands of dollars before 2023 filings and in dollars after; detect per window by median px scale vs plausible range
    med = float(m["px"].median())
    if med > 5000:                                                           # values already in dollars -> undo the x1000
        m = m.with_columns((pl.col("px") / 1000.0).alias("px"))
    frames.append(m.select(["CUSIP", "period", "px", "filed"]))
M = pl.concat(frames)
q = (M.group_by(["CUSIP", "period"]).agg(pl.col("px").median().alias("price"), pl.len().alias("n_managers"), pl.col("filed").max().alias("last_filed"))
      .filter((pl.col("n_managers") >= 5) & (pl.col("price") > 0.5) & (pl.col("price") < 20000)))
q = q.with_columns(pl.col("CUSIP").map_elements(lambda c: cmap.get(c), return_dtype=pl.Utf8).alias("symbol")).rename({"CUSIP": "cusip", "period": "event_time"})
q = q.with_columns((pl.col("event_time") + pl.duration(days=46)).alias("knowledge_time"))
store = BitemporalStore("runs/us_survfree/store"); store.register_table("qprices", ["cusip"]); h = store.append("qprices", q.select(["cusip", "symbol", "event_time", "knowledge_time", "price", "n_managers"]))
print(f"[qprices] {q.height:,} (cusip, quarter) prices, {q['cusip'].n_unique():,} cusips ({int(q['symbol'].is_not_null().sum()):,} mapped to tickers), quarters {q['event_time'].min():%Y-%m-%d}..{q['event_time'].max():%Y-%m-%d} hash={h[:10]} ({time.perf_counter()-t0:.0f}s)", flush=True)
# how many names disappear (survivorship count): cusips whose last quarter is before the latest quarter
last = q.group_by("cusip").agg(pl.col("event_time").max().alias("last_q")); latest = q["event_time"].max()
print(f"[survivorship] {int((last['last_q'] < latest).sum()):,} of {last.height:,} cusips have a last quarter before {latest:%Y-%m-%d} (delisted/absorbed/dropped) — these are missing from the Nasdaq-screener pool", flush=True)
print("[done]")
