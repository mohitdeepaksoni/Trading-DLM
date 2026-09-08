"""EU/UK price panel: every ESEF/UKSEF issuer of the given countries -> ISINs (GLEIF) -> listed equity symbol (Yahoo search)
-> daily bars since 2016 (yfinance) into runs/eu_uk/store: tables `universe` (lei, isin, symbol, exchange) and `bars`.
Optionally the STOXX Europe 600 sector ETFs (iShares, Xetra) + the STOXX 600 benchmark into runs/eu_sectors/store for the
Agora event harness.
    python scripts/build_euuk_panel.py [--countries GB,FR,...] [--start 2016-01-01] [--skip-bars] [--sectors]
"""
import argparse, datetime as dt, os, sys, time
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.registry import get
from emporion.data.store import BitemporalStore
import emporion.data.loaders  # noqa: F401

# iShares STOXX Europe 600 supersector ETFs on Xetra (Yahoo symbols verified 2026-09-08) + benchmark
EU_SECTOR_ETFS = {"EXV1.DE": "Banks", "EXH1.DE": "Oil & Gas", "EXV3.DE": "Technology", "EXV5.DE": "Automobiles & Parts", "EXV6.DE": "Basic Resources",
                  "EXV7.DE": "Chemicals", "EXH9.DE": "Utilities", "EXH4.DE": "Industrial Goods & Services", "EXH3.DE": "Food & Beverage",
                  "EXV4.DE": "Health Care", "EXH5.DE": "Insurance", "EXH8.DE": "Retail", "EXV8.DE": "Construction & Materials",
                  "EXV2.DE": "Telecommunications", "EXV9.DE": "Travel & Leisure", "EXH6.DE": "Media", "EXH7.DE": "Personal & Household Goods",
                  "EXI5.DE": "Real Estate", "EXH2.DE": "Financial Services"}
EU_BENCHMARK = "EXSA.DE"

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs/eu_uk"); ap.add_argument("--countries", default="GB,FR,NL,SE,DK,FI,IT,ES,BE,AT,PT,NO")
ap.add_argument("--start", default="2016-01-01"); ap.add_argument("--skip-bars", action="store_true"); ap.add_argument("--skip-map", action="store_true", help="reuse the stored universe table")
ap.add_argument("--sectors", action="store_true"); ap.add_argument("--sectors-root", default="runs/eu_sectors"); ap.add_argument("--workers", type=int, default=4); args = ap.parse_args()
log = lambda s: print(s, flush=True)
now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None); t0 = time.perf_counter()
store = BitemporalStore(os.path.join(args.root, "store"))
countries = [c.strip() for c in args.countries.split(",") if c.strip()]
if not args.skip_map:
    idx = store.as_of("esef_index", now + dt.timedelta(days=3650)).filter(pl.col("country").is_in(countries))
    issuers = idx.group_by("lei").agg(pl.col("name").last(), pl.col("country").last(), pl.len().alias("n_filings")).sort("lei")
    log(f"[issuers] {issuers.height:,} ESEF issuers in {countries}")
    isins = get("loader", "gleif_isins")(issuers["lei"].to_list(), workers=args.workers, cache_path=os.path.join(args.root, "isin-lei-latest.zip"), log=log)
    uni = get("loader", "yahoo_symbols")(issuers.select(["lei", "name", "country"]), isins, workers=args.workers, log=log)
    uni = uni.join(issuers.select(["lei", "n_filings"]), on="lei", how="left").with_columns(pl.lit(now).alias("event_time"), pl.lit(now).alias("knowledge_time"))
    # a listing belongs to one issuer (parent and subsidiary both filing -> keep the one with more filings)
    uni = uni.sort(["symbol", "n_filings"], descending=[False, True]).unique("symbol", keep="first").sort(["lei", "rank"])
    uni = uni.with_columns(pl.lit(None, dtype=pl.Boolean).alias("primary"), pl.lit(None, dtype=pl.Float64).alias("traded_value_median"))
    store.register_table("universe", ["lei", "symbol"])
    log(f"[universe] {uni['lei'].n_unique():,} issuers with a listed equity, {uni.height:,} candidate lines; by country {uni.unique('lei').group_by('country').agg(pl.len()).sort('len', descending=True).to_dict(as_series=False)}; by exchange {uni['exchange'].value_counts().sort('count', descending=True).head(14).to_dict(as_series=False)} ({time.perf_counter()-t0:.0f}s)")
    if args.skip_bars:
        store.append("universe", uni)
    isins.with_columns(pl.lit(now).alias("event_time"), pl.lit(now).alias("knowledge_time")).pipe(lambda d: (store.register_table("isins", ["lei", "isin"]), store.append("isins", d)))
else:
    uni = store.as_of("universe", now)
if not args.skip_bars:
    bars = get("loader", "yahoo")(uni["symbol"].unique().to_list(), start=args.start, log=log)
    store.register_table("bars", ["symbol"]); h = store.append("bars", bars)
    cov = bars.group_by("symbol").agg(pl.len().alias("n"), pl.col("event_time").min().alias("first"))
    log(f"[bars] {bars.height:,} rows, {bars['symbol'].n_unique():,} symbols, {bars['event_time'].min():%Y-%m-%d}..{bars['event_time'].max():%Y-%m-%d}; "
        f"symbols with >= 1500 sessions: {int((cov['n'] >= 1500).sum())}, listed after 2018: {int((cov['first'] > dt.datetime(2018, 1, 1)).sum())} hash={h[:10]} ({time.perf_counter()-t0:.0f}s)")
    # primary line per issuer = the candidate with the highest median traded value over the last 250 sessions (ordinary over preference, A vs B)
    tv = (bars.filter(pl.col("event_time") >= bars["event_time"].max() - dt.timedelta(days=400)).group_by("symbol")
              .agg((pl.col("close") * pl.col("volume")).median().alias("tv"), pl.len().alias("n")))
    prim = (uni.drop(["primary", "traded_value_median"]).join(tv, on="symbol", how="left").with_columns(pl.col("tv").fill_null(-1.0)).sort(["lei", "tv"], descending=[False, True])
               .with_columns((pl.col("tv") == pl.col("tv").max().over("lei")).alias("primary"), pl.lit(now).alias("knowledge_time")))
    h = store.append("universe", prim.with_columns(pl.col("tv").alias("traded_value_median")).select(uni.columns))
    log(f"[primary] {int(prim['primary'].sum()):,} primary lines chosen by traded value; {int((prim.filter(pl.col('primary'))['tv'] <= 0).sum())} issuers without any bars")
if args.sectors:
    sst = BitemporalStore(os.path.join(args.sectors_root, "store")); syms = list(EU_SECTOR_ETFS) + [EU_BENCHMARK]
    sb = get("loader", "yahoo")(syms, start="2014-01-01", log=log); sst.register_table("bars", ["symbol"]); h = sst.append("bars", sb)
    log(f"[sectors] {sb.height:,} rows, {sb['symbol'].n_unique()} ETFs, {sb['event_time'].min():%Y-%m-%d}..{sb['event_time'].max():%Y-%m-%d} hash={h[:10]}")
log("[done]")
