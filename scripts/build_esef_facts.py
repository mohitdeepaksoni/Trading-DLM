"""Download + parse the xBRL-JSON facts of every ESEF/UKSEF filing for the given countries into runs/eu_uk/store/esef_facts
(bitemporal: event_time = reporting period end from the facts, knowledge_time = date the filing was added to the OAM mirror).
Coverage is by construction the whole regulated market of each country (research/03 §16: features for the union of every
point-in-time universe), not today's index constituents.
    python scripts/build_esef_facts.py --countries GB,FR,NL,SE,DK,FI,IT,ES,BE,AT,PT,NO [--workers 6] [--limit N]
Prints the sanity checks that must be read before any of this is trusted (§14): period types, EPS-unit ratios, restatements.
"""
import argparse, datetime as dt, os, sys, time
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.registry import get
from emporion.data.store import BitemporalStore
import emporion.data.loaders  # noqa: F401

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs/eu_uk"); ap.add_argument("--countries", default="GB,FR,NL,SE,DK,FI,IT,ES,BE,AT,PT,NO")
ap.add_argument("--workers", type=int, default=6); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--sleep", type=float, default=0.05); args = ap.parse_args()
store = BitemporalStore(os.path.join(args.root, "store")); now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
idx = store.as_of("esef_index", now + dt.timedelta(days=365 * 10))             # the 9 junk period_end rows carry future knowledge times; see below
countries = [c.strip() for c in args.countries.split(",") if c.strip()]
idx = idx.filter(pl.col("country").is_in(countries))
junk = idx.filter(pl.col("knowledge_time") > now); idx = idx.filter(pl.col("knowledge_time") <= now)
print(f"[index] {idx.height:,} filings, {idx['lei'].n_unique():,} issuers in {countries}; dropped {junk.height} filings whose index period_end lies in the future (knowledge_time unrecoverable)", flush=True)
if args.limit:
    idx = idx.sort("knowledge_time", descending=True).head(args.limit)
t0 = time.perf_counter()
facts = get("loader", "esef_facts")(idx, cache_dir=os.path.join(args.root, "esef_cache"), workers=args.workers, sleep=args.sleep, log=lambda s: print(s, flush=True))
if facts.height == 0:
    raise SystemExit("no facts parsed")
store.register_table("esef_facts", ["lei"]); h = store.append("esef_facts", facts)
print(f"[esef_facts] {facts.height:,} rows, {facts['lei'].n_unique():,} issuers, periods {facts['event_time'].min():%Y-%m-%d}..{facts['event_time'].max():%Y-%m-%d} hash={h[:10]} ({time.perf_counter()-t0:.0f}s)", flush=True)
# ---- sanity checks (§14): read these before trusting anything ----
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(160)
print("[sanity] period types:", facts["period_type"].value_counts().sort("count", descending=True).to_dict(as_series=False))
print("[sanity] filings per issuer:", facts.group_by("lei").agg(pl.len()).get_column("len").value_counts().sort("len").to_dict(as_series=False))
print("[sanity] coverage of core fields (share non-null):", {c: round(float(facts[c].is_not_null().mean()), 3) for c in ("eps_diluted", "eps_basic", "revenue", "profit_owners", "profit", "equity", "assets", "shares_outstanding", "weighted_shares", "cfo")})
r = facts.filter(pl.col("eps_ratio").is_not_null()).with_columns(pl.col("eps_ratio").abs().alias("ar"))
b = r.with_columns(pl.when(pl.col("ar") < 0.5).then(pl.lit("<0.5")).when(pl.col("ar") < 2).then(pl.lit("0.5-2 (consistent)")).when(pl.col("ar") < 50).then(pl.lit("2-50")).when(pl.col("ar") < 200).then(pl.lit("50-200 (pence/cents?)")).otherwise(pl.lit(">200")).alias("bucket"))
print("[sanity] reported EPS / implied EPS by country (rows):"); print(b.pivot(index="country", on="bucket", values="lei", aggregate_function="len").fill_null(0).sort("country"))
print("[sanity] EPS units:", facts["eps_unit"].value_counts().sort("count", descending=True).head(8).to_dict(as_series=False))
print("[sanity] currencies:", facts["currency"].value_counts().sort("count", descending=True).head(10).to_dict(as_series=False))
dup = facts.group_by(["lei", "event_time"]).agg(pl.len().alias("n")).filter(pl.col("n") > 1)
print(f"[sanity] restated/duplicated (lei, period): {dup.height:,} of {facts.n_unique(['lei','event_time']):,} periods (each revision keeps its own knowledge_time)")
lag = facts.with_columns(((pl.col("knowledge_time") - pl.col("event_time")).dt.total_days()).alias("lag_days"))
print("[sanity] filing lag days (period end -> added), quantiles 5/50/95:", [float(lag["lag_days"].quantile(q)) for q in (0.05, 0.5, 0.95)])
jump = (facts.filter(pl.col("revenue").is_not_null() & pl.col("revenue_prior").is_not_null() & (pl.col("revenue") > 0) & (pl.col("revenue_prior") > 0))
             .with_columns((pl.col("revenue") / pl.col("revenue_prior")).log().abs().alias("j")))
print(f"[sanity] |log revenue / prior| > 1.0 (year-to-date contamination or unit change): {int((jump['j'] > 1.0).sum())} of {jump.height:,} filings with both")
print("[done]")
