"""Crawl the ESEF/UKSEF filings index into runs/eu_uk/store/esef_index (bitemporal: period_end / date_added)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.registry import get
from emporion.data.store import BitemporalStore
import emporion.data.loaders  # noqa: F401
t0 = time.perf_counter(); df = get("loader", "esef_index")()
store = BitemporalStore("runs/eu_uk/store"); store.register_table("esef_index", ["lei", "sha256"]); h = store.append("esef_index", df)
print(f"[esef_index] {df.height:,} filings, {df['lei'].n_unique():,} issuers, countries {df['country'].value_counts().sort('count', descending=True).head(12).to_dict(as_series=False)} hash={h[:10]} ({time.perf_counter()-t0:.0f}s)")
