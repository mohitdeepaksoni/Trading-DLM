"""Daily snapshot of analyst consensus for the US stock universe into the bitemporal store (knowledge_time = now).
Run nightly (RT2 shadow job); the history it builds is the only honest forward-consensus anchor."""
import datetime as dt, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.registry import get
from emporion.data.store import BitemporalStore
import emporion.data.loaders  # noqa: F401

root = sys.argv[1] if len(sys.argv) > 1 else "runs/us_stocks"
store = BitemporalStore(os.path.join(root, "store"))
now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
syms = store.as_of("universe", now)["symbol"].to_list()
t0 = time.perf_counter()
df = get("loader", "nasdaq_consensus")(syms)
if "consensus" not in store.tables():
    store.register_table("consensus", ["symbol", "kind", "period_end"])
else:
    store.keys("consensus")
h = store.append("consensus", df)
print(f"[consensus] snapshot {now:%Y-%m-%d %H:%M} UTC: {df.height:,} rows for {df['symbol'].n_unique()} symbols "
      f"({int((df['kind']=='UpcomingQuarter').sum())} upcoming-quarter estimates) hash={h[:10]} {time.perf_counter()-t0:.0f}s")
