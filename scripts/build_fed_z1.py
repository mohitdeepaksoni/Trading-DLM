"""Parse the Z.1 package into the US store (table z1) and print the mandated share phi over time."""
import os, sys, datetime as dt, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.registry import get
from emporion.data.store import BitemporalStore
import emporion.data.loaders  # noqa: F401
from emporion.data.loaders.fed_z1 import mandated_share
t0 = time.perf_counter()
df = get("loader", "fed_z1")(sys.argv[1] if len(sys.argv) > 1 else "runs/fed_z1/z1/Z1_data.xml")
store = BitemporalStore("runs/us_stocks_wide/store")
if "z1" not in store.tables(): store.register_table("z1", ["sector"])
else: store.keys("z1")
h = store.append("z1", df)
print(f"[z1] {df.height} obs, {df['sector'].n_unique()} sectors, {df['event_time'].min():%Y-%m-%d}..{df['event_time'].max():%Y-%m-%d} hash={h[:10]} ({time.perf_counter()-t0:.0f}s)")
m = mandated_share(df.filter(df["event_time"] >= dt.datetime(2005, 1, 1)))
print(m.filter(m["event_time"].dt.month() == 12).select(["event_time", "phi_mandated", "phi_funds", "share_households", "share_foreign", "total"]).to_pandas().to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
