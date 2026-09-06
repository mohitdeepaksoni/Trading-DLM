"""Full crypto panel: every online Coinbase USD pair back to 2015 + a CoinGecko supply/market-cap snapshot."""
import datetime as dt, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.registry import get
from emporion.data.store import BitemporalStore
import emporion.data.loaders  # noqa: F401
root = "runs/crypto_full"; store = BitemporalStore(os.path.join(root, "store")); t0 = time.perf_counter()
bars = get("loader", "coinbase")(start="2015-01-01")
store.register_table("bars", ["symbol"]); h = store.append("bars", bars)
print(f"[bars] {bars.height:,} rows, {bars['symbol'].n_unique()} pairs, {bars['event_time'].min():%Y-%m-%d}..{bars['event_time'].max():%Y-%m-%d} hash={h[:10]} ({time.perf_counter()-t0:.0f}s)", flush=True)
snap = get("loader", "coingecko_snapshot")(pages=4)
store.register_table("supply", ["coin_id"]); h = store.append("supply", snap)
print(f"[supply] snapshot {snap.height} coins hash={h[:10]}", flush=True)
print("[done]")
