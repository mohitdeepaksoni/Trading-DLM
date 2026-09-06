"""The nightly shadow job (RT2): refresh point-in-time data, snapshot forward-looking sources, build and publish the
pre-registered signal files for every market, chain + ledger + OpenTimestamps.  Trades nothing.
    python scripts/shadow_daily.py            # all markets
"""
from __future__ import annotations

import datetime as dt, os, subprocess, sys, time
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.registry import get
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
import emporion.data.loaders  # noqa: F401
from emporion.signals import build_signals, publish

OUT = "runs/signals"; LEDGER = Ledger(os.path.join(OUT, "ledger"))
now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
log = lambda m: print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {m}", flush=True)


def refresh_bars(root, loader, **kw):
    store = BitemporalStore(os.path.join(root, "store"))
    last = store.as_of("bars", now, columns=["close"])["event_time"].max()
    start = (last - dt.timedelta(days=5)).strftime("%Y-%m-%d")
    syms = store.as_of("bars", now, columns=["close"], event_start=now - dt.timedelta(days=30))["symbol"].unique().to_list()
    try:
        df = get("loader", loader)(symbols=syms, start=start, **kw)
        df = df.filter(pl.col("event_time") > last)
        if df.height:
            store.append("bars", df)
        log(f"{root}: +{df.height} bars via {loader} (last was {last:%Y-%m-%d})")
    except Exception as e:  # noqa: BLE001
        log(f"{root}: bars refresh FAILED {str(e)[:80]}")


t0 = time.perf_counter()
refresh_bars("runs/us_stocks_wide", "nasdaq", assetclass="stocks")
refresh_bars("runs/crypto_full", "coinbase")
for name, fn in (("consensus", lambda: subprocess.run([sys.executable, "scripts/snapshot_consensus.py", "runs/us_stocks_wide"], check=False)),):
    try:
        fn(); log("consensus snapshot done")
    except Exception as e:  # noqa: BLE001
        log(f"consensus FAILED {str(e)[:60]}")
try:
    snap = get("loader", "coingecko_snapshot")(pages=2); s = BitemporalStore("runs/crypto_full/store"); s.keys("supply"); s.append("supply", snap); log(f"coingecko snapshot {snap.height} coins")
except Exception as e:  # noqa: BLE001
    log(f"coingecko FAILED {str(e)[:60]}")
try:
    fl = get("loader", "nse_flows")(); s = BitemporalStore("runs/nse/store")
    if "flows" not in s.tables():
        s.register_table("flows", ["category"])
    else:
        s.keys("flows")
    s.append("flows", fl); log(f"NSE flows snapshot {fl.height} rows")
except Exception as e:  # noqa: BLE001
    log(f"NSE flows FAILED {str(e)[:60]}")
for market, root, key in (("us_stocks_wide", "runs/us_stocks_wide", "primary"), ("us_stocks_wide", "runs/us_stocks_wide", "composite"),
                          ("crypto_full", "runs/crypto_full", "primary"), ("nse", "runs/nse", "primary")):
    try:
        sig = build_signals(market, root, K=200 if market != "crypto_full" else 100, strategy_key=key)
        rec = publish(sig, OUT, LEDGER)
        log(f"{sig['market']}: {sig['strategy']} | universe {sig['universe_size']} scored {sig['n_scored']} positions {sig['n_positions']} abstain={sig['abstain']} | sha256 {rec['sha256'][:16]} chain {rec['chain'][:16]} ledger#{rec['ledger_seq']}")
        try:
            subprocess.run(["ots", "stamp", rec["path"]], check=True, capture_output=True, timeout=120); log(f"{sig['market']}: OpenTimestamps proof {os.path.basename(rec['path'])}.ots")
        except Exception as e:  # noqa: BLE001
            log(f"{market}: ots stamp skipped ({str(e)[:50]})")
    except Exception as e:  # noqa: BLE001
        log(f"{market}: signal FAILED {str(e)[:100]}")
assert LEDGER.verify_chain(); log(f"done in {time.perf_counter()-t0:.0f}s; chain ok")
