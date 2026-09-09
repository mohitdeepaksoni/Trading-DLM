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


def refresh_delivery(root="runs/nse"):
    """NSE delivery position (MTO) for the sessions since the last one in the store; knowledge_time = session + 1 day."""
    store = BitemporalStore(os.path.join(root, "store"))
    if "delivery" not in store.tables():
        return
    last = store.as_of("delivery", now, columns=["deliv_per"])["event_time"].max()
    try:
        df = get("loader", "nse_delivery")((last + dt.timedelta(days=1)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"), workers=2, log=lambda m: None)
        df = df.filter(pl.col("event_time") > last)
        if df.height:
            store.append("delivery", df)
        log(f"{root}: +{df.height} delivery rows (last was {last:%Y-%m-%d})")
    except Exception as e:  # noqa: BLE001
        log(f"{root}: delivery refresh FAILED {str(e)[:80]}")


t0 = time.perf_counter()
refresh_bars("runs/us_stocks_wide", "nasdaq", assetclass="stocks")
refresh_bars("runs/crypto_full", "coinbase")
# NSE bars were never refreshed by this job until 2026-09-09 (the NSE file carried last_bar 2026-09-04 for three nights):
# bhavcopy sessions since the last stored one; min_sessions=1 because a 5-day window holds at most 4 sessions
refresh_bars("runs/nse", "nse", end=now.strftime("%Y-%m-%d"), min_sessions=1, workers=2)
refresh_delivery("runs/nse")


def refresh_events(root="runs/nse"):
    """NSE corporate-event tables (Phase IN): SAST + announcements since the last stored broadcast; pledge + bulk/block snapshots
    every night (no history on those endpoints); insider filings per symbol of the live universe on Sundays only (200 calls)."""
    from emporion.data.loaders.nse_events import NseSession  # noqa: PLC0415
    store = BitemporalStore(os.path.join(root, "store")); sess = None
    def S():
        nonlocal sess; sess = sess or NseSession(); return sess
    for t in ("sast", "announcements"):
        if t not in store.tables():
            continue
        try:
            last = store.as_of(t, now, columns=["knowledge_time"])["knowledge_time"].max()
            df = get("loader", f"nse_{t}")((last - dt.timedelta(days=2)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d"), session=S(), log=lambda m: None)
            df = df.filter(pl.col("knowledge_time") > last) if df.height else df
            if df.height:
                store.append(t, df)
            log(f"{root}: +{df.height} {t} rows (last was {last:%Y-%m-%d %H:%M})")
        except Exception as e:  # noqa: BLE001
            log(f"{root}: {t} refresh FAILED {str(e)[:80]}")
    for t in ("pledge", "largedeals"):
        if t not in store.tables():
            continue
        try:
            df = get("loader", f"nse_{t}")(session=S(), log=lambda m: None)
            if df.height:
                store.append(t, df)
            log(f"{root}: {t} snapshot {df.height} rows")
        except Exception as e:  # noqa: BLE001
            log(f"{root}: {t} snapshot FAILED {str(e)[:80]}")
    if now.weekday() == 6 and "insider" in store.tables() and os.path.exists(os.path.join(root, "pit_union_all.txt")):
        try:
            syms = [l.strip() for l in open(os.path.join(root, "pit_union_all.txt")) if l.strip()]
            df = get("loader", "nse_insider")(syms, session=S(), log=lambda m: None)
            if df.height:
                store.append("insider", df)
            log(f"{root}: insider refresh {df.height} rows for {len(syms)} symbols")
        except Exception as e:  # noqa: BLE001
            log(f"{root}: insider refresh FAILED {str(e)[:80]}")


refresh_events("runs/nse")
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
if now.weekday() == 6 and os.path.exists("runs/nse/pit_union_all.txt"):        # Sunday: refresh NSE quarterly results + shareholding (cached XBRL travels in the state)
    try:
        r = subprocess.run([sys.executable, "scripts/build_nse_fundamentals.py", "--symbols-file", "runs/nse/pit_union_all.txt", "--only", "results,shareholding", "--replace"], capture_output=True, text=True, timeout=2400)
        log("NSE fundamentals weekly refresh " + ("done" if r.returncode == 0 else f"FAILED rc={r.returncode}: {r.stderr[-200:]}"))
    except Exception as e:  # noqa: BLE001
        log(f"NSE fundamentals refresh FAILED {str(e)[:80]}")
for market, root, key in (("us_stocks_wide", "runs/us_stocks_wide", "primary"), ("us_stocks_wide", "runs/us_stocks_wide", "composite"),
                          ("crypto_full", "runs/crypto_full", "primary"), ("nse", "runs/nse", "primary")):
    tag = market if key == "primary" else f"{market}-{key}"
    if os.path.exists(os.path.join(OUT, f"{tag}-{now:%Y-%m-%d}.json")):
        log(f"{tag}: already published for {now:%Y-%m-%d} — skipped (one file per day; reruns never double-publish)"); continue
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
