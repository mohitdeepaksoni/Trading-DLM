"""NSE corporate-event tables -> runs/nse/store (Phase IN data steps 3 and 6): sast, announcements (history 2016 ->), insider (per symbol
of the 2016-2026 point-in-time union), pledge and largedeals (current snapshots; accumulate nightly).
    python scripts/build_nse_events.py --sources sast,announcements,insider,pledge,largedeals --start 2016-01-01
Idempotent: sast/announcements skip months already present (by event_time); insider/pledge/largedeals append snapshots (the store
resolves revisions per key). Runs on the box (nseindia.com answers from it)."""
import argparse, datetime as dt, os, sys, time
import polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.registry import get
from emporion.data.store import BitemporalStore
import emporion.data.loaders  # noqa: F401
from emporion.data.loaders.nse_events import NseSession

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs/nse"); ap.add_argument("--sources", default="sast,announcements,insider,pledge,largedeals")
ap.add_argument("--start", default="2016-01-01"); ap.add_argument("--end", default=dt.date.today().isoformat()); ap.add_argument("--symbols-file", default="runs/nse/pit_union_2016.txt")
ap.add_argument("--chunk-months", type=int, default=3); args = ap.parse_args()
log = lambda m: print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {m}", flush=True)
store = BitemporalStore(os.path.join(args.root, "store")); now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None); sess = NseSession(); t0 = time.perf_counter()
KEYS = {"sast": ["symbol", "application_no"], "announcements": ["symbol", "seq_id"], "insider": ["symbol", "did"], "pledge": ["company"], "largedeals": ["symbol", "kind", "client", "side"]}
for src in [x.strip() for x in args.sources.split(",") if x.strip()]:
    store.register_table(src, KEYS[src])
    if src in ("sast", "announcements"):
        have = store.as_of(src, now, columns=["knowledge_time"]) if os.path.isdir(os.path.join(args.root, "store", src)) and any(f.endswith(".parquet") for f in os.listdir(os.path.join(args.root, "store", src))) else pl.DataFrame()
        have_months = set(have["event_time"].dt.strftime("%Y-%m").unique().to_list()) if have.height else set()
        a = dt.date.fromisoformat(args.start); end = dt.date.fromisoformat(args.end); total = 0
        while a <= end:
            m = a.month - 1 + args.chunk_months; c = min(end, dt.date(a.year + m // 12, m % 12 + 1, 1) - dt.timedelta(days=1))
            if all(f"{a.year + (a.month - 1 + k) // 12}-{(a.month - 1 + k) % 12 + 1:02d}" in have_months for k in range(args.chunk_months)) and c < end:
                a = c + dt.timedelta(days=1); continue
            try:
                df = get("loader", f"nse_{src}")(a.isoformat(), c.isoformat(), session=sess, log=lambda m: None)
            except Exception as e:  # noqa: BLE001
                log(f"[{src}] {a}..{c} FAILED {str(e)[:80]}"); a = c + dt.timedelta(days=1); continue
            if df.height:
                h = store.append(src, df); total += df.height
                log(f"[{src}] {a}..{c}: {df.height:,} rows, {df['symbol'].n_unique()} symbols hash={h[:10]} ({time.perf_counter()-t0:.0f}s)")
            else:
                log(f"[{src}] {a}..{c}: 0 rows")
            a = c + dt.timedelta(days=1)
        log(f"[{src}] done: +{total:,} rows")
    elif src == "insider":
        syms = [l.strip() for l in open(args.symbols_file) if l.strip()]
        df = get("loader", "nse_insider")(syms, session=sess, log=log)
        if df.height:
            h = store.append("insider", df); log(f"[insider] {df.height:,} rows, {df['symbol'].n_unique()} symbols, {df['event_time'].min():%Y-%m-%d}..{df['event_time'].max():%Y-%m-%d} hash={h[:10]}")
    else:
        df = get("loader", f"nse_{src}")(session=sess, log=log)
        if df.height:
            h = store.append(src, df); log(f"[{src}] {df.height:,} rows appended hash={h[:10]}")
log("[done]")
