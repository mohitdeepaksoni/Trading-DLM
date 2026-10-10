"""Data inventory and completeness audit — runs inside the nightly job and ships with the chain (runs/signals/DATA-INVENTORY.md + .json).
Answers, mechanically and every night, "is the data we are working on complete end to end?":
  * every bitemporal table per market: rows, symbols, event-time span, latest knowledge time, parquet parts
  * freshness: the latest stored session vs the latest session the market actually had (weekday calendar; holidays listed as unverified)
  * bar continuity since the store's start: sessions present vs weekdays (gaps listed, so a missed bhavcopy is visible, not silent)
  * fundamentals coverage of the point-in-time union (research/03 §16): names with filings / names ever in the PIT top-K
  * paper book state and the signal-chain length
Known structural limits are printed alongside so a gap that cannot be filled is not mistaken for a job failure.
    python scripts/data_inventory.py [--out runs/signals]
"""
from __future__ import annotations

import argparse, datetime as dt, glob, json, os, sys
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.data.store import BitemporalStore

MARKETS = {"nse": "runs/nse", "us_stocks_wide": "runs/us_stocks_wide", "crypto_full": "runs/crypto_full", "eu_uk": "runs/eu_uk", "eu_sectors": "runs/eu_sectors"}
KNOWN_LIMITS = {
    "nse": ["results/shareholding: 370 of the 417 names ever in the 2016-2026 PIT top-200 have filings on NSE's API (387 symbols in the table incl. non-union names); the rest have none on the API (not a job failure)",
            "insider: per-symbol history for 336 of 417 names; ~72 % of rows are employee ESOP allotments", "pledge: no history on the endpoint — snapshots accumulate from 2026-09-08",
            "largedeals: historical endpoint answers 503 — snapshots accumulate from 2026-09-08", "sast: NSE's own feed published nothing dated 09-09..09-22 and back-filled later; refresh re-reads 2 days back",
            "bars 2016 →: legacy bhavcopy; symbol renames appear as two symbols"],
    "us_stocks_wide": ["consensus snapshots exist from 2026-09-06 only", "13F aggregates: fullest-batch rule (§15); quarterly, ~45-day lag"],
    "crypto_full": ["CoinGecko supply snapshots from 2026-09-06 only"],
    "eu_uk": ["Yahoo research bars for 2,018 of 3,040 ESEF issuers; DE and IE not on filings.xbrl.org; annual filings with a ~110-day median lag"],
}


def weekdays(a: dt.date, b: dt.date):
    d = a
    while d <= b:
        if d.weekday() < 5:
            yield d
        d += dt.timedelta(days=1)


def table_stats(store: BitemporalStore, t: str, now: dt.datetime) -> dict:
    parts = glob.glob(os.path.join(store.root, t, "part-*.parquet"))
    if not parts:
        return {"table": t, "parts": 0, "rows": 0}
    df = store.as_of(t, now)
    keys = store.keys(t); sym_col = "symbol" if "symbol" in df.columns else (keys[0] if keys else None)
    return {"table": t, "parts": len(parts), "rows": int(df.height), "symbols": int(df[sym_col].n_unique()) if sym_col and df.height else None,
            "event_min": str(df["event_time"].min())[:10] if df.height else None, "event_max": str(df["event_time"].max())[:10] if df.height else None,
            "knowledge_max": str(df["knowledge_time"].max())[:16] if df.height else None}


def bars_continuity(store: BitemporalStore, now: dt.datetime, start: dt.date, market: str) -> dict:
    b = store.as_of("bars", now, columns=["close"], event_start=dt.datetime(start.year, start.month, start.day))
    have = set(b["event_time"].dt.date().unique().to_list())
    last_have = max(have) if have else None
    if market.startswith("crypto"):
        exp = {start + dt.timedelta(days=i) for i in range((now.date() - start).days)}
    else:
        exp = set(weekdays(start, now.date() - dt.timedelta(days=1)))
    missing = sorted(exp - have)
    # weekdays with no bar: holidays or misses — we cannot tell without a calendar, so list the recent ones for a human to check
    recent_missing = [d.isoformat() for d in missing if d >= now.date() - dt.timedelta(days=45)]
    return {"sessions_stored": len(have), "weekdays_expected": len(exp), "weekdays_without_bar": len(missing), "last_session_stored": last_have.isoformat() if last_have else None,
            "days_since_last_session": (now.date() - last_have).days if last_have else None, "recent_weekdays_without_bar": recent_missing}


def pit_coverage(root: str, store: BitemporalStore, now: dt.datetime) -> dict:
    out = {}
    uf = [os.path.join(root, "pit_union_2016.txt"), os.path.join(root, "pit_union_all.txt")]
    uf = [f for f in uf if os.path.exists(f)]
    if not uf:
        return out
    union = set(l.strip() for l in open(uf[0]) if l.strip()); out["union_file"] = os.path.basename(uf[0]); out["union_names"] = len(union)
    for t in ("results", "shareholding", "insider", "delivery"):
        if t in store.tables():
            have = set(store.as_of(t, now, columns=[]).get_column("symbol").unique().to_list()) if glob.glob(os.path.join(store.root, t, "part-*.parquet")) else set()
            out[t] = {"covered": len(union & have), "of": len(union)}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/signals"); args = ap.parse_args()
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None); inv = {"as_of": now.isoformat(timespec="seconds"), "markets": {}}
    for m, root in MARKETS.items():
        sp = os.path.join(root, "store")
        if not os.path.isdir(sp):
            continue
        st = BitemporalStore(sp); tables = [t for t in st.tables() if glob.glob(os.path.join(sp, t, "part-*.parquet"))]
        rec = {"tables": [table_stats(st, t, now) for t in tables]}
        if "bars" in tables:
            start = dt.date(2016, 1, 1) if m in ("nse", "eu_uk") else dt.date(2014, 1, 1) if m == "eu_sectors" else dt.date(2015, 1, 1)
            rec["bars"] = bars_continuity(st, now, start, m)
        if m == "nse":
            rec["pit_coverage"] = pit_coverage(root, st, now)
        rec["known_limits"] = KNOWN_LIMITS.get(m, []); inv["markets"][m] = rec
    chain = os.path.join(args.out, "chain.txt"); inv["signal_chain_entries"] = len([l for l in open(chain).read().splitlines() if l.strip()]) if os.path.exists(chain) else 0
    sp = "runs/paper/nse/state.json"
    if os.path.exists(sp):
        s = json.load(open(sp)); inv["paper_nse"] = {b: {"nav": bk["nav"][-1]["nav"] if bk["nav"] else None, "date": bk["nav"][-1]["date"] if bk["nav"] else None, "fills": len(bk["fills"]), "pending": len(bk["pending"])} for b, bk in s["books"].items()}
    os.makedirs(args.out, exist_ok=True); json.dump(inv, open(os.path.join(args.out, "data-inventory.json.txt"), "w"), indent=1, default=str)   # .json.txt: the signals folder's *.json namespace belongs to signal files only
    md = [f"# Data inventory — {now:%Y-%m-%d %H:%M} UTC", "", f"Signal chain: {inv['signal_chain_entries']} entries. Every table below is a bitemporal store (event_time / knowledge_time); rows are as of now.", ""]
    for m, rec in inv["markets"].items():
        md += [f"## {m}", "", "| table | parts | rows | symbols | event span | latest knowledge |", "|---|---|---|---|---|---|"]
        md += [f"| {t['table']} | {t['parts']} | {t['rows']:,} | {t.get('symbols') or ''} | {t.get('event_min') or ''} → {t.get('event_max') or ''} | {t.get('knowledge_max') or ''} |" for t in rec["tables"]]
        if "bars" in rec:
            b = rec["bars"]; md += ["", f"Bars: {b['sessions_stored']:,} sessions stored; last session {b['last_session_stored']} ({b['days_since_last_session']} days ago); "
                                   f"weekdays without a bar since the store's start: {b['weekdays_without_bar']} (holidays included); recent ones to check: {', '.join(b['recent_weekdays_without_bar']) or 'none'}"]
        if rec.get("pit_coverage"):
            c = rec["pit_coverage"]; md += ["", "Point-in-time union coverage (" + c.get("union_file", "") + f", {c.get('union_names')} names): " + "; ".join(f"{k} {v['covered']}/{v['of']}" for k, v in c.items() if isinstance(v, dict))]
        if rec["known_limits"]:
            md += ["", "Known structural limits (not job failures):"] + [f"- {x}" for x in rec["known_limits"]]
        md.append("")
    if "paper_nse" in inv:
        md += ["## Paper book (NSE)", "", "| book | NAV | as of | fills | pending |", "|---|---|---|---|---|"] + [f"| {b} | {v['nav']:,.0f} | {v['date']} | {v['fills']} | {v['pending']} |" for b, v in inv["paper_nse"].items() if v["nav"]]
    open(os.path.join(args.out, "DATA-INVENTORY.md"), "w").write("\n".join(md) + "\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
