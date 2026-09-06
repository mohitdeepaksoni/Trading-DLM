"""REHEARSAL of the shadow chain on past dates — NOT evidence. Rebuilds the two pre-registered US signal files for
each trading day in [start, end] from the bitemporal store as it was known that evening, publishes them into a
SEPARATE directory with its own chain and ledger, flags every file BACKFILL=true (inside the hash), never stamps them,
then reports what the live chain will look like: forward returns of each day's book vs the equal-weight universe.
    python scripts/shadow_backfill.py --start 2026-08-06 --end 2026-09-05
"""
import argparse, datetime as dt, json, os, sys
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from emporion.signals import build_signals, publish

ap = argparse.ArgumentParser(); ap.add_argument("--start", default="2026-08-06"); ap.add_argument("--end", default="2026-09-05")
ap.add_argument("--root", default="runs/us_stocks_wide"); ap.add_argument("--out", default="runs/signals-backfill"); args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); ledger = Ledger(os.path.join(args.out, "ledger"))
store = BitemporalStore(os.path.join(args.root, "store")); now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
bars = store.as_of("bars", now, columns=["close"]); wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time")
tdays = [d for d in wide["event_time"].to_list() if dt.datetime.fromisoformat(args.start) <= d <= dt.datetime.fromisoformat(args.end)]
P = wide.select([c for c in wide.columns if c != "event_time"]).to_numpy().astype(float); syms = [c for c in wide.columns if c != "event_time"]
di = {d: i for i, d in enumerate(wide["event_time"].to_list())}; si = {s: i for i, s in enumerate(syms)}
rows = []
for d in tdays:
    asof = d.replace(hour=23, minute=30)
    for key in ("primary", "composite"):
        sig = build_signals("us_stocks_wide", args.root, K=200, asof=asof, strategy_key=key)
        sig["BACKFILL"] = True; sig["note"] = "rehearsal built after the fact from point-in-time data; not pre-registered evidence; never stamped"
        rec = publish(sig, args.out, ledger)
        # forward returns of the book vs the EW universe (where the future exists in the store)
        i0 = di[d]; univ = [si[r["symbol"]] for r in sig["rows"]]; book = {si[r["symbol"]]: r["target_weight"] for r in sig["rows"] if r["target_weight"] > 0}
        fr = {}
        for h in (1, 5, 20):
            if i0 + h < P.shape[0]:
                r = P[i0 + h] / P[i0] - 1
                fr[f"book_{h}d"] = float(sum(w * r[j] for j, w in book.items())); fr[f"ew_{h}d"] = float(np.nanmean(r[univ]))
        rows.append({"date": d.strftime("%Y-%m-%d"), "strategy": sig["market"], "positions": sig["n_positions"], "abstain": sig["abstain"], "sha256": rec["sha256"][:12], **fr})
        print(f"[backfill] {d:%Y-%m-%d} {sig['market']:28s} pos={sig['n_positions']:3d} sha={rec['sha256'][:12]} " + " ".join(f"{k}={v:+.4f}" for k, v in fr.items()), flush=True)
assert ledger.verify_chain()
df = pl.DataFrame(rows)
lines = ["# Shadow-chain REHEARSAL (backfill) — not evidence", "", f"Dates {args.start}..{args.end}, files in `{args.out}/`, own chain `{args.out}/chain.txt`, every file carries BACKFILL=true inside its hash. Never stamped, never merged into the live chain.", ""]
for strat in df["strategy"].unique().sort().to_list():
    x = df.filter(pl.col("strategy") == strat)
    lines.append(f"## {strat}"); lines.append("")
    for h in (1, 5, 20):
        b, e = x[f"book_{h}d"].drop_nulls(), x[f"ew_{h}d"].drop_nulls()
        if b.len():
            lines.append(f"- {h}-day forward: book mean {b.mean()*100:+.2f} % vs EW {e.mean()*100:+.2f} % over {b.len()} dates; book beat EW on {int((b.to_numpy() > e.to_numpy()).sum())}/{b.len()}")
    lines.append("")
lines.append("| date | strategy | positions | 1d book | 1d EW | 5d book | 5d EW | 20d book | 20d EW |"); lines.append("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    f = lambda k: ("" if k not in r else f"{r[k]*100:+.2f}%")
    lines.append(f"| {r['date']} | {r['strategy']} | {r['positions']} | {f('book_1d')} | {f('ew_1d')} | {f('book_5d')} | {f('ew_5d')} | {f('book_20d')} | {f('ew_20d')} |")
with open(os.path.join(args.out, "REPORT.md"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"[done] {len(rows)} files; report {args.out}/REPORT.md")
