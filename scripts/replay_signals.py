"""Frozen-clock replay (plan/PHASES-BY-MARKET.md Phase IN task 0): run the EXACT nightly builder (`emporion.signals.daily.build_signals`)
with the clock frozen at every month-end in a range, reading the bitemporal store `as_of` that date only, then score each file's
target book against the equal-weight book of the same point-in-time universe over the next 20 and 60 sessions, with costs.

What it proves: research <-> live parity and the point-in-time machinery end to end (a gap to the research ladder is a leak).
What it does not prove: the signal itself (its choice post-dates the data) — every row is ledgered as kind "rehearsal", never "trial".
    python scripts/replay_signals.py --market nse --root runs/nse --from 2023-01-31 --to 2026-06-30 [--cost-bps 15] [--holdout-to 2019-12-31]
Outputs: runs/signals-replay/<market>/<market>-<date>.json (+ chain.txt), runs/<root>/replay-<market>-<stamp>.json, reports/REPLAY-<market>-<stamp>.md.
"""
from __future__ import annotations

import argparse, datetime as dt, json, os, sys
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from emporion.signals.daily import build_signals
import hashlib


def month_ends(a: dt.date, b: dt.date):
    d = a.replace(day=1); out = []
    while d <= b:
        nxt = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1); me = nxt - dt.timedelta(days=1)
        if a <= me <= b:
            out.append(me)
        d = nxt
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--market", default="nse"); ap.add_argument("--root", default="runs/nse"); ap.add_argument("--from", dest="d0", default="2023-01-31")
    ap.add_argument("--to", dest="d1", default="2026-06-30"); ap.add_argument("--cost-bps", type=float, default=15.0); ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--label", default="", help="e.g. holdout-2016-2019; goes into the scope"); args = ap.parse_args()
    store = BitemporalStore(os.path.join(args.root, "store")); now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    out_dir = os.path.join("runs", "signals-replay", args.market + (f"-{args.label}" if args.label else "")); os.makedirs(out_dir, exist_ok=True)
    ledger = Ledger(os.path.join(args.root, "ledger")); scope = f"replay-{args.market}" + (f"-{args.label}" if args.label else "")
    bars = store.as_of("bars", now, columns=["close"]); wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time")
    syms = [c for c in wide.columns if c != "event_time"]; wide = wide.with_columns([pl.col(c).fill_null(strategy="forward") for c in syms])
    dates = wide["event_time"].to_list(); P = wide.select(syms).to_numpy().astype(float); si = {s: i for i, s in enumerate(syms)}
    def fwd(t0, h, idx):
        t1 = min(t0 + h, len(dates) - 1); r = P[t1, idx] / P[t0, idx] - 1.0; return np.where(np.isfinite(r), r, 0.0), t1
    rows = []; books = []
    for me in month_ends(dt.date.fromisoformat(args.d0), dt.date.fromisoformat(args.d1)):
        asof = dt.datetime(me.year, me.month, me.day, 23, 59)                       # the clock: end of the month-end day, nothing after it is knowable
        try:
            sig = build_signals(args.market, args.root, K=args.K, asof=asof)
        except Exception as e:  # noqa: BLE001
            print(f"[replay] {me}: builder failed {str(e)[:80]}"); continue
        sig["replay"] = {"frozen_clock": asof.isoformat(timespec="seconds"), "built": now.isoformat(timespec="seconds"), "kind": "rehearsal"}
        # own file + own chain (NOT publish(): that ledgers under the live scope 'shadow-signals'); ledger rows are kind 'note' marked rehearsal
        body = json.dumps(sig, sort_keys=True, separators=(",", ":")).encode(); h = hashlib.sha256(body).hexdigest()
        open(os.path.join(out_dir, f"{sig['market']}-{me.isoformat()}.json"), "wb").write(body)
        cp = os.path.join(out_dir, "chain.txt"); prev = open(cp).read().split()[-1] if os.path.exists(cp) and open(cp).read().strip() else "0" * 64
        link = hashlib.sha256((prev + h).encode()).hexdigest(); open(cp, "a").write(f"{asof.isoformat(timespec='seconds')} {sig['market']} {h} {link}\n")
        pub = {"sha256": h}
        t0 = max(i for i, d in enumerate(dates) if d <= asof)                     # decision at the close of the last bar known at asof
        idx = np.array([si[r["symbol"]] for r in sig["rows"]]); w = np.array([r["target_weight"] for r in sig["rows"]]); ew = np.ones(len(idx)) / len(idx)
        rec = {"date": me.isoformat(), "last_bar": sig["last_bar"], "n_scored": sig["n_scored"], "n_positions": sig["n_positions"], "abstain": sig["abstain"], "sha256": pub["sha256"]}
        for h in (20, 60):
            r, t1 = fwd(t0, h, idx); cost = args.cost_bps / 1e4 * 2                # in and out, one-way each
            rec[f"book_h{h}"] = float((w * r).sum() - cost * (w > 0).sum() / max(1, (w > 0).sum()) * 1.0); rec[f"ew_h{h}"] = float((ew * r).sum() - cost)
            rec[f"excess_h{h}"] = rec[f"book_h{h}"] - rec[f"ew_h{h}"]; rec[f"realised_to_h{h}"] = dates[t1].strftime("%Y-%m-%d")
        rows.append(rec); books.append((me, sig["n_positions"]))
        ledger.append("note", f"replay:{args.market}:{me}", {"kind": "rehearsal", **rec}, scope=scope)
        print(f"[replay] {me} last_bar={sig['last_bar']} scored={sig['n_scored']:3d} pos={sig['n_positions']:3d} abstain={sig['abstain']} | h20 book {rec['book_h20']*100:+.2f}% ew {rec['ew_h20']*100:+.2f}% | h60 book {rec['book_h60']*100:+.2f}% ew {rec['ew_h60']*100:+.2f}%", flush=True)
    if not rows:
        raise SystemExit("no replay rows")
    df = pl.DataFrame(rows); summ = {}
    for h in (20, 60):
        ex = df[f"excess_h{h}"].to_numpy(); bk = df[f"book_h{h}"].to_numpy()
        eq = np.cumprod(1 + bk); dd = float((eq / np.maximum.accumulate(eq) - 1).min())
        summ[f"h{h}"] = {"n": int(len(ex)), "mean_excess": float(ex.mean()), "t_excess": float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))) if len(ex) > 1 else None,
                         "months_beating_ew": int((ex > 0).sum()), "worst_excess": float(ex.min()), "best_excess": float(ex.max()), "book_max_drawdown_chained": dd,
                         "longest_losing_run": int(max((len(list(g)) for k, g in __import__("itertools").groupby(ex < 0) if k), default=0))}
    stamp = now.strftime("%Y%m%d-%H%M"); rep = os.path.join(args.root, f"replay-{args.market}{'-' + args.label if args.label else ''}-{stamp}.json")
    json.dump({"market": args.market, "scope": scope, "cost_bps": args.cost_bps, "summary": summ, "rows": rows}, open(rep, "w"), indent=1)
    md = [f"# Frozen-clock replay — {args.market}{' (' + args.label + ')' if args.label else ''} — {now:%Y-%m-%d}", "",
          f"Built with the live builder (`emporion/signals/daily.py`) at {len(rows)} month-ends {rows[0]['date']} → {rows[-1]['date']}, clock frozen at each; "
          f"costs {args.cost_bps} bp one-way in and out; benchmark = equal weight of the same point-in-time universe. **Rehearsal, not evidence** (scope `{scope}`).", ""]
    for h in (20, 60):
        s = summ[f"h{h}"]
        md.append(f"- **{h} sessions**: mean excess {s['mean_excess']*100:+.2f} %/month (t {s['t_excess']:+.2f}), beat EW in {s['months_beating_ew']}/{s['n']} months, "
                  f"worst {s['worst_excess']*100:+.2f} %, best {s['best_excess']*100:+.2f} %, longest losing run {s['longest_losing_run']} months, chained book max drawdown {s['book_max_drawdown_chained']*100:.1f} %")
    md += ["", "| month-end | last bar | scored | positions | book 20d | EW 20d | book 60d | EW 60d |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['date']} | {r['last_bar']} | {r['n_scored']} | {r['n_positions']} | {r['book_h20']*100:+.2f} % | {r['ew_h20']*100:+.2f} % | {r['book_h60']*100:+.2f} % | {r['ew_h60']*100:+.2f} % |" for r in rows]
    os.makedirs("reports", exist_ok=True); mdp = os.path.join("reports", f"REPLAY-{args.market}{'-' + args.label if args.label else ''}-{stamp}.md"); open(mdp, "w").write("\n".join(md) + "\n")
    assert ledger.verify_chain(); print(f"[done] {len(rows)} months; {rep}; {mdp}; summary {json.dumps(summ)}")


if __name__ == "__main__":
    main()
