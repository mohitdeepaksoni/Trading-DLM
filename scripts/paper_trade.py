"""Stage S1 paper trading (research/05: "live signals, simulated fills with real quotes, 3–6 months; exit: implementation shortfall
inside the bootstrap CI"). Broker-free and deterministic, so it runs inside the nightly job and its state travels with the chain.

Pre-declared rules (2026-09-09; changing any of them is a new scope):
  * Book = the market's newest pre-registered signal file (e.g. runs/signals/nse-<date>.json, built 07:00 IST from the previous close).
  * Rebalance on the FIRST NSE session of each calendar month from that morning's file (and once at inception); hold otherwise.
    Target quantity = floor(target_weight x NAV / decision price), decision price = the file's `close` (last known bar).
  * Orders are market-on-open the same day: fill = that session's OPEN from the bhavcopy (the first traded price), settled by the next
    nightly run once the bar exists; a holiday rolls the order to the next session. Costs: emporion/evaluate/costs_in (STT, stamp on buys,
    exchange, 5 bp slippage). Implementation shortfall per order = signed (fill - decision price) / decision price, in bp, before costs.
  * Two books are run identically: `signal` (the file's weights) and `ew` (equal weight over the file's universe) — the benchmark is
    traded the same way, so the difference is the signal, not the execution.
  * NAV marked at the latest close every night; every fill and every NAV row is a ledger entry (kind `paper`, scope `paper-<market>`).
State: runs/paper/<market>/state.json (positions, cash, pending orders, NAV history). Output: runs/paper/<market>/PAPER.md.
    python scripts/paper_trade.py --market nse [--capital 1000000]
"""
from __future__ import annotations

import argparse, datetime as dt, glob, json, math, os, sys
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from emporion.evaluate.costs_in import one_way_cost_bps, buy_tax_bps

ROOTS = {"nse": "runs/nse"}


def load_state(path, capital):
    if os.path.exists(path):
        return json.load(open(path))
    return {"capital": capital, "started": None, "last_rebalance_month": None,
            "books": {b: {"cash": capital, "positions": {}, "pending": [], "fills": [], "nav": []} for b in ("signal", "ew")}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--market", default="nse"); ap.add_argument("--signals", default="runs/signals"); ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--out", default="runs/paper"); ap.add_argument("--slippage-bps", type=float, default=5.0); args = ap.parse_args()
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None); out = os.path.join(args.out, args.market); os.makedirs(out, exist_ok=True)
    state_path = os.path.join(out, "state.json"); st = load_state(state_path, args.capital); ledger = Ledger(os.path.join(out, "ledger")); scope = f"paper-{args.market}"
    store = BitemporalStore(os.path.join(ROOTS[args.market], "store"))
    bars = store.as_of("bars", now, columns=["open", "close"], event_start=now - dt.timedelta(days=200))
    sessions = sorted(bars["event_time"].unique().to_list()); last_session = sessions[-1]
    px = {(r["symbol"], r["event_time"]): (r["open"], r["close"]) for r in bars.iter_rows(named=True)}
    last_close = {r["symbol"]: r["close"] for r in bars.filter(pl.col("event_time") == last_session).iter_rows(named=True)}
    files = sorted(glob.glob(os.path.join(args.signals, f"{args.market}-*.json"))); sig = json.load(open(files[-1])) if files else None
    log = lambda m: print(f"[paper:{args.market}] {m}", flush=True)
    cost_bps = one_way_cost_bps(slippage_bps=args.slippage_bps); tax_bps = buy_tax_bps()

    def first_session_on_or_after(d):
        for s_ in sessions:
            if s_ >= d:
                return s_
        return None

    # 1. settle pending orders whose session bar now exists
    for bname, book in st["books"].items():
        keep = []
        for o in book["pending"]:
            s_ = first_session_on_or_after(dt.datetime.fromisoformat(o["order_date"]))
            if s_ is None or s_ > last_session or (o["symbol"], s_) not in px:
                keep.append(o); continue
            fill = px[(o["symbol"], s_)][0]
            if not fill or not math.isfinite(fill) or fill <= 0:
                keep.append(o); continue
            q = o["qty"]; notional = abs(q) * fill; cost = notional * cost_bps / 1e4 + (notional * tax_bps / 1e4 if q > 0 else 0.0)
            book["cash"] -= q * fill + cost; book["positions"][o["symbol"]] = book["positions"].get(o["symbol"], 0) + q
            if book["positions"][o["symbol"]] == 0:
                del book["positions"][o["symbol"]]
            shortfall_bp = (fill - o["decision_price"]) / o["decision_price"] * 1e4 * (1 if q > 0 else -1)
            f = {"symbol": o["symbol"], "qty": q, "fill": fill, "session": s_.strftime("%Y-%m-%d"), "decision_price": o["decision_price"], "decision_file": o["decision_file"],
                 "shortfall_bp": round(shortfall_bp, 2), "cost": round(cost, 2)}
            book["fills"].append(f); ledger.append("paper", f"{bname}:fill:{o['symbol']}:{f['session']}", f, scope=scope)
        book["pending"] = keep
        if book["fills"]:
            log(f"{bname}: {len(book['fills'])} fills total, {len(keep)} pending")
    # 2. NAV at the latest close
    for bname, book in st["books"].items():
        mv = sum(q * last_close.get(s_, 0.0) for s_, q in book["positions"].items()); nav = book["cash"] + mv
        row = {"date": last_session.strftime("%Y-%m-%d"), "nav": round(nav, 2), "cash": round(book["cash"], 2), "n_positions": len(book["positions"]), "pending": len(book["pending"])}
        if not book["nav"] or book["nav"][-1]["date"] != row["date"]:
            book["nav"].append(row); ledger.append("paper", f"{bname}:nav:{row['date']}", row, scope=scope)
        log(f"{bname}: NAV {nav:,.0f} ({(nav / st['capital'] - 1) * 100:+.2f} % since inception), {len(book['positions'])} positions, cash {book['cash']:,.0f}")
    # 3. orders from the newest file: at inception, then on the first session of each month
    if sig and not sig.get("abstain"):
        file_day = dt.datetime.fromisoformat(sig["as_of"][:10]); month = file_day.strftime("%Y-%m")
        due = st["started"] is None or (st["last_rebalance_month"] != month and first_session_on_or_after(file_day.replace(day=1)) is not None
                                        and file_day >= first_session_on_or_after(file_day.replace(day=1)) and not any(st["books"][b]["pending"] for b in st["books"]))
        if due:
            rows = sig["rows"]; n_u = len(rows)
            for bname, book in st["books"].items():
                nav = book["nav"][-1]["nav"] if book["nav"] else book["cash"]; target = {}
                for r in rows:
                    w = r["target_weight"] if bname == "signal" else 1.0 / n_u
                    if w > 0 and r.get("close") and r["close"] > 0:
                        target[r["symbol"]] = (math.floor(w * nav / r["close"]), r["close"])
                orders = []
                for s_, (q, p_) in target.items():
                    d = q - book["positions"].get(s_, 0)
                    if d != 0:
                        orders.append({"symbol": s_, "qty": d, "decision_price": p_, "decision_file": os.path.basename(files[-1]), "order_date": file_day.isoformat()})
                for s_, q in book["positions"].items():
                    if s_ not in target and q != 0:
                        orders.append({"symbol": s_, "qty": -q, "decision_price": last_close.get(s_, 0.0) or 0.0, "decision_file": os.path.basename(files[-1]), "order_date": file_day.isoformat()})
                book["pending"] = [o for o in orders if o["decision_price"] > 0]
                log(f"{bname}: {len(book['pending'])} orders for the open of {file_day:%Y-%m-%d} from {os.path.basename(files[-1])} ({sig['strategy'][:40]})")
            st["started"] = st["started"] or now.isoformat(timespec="seconds"); st["last_rebalance_month"] = month
    elif sig:
        log(f"file {os.path.basename(files[-1])} abstains; no orders")
    json.dump(st, open(state_path, "w"), indent=1)
    # 4. page
    md = [f"# Paper trading — {args.market.upper()} — as of {last_session:%Y-%m-%d}", "", f"Stage S1 (research/05): simulated market-on-open fills at the bhavcopy open, Indian costs ({one_way_cost_bps(slippage_bps=args.slippage_bps):.1f} bp one-way + {tax_bps} bp stamp on buys), "
          f"monthly rebalance from the morning file, two books traded identically. Started {st['started'] or '—'}; capital {st['capital']:,.0f}. **A record, not a claim** (scope `{scope}`).", ""]
    md += ["| book | NAV | since inception | positions | fills | mean shortfall bp | pending |", "|---|---|---|---|---|---|---|"]
    for bname, book in st["books"].items():
        nav = book["nav"][-1]["nav"] if book["nav"] else book["cash"]; sf = [f["shortfall_bp"] for f in book["fills"]]
        md.append(f"| {bname} | {nav:,.0f} | {(nav / st['capital'] - 1) * 100:+.2f} % | {len(book['positions'])} | {len(book['fills'])} | {sum(sf) / len(sf) if sf else 0:+.1f} | {len(book['pending'])} |")
    sig_nav = st["books"]["signal"]["nav"]; ew_nav = st["books"]["ew"]["nav"]
    if sig_nav and ew_nav:
        md += ["", "| date | signal NAV | EW NAV | signal − EW % |", "|---|---|---|---|"]
        md += [f"| {a['date']} | {a['nav']:,.0f} | {b['nav']:,.0f} | {(a['nav'] / b['nav'] - 1) * 100:+.2f} |" for a, b in zip(sig_nav[-40:], ew_nav[-40:])]
    open(os.path.join(out, "PAPER.md"), "w").write("\n".join(md) + "\n"); assert ledger.verify_chain(); log("done")


if __name__ == "__main__":
    main()
