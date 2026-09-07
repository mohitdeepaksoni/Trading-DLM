"""P3 product surface — sector-ETF rotation file for ONE dated tariff action (research/11 decision: ETF level only).
Given an event from the event table (--event-date) or a shock file (--shock-json + --name --date), solve Agora once,
map the producer-price response to sector ETFs, and emit a book in the shadow-signal format: long the top quartile of
predicted dlogPX, short the bottom quartile, dollar-neutral, DAY-0 reaction horizon (the only window with evidence:
rank-IC +0.125, t 2.3, p 0.004, n = 20). Published into its own directory/chain (`runs/signals-events/`), ledgered under
scope `p3-event-rotation`. Pre-registration rule for live use: plan/EVENT-PREREG-RULE.md (event text hash stamped
BEFORE the US open of the event day; files built after the open are labelled post_hoc=true inside the hash).
    python scripts/event_rotation.py --event-date 2026-05-07
    python scripts/event_rotation.py --shock-json shock.json --name "..." --date 2026-09-08 --source-text "quoted action"
"""
import argparse, datetime as dt, hashlib, json, os, sys
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from emporion.signals import publish
from emporion.structure.agora_bridge import sector_response
from emporion.structure.sector_map import ETF_EXPOSURE, BENCHMARK
from emporion.structure.events_us_tariffs import events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-date", default=""); ap.add_argument("--shock-json", default=""); ap.add_argument("--name", default=""); ap.add_argument("--date", default="")
    ap.add_argument("--source-text", default="", help="the quoted official action text (hashed into the file for pre-registration)")
    ap.add_argument("--out", default="runs/signals-events"); ap.add_argument("--sectors-root", default="runs/us_sectors"); ap.add_argument("--post-hoc", action="store_true")
    ap.add_argument("--quantile", type=float, default=0.25); args = ap.parse_args()
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    if args.event_date:
        ev = [e for e in events() if e["date"] == args.event_date]
        if not ev:
            raise SystemExit(f"no event on {args.event_date} in the event table")
        ev = ev[0]; shock, te, name, date, src = ev["shock"], ev.get("te") or {}, ev["name"], ev["date"], ev.get("source", "")
        post_hoc = True                                                   # anything from the historical table is by construction after the fact
    else:
        j = json.load(open(args.shock_json)); shock, te = j.get("shock", j), j.get("te", {}); name, date, src = args.name, args.date, args.source_text; post_hoc = args.post_hoc
    resp = sector_response("USA", [shock], [te]); codes = resp["codes"]; ci = {c: i for i, c in enumerate(codes)}
    dpx = np.asarray(resp["dlogPX"][0]); etfs = sorted(ETF_EXPOSURE)
    pred = np.array([sum(w * dpx[ci[c]] for c, w in ETF_EXPOSURE[e].items()) for e in etfs])
    k = max(1, int(round(len(etfs) * args.quantile))); order = np.argsort(pred)
    w = np.zeros(len(etfs)); w[order[-k:]] = 1.0 / k; w[order[:k]] = -1.0 / k
    text_hash = hashlib.sha256((src or name).encode()).hexdigest()
    sig = {"market": "us_sectors-event", "as_of": now.isoformat(timespec="seconds"), "event": name, "event_date": date, "source_text_sha256": text_hash,
           "shock": shock, "te": te, "response": "dlogPX", "solver_residual": float(resp["max_residual"]), "post_hoc": bool(post_hoc),
           "strategy": f"event_rotation: long top {int(args.quantile*100)}% / short bottom {int(args.quantile*100)}% of Agora dlogPX mapped to sector ETFs, dollar-neutral, DAY-0 reaction only",
           "evidence": "reaction h=1 rank-IC +0.125 (t 2.30, perm-p 0.003, n=20 events, 2025-02..2026-05); tradeable-window (next day) NOT significant — this is a same-day instrument",
           "primary_score": "predicted_dlogPX", "abstain": bool(np.allclose(pred, pred[0])), "n_scored": len(etfs), "n_positions": int((w != 0).sum()), "universe_size": len(etfs),
           "last_bar": None, "rule": "ETF_EXPOSURE hand map (emporion/structure/sector_map.py), fixed 2026-09-05",
           "rows": sorted([{"symbol": e, "predicted_dlogPX": round(float(pred[i]), 6), "target_weight": round(float(w[i]), 6)} for i, e in enumerate(etfs)], key=lambda r: -r["target_weight"])}
    # realised day-0 reaction where the store already has it (historical events) — reported, never used for the weights
    store = BitemporalStore(os.path.join(args.sectors_root, "store"))
    if "bars" in store.tables():
        bars = store.as_of("bars", now, columns=["close"]); wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time")
        dates = [d.date() for d in wide["event_time"].to_list()]; d0 = dt.date.fromisoformat(date)
        if d0 in dates and dates.index(d0) >= 1 and all(e in wide.columns for e in etfs + [BENCHMARK]):
            i = dates.index(d0); P = wide.select(etfs + [BENCHMARK]).to_numpy().astype(float); r = P[i] / P[i - 1] - 1; rex = r[:len(etfs)] - r[-1]
            sig["last_bar"] = str(dates[-1]); sig["realised_day0"] = {"book_excess_return": float((w * rex).sum()), "rank_ic": float(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(rex)))[0, 1]),
                                                                       "long_leg": float(rex[order[-k:]].mean()), "short_leg": float(rex[order[:k]].mean())}
    os.makedirs(args.out, exist_ok=True); ledger = Ledger(os.path.join(args.out, "ledger"))
    sig["market"] = f"us_sectors-event-{date}"
    rec = publish(sig, args.out, ledger)
    ledger.append("event", name[:60], {"event_date": date, "post_hoc": post_hoc, "sha256": rec["sha256"], "source_text_sha256": text_hash, "n_positions": sig["n_positions"],
                                          **({"realised_day0_book": sig["realised_day0"]["book_excess_return"]} if "realised_day0" in sig else {})}, scope="p3-event-rotation")
    assert ledger.verify_chain()
    print(f"[event] {date} {name[:70]}\n[book] long {[etfs[i] for i in order[-k:]]}\n[book] short {[etfs[i] for i in order[:k]]}")
    if "realised_day0" in sig:
        rd = sig["realised_day0"]; print(f"[realised day-0] book {rd['book_excess_return']*100:+.2f}%  long {rd['long_leg']*100:+.2f}%  short {rd['short_leg']*100:+.2f}%  rank-IC {rd['rank_ic']:+.3f}")
    print(f"[file] {rec['path']}  sha256 {rec['sha256'][:16]}  post_hoc={post_hoc}")


if __name__ == "__main__":
    main()
