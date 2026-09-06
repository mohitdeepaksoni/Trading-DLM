"""P3 go/no-go: does Agora's general-equilibrium sector response to a dated tariff action
line up with the cross-section of realised US sector-ETF returns around that date?

    python scripts/p3_event_harness.py --root runs --reuse-store

Protocol (doc 06 §5, doc 08 §3):
  * events: deterministic table (emporion/structure/events_us_tariffs.py), no LLM labels
  * prediction: Agora exact CGE (USA anchor) -> dlog revenue / output per ISIC sector -> ETF via exposure map
  * realised: beta-stripped ETF log returns (rolling 250-day beta to SPY estimated BEFORE the event)
    over two windows per horizon h in {1,5,20}: 'reaction' [close(d-1) -> close(d+h-1)] includes day 0,
    'tradeable' [close(d) -> close(d+h)] excludes it
  * score: Spearman rank correlation across ETFs per event; mean over events; sign accuracy
  * null: permutation of the predicted vector across ETFs within each event (2000 draws) ->
    p-value of the observed mean rank-IC; plus a 'sector-momentum' null (prior-20d residual return)
  * every prediction is a ledger row; the scoreboard JSON is the artefact the console will read
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core import perf                                # noqa: E402
from emporion.core.ledger import Ledger                        # noqa: E402
from emporion.core.registry import get                         # noqa: E402
from emporion.data.store import BitemporalStore                # noqa: E402
import emporion.data.loaders                                   # noqa: E402,F401
from emporion.structure.agora_bridge import Economy, sector_response  # noqa: E402
from emporion.structure.events_us_tariffs import events        # noqa: E402
from emporion.structure.sector_map import ETF_EXPOSURE, BENCHMARK    # noqa: E402


def rank_ic(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    ra = np.argsort(np.argsort(a[ok])).astype(float); rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra = (ra - ra.mean()) / ra.std(); rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--reuse-store", action="store_true")
    ap.add_argument("--response", default="dlogREV", choices=["dlogREV", "dlogQX", "dlogPX"])
    ap.add_argument("--drop", default="", help="comma list of ETFs to exclude (robustness)")
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    args = ap.parse_args()
    root = os.path.join(args.root, "us_sectors")
    store = BitemporalStore(os.path.join(root, "store"))
    ledger = Ledger(os.path.join(root, "ledger"))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    drop = set(filter(None, args.drop.split(",")))
    etfs = sorted(e for e in ETF_EXPOSURE if e not in drop)                 # --drop: robustness (which ETFs carry the result)
    syms = etfs + [BENCHMARK]
    have = args.reuse_store and "bars" in store.tables() and store.as_of("bars", now, columns=["close"]).height > 0
    if not have:
        bars = get("loader", "nasdaq")(symbols=syms, start="2023-06-01")
        store.register_table("bars", ["symbol"]); h = store.append("bars", bars)
        print(f"[data] {bars.height:,} bars for {len(syms)} ETFs  hash={h[:12]}")
    bars = store.as_of("bars", now, columns=["close"])
    wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time").drop_nulls()
    dates = wide["event_time"].to_list()
    P = wide.select(syms).to_numpy()
    L = np.log(P)
    R = np.diff(L, axis=0); rdates = dates[1:]                         # daily log returns, R[t] = log P[t+1]/P[t]
    ib = syms.index(BENCHMARK)
    # residual returns: r_i - beta_i * r_spy with beta from trailing 250 days ending the day BEFORE t
    T, N = R.shape
    beta = np.full((T, N), np.nan)
    for t in range(250, T):
        w = R[t - 250:t]
        b = w[:, ib]; vb = b.var()
        beta[t] = ((w - w.mean(0)) * (b - b.mean())[:, None]).mean(0) / vb if vb > 0 else 1.0
    resid = R - beta * R[:, [ib]]
    date_idx = {d.date(): i for i, d in enumerate(rdates)}            # index i: return over (rdates[i-1], rdates[i]]

    ev = [e for e in events() if e["confidence"] >= args.min_confidence]
    ec = Economy("USA")
    resp = sector_response("USA", [e["shock"] for e in ev], [e["te"] for e in ev], econ=ec)
    codes = resp["codes"]; ci = {c: i for i, c in enumerate(codes)}
    E = np.zeros((len(etfs), len(codes)))
    for j, etf in enumerate(etfs):
        for c, w in ETF_EXPOSURE[etf].items():
            E[j, ci[c]] = w
    pred = resp[args.response] @ E.T                                     # (n_events, n_etf)
    ok_ev = np.asarray(resp["residual"]) <= 1e-6
    print(f"[agora] {len(ev)} events solved; converged={int(ok_ev.sum())}  max residual={resp['max_residual']:.1e}  response={args.response}")

    horizons = (1, 5, 20)
    results = {}
    scope = "p3-event-harness-us" + (f"-drop-{'-'.join(sorted(drop))}" if drop else "")
    rows = []
    for name, offset in (("reaction", 0), ("tradeable", 1)):
        for h in horizons:
            ics, sign, ics_mom, kept = [], [], [], []
            for k, e in enumerate(ev):
                d = dt.date.fromisoformat(e["date"])
                if d not in date_idx or not ok_ev[k]:
                    continue
                i0 = date_idx[d] + offset                                # first return index in the window
                i1 = i0 + h                                              # exclusive
                if i1 > T or i0 - 20 < 250:
                    continue
                real = resid[i0:i1, :len(etfs)].sum(0)                   # ETFs are the first len(etfs) columns of syms
                mom = resid[i0 - 20:i0, :len(etfs)].sum(0)               # prior-20d residual (sector momentum null)
                p = pred[k]
                if np.allclose(p, p[0]):
                    continue
                ic = rank_ic(p, real); ics.append(ic); ics_mom.append(rank_ic(mom, real))
                sign.append(float((np.sign(p) == np.sign(real))[p != 0].mean()))
                kept.append(k)
                rows.append({"event": e["name"], "date": e["date"], "window": name, "h": h, "rank_ic": ic,
                             "sign_acc": sign[-1], "confidence": e["confidence"]})
            ics = np.array(ics); n = len(ics)
            if n == 0:
                continue
            # permutation null: shuffle predictions across ETFs per event
            rng = np.random.default_rng(0)
            null = np.empty(args.perms)
            preds_kept = [pred[k] for k in kept]
            reals = []
            for k in kept:
                d = dt.date.fromisoformat(ev[k]["date"]); i0 = date_idx[d] + offset
                reals.append(resid[i0:i0 + h, :len(etfs)].sum(0))
            for b in range(args.perms):
                null[b] = np.mean([rank_ic(rng.permutation(pp), rr) for pp, rr in zip(preds_kept, reals)])
            mean_ic = float(ics.mean()); t = float(mean_ic / (ics.std(ddof=1) / np.sqrt(n))) if n > 1 else np.nan
            pval = float((null >= mean_ic).mean())
            results[f"{name}_h{h}"] = {"n_events": n, "mean_rank_ic": mean_ic, "t_stat": t, "perm_p": pval,
                                        "hit_rate": float((ics > 0).mean()), "mean_sign_acc": float(np.mean(sign)),
                                        "sector_momentum_null_ic": float(np.nanmean(ics_mom)),
                                        "null_mean": float(null.mean()), "null_sd": float(null.std())}
            r = results[f"{name}_h{h}"]
            print(f"[score] {name:9s} h={h:2d}  n={n:2d}  meanIC={mean_ic:+.3f} t={t:+.2f} perm-p={pval:.3f} hit={r['hit_rate']:.2f} "
                  f"sign={r['mean_sign_acc']:.2f} | momentum-null IC={r['sector_momentum_null_ic']:+.3f}")
            ledger.append("trial", f"event_harness:{name}_h{h}", {"sharpe_is": t, **r, "response": args.response}, scope=scope)
    for row in rows:
        ledger.append("event", row["event"][:60], row, scope=scope)
    stamp = now.strftime("%Y%m%d-%H%M")
    out = {"n_events": len(ev), "converged": int(ok_ev.sum()), "response": args.response, "etfs": etfs,
           "results": results, "per_event": rows,
           "predicted": {ev[k]["date"]: dict(zip(etfs, map(float, pred[k]))) for k in range(len(ev))},
           "sector_response": {ev[k]["date"]: dict(zip(codes, map(float, resp[args.response][k]))) for k in range(len(ev))},
           "converged_by_date": {ev[k]["date"]: bool(ok_ev[k]) for k in range(len(ev))}}
    with open(os.path.join(root, f"p3-event-harness-{args.response}-{stamp}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    perf.save_report(os.path.join("benchmarks", f"p3-event-harness-{stamp}.json"), extra={"n_events": len(ev)})
    assert ledger.verify_chain()
    print(f"[done] ledger rows={ledger.frame().height}  report={root}/p3-event-harness-{args.response}-{stamp}.json")


if __name__ == "__main__":
    main()
