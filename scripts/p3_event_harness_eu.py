"""Agora event harness for Europe/UK (plan/DAY-2026-09-08.md block D; same protocol as scripts/p3_event_harness.py):
does the general-equilibrium sector response of the European economies to a dated trade-policy action line up with the
cross-section of realised STOXX Europe 600 supersector-ETF returns around that date?

  * events: emporion/structure/events_eu.py (deterministic, first pass, 9 events 2025-02 -> 2026-02; confidence-flagged)
  * prediction: per economy in the event (DEU, FRA, GBR, NLD, ITA, ESP, SWE, CHE) Agora's exact solve -> dlogPX / dlogREV / dlogQX per
    ISIC sector, combined with STOXX country weights -> ETF via sector_map_eu.ETF_EXPOSURE_EU
  * realised: beta-stripped ETF log returns (rolling 250-day beta to EXSA.DE estimated before the event) over 'reaction'
    [close(d-1) -> close(d+h-1)] and 'tradeable' [close(d) -> close(d+h)] windows, h in {1, 5, 20}
  * score: Spearman rank-IC across ETFs per event, mean over events, permutation null (predictions shuffled across ETFs)
  * every trial and every event row is a ledger entry (scope p3-event-harness-eu); bars from Yahoo (research source)
    python scripts/p3_event_harness_eu.py --reuse-store [--response dlogPX]
"""
from __future__ import annotations

import argparse, datetime as dt, json, os, sys
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger
from emporion.core.registry import get
from emporion.data.store import BitemporalStore
import emporion.data.loaders  # noqa: F401
from emporion.structure.agora_bridge import sector_response
from emporion.structure.events_eu import events
from emporion.structure.sector_map_eu import ETF_EXPOSURE_EU, BENCHMARK_EU
from p3_event_harness import rank_ic


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs"); ap.add_argument("--reuse-store", action="store_true")
    ap.add_argument("--response", default="dlogPX", choices=["dlogREV", "dlogQX", "dlogPX"]); ap.add_argument("--drop", default="")
    ap.add_argument("--perms", type=int, default=2000); ap.add_argument("--min-confidence", type=float, default=0.0); args = ap.parse_args()
    root = os.path.join(args.root, "eu_sectors"); store = BitemporalStore(os.path.join(root, "store")); ledger = Ledger(os.path.join(root, "ledger"))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    drop = set(filter(None, args.drop.split(","))); etfs = sorted(e for e in ETF_EXPOSURE_EU if e not in drop); syms = etfs + [BENCHMARK_EU]
    have = args.reuse_store and "bars" in store.tables() and store.as_of("bars", now, columns=["close"]).height > 0
    if not have:
        bars = get("loader", "yahoo")(syms, start="2014-01-01"); store.register_table("bars", ["symbol"]); h = store.append("bars", bars)
        print(f"[data] {bars.height:,} bars for {len(syms)} ETFs hash={h[:12]}")
    bars = store.as_of("bars", now, columns=["close"]).filter(pl.col("symbol").is_in(syms))
    wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time").drop_nulls()
    dates = wide["event_time"].to_list(); P = wide.select(syms).to_numpy(); L = np.log(P); R = np.diff(L, axis=0); rdates = dates[1:]
    ib = syms.index(BENCHMARK_EU); T, N = R.shape
    beta = np.full((T, N), np.nan)
    for t in range(250, T):
        w = R[t - 250:t]; b = w[:, ib]; vb = b.var()
        beta[t] = ((w - w.mean(0)) * (b - b.mean())[:, None]).mean(0) / vb if vb > 0 else 1.0
    resid = R - beta * R[:, [ib]]; date_idx = {d.date(): i for i, d in enumerate(rdates)}
    ev = [e for e in events() if e["confidence"] >= args.min_confidence]
    # one Agora batch per economy over the events that involve it; combine with the STOXX country weights
    codes = None; pred_sec = None; conv = np.ones(len(ev), bool); max_res = 0.0
    econs = sorted({c for e in ev for c in e["economies"]})
    # the bridge clips export-tax shocks to [0, 0.9] (its baseline has no export tax), so a tariff REDUCTION is solved as
    # minus the response to the same-sized increase: each (economy, event) is split into a positive and a negative part
    pos = lambda d: {c: v for c, v in d.items() if v > 0}; neg = lambda d: {c: -v for c, v in d.items() if v < 0}
    for c in econs:
        ks = [k for k, e in enumerate(ev) if c in e["economies"]]
        tm_p = [pos(ev[k]["economies"][c]["tm"]) for k in ks] + [neg(ev[k]["economies"][c]["tm"]) for k in ks]
        te_p = [pos(ev[k]["economies"][c]["te"]) for k in ks] + [neg(ev[k]["economies"][c]["te"]) for k in ks]
        resp = sector_response(c, tm_p, te_p)
        if codes is None:
            codes = resp["codes"]; pred_sec = np.zeros((len(ev), len(codes)))
        assert resp["codes"] == codes
        arr = np.asarray(resp[args.response]); res = np.asarray(resp["residual"]); max_res = max(max_res, float(np.max(res))); n = len(ks)
        for j, k in enumerate(ks):
            pred_sec[k] += ev[k]["economies"][c]["w"] * (arr[j] - arr[n + j]); conv[k] &= (res[j] <= 1e-6) and (res[n + j] <= 1e-6)
        print(f"[agora] {c}: {n} events (x2 signs), max residual {float(np.max(res)):.1e}")
    ci = {c: i for i, c in enumerate(codes)}; E = np.zeros((len(etfs), len(codes)))
    for j, etf in enumerate(etfs):
        for c, w in ETF_EXPOSURE_EU[etf].items():
            E[j, ci[c]] = w
    pred = pred_sec @ E.T
    print(f"[agora] {len(ev)} events; converged={int(conv.sum())} max residual={max_res:.1e} response={args.response}; ETFs {len(etfs)}; bars {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}")
    scope = "p3-event-harness-eu" + (f"-drop-{'-'.join(sorted(drop))}" if drop else ""); results = {}; rows = []
    for name, offset in (("reaction", 0), ("tradeable", 1)):
        for h in (1, 5, 20):
            ics, sign, ics_mom, kept = [], [], [], []
            for k, e in enumerate(ev):
                d = dt.date.fromisoformat(e["date"])
                if d not in date_idx or not conv[k]:
                    continue
                i0 = date_idx[d] + offset; i1 = i0 + h
                if i1 > T or i0 - 20 < 250:
                    continue
                real = resid[i0:i1, :len(etfs)].sum(0); mom = resid[i0 - 20:i0, :len(etfs)].sum(0); p = pred[k]
                if np.allclose(p, p[0]):
                    continue
                ic = rank_ic(p, real); ics.append(ic); ics_mom.append(rank_ic(mom, real)); sign.append(float((np.sign(p) == np.sign(real))[p != 0].mean())); kept.append(k)
                rows.append({"event": e["name"], "date": e["date"], "window": name, "h": h, "rank_ic": ic, "sign_acc": sign[-1], "confidence": e["confidence"]})
            ics = np.array(ics); n = len(ics)
            if n == 0:
                continue
            rng = np.random.default_rng(0); null = np.empty(args.perms); preds_kept = [pred[k] for k in kept]
            reals = [resid[date_idx[dt.date.fromisoformat(ev[k]["date"])] + offset:date_idx[dt.date.fromisoformat(ev[k]["date"])] + offset + h, :len(etfs)].sum(0) for k in kept]
            for b in range(args.perms):
                null[b] = np.mean([rank_ic(rng.permutation(pp), rr) for pp, rr in zip(preds_kept, reals)])
            mean_ic = float(ics.mean()); t = float(mean_ic / (ics.std(ddof=1) / np.sqrt(n))) if n > 1 else float("nan"); pval = float((null >= mean_ic).mean())
            results[f"{name}_h{h}"] = {"n_events": n, "mean_rank_ic": mean_ic, "t_stat": t, "perm_p": pval, "hit_rate": float((ics > 0).mean()), "mean_sign_acc": float(np.mean(sign)),
                                        "sector_momentum_null_ic": float(np.nanmean(ics_mom)), "null_mean": float(null.mean()), "null_sd": float(null.std())}
            r = results[f"{name}_h{h}"]
            print(f"[score] {name:9s} h={h:2d}  n={n:2d}  meanIC={mean_ic:+.3f} t={t:+.2f} perm-p={pval:.3f} hit={r['hit_rate']:.2f} sign={r['mean_sign_acc']:.2f} | momentum-null IC={r['sector_momentum_null_ic']:+.3f}")
            ledger.append("trial", f"event_harness_eu:{name}_h{h}", {"sharpe_is": t, **r, "response": args.response}, scope=scope)
    for row in rows:
        ledger.append("event", row["event"][:60], row, scope=scope)
    stamp = now.strftime("%Y%m%d-%H%M")
    out = {"n_events": len(ev), "converged": int(conv.sum()), "response": args.response, "etfs": etfs, "results": results, "per_event": rows,
           "predicted": {ev[k]["date"]: dict(zip(etfs, map(float, pred[k]))) for k in range(len(ev))},
           "sector_response": {ev[k]["date"]: dict(zip(codes, map(float, pred_sec[k]))) for k in range(len(ev))}}
    with open(os.path.join(root, f"p3-event-harness-eu-{args.response}-{stamp}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    assert ledger.verify_chain(); print(f"[done] ledger rows={ledger.frame().height} report={root}/p3-event-harness-eu-{args.response}-{stamp}.json")


if __name__ == "__main__":
    main()
