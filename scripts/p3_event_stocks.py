"""P3 step 2 — does the Agora sector price response rank INDIVIDUAL US stocks on tariff event days?
Per stock: predicted = industry->ICIO exposure · sector dlogPX (from the latest harness JSON with sector_response).
Realised: day-0 (and h=5, 20) log return residual to the equal-weight PIT-universe market (beta from trailing 250 d).
Score: rank-IC across stocks per event; permutation null across stocks; t over events. Ledger scope p3-event-stocks-us.
    python scripts/p3_event_stocks.py --root runs --top 500
"""
import argparse, datetime as dt, glob, json, os, sys
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from scipy.stats import spearmanr


def rank_ic(a, b):
    return float(spearmanr(a, b).correlation)
from emporion.structure.industry_map import exposure_matrix
from emporion.structure.sector_map import ETF_EXPOSURE


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs"); ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--perms", type=int, default=2000); ap.add_argument("--harness", default="")
    ap.add_argument("--by-sector", action="store_true", help="diagnostic: aggregate stocks into ICIO sector portfolios (exposure-weighted) and score across sectors")
    ap.add_argument("--min-stocks", type=int, default=5)
    ap.add_argument("--goods-only", action="store_true", help="restrict to goods sectors (ICIO group I: 05T06..31T33) / stocks with >=0.5 weight there")
    ap.add_argument("--synthetic-etf", action="store_true", help="diagnostic: rank across ETFs rebuilt from stocks via ETF_EXPOSURE @ sector portfolios")
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.root, "us_sectors", "p3-event-harness-dlogPX-*.json")))
    hj = None
    for f in reversed([args.harness] if args.harness else files):
        j = json.load(open(f))
        if "sector_response" in j:
            hj = j; hf = f; break
    if hj is None:
        raise SystemExit("no harness JSON with sector_response — rerun scripts/p3_event_harness.py --response dlogPX")
    codes = list(next(iter(hj["sector_response"].values())).keys())
    store = BitemporalStore(os.path.join(args.root, "us_stocks_wide", "store")); ledger = Ledger(os.path.join(args.root, "us_stocks_wide", "ledger"))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    uni = store.as_of("universe", now, columns=["sector", "industry"]).group_by("symbol").last()
    bars = store.as_of("bars", now, columns=["close", "volume"], event_start=dt.datetime(2023, 6, 1))
    wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time")
    syms = [c for c in wide.columns if c != "event_time"]
    wide = wide.with_columns([pl.col(c).fill_null(strategy="forward") for c in syms])
    dates = wide["event_time"].to_list(); P = wide.select(syms).to_numpy().astype(float)
    volw = bars.pivot(index="event_time", on="symbol", values="volume").sort("event_time")
    dv = np.nan_to_num(P) * np.nan_to_num(volw.select(syms).to_numpy().astype(float))
    listed = np.isfinite(P); L = np.log(P); R = np.where(listed[1:] & listed[:-1], np.diff(L, axis=0), np.nan); rdates = dates[1:]
    T, N = R.shape
    umap = {r["symbol"]: (r["industry"], r["sector"]) for r in uni.iter_rows(named=True)}
    E = exposure_matrix([umap.get(s, ("", ""))[0] for s in syms], [umap.get(s, ("", ""))[1] for s in syms], codes)
    mapped = E.sum(1) > 0
    GOODS = {c for c in codes if c not in ("01T02", "03") and c <= "31T33"}                 # ICIO group I
    goods_w = E[:, [i for i, c in enumerate(codes) if c in GOODS]].sum(1)
    if args.goods_only:
        mapped = mapped & (goods_w >= 0.5)
    etfs = sorted(ETF_EXPOSURE); X = np.zeros((len(etfs), len(codes))); cix = {c: i for i, c in enumerate(codes)}
    for j, e in enumerate(etfs):
        for c, w in ETF_EXPOSURE[e].items():
            X[j, cix[c]] = w
    print(f"[panel] {N} stocks x {T} days {rdates[0]:%Y-%m-%d}..{rdates[-1]:%Y-%m-%d}; mapped to ICIO: {int(mapped.sum())}/{N}; harness {os.path.basename(hf)}")
    date_idx = {d.date(): i for i, d in enumerate(rdates)}
    scope = "p3-event-stocks-us" + ("-by-sector" if args.by_sector else "") + ("-goods" if args.goods_only else "") + ("-synthetic-etf" if args.synthetic_etf else ""); results = {}; rows = []
    for name, offset in (("reaction", 0), ("tradeable", 1)):
        for h in (1, 5, 20):
            ics, kept_pred, kept_real, per = [], [], [], []
            for date_s, sr in hj["sector_response"].items():
                if not hj.get("converged_by_date", {}).get(date_s, True):
                    continue
                d = dt.date.fromisoformat(date_s)
                if d not in date_idx:
                    continue
                i0 = date_idx[d] + offset; i1 = i0 + h
                if i1 > T or i0 < 250:
                    continue
                # point-in-time universe: top-K by trailing-250d median dollar volume, listed the whole window
                hist = listed[i0 - 250:i0 + 1].all(0); med = np.median(dv[i0 - 250:i0], axis=0); med[~hist | ~mapped] = -1
                U = np.zeros(N, bool); top = np.argsort(-med)[:args.top]; U[top] = med[top] > 0
                w = R[i0 - 250:i0]; mkt = np.nanmean(np.where(U, w, np.nan), axis=1)
                beta = np.array([np.cov(np.nan_to_num(w[:, j]), mkt)[0, 1] / mkt.var() if U[j] else np.nan for j in range(N)])
                real = np.nansum(R[i0:i1], axis=0) - beta * np.nansum(R[i0:i1], axis=0)[U].mean()      # residual to EW market
                pred = E @ np.array([sr[c] for c in codes])
                ok = U & np.isfinite(real) & np.isfinite(pred)
                if args.synthetic_etf:
                    Wt = E[ok].T; cnt = (Wt > 0).sum(1); good = cnt >= args.min_stocks
                    port = np.where(good, (Wt @ real[ok]) / np.maximum(Wt.sum(1), 1e-9), np.nan)
                    synth = np.array([np.nansum(X[j] * port) / max(np.nansum(X[j] * np.isfinite(port)), 1e-9) for j in range(len(etfs))])
                    ps = X @ np.array([sr[c] for c in codes])
                    ic = float(rank_ic(ps, synth)); ics.append(ic); kept_pred.append(ps); kept_real.append(synth)
                    rows.append({"date": date_s, "window": name, "h": h, "rank_ic": ic, "n_etfs": len(etfs)})
                    continue
                if args.by_sector:
                    # sector portfolios: exposure-weighted mean residual return of the PIT stocks in each ICIO code
                    Wt = E[ok].T; cnt = (Wt > 0).sum(1); good = cnt >= args.min_stocks
                    if args.goods_only:
                        good = good & np.array([c in GOODS for c in codes])
                    if good.sum() < 5:
                        continue
                    port = (Wt[good] @ real[ok]) / Wt[good].sum(1); ps = np.array([sr[c] for c in codes])[good]
                    if np.allclose(ps, ps[0]):
                        continue
                    ic = float(rank_ic(ps, port)); ics.append(ic); kept_pred.append(ps); kept_real.append(port)
                    rows.append({"date": date_s, "window": name, "h": h, "rank_ic": ic, "n_sectors": int(good.sum())})
                    continue
                if ok.sum() < 50 or np.allclose(pred[ok], pred[ok][0]):
                    continue
                ic = float(rank_ic(pred[ok], real[ok])); ics.append(ic); kept_pred.append(pred[ok]); kept_real.append(real[ok])
                rows.append({"date": date_s, "window": name, "h": h, "rank_ic": ic, "n_stocks": int(ok.sum())})
            ics = np.array(ics); n = len(ics)
            if n == 0:
                continue
            rng = np.random.default_rng(0); null = np.empty(args.perms)
            for b in range(args.perms):
                null[b] = np.mean([rank_ic(rng.permutation(p), r) for p, r in zip(kept_pred, kept_real)])
            mean_ic = float(ics.mean()); t = float(mean_ic / (ics.std(ddof=1) / np.sqrt(n))) if n > 1 else np.nan
            pval = float((null >= mean_ic).mean())
            results[f"{name}_h{h}"] = {"n_events": n, "mean_rank_ic": mean_ic, "t_stat": t, "perm_p": pval, "hit_rate": float((ics > 0).mean()),
                                        "null_mean": float(null.mean()), "null_sd": float(null.std()), "mean_n_stocks": float(np.mean([len(p) for p in kept_pred]))}
            r = results[f"{name}_h{h}"]
            print(f"[score] {name:9s} h={h:2d}  n={n:2d}  meanIC={mean_ic:+.3f} t={t:+.2f} perm-p={pval:.3f} hit={r['hit_rate']:.2f} stocks/event={r['mean_n_stocks']:.0f}")
            ledger.append("trial", f"event_stocks:{name}_h{h}", {"sharpe_is": t, **r, "response": "dlogPX", "top": args.top}, scope=scope)
    stamp = now.strftime("%Y%m%d-%H%M")
    out = {"harness": os.path.basename(hf), "top": args.top, "results": results, "per_event": rows}
    with open(os.path.join(args.root, "us_stocks_wide", f"p3-event-stocks{'-by-sector' if args.by_sector else ''}{'-goods' if args.goods_only else ''}{'-synthetic-etf' if args.synthetic_etf else ''}-{stamp}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("[done]")


if __name__ == "__main__":
    main()
