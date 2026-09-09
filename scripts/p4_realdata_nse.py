"""NSE ladder with holdings: the first market where phi (non-trading float) and demand pressure come from data.

Anchors (point-in-time from NSE results/shareholding, knowledge_time = broadcast):
  value        log(EPS_ttm / median_ey / p)                (consolidated EPS, trailing 4 quarters)
  sue          standardised unexpected earnings (seasonal difference / its 8-quarter std)
  inst_change  change in institutional share (domestic + foreign) over the last reported quarter   [demand pressure]
  fpi_level    foreign portfolio investor share                                                     [state]
  clearing     mu = Rf p (1 + 0.03 z_sue), s = (1 - promoter) * public float, kappa from realised excess payoffs
Monthly decisions, RankIC 20/60d, top-quintile long-only tilt with costs, PBO across rungs, certify().
"""
from __future__ import annotations

import argparse, datetime as dt, json, os, sys
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from emporion.evaluate import portfolio_returns, certify, pbo_cscv
from emporion.evaluate.costs import _drifted
from emporion.evaluate.costs_in import cost_vectors, round_trip_bps


def portfolio_returns_vec(W, R, cost_bps, buy_tax_bps):
    """portfolio_returns with per-asset one-way costs and a buy-side tax (Indian STT + stamp; plan Phase IN step 5); next-bar execution."""
    W = np.nan_to_num(np.asarray(W, float)); R = np.nan_to_num(np.asarray(R, float)); T, N = W.shape
    Wx = np.zeros_like(W); Wx[1:] = W[:T - 1]; gross = (Wx * R).sum(1)
    prev = np.zeros_like(W); prev[1:] = _drifted(Wx[:-1], R[:-1]); d = Wx - prev
    return gross - (np.abs(d) * cost_bps[None, :]).sum(1) / 1e4 - (np.maximum(d, 0.0) * buy_tax_bps[None, :]).sum(1) / 1e4, np.abs(d).sum(1)
from p3_event_harness import rank_ic
from emporion.anchors import price_themes
from emporion.structure.embeddings import asset_embeddings, peer_implied_yield


from emporion.anchors.nse import last_known_t, split_adjust  # noqa: E402  (moved into the package 2026-09-07)


def last_known(df, asof, col, syms, n=1, days=400, min_n=None):
    """Last n known (non-null) values of `col` per symbol as of `asof` (latest revision per quarter); None if fewer
    than min_n (default n) values exist or the newest one is older than `days`."""
    min_n = n if min_n is None else min_n
    f = df.filter(pl.col("knowledge_time") <= asof).sort("knowledge_time").group_by(["symbol", "event_time"]).last()
    f = f.filter(pl.col(col).is_not_null()).sort("event_time").group_by("symbol").agg(pl.col(col).tail(n).alias("v"), pl.col("event_time").tail(n).alias("t"))
    out = {r["symbol"]: (list(r["v"]) if len(r["v"]) >= min_n and (asof - r["t"][-1]).days < days else None) for r in f.iter_rows(named=True)}
    return [out.get(s) for s in syms]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs/nse"); ap.add_argument("--every", type=int, default=21)
    ap.add_argument("--cost-bps", type=float, default=None, help="flat one-way cost; default = the Indian cost model (STT + stamp + exchange + slippage)")
    ap.add_argument("--slippage-bps", type=float, default=5.0); args = ap.parse_args()
    store = BitemporalStore(os.path.join(args.root, "store")); ledger = Ledger(os.path.join(args.root, "ledger"))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    res = store.all_versions("results")                                   # ALL revisions; resolved per decision date
    # prefer consolidated where available
    res = res.sort(["symbol", "event_time", "consolidated", "knowledge_time"])
    sh = store.all_versions("shareholding")
    bars = store.as_of("bars", now, columns=["close", "volume"])
    cov = bars.group_by("symbol").agg(pl.len().alias("n")).filter(pl.col("n") >= 500)
    bars = bars.filter(pl.col("symbol").is_in(cov["symbol"].to_list()))
    wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time")
    syms = [c for c in wide.columns if c != "event_time"]
    wide = wide.with_columns([pl.col(c).fill_null(strategy="forward") for c in syms])
    dates = wide["event_time"].to_list(); P = wide.select(syms).to_numpy().astype(float); T, N = P.shape
    listed = np.isfinite(P); R = np.where(listed[1:] & listed[:-1], P[1:] / P[:-1] - 1, 0.0)
    volw = bars.pivot(index="event_time", on="symbol", values="volume").sort("event_time")
    Vm = np.nan_to_num(volw.select(syms).to_numpy().astype(float)); dv = np.nan_to_num(P) * Vm
    def universe_at(t, K=200):
        hist = listed[max(0, t - 300):t + 1].all(0); med = np.median(dv[t - 250:t + 1], axis=0); med[~hist] = -1
        keep = np.zeros(N, bool); top = np.argsort(-med)[:K]; keep[top] = med[top] > 0; return keep
    # delivery percentage (MTO files; known the day after the session): ONE pre-declared rung (plan/PHASES-BY-MARKET.md Phase IN, data step 2)
    #   delivery_trend = mean(deliv %) over the last 20 known sessions - mean over the 230 sessions before them; published direction: higher = more conviction
    D = None
    if "delivery" in store.tables():
        dl = store.as_of("delivery", now, columns=["deliv_per"]).filter(pl.col("symbol").is_in(syms))
        dw = dl.pivot(index="event_time", on="symbol", values="deliv_per").sort("event_time")
        dw = wide.select("event_time").join(dw, on="event_time", how="left")
        D = dw.select([c if c in dw.columns else pl.lit(None).alias(c) for c in syms]).to_numpy().astype(float)
    # corporate-event tables (Phase IN data steps 3 and 6), each ONE pre-declared rung — published directions fixed here, before any run:
    #   insider_net_buy        sum(buy value - sell value) of PIT disclosures known in the last 90 days / trailing 60-session traded value; higher = better
    #   sast_promoter_net      promoter SAST shares acquired - sold, known in the last 180 days, / last known total shares;               higher = better
    #   announcement_intensity announcements in the last 20 sessions vs the mean 20-session count over the prior 250 (log ratio);        HIGHER = WORSE (attention)
    ins = store.all_versions("insider") if "insider" in store.tables() else None
    sast = store.all_versions("sast") if "sast" in store.tables() else None
    A = None
    if "announcements" in store.tables():
        an = store.as_of("announcements", now, columns=["category"]).filter(pl.col("symbol").is_in(syms)).with_columns(pl.col("knowledge_time").dt.date().alias("d"))
        cnt = an.group_by(["symbol", "d"]).agg(pl.len().alias("n")).with_columns(pl.col("d").cast(pl.Datetime("us")).alias("event_time"))
        aw = wide.select("event_time").join(cnt.pivot(index="event_time", on="symbol", values="n"), on="event_time", how="left")
        A = np.nan_to_num(aw.select([c if c in aw.columns else pl.lit(None).alias(c) for c in syms]).to_numpy().astype(float))
    print(f"[panel] NSE {N} stocks x {T} days {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}; results {res.height:,} rows, shareholding {sh.height:,} rows; "
          f"insider {ins.height if ins is not None else 0:,}, sast {sast.height if sast is not None else 0:,}, announcements {int(A.sum()) if A is not None else 0:,}; delivery {'%d sessions' % int(np.isfinite(D).any(1).sum()) if D is not None else 'absent'}")
    res_c = res.filter(pl.col("consolidated")); cons = set(res_c["symbol"].to_list())
    res_use = pl.concat([res_c, res.filter(~pl.col("consolidated") & ~pl.col("symbol").is_in(list(cons)))])
    Rf_d = 1.0 + 0.065 / 252
    dec = list(range(300, T - 61, args.every))
    names = ("value", "sue", "inst_change", "fpi_level", "clearing_sue_float", "momentum", "st_reversal", "low_risk", "size", "illiquidity", "embedding_peer_value") + (("delivery_trend",) if D is not None else ()) + (("insider_net_buy",) if ins is not None else ()) + (("sast_promoter_net",) if sast is not None else ()) + (("announcement_intensity",) if A is not None else ())
    SH_COLS = [c for c in sh.columns if c.startswith("pct_")]
    sig = {k: np.full((T, N), np.nan) for k in names}; kappas = []
    for t in dec:
        asof = dates[t]; p = P[t]; U = universe_at(t)
        shs = last_known_t(sh, asof, "shares_total", syms, n=40, days=100000, min_n=1) if "shares_total" in sh.columns else [None] * N
        e4 = [split_adjust(a, b) for a, b in zip(last_known_t(res_use, asof, "eps_diluted", syms, n=4), shs)]
        e12 = [split_adjust(a, b) for a, b in zip(last_known_t(res_use, asof, "eps_diluted", syms, n=12, days=400, min_n=8), shs)]
        eps = np.array([sum(v) if v else np.nan for v in e4]); ok = U & np.isfinite(eps) & (eps > 0) & np.isfinite(p)
        Pu = np.where(U[None, :] & listed, P, np.nan)
        th = price_themes(Pu, Vm, t)
        for k in ("momentum", "st_reversal", "low_risk", "size", "illiquidity"):
            if k in th: sig[k][t] = np.where(U, th[k], np.nan)
        if D is not None and t >= 250:
            with np.errstate(all="ignore"):
                d20 = np.nanmean(D[t - 20:t], axis=0); d250 = np.nanmean(D[t - 250:t - 20], axis=0)      # rows < t: delivery of day t is known only at t+1
            sig["delivery_trend"][t] = np.where(U & np.isfinite(d20) & np.isfinite(d250), d20 - d250, np.nan)
        if ins is not None and t >= 60:
            k = ins.filter((pl.col("knowledge_time") <= asof) & (pl.col("knowledge_time") > asof - dt.timedelta(days=90))).group_by("symbol").agg((pl.col("buy_value").fill_null(0) - pl.col("sell_value").fill_null(0)).sum().alias("net"))
            net = {r["symbol"]: r["net"] for r in k.iter_rows(named=True)}; tv60 = dv[t - 60:t].sum(0)
            v = np.array([net.get(s_, 0.0) for s_ in syms]) / np.maximum(tv60, 1.0)
            sig["insider_net_buy"][t] = np.where(U & (tv60 > 0), np.clip(v, -1, 1), np.nan)
        if sast is not None:
            k = sast.filter((pl.col("knowledge_time") <= asof) & (pl.col("knowledge_time") > asof - dt.timedelta(days=180)) & pl.col("promoter")).group_by("symbol").agg((pl.col("shares_acq").fill_null(0) - pl.col("shares_sale").fill_null(0)).sum().alias("net"))
            net = {r["symbol"]: r["net"] for r in k.iter_rows(named=True)}
            tot = [v_[0][-1] if v_ else np.nan for v_ in shs]                                            # last known total shares (shareholding table)
            v = np.array([net.get(s_, 0.0) / tot[i_] if (np.isfinite(tot[i_]) and tot[i_] > 0) else np.nan for i_, s_ in enumerate(syms)])
            sig["sast_promoter_net"][t] = np.where(U, np.clip(v, -0.5, 0.5), np.nan)
        if A is not None and t >= 270:
            c20 = A[t - 20:t].sum(0); base = A[t - 270:t - 20].sum(0) / 250.0 * 20.0
            sig["announcement_intensity"][t] = np.where(U, -np.log((c20 + 1.0) / (base + 1.0)), np.nan)
        if ok.sum() < 30:
            continue
        ey_med = np.median(eps[ok] / p[ok]); p_fund = np.where(ok, eps / ey_med, np.nan)
        sig["value"][t] = np.log(p_fund / p)
        # asset embeddings from the latest known shareholding pattern (categories = investor types) -> peer-implied value
        H = np.full((N, len(SH_COLS)), np.nan)
        for j, c in enumerate(SH_COLS):
            col = last_known(sh, asof, c, syms, n=1, days=200); H[:, j] = np.array([v[0] if v and v[0] is not None else np.nan for v in col])
        okh = np.isfinite(H).all(1) & ok
        if okh.sum() >= 30:
            E, _, _ = asset_embeddings(H[okh], k=4)
            ey = eps[okh] / p[okh]
            pv = np.full(N, np.nan); pv[okh] = peer_implied_yield(E, ey, n_peers=10); sig["embedding_peer_value"][t] = pv
        sue = np.full(N, np.nan)
        for i, v in enumerate(e12):
            if v and len(v) >= 8:
                d = np.array(v[4:]) - np.array(v[:-4]); sd = d[:-1].std(ddof=1)
                if sd > 1e-9: sue[i] = d[-1] / sd
        sue = np.clip(sue, -5, 5); sig["sue"][t] = sue
        inst2 = last_known(sh.with_columns((pl.col("pct_institutions_domestic").fill_null(0) + pl.col("pct_institutions_foreign").fill_null(0)).alias("inst")), asof, "inst", syms, n=2, days=200)
        sig["inst_change"][t] = np.array([(v[1] - v[0]) if v else np.nan for v in inst2])
        fpi = last_known(sh, asof, "pct_fpi_cat1", syms, n=1, days=200); sig["fpi_level"][t] = np.array([v[0] if v else np.nan for v in fpi])
        prom = last_known(sh, asof, "pct_promoter", syms, n=1, days=200); prom = np.array([v[0] if v else np.nan for v in prom])
        okc = ok & np.isfinite(prom)
        if okc.sum() >= 30:
            w = R[t - 250:t][:, okc]; Sig = np.cov(w.T) * P[t][okc][:, None] * P[t][okc][None, :]
            s = (1 - prom[okc]); s = s / s.sum() * okc.sum()                                      # public float as supply
            fwd = P[t - 230:t + 1][:, okc] / P[t - 250:t + 1 - 20][:, okc] - Rf_d ** 20
            prem = (fwd * P[t - 250:t + 1 - 20][:, okc]).mean(0); x = Sig @ s
            kappa = max(float((prem * x).sum() / (x * x).sum()), 0.0); kappas.append(kappa)
            zs = np.where(np.isfinite(sue), np.clip(np.nan_to_num(sue) / 2.0, -2, 2), 0.0)
            p_hat = np.full(N, np.nan); p_hat[okc] = p[okc] * (1 + 0.03 * zs[okc]) - kappa * x / Rf_d ** 20
            sig["clearing_sue_float"][t] = np.log(np.maximum(p_hat, 1e-6) / p)
    print(f"[kappa] median={np.nanmedian(kappas) if kappas else float('nan'):.4g} (n={len(kappas)})")
    Rfull = R if R.shape[0] == T else np.vstack([R, np.zeros((1, N))])
    if args.cost_bps is None:                                                                      # Indian cost model (ledgered as scope suffix)
        cost_v, tax_v = cost_vectors(N, slippage_bps=args.slippage_bps); scope = f"p4-realdata-nse-N{N}-incost"
        cost_desc = {"cost_model": "IN delivery: STT 10 bp each side, stamp 1.5 bp buy, exchange+GST 0.35 bp, SEBI 0.01 bp, slippage %.1f bp" % args.slippage_bps, "round_trip_bps": round_trip_bps(slippage_bps=args.slippage_bps)}
    else:
        cost_v, tax_v = np.full(N, args.cost_bps), np.zeros(N); scope = f"p4-realdata-nse-N{N}"; cost_desc = {"cost_bps": args.cost_bps}
    print(f"[costs] {cost_desc}")
    def tilt(S):
        W = np.zeros((T, N)); cur = np.zeros(N)
        for t in range(T):
            if np.isfinite(S[t]).any():
                sc = S[t]; k = max(1, int(np.isfinite(sc).sum() * 0.2)); top = np.argsort(np.nan_to_num(sc, nan=-1e9))[-k:]
                cur = np.zeros(N); cur[top] = 1.0 / k
            W[t] = cur
        return portfolio_returns_vec(W, Rfull, cost_v, tax_v)[0], W
    nets = {k: tilt(S)[0] for k, S in sig.items()}
    pbo = pbo_cscv(np.stack([nets[k] for k in names], axis=1), n_blocks=16)["pbo"]; print(f"[pbo] across {len(names)} rungs: {pbo:.3f}")
    results = {}
    for name, S in sig.items():
        for h in (20, 60):
            ics = [rank_ic(S[t], np.log(P[t + h] / P[t]) - np.nanmean(np.where(np.isfinite(S[t]), np.log(P[t + h] / P[t]), np.nan))) for t in dec if np.isfinite(S[t]).sum() >= 30]
            ics = np.array([x for x in ics if np.isfinite(x)])
            if len(ics) < 5:
                continue
            r = {"n": int(len(ics)), "mean_ic": float(ics.mean()), "t": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), "hit": float((ics > 0).mean())}
            results[f"{name}_h{h}"] = r; print(f"[rankic] {name:20s} h={h:2d} n={r['n']:3d} meanIC={r['mean_ic']:+.4f} t={r['t']:+.2f} hit={r['hit']:.2f}")
        net, W = nets[name], tilt(S)[1]
        Wew = np.zeros((T, N)); cur = np.zeros(N)
        for t in range(T):
            if np.isfinite(S[t]).sum() >= 30:
                m = np.isfinite(S[t]); cur = m / m.sum()
            Wew[t] = cur
        ew, _ = portfolio_returns_vec(Wew, Rfull, cost_v, tax_v)
        rep = certify(f"{name}_top20pct", T, lambda tr, te, net=net: net[te], net, ledger, scope, benchmark_returns=ew, B=1000, pbo=pbo, extra=cost_desc)
        rel = rep["vs_benchmark_ann_excess_ci"]
        print(f"[certify] {name:20s} SR={rep['sharpe']:+.2f} mdd={rep['max_drawdown']*100:.1f}% DSR={rep['dsr']['dsr']:.3f}(N={rep['n_trials']}) vsEW={rel['lo']*100:+.1f}..{rel['hi']*100:+.1f}%/yr -> {'CERTIFIED' if rep['certified'] else 'not certified'}")
        results[f"{name}_certify"] = {k: rep[k] for k in ("sharpe", "max_drawdown", "certified")}
    with open(os.path.join(args.root, f"p4-realdata-nse-{now:%Y%m%d-%H%M}.json"), "w") as f:
        json.dump({"N": N, "T": T, "pbo": pbo, "results": results}, f, indent=1, default=str)
    assert ledger.verify_chain(); print("[done]")


if __name__ == "__main__":
    main()
