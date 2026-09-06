"""Block C on real data: the structural clearing model on the top-200 US stock panel, fundamentals-anchored.

Beliefs (per stock, per day, point-in-time): mu_i = Rf * p_fund_i, with the fundamentals-based fair price
    p_fund_i = EPS_ttm_i / ey_sector(t)   (trailing-4-quarter diluted EPS as of the filing date, capitalised at the
                                          cross-sectional median earnings yield of the day — a value anchor)
State: kappa from the structural estimator on the trailing window (returns only), supply s = shares outstanding
(SEC, point-in-time), Sigma from the trailing 250 days.  Equilibrium p_hat = (mu - kappa Sigma s)/Rf.
Signal = -r_mkt = log(p_hat / p_obs).  Ablation rungs on ONE ledger scope:
    rung 3  value anchor only  (kappa = 0: p_hat = p_fund)
    rung 4  clearing model     (kappa from estimator)
Scored as RankIC at h = 20 / 60 trading days, monthly decision dates, plus certify() on a long-only top-quintile tilt.
"""
from __future__ import annotations

import argparse, datetime as dt, json, os, sys
import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from emporion.evaluate import portfolio_returns, certify, sharpe, pbo_cscv
from p3_event_harness import rank_ic
from emporion.anchors import price_themes
from emporion.structure.embeddings import asset_embeddings, peer_implied_yield
import glob

torch.set_default_dtype(torch.float64)


from emporion.anchors.fundamentals import ttm_eps as _ttm, shares as _shares, sue as _sue, accruals as _acc


def ttm_eps_asof(facts: pl.DataFrame, asof: dt.datetime, syms):
    return _ttm(facts, asof, syms)


def _ttm_eps_asof_old(facts: pl.DataFrame, asof: dt.datetime, syms):
    """Trailing-4-quarter diluted EPS known at `asof`, per symbol (NaN if < 4 quarters)."""
    f = facts.filter((pl.col("knowledge_time") <= asof) & (pl.col("concept") == "eps_diluted"))
    f = f.sort("knowledge_time").group_by(["symbol", "event_time"]).last()      # latest revision known
    f = f.sort("event_time").group_by("symbol").agg(pl.col("value").tail(4).alias("v"), pl.col("event_time").tail(4).alias("t"))
    out = {r["symbol"]: (float(sum(r["v"])) if len(r["v"]) == 4 and (asof - r["t"][-1]).days < 400 else np.nan) for r in f.iter_rows(named=True)}
    return np.array([out.get(s, np.nan) for s in syms])


def quarters_asof(facts, concept, asof, syms, n=8):
    """Last n quarterly values (latest revision known at asof) per symbol, oldest->newest; None if fewer than n."""
    f = facts.filter((pl.col("knowledge_time") <= asof) & (pl.col("concept") == concept))
    f = f.sort("knowledge_time").group_by(["symbol", "event_time"]).last()
    f = f.sort("event_time").group_by("symbol").agg(pl.col("value").tail(n).alias("v"), pl.col("event_time").tail(n).alias("t"))
    out = {r["symbol"]: (list(r["v"]) if len(r["v"]) == n and (asof - r["t"][-1]).days < 400 else None) for r in f.iter_rows(named=True)}
    return [out.get(s_) for s_ in syms]


def sue_asof(facts, asof, syms):
    return _sue(facts, asof, syms)


def _sue_asof_old(facts, asof, syms):
    """Standardised unexpected earnings: (EPS_q - EPS_{q-4}) / std of that difference over the last 8 quarters."""
    q = quarters_asof(facts, "eps_diluted", asof, syms, n=12)
    out = np.full(len(syms), np.nan)
    for i, v in enumerate(q):
        if v is None:
            continue
        d = np.array(v[4:]) - np.array(v[:-4])                    # 8 seasonal differences
        sd = d[:-1].std(ddof=1)
        if sd > 1e-9:
            out[i] = d[-1] / sd
    return np.clip(out, -5, 5)


def accruals_asof(facts, asof, syms):
    return _acc(facts, asof, syms)


def _accruals_asof_old(facts, asof, syms):
    """(Net income_ttm - CFO_ttm) / Assets, point-in-time; high accruals predict lower returns (Sloan 1996)."""
    ni = quarters_asof(facts, "net_income", asof, syms, n=4); cfo = quarters_asof(facts, "cfo", asof, syms, n=4)
    f = facts.filter((pl.col("knowledge_time") <= asof) & (pl.col("concept") == "assets")).sort("knowledge_time").group_by("symbol").last()
    assets = {r["symbol"]: r["value"] for r in f.iter_rows(named=True)}
    out = np.full(len(syms), np.nan)
    for i, s_ in enumerate(syms):
        if ni[i] is None or cfo[i] is None or not assets.get(s_):
            continue
        out[i] = (sum(ni[i]) - sum(cfo[i])) / assets[s_]
    return out


def shares_asof(facts, asof, syms):
    return _shares(facts, asof, syms)


def _shares_asof_old(facts, asof, syms):
    f = facts.filter((pl.col("knowledge_time") <= asof) & (pl.col("concept") == "shares")).sort("knowledge_time").group_by("symbol").last()
    d = {r["symbol"]: r["value"] for r in f.iter_rows(named=True)}
    return np.array([d.get(s, np.nan) for s in syms])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/us_stocks")
    ap.add_argument("--every", type=int, default=21)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--universe-k", type=int, default=200, help="point-in-time universe: top-K by trailing dollar volume at each decision date")
    ap.add_argument("--scope-version", default="pit2", help="procedure-family version: pit2 = point-in-time universe + revisions resolved per decision date (2026-09-06)")
    args = ap.parse_args()
    store = BitemporalStore(os.path.join(args.root, "store")); ledger = Ledger(os.path.join(args.root, "ledger"))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    bars = store.as_of("bars", now, columns=["close"]); facts = store.all_versions("facts")   # ALL revisions; resolved per decision date
    wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time")
    syms = [c for c in wide.columns if c != "event_time"]
    wide = wide.with_columns([pl.col(c).fill_null(strategy="forward") for c in syms])          # NaN before listing stays NaN
    dates = [d for d in wide["event_time"].to_list()]; P = wide.select(syms).to_numpy().astype(float); T, N = P.shape
    listed = np.isfinite(P)
    R = np.where(listed[1:] & listed[:-1], P[1:] / P[:-1] - 1, 0.0)
    volw = store.as_of("bars", now, columns=["volume"]).pivot(index="event_time", on="symbol", values="volume").sort("event_time")
    Vm = np.nan_to_num(volw.select(syms).to_numpy().astype(float)) if volw.height == wide.height else np.ones_like(P)
    dv = np.nan_to_num(P) * Vm
    def universe_at(t):
        """POINT-IN-TIME universe: top-K by trailing 250-day median dollar volume among stocks listed >= 300 days (no look-ahead)."""
        hist = listed[max(0, t - 300):t + 1].all(0)
        med = np.median(dv[t - 250:t + 1], axis=0); med[~hist] = -1
        keep = np.zeros(N, bool); keep[np.argsort(-med)[:args.universe_k]] = med[np.argsort(-med)[:args.universe_k]] > 0
        return keep
    print(f"[panel] {N} stocks x {T} days {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}; facts rows {facts.height:,}")
    Rf_d = 1.0 + 0.04 / 252
    dec = list(range(300, T - 61, args.every))
    sig = {k: np.full((T, N), np.nan) for k in ("rung3_value", "rung3b_sue", "rung3c_accruals", "rung4_clearing", "rung4b_clearing_sue", "rung4c_clearing_accruals", "rung4d_clearing_on_accruals",
                                                 "momentum", "st_reversal", "low_risk", "size", "illiquidity", "skewness",
                                                 "inst_level", "inst_change", "holders_change", "embedding_peer_value", "composite_nofit",
                                                 "rung4e_clearing_float13f", "rung4f_clearing_inst13f")}
    inst = store.all_versions("inst_holdings") if "inst_holdings" in store.tables() else None
    mfiles = sorted(glob.glob("runs/sec_13f/matrix/*.parquet"))
    cusip_of = {}
    if inst is not None:
        for r in inst.filter(pl.col("symbol").is_not_null()).select(["symbol", "cusip"]).unique().iter_rows(named=True):
            cusip_of.setdefault(r["symbol"], r["cusip"])

    kappas = []
    for t in dec:
        asof = dates[t]
        U = universe_at(t)
        eps = ttm_eps_asof(facts, asof, syms); sh = shares_asof(facts, asof, syms)
        p = P[t]; ok = U & np.isfinite(eps) & (eps > 0) & np.isfinite(sh) & (sh > 0) & np.isfinite(p)
        if ok.sum() < 30:
            continue
        if t == dec[0] or t == dec[len(dec) // 2]:
            print(f"[coverage] {asof:%Y-%m-%d}: {int(ok.sum())}/{int(U.sum())} of the point-in-time universe have 4 known EPS quarters + shares")
        ey = eps[ok] / p[ok]; ey_med = np.median(ey)
        p_fund = np.where(ok, eps / ey_med, np.nan)                            # value anchor: same earnings yield for all
        # clearing: kappa from trailing 250d panel (returns vs Sigma s), Sigma from trailing 250d, s = shares (scaled)
        w = R[t - 250:t][:, ok]; Sig = np.cov(w.T) * P[t][ok][:, None] * P[t][ok][None, :]   # payoff covariance (price units)
        s = sh[ok] / sh[ok].sum() * ok.sum()                                    # supply in relative share units
        # kappa from REALISED excess payoffs (research/09 estimator): mean over the trailing window of the 20-day
        # excess payoff per share, regressed on (Sigma s) across stocks
        h20 = 20
        fwd = P[t - 250 + h20:t + 1][:, ok] / P[t - 250:t + 1 - h20][:, ok] - Rf_d ** h20          # (230, n) realised 20d excess returns
        prem = (fwd * P[t - 250:t + 1 - h20][:, ok]).mean(0)                                        # in payoff (price) units
        x = Sig @ s
        kappa = float((prem * x).sum() / (x * x).sum()); kappa = max(kappa, 0.0); kappas.append(kappa)
        p_hat = np.full(N, np.nan); p_hat[ok] = p_fund[ok] - kappa * x / Rf_d ** 20
        sig["rung3_value"][t] = np.log(p_fund / p) ; sig["rung4_clearing"][t] = np.log(np.maximum(p_hat, 1e-6) / p)
        Pu = np.where(U[None, :] & listed, P, np.nan)
        # --- US holdings state (13F, point-in-time by filing date) ---
        if inst is not None:
            known = inst.filter(pl.col("knowledge_time") <= asof).sort("knowledge_time").group_by(["cusip", "event_time"]).last()
            g = known.sort("event_time").group_by("cusip").agg(pl.col("inst_shares").tail(2).alias("sh2"), pl.col("n_holders").tail(2).alias("nh2"), pl.col("event_time").tail(1).alias("t1"))
            rec = {r["cusip"]: r for r in g.iter_rows(named=True) if (asof - r["t1"][0]).days < 200}
            lvl = np.full(N, np.nan); chg = np.full(N, np.nan); hchg = np.full(N, np.nan)
            for i, sym in enumerate(syms):
                r = rec.get(cusip_of.get(sym))
                if r is None or not (np.isfinite(sh[i]) and sh[i] > 0):
                    continue
                lvl[i] = r["sh2"][-1] / sh[i]
                if len(r["sh2"]) == 2 and r["sh2"][0] > 0:
                    chg[i] = (r["sh2"][-1] - r["sh2"][0]) / sh[i]; hchg[i] = np.log(max(r["nh2"][-1], 1) / max(r["nh2"][0], 1))
            sig["inst_level"][t] = np.where(U, lvl, np.nan); sig["inst_change"][t] = np.where(U, chg, np.nan); sig["holders_change"][t] = np.where(U, hchg, np.nan)
            # holdings-calibrated clearing (research/09 §5 follow-up): the supply the price-sensitive sector must absorb is
            # not total shares but (e) the NON-institutional float, or (f) the institutional block itself; kappa re-estimated
            # for each supply definition with the same realised-payoff regression, same Sigma, same value anchor.
            okl = ok & np.isfinite(lvl)
            if okl.sum() >= 30:
                for key, frac in (("rung4e_clearing_float13f", np.clip(1.0 - lvl, 0.05, 1.0)), ("rung4f_clearing_inst13f", np.clip(lvl, 0.05, 1.0))):
                    sub = okl[ok]                                                       # rows of the `ok` block that also have holdings
                    s_alt = (sh[ok] * frac[ok]); s_alt = s_alt / s_alt[sub].sum() * sub.sum(); s_alt[~sub] = 0.0
                    x_alt = Sig @ s_alt; k_alt = max(float((prem * x_alt).sum() / (x_alt * x_alt).sum()), 0.0)
                    ph = np.full(N, np.nan); ph[ok] = p_fund[ok] - k_alt * x_alt / Rf_d ** 20
                    sig[key][t] = np.where(okl, np.log(np.maximum(ph, 1e-6) / p), np.nan)
            # asset embeddings from the latest known manager x stock matrix (top-300 managers by value), peer-implied value
            per = max(rec[c]["t1"][0] for c in rec) if rec else None
            if per is not None and mfiles:
                if "_mcache" not in globals():
                    globals()["_mcache"] = {}
                key = per.strftime("%Y-%m-%d")
                if key not in _mcache:
                    mats = [pl.scan_parquet(f).filter(pl.col("period") == per).collect() for f in mfiles]
                    _mcache[key] = pl.concat([m for m in mats if m.height]) if any(m.height for m in mats) else None
                M = _mcache[key].filter(pl.col("filed") <= asof) if _mcache[key] is not None else pl.DataFrame()
                if M.height:
                    topm = M.group_by("CIK").agg(pl.col("value").sum().alias("v")).sort("v", descending=True).head(300)["CIK"].to_list()
                    M = M.filter(pl.col("CIK").is_in(topm))
                    cus = [cusip_of.get(sym) for sym in syms]
                    idx_c = {c: i for i, c in enumerate(cus) if c}
                    H = np.zeros((N, len(topm))); mi = {m: j for j, m in enumerate(topm)}
                    for r in M.select(["CIK", "CUSIP", "value"]).iter_rows():
                        if r[1] in idx_c:
                            H[idx_c[r[1]], mi[r[0]]] += r[2]
                    okh = ok & (H.sum(1) > 0)
                    if okh.sum() >= 30:
                        E, _, _ = asset_embeddings(H[okh] / (H[okh].sum(1, keepdims=True) + 1e-9), k=8)
                        ey = eps[okh] / p[okh]
                        pv = np.full(N, np.nan); pv[okh] = peer_implied_yield(E, ey, n_peers=10); sig["embedding_peer_value"][t] = pv
        th = price_themes(np.where(np.isfinite(Pu), Pu, np.nan), Vm, t)
        for k in ("momentum", "st_reversal", "low_risk", "size", "illiquidity", "skewness"):
            if k in th: sig[k][t] = np.where(U, th[k], np.nan)
        # no-fit composite: equal-weight mean of standardised anchors with published direction (value, SUE, accruals, momentum)
        zc = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
        comp = np.nanmean(np.stack([zc(sig["rung3_value"][t]), zc(sig["rung3b_sue"][t]), zc(sig["rung3c_accruals"][t]), zc(np.where(U, th.get("momentum", np.full(N, np.nan)), np.nan))]), axis=0)
        sig["composite_nofit"][t] = np.where(U & np.isfinite(comp), comp, np.nan)
        # accounting anchors (point-in-time)
        sue = sue_asof(facts, asof, syms); acc = accruals_asof(facts, asof, syms)
        sig["rung3b_sue"][t] = sue
        sig["rung3c_accruals"][t] = -acc
        # surprise-adjusted beliefs into the clearing model: mu scaled by (1 + 0.02 * SUE), a fixed prior loading (ledgered)
        adj = np.where(np.isfinite(sue), 1.0 + 0.02 * sue, 1.0)
        p_hat_b = np.full(N, np.nan); p_hat_b[ok] = p_fund[ok] * adj[ok] - kappa * x / Rf_d ** 20
        sig["rung4b_clearing_sue"][t] = np.log(np.maximum(p_hat_b, 1e-6) / p)
        # accrual-adjusted beliefs (cross-sectional z-score of -accruals, fixed loading 3 % per sd) into the clearing model
        za = -acc; za = (za - np.nanmean(za)) / (np.nanstd(za) + 1e-12); za = np.where(np.isfinite(za), np.clip(za, -3, 3), 0.0)
        p_hat_c = np.full(N, np.nan); p_hat_c[ok] = p_fund[ok] * (1.0 + 0.03 * za[ok]) - kappa * x / Rf_d ** 20
        sig["rung4c_clearing_accruals"][t] = np.log(np.maximum(p_hat_c, 1e-6) / p)
        # beliefs BUILT from the informative anchor: mu = Rf p * (1 + 0.03 z_accruals) (no value anchor), then the
        # clearing correction -kappa Sigma s.  Tests whether the structure preserves and prices an informative anchor.
        p_hat_d = np.full(N, np.nan); p_hat_d[ok] = p[ok] * (1.0 + 0.03 * za[ok]) - kappa * x / Rf_d ** 20
        sig["rung4d_clearing_on_accruals"][t] = np.log(np.maximum(p_hat_d, 1e-6) / p)
    print(f"[kappa] median over decision dates = {np.nanmedian(kappas):.4g} (n={len(kappas)}; zero-clipped {int((np.array(kappas)==0).sum())})")
    # holdings state from Fed Z.1 (point-in-time): phi = mandated institutional share -> A = (1-phi)/kappa (research/09 §5)
    if "z1" in store.tables():
        from emporion.data.loaders.fed_z1 import mandated_share
        z1 = mandated_share(store.as_of("z1", now))
        phi_last = float(z1.sort("event_time")["phi_mandated"].drop_nulls().tail(1)[0])
        kap = np.array([k for k in kappas if k > 0])
        if kap.size:
            print(f"[state] Z.1 mandated share phi (latest known) = {phi_last:.3f}; aggregate risk tolerance A = (1-phi)/kappa: median {np.median((1 - phi_last) / kap):.3g}, IQR {np.percentile((1 - phi_last) / kap, 25):.3g}..{np.percentile((1 - phi_last) / kap, 75):.3g}")
    scope = f"p4-realdata-us-stocks-{args.scope_version}"      # one scope per procedure FAMILY; older (buggy-pipeline) scopes stay on the ledger
    results = {}
    Rfull = R[:T] if R.shape[0] == T else np.vstack([R, np.zeros((1, N))])
    def tilt_returns(S):
        W = np.zeros((T, N)); cur = np.zeros(N)
        for t in range(T):
            if np.isfinite(S[t]).sum() >= 30:
                sc = S[t]; k = int(np.isfinite(sc).sum() * 0.2)
                top = np.argsort(np.nan_to_num(sc, nan=-1e9))[-k:]
                cur = np.zeros(N); cur[top] = 1.0 / k
            W[t] = cur
        return portfolio_returns(W, Rfull, cost_bps=args.cost_bps)[0], W
    nets = {name: tilt_returns(S)[0] for name, S in sig.items()}
    pbo = pbo_cscv(np.stack([nets[n] for n in sig], axis=1), n_blocks=16)["pbo"]
    print(f"[pbo] across {len(sig)} rungs (CSCV, 16 blocks): PBO={pbo:.3f}")
    for name, S in sig.items():
        for h in (20, 60):
            ics = []
            for t in dec:
                if not np.isfinite(S[t]).any():
                    continue
                real = np.log(P[t + h] / P[t]); real = real - np.nanmean(real[np.isfinite(S[t])])  # demeaned within the universe
                ics.append(rank_ic(S[t], real))
            ics = np.array([x for x in ics if np.isfinite(x)])
            r = {"n": int(len(ics)), "mean_ic": float(ics.mean()), "t": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), "hit": float((ics > 0).mean())}
            results[f"{name}_h{h}"] = r
            print(f"[rankic] {name:15s} h={h:2d} n={r['n']:3d} meanIC={r['mean_ic']:+.4f} t={r['t']:+.2f} hit={r['hit']:.2f}")
        # long-only top-quintile tilt, rebalanced monthly, through the certification stack
        net, W = nets[name], tilt_returns(S)[1]
        Wew = np.zeros((T, N))                                                   # equal-weight of the SAME point-in-time universe
        cur = np.zeros(N)
        for t in range(T):
            if np.isfinite(S[t]).sum() >= 30:
                m = np.isfinite(S[t]); cur = m / m.sum()
            Wew[t] = cur
        ew, _ = portfolio_returns(Wew, Rfull, cost_bps=args.cost_bps)
        rep = certify(f"{name}_top20pct", T, lambda tr, te, net=net: net[te], net, ledger, scope, benchmark_returns=ew, B=1000,
                      pbo=pbo, extra={"h": "monthly", "cost_bps": args.cost_bps})
        rel = rep["vs_benchmark_ann_excess_ci"]
        print(f"[certify] {name:15s} SR={rep['sharpe']:+.2f} mdd={rep['max_drawdown']*100:.1f}% DSR={rep['dsr']['dsr']:.3f}(N={rep['n_trials']}) "
              f"vsEW={rel['lo']*100:+.1f}..{rel['hi']*100:+.1f}%/yr -> {'CERTIFIED' if rep['certified'] else 'not certified'}")
        results[f"{name}_certify"] = {k: rep[k] for k in ("sharpe", "max_drawdown", "certified")}
    with open(os.path.join(args.root, f"p4-realdata-{now:%Y%m%d-%H%M}.json"), "w") as f:
        json.dump({"N": N, "T": T, "results": results, "kappa_median": float(np.nanmedian(kappas))}, f, indent=1, default=str)
    assert ledger.verify_chain(); print("[done]")


if __name__ == "__main__":
    main()
