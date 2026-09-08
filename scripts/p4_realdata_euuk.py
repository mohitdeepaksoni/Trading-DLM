"""EU/UK anchor ladder — first pass (plan/PHASE-EU-UK.md steps 3–4; plan/DAY-2026-09-08.md block C). Scope `p4-realdata-euuk`.

Data: runs/eu_uk/store — `bars` (Yahoo, split-adjusted, local currency: GBp on LSE, EUR/SEK/DKK/NOK/CHF elsewhere), `universe`
(ESEF issuer -> primary listed line), `esef_facts` (annual / half-year IFRS statements, knowledge_time = date added to the OAM).
Point-in-time universe: top-K by trailing 250-day median traded value in EUR (static FX, ranking only) among lines listed >= 300 d.
Anchors, all from the last filing known at the decision date (revisions resolved per date, research/03 §12):
  value             log(EPS_annual / median_ey / p)     EPS converted to the price currency (static FX); UK pence/pounds resolved by
                                                        a plausibility rule on the earnings yield (counted and printed, never silent)
  earnings_growth   (EPS - EPS_prior) / (|EPS| + |EPS_prior|)  from the filing's own comparative (bounded, no history needed)
  accruals          -(profit - CFO) / assets            the only anchor that survived on the US
  sue_annual        (EPS - EPS_prior) / std of the issuer's earlier annual differences (>= 3 known): thin by construction (FY2020->)
  price themes      momentum, st_reversal, low_risk, size, illiquidity (traded value in EUR)
  composite_nofit   equal z-weights value / earnings_growth / accruals / momentum, fixed forever
Costs inside returns: 15 bp one-way (EU), 20 bp (UK) + 50 bp stamp duty on UK purchases. Monthly decisions, RankIC 20/60 d,
top-quintile long-only tilt vs the equal-weight universe under the same costs, PBO across rungs, certify().
Caveats stated in every number: dividend-unadjusted prices (EW benchmark carries the same bias), static FX, annual filings with a
~5-month median lag, issuers delisted before 2021 absent (ESEF starts FY2020). Nothing here is pre-registered.
    python scripts/p4_realdata_euuk.py [--universe-k 200] [--every 21]
"""
from __future__ import annotations

import argparse, datetime as dt, json, os, sys
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from emporion.evaluate import certify, pbo_cscv
from emporion.evaluate.costs import _drifted
from emporion.anchors import price_themes
from p3_event_harness import rank_ic

# static FX to EUR (2026-09 levels, ranking / per-share conversion only; the time variation is second order for cross-sectional ranks)
FX_EUR = {"EUR": 1.0, "GBP": 1.17, "GBp": 0.0117, "USD": 0.92, "SEK": 0.088, "DKK": 0.134, "NOK": 0.086, "CHF": 1.05, "HKD": 0.118, "PLN": 0.235, "ISK": 0.0068, "CZK": 0.040}
PRICE_CCY = {"LSE": "GBp", "PAR": "EUR", "AMS": "EUR", "BRU": "EUR", "LIS": "EUR", "MIL": "EUR", "MCE": "EUR", "VIE": "EUR", "HEL": "EUR", "GER": "EUR", "FRA": "EUR",
             "ISE": "EUR", "LUX": "EUR", "STO": "SEK", "CPH": "DKK", "OSL": "NOK", "EBS": "CHF", "WSE": "PLN", "ICE": "ISK", "ATH": "EUR"}
UK_EXCH = {"LSE"}


def portfolio_returns_vec(W, R, cost_bps, buy_tax_bps):
    """portfolio_returns with per-asset one-way costs (bp) and a per-asset purchase tax (bp, e.g. UK stamp duty); next-bar execution."""
    W = np.nan_to_num(np.asarray(W, float)); R = np.nan_to_num(np.asarray(R, float)); T, N = W.shape
    Wx = np.zeros_like(W); Wx[1:] = W[:T - 1]
    gross = (Wx * R).sum(1)
    prev = np.zeros_like(W); prev[1:] = _drifted(Wx[:-1], R[:-1])
    d = Wx - prev
    net = gross - (np.abs(d) * cost_bps[None, :]).sum(1) / 1e4 - (np.maximum(d, 0.0) * buy_tax_bps[None, :]).sum(1) / 1e4
    return net, np.abs(d).sum(1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs/eu_uk"); ap.add_argument("--every", type=int, default=21)
    ap.add_argument("--universe-k", type=int, default=200); ap.add_argument("--cost-eu-bps", type=float, default=15.0); ap.add_argument("--cost-uk-bps", type=float, default=20.0)
    ap.add_argument("--uk-stamp-bps", type=float, default=50.0); ap.add_argument("--min-history", type=int, default=500); args = ap.parse_args()
    store = BitemporalStore(os.path.join(args.root, "store")); ledger = Ledger(os.path.join(args.root, "ledger"))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    uni = store.as_of("universe", now).filter(pl.col("primary") == True)  # noqa: E712
    facts = store.all_versions("esef_facts")                                                   # ALL revisions; resolved per decision date
    bars = store.as_of("bars", now, columns=["close", "volume"]).filter(pl.col("symbol").is_in(uni["symbol"].to_list()))
    cov = bars.group_by("symbol").agg(pl.len().alias("n")).filter(pl.col("n") >= args.min_history)
    bars = bars.filter(pl.col("symbol").is_in(cov["symbol"].to_list()))
    wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time")
    syms = [c for c in wide.columns if c != "event_time"]; N = len(syms)
    wide = wide.with_columns([pl.col(c).fill_null(strategy="forward") for c in syms])
    dates = wide["event_time"].to_list(); P = wide.select(syms).to_numpy().astype(float); T = P.shape[0]
    listed = np.isfinite(P); R = np.where(listed[1:] & listed[:-1], P[1:] / P[:-1] - 1, 0.0); R = np.vstack([R, np.zeros((1, N))])
    volw = bars.pivot(index="event_time", on="symbol", values="volume").sort("event_time")
    Vm = np.nan_to_num(volw.select(syms).to_numpy().astype(float))
    meta = {r["symbol"]: r for r in uni.iter_rows(named=True)}
    exch = np.array([meta[s]["exchange"] for s in syms]); country = np.array([meta[s]["country"] for s in syms])
    pccy = np.array([PRICE_CCY.get(e, "EUR") for e in exch]); scale = np.array([FX_EUR.get(c, 1.0) for c in pccy])
    Vs = Vm * scale[None, :]                                                                  # volume scaled so that P * Vs is traded value in EUR
    dv = np.nan_to_num(P) * Vs
    cost = np.where(np.isin(exch, list(UK_EXCH)), args.cost_uk_bps, args.cost_eu_bps); stamp = np.where(np.isin(exch, list(UK_EXCH)), args.uk_stamp_bps, 0.0)
    def universe_at(t, K=args.universe_k):
        hist = listed[max(0, t - 300):t + 1].all(0); med = np.median(dv[t - 250:t + 1], axis=0); med[~hist] = -1
        keep = np.zeros(N, bool); top = np.argsort(-med)[:K]; keep[top] = med[top] > 0; return keep
    print(f"[panel] EU/UK {N} lines x {T} days {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}; by country {dict(zip(*np.unique(country, return_counts=True)))}; facts {facts.height:,} rows")
    # facts -> symbol; per-share figures into the PRICE currency of the line (static FX); UK pence/pounds plausibility rule applied per date
    f = (facts.join(uni.select(["lei", "symbol"]), on="lei", how="inner").filter(pl.col("symbol").is_in(syms))
              .with_columns(pl.coalesce([pl.col("eps_diluted"), pl.col("eps_basic")]).alias("eps"))
              .with_columns(((pl.col("profit_owners").fill_null(pl.col("profit")) - pl.col("cfo")) / pl.col("assets")).alias("accruals"),
                            ((pl.col("eps") - pl.col("eps_diluted_prior")) / (pl.col("eps").abs() + pl.col("eps_diluted_prior").abs() + 1e-12)).alias("growth"),
                            (pl.col("eps") - pl.col("eps_diluted_prior")).alias("dif")))
    fx_eps = np.array([FX_EUR.get(c or "EUR", 1.0) for c in f["currency"].to_list()]); sym_pccy = {s: c for s, c in zip(syms, pccy)}
    fx_px = np.array([FX_EUR[sym_pccy[s]] for s in f["symbol"].to_list()])
    f = f.with_columns(pl.Series("fx", fx_eps / fx_px))                                          # multiply a per-share amount by fx -> price-currency units
    f_ann = f.filter(pl.col("period_type") == "annual")
    def last_known(df, asof, cols, max_age=550):
        k = df.filter(pl.col("knowledge_time") <= asof).sort("knowledge_time").group_by(["symbol", "event_time"]).last()
        k = k.sort("event_time").group_by("symbol").last().filter((pl.lit(asof) - pl.col("event_time")).dt.total_days() < max_age)
        return {r["symbol"]: r for r in k.select(["symbol"] + cols).iter_rows(named=True)}
    def sue_annual(asof):
        k = f_ann.filter(pl.col("knowledge_time") <= asof).sort("knowledge_time").group_by(["symbol", "event_time"]).last().filter(pl.col("dif").is_not_null())
        g = k.sort("event_time").group_by("symbol").agg(pl.col("dif"), pl.col("fx").last().alias("fx"), pl.col("event_time").last().alias("t1"))
        out = {}
        for r in g.iter_rows(named=True):
            d = np.array(r["dif"], float)
            if len(d) >= 4 and (asof - r["t1"]).days < 550:
                sd = d[:-1].std(ddof=1)
                if sd > 1e-9:
                    out[r["symbol"]] = float(np.clip(d[-1] / sd, -5, 5))
        return out
    dec = list(range(300, T - 61, args.every))
    names = ("value", "earnings_growth", "accruals", "sue_annual", "momentum", "st_reversal", "low_risk", "size", "illiquidity", "composite_nofit")
    sig = {k: np.full((T, N), np.nan) for k in names}; unit_fix = {"x100": 0, "/100": 0, "n": 0}; cover = []
    zc = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
    for t in dec:
        asof = dates[t]; p = P[t]; U = universe_at(t)
        lk = last_known(f_ann, asof, ["eps", "fx", "accruals", "growth", "country"])
        eps = np.full(N, np.nan); acc = np.full(N, np.nan); gr = np.full(N, np.nan)
        for i, s in enumerate(syms):
            r = lk.get(s)
            if r is None or r["eps"] is None:
                continue
            e = r["eps"] * r["fx"]
            if np.isfinite(p[i]) and p[i] > 0 and e != 0:                                    # UK pence vs pounds: a 100x unit error puts the yield far outside (0.2 %, 50 %)
                ey = abs(e) / p[i]
                if r["country"] == "GB" and ey < 0.002:
                    e *= 100.0; unit_fix["x100"] += 1
                elif r["country"] == "GB" and ey > 0.5:
                    e /= 100.0; unit_fix["/100"] += 1
                unit_fix["n"] += 1
            eps[i] = e; acc[i] = r["accruals"] if r["accruals"] is not None else np.nan; gr[i] = r["growth"] if r["growth"] is not None else np.nan
        ok = U & np.isfinite(eps) & (eps > 0) & np.isfinite(p) & (p > 0)
        cover.append((asof, int(U.sum()), int((U & np.isfinite(eps)).sum()), int((U & np.isfinite(acc)).sum())))
        Pu = np.where(U[None, :] & listed, P, np.nan); th = price_themes(Pu, Vs, t)
        for k in ("momentum", "st_reversal", "low_risk", "size", "illiquidity"):
            if k in th: sig[k][t] = np.where(U, th[k], np.nan)
        if ok.sum() >= 30:
            ey_med = np.median(eps[ok] / p[ok]); sig["value"][t] = np.where(ok, np.log(np.where(ok, eps, 1.0) / ey_med / np.where(ok, p, 1.0)), np.nan)
        sig["accruals"][t] = np.where(U, -acc, np.nan); sig["earnings_growth"][t] = np.where(U, gr, np.nan)
        su = sue_annual(asof); sig["sue_annual"][t] = np.array([su.get(s, np.nan) if U[i] else np.nan for i, s in enumerate(syms)])
        comp = np.nanmean(np.stack([zc(sig["value"][t]), zc(sig["earnings_growth"][t]), zc(sig["accruals"][t]), zc(sig["momentum"][t] if "momentum" in th else np.full(N, np.nan))]), axis=0)
        sig["composite_nofit"][t] = np.where(U & np.isfinite(comp), comp, np.nan)
    c0, c1 = cover[0], cover[len(cover) // 2]
    print(f"[coverage] {c0[0]:%Y-%m-%d}: EPS known {c0[2]}/{c0[1]}, accruals {c0[3]}/{c0[1]}; {c1[0]:%Y-%m-%d}: EPS {c1[2]}/{c1[1]}, accruals {c1[3]}/{c1[1]}; last: EPS {cover[-1][2]}/{cover[-1][1]}")
    print(f"[units] UK EPS unit fixes over {unit_fix['n']} (line, date) pairs: x100 {unit_fix['x100']}, /100 {unit_fix['/100']}")
    scope = f"p4-realdata-euuk-N{N}"
    def tilt(S):
        W = np.zeros((T, N)); cur = np.zeros(N)
        for t in range(T):
            if np.isfinite(S[t]).sum() >= 30:
                sc = S[t]; k = max(1, int(np.isfinite(sc).sum() * 0.2)); top = np.argsort(np.nan_to_num(sc, nan=-1e9))[-k:]
                cur = np.zeros(N); cur[top] = 1.0 / k
            W[t] = cur
        return portfolio_returns_vec(W, R, cost, stamp)[0], W
    def ew_of(S):
        W = np.zeros((T, N)); cur = np.zeros(N)
        for t in range(T):
            if np.isfinite(S[t]).sum() >= 30:
                m = np.isfinite(S[t]); cur = m / m.sum()
            W[t] = cur
        return portfolio_returns_vec(W, R, cost, stamp)[0]
    nets = {k: tilt(S)[0] for k, S in sig.items()}
    pbo = pbo_cscv(np.stack([nets[k] for k in names], axis=1), n_blocks=16)["pbo"]; print(f"[pbo] across {len(names)} rungs: {pbo:.3f}")
    results = {}
    for name, S in sig.items():
        for h in (20, 60):
            ics = []
            for t in dec:
                if np.isfinite(S[t]).sum() < 30:
                    continue
                fwd = np.log(P[t + h] / P[t]); fwd = fwd - np.nanmean(np.where(np.isfinite(S[t]), fwd, np.nan)); ics.append(rank_ic(S[t], fwd))
            ics = np.array([x for x in ics if np.isfinite(x)])
            if len(ics) < 5:
                continue
            r = {"n": int(len(ics)), "mean_ic": float(ics.mean()), "t": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), "hit": float((ics > 0).mean())}
            results[f"{name}_h{h}"] = r; print(f"[rankic] {name:18s} h={h:2d} n={r['n']:3d} meanIC={r['mean_ic']:+.4f} t={r['t']:+.2f} hit={r['hit']:.2f}")
        net = nets[name]; ew = ew_of(S)
        if not np.isfinite(net).all() or np.abs(net).sum() == 0:
            continue
        rep = certify(f"{name}_top20pct", T, lambda tr, te, net=net: net[te], net, ledger, scope, benchmark_returns=ew, B=1000, pbo=pbo,
                      extra={"cost_eu_bps": args.cost_eu_bps, "cost_uk_bps": args.cost_uk_bps, "uk_stamp_bps": args.uk_stamp_bps, "universe_k": args.universe_k, "prices": "yahoo split-adjusted, dividend-unadjusted", "fx": "static"})
        rel = rep["vs_benchmark_ann_excess_ci"]
        print(f"[certify] {name:18s} SR={rep['sharpe']:+.2f} mdd={rep['max_drawdown']*100:.1f}% DSR={rep['dsr']['dsr']:.3f}(N={rep['n_trials']}) vsEW={rel['lo']*100:+.1f}..{rel['hi']*100:+.1f}%/yr -> {'CERTIFIED' if rep['certified'] else 'not certified'}")
        results[f"{name}_certify"] = {k: rep[k] for k in ("sharpe", "max_drawdown", "certified")}
    with open(os.path.join(args.root, f"p4-realdata-euuk-{now:%Y%m%d-%H%M}.json"), "w") as fo:
        json.dump({"N": N, "T": T, "pbo": pbo, "unit_fixes": unit_fix, "coverage": [(str(a), b, c, d) for a, b, c, d in cover[::6]], "results": results}, fo, indent=1, default=str)
    assert ledger.verify_chain(); print("[done]")


if __name__ == "__main__":
    main()
