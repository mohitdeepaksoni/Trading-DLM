"""P2: regime gate + conformal intervals, evaluated honestly through the ledger.
    python scripts/p2_regime_conformal.py --market us
1. Walk-forward 2-state Gaussian HMM on the equal-weight market return (filtered probs only).
2. Regime-sliced stats of EW: Sharpe/vol/MDD in calm vs stress, time in stress.
3. Abstention in miniature: EW gated flat when P(stress) > thr (costs included, next-bar execution),
   certified vs ungated EW with the same CPCV/DSR/bootstrap stack.  Threshold is a trial (logged).
4. Split-conformal interval on next-day EW return around a zero forecast, scaled by EWMA vol:
   coverage and mean width — the sizing input for conformal Kelly."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger                                   # noqa: E402
from emporion.data.store import BitemporalStore                           # noqa: E402
from emporion.baselines import equal_weight, GaussianHMM, conformal_quantile, coverage, ewma  # noqa: E402
from emporion.evaluate import portfolio_returns, certify, sharpe, max_drawdown  # noqa: E402
from p0_certify_baselines import returns_panel                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="us")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--thresholds", default="0.5,0.7,0.9")
    ap.add_argument("--start", default=None)
    args = ap.parse_args()
    periods = 365 if args.market.startswith("crypto") else 252
    root = os.path.join(args.root, args.market)
    store = BitemporalStore(os.path.join(root, "store"))
    ledger = Ledger(os.path.join(root, "ledger"))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    R, syms, dates = returns_panel(store, now, min_history=1000 if args.market != "synthetic" else 100, top_n=args.top_n, start=(dt.datetime.fromisoformat(args.start) if args.start else None))
    T, N = R.shape
    scope = f"p2-regime-{args.market}-N{N}"
    W = equal_weight(R)
    ew, _ = portfolio_returns(W, R, cost_bps=args.cost_bps)
    mkt = R.mean(axis=1)                                            # gross EW market return, the HMM input
    P = GaussianHMM(2).rolling_filter(mkt, refit_every=63, min_train=min(500, T // 3))
    p_stress = P[:, 1]
    ok = np.isfinite(p_stress)
    lo = int(np.argmax(ok))
    print(f"[panel] {args.market}: T={T} N={N}; HMM walk-forward from {dates[lo]:%Y-%m-%d}; time in stress(P>0.5)={float((p_stress[ok] > 0.5).mean()):.2f}")
    # regime-sliced EW stats (using the filtered prob known at t-1 for the return at t)
    calm = ok.copy(); calm[1:] &= p_stress[:-1] <= 0.5; calm[0] = False
    stress = ok.copy(); stress[1:] &= p_stress[:-1] > 0.5; stress[0] = False
    for nm, m in (("calm", calm), ("stress", stress)):
        r = ew[m]
        print(f"[regime] {nm:6s} days={m.sum():5d} SR={float(sharpe(r, periods)[0]):+.2f} vol={float(r.std()*np.sqrt(periods))*100:5.1f}% "
              f"mdd={float(max_drawdown(r)[0])*100:6.1f}% mean_daily={r.mean()*1e4:+.1f}bp")
    ledger.append("note", "regime_slices", {"calm_sr": float(sharpe(ew[calm], periods)[0]), "stress_sr": float(sharpe(ew[stress], periods)[0]),
                                            "stress_frac": float(stress.sum() / ok.sum())}, scope=scope)
    # gated EW: flat when P(stress) > thr, decided at t with P known at t (executes t+1 via shift in portfolio_returns)
    ung = certify("ew_ungated", T, lambda tr, te: ew[te], ew, ledger, scope, periods=periods, B=1000)
    print(f"[certify] ew_ungated SR={ung['sharpe']:+.2f} mdd={ung['max_drawdown']*100:.1f}% CI[{ung['bootstrap_sharpe_ci']['lo']:+.2f},{ung['bootstrap_sharpe_ci']['hi']:+.2f}]")
    for thr in [float(x) for x in args.thresholds.split(",")]:
        g = np.where(np.nan_to_num(p_stress, nan=0.0) > thr, 0.0, 1.0)[:, None]
        Wg = W * g
        rg, _ = portfolio_returns(Wg, R, cost_bps=args.cost_bps)
        rep = certify(f"ew_gated_thr{thr}", T, lambda tr, te, rg=rg: rg[te], rg, ledger, scope,
                      benchmark_returns=ew, periods=periods, B=1000, extra={"threshold": thr, "flat_frac": float(1 - g.mean())})
        rel = rep["vs_benchmark_ann_excess_ci"]
        print(f"[certify] gated thr={thr:.1f} flat={1-g.mean():.2f} SR={rep['sharpe']:+.2f} mdd={rep['max_drawdown']*100:.1f}% "
              f"DSR={rep['dsr']['dsr']:.3f}(N={rep['n_trials']}) vsEW={rel['lo']*100:+.1f}..{rel['hi']*100:+.1f}%/yr -> {'CERTIFIED' if rep['certified'] else 'not certified'}")
    # conformal interval on next-day EW return: residual = r_t / sigma_t (EWMA, known at t-1), zero forecast
    h = ewma(mkt[:, None])[:, 0]
    z = mkt / np.sqrt(np.maximum(h, 1e-12))
    q = conformal_quantile(z, alpha=0.1, window=250)
    cov = coverage(z, q)
    width = float(np.nanmean(q * np.sqrt(h)) * 2)
    print(f"[conformal] 90% interval on EW next-day return: empirical coverage={cov:.3f}, mean width={width*100:.2f}% (vol-scaled, rolling 250)")
    ledger.append("note", "conformal_ew", {"alpha": 0.1, "coverage": cov, "mean_width": width}, scope=scope)
    stamp = now.strftime("%Y%m%d-%H%M")
    with open(os.path.join(root, f"p2-regime-{stamp}.json"), "w") as f:
        json.dump({"market": args.market, "T": T, "N": N, "stress_frac": float(stress.sum() / ok.sum()), "ungated": ung,
                   "conformal": {"coverage": cov, "mean_width": width}}, f, indent=1, default=str)
    assert ledger.verify_chain()
    print(f"[done] ledger rows={ledger.frame().height}")


if __name__ == "__main__":
    main()
