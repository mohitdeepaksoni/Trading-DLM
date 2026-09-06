"""Crypto ladder on the full Coinbase panel: the replication themes that the crypto literature says matter
(size, momentum, trend, illiquidity, low-risk, reversal, skewness), monthly decisions, top-quintile tilt, certify().
No fundamentals anchor exists in crypto (research/09), so these are the honest rungs."""
from __future__ import annotations

import argparse, datetime as dt, json, os, sys
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from emporion.evaluate import portfolio_returns, certify, pbo_cscv
from emporion.anchors import price_themes, THEMES
from p3_event_harness import rank_ic


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs/crypto_full"); ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--cost-bps", type=float, default=25.0); ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--long-short", action="store_true", help="also evaluate long top-quintile / short bottom-quintile, vol-targeted to 30 %")
    args = ap.parse_args()
    store = BitemporalStore(os.path.join(args.root, "store")); ledger = Ledger(os.path.join(args.root, "ledger"))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    bars = store.as_of("bars", now, columns=["close", "volume"])
    n_days = bars["event_time"].n_unique()
    cov = bars.group_by("symbol").agg(pl.len().alias("n")).filter(pl.col("n") >= 400)
    syms = cov["symbol"].to_list()                                           # ALL pairs with >= 400 days; selection is point-in-time below
    b = bars.filter(pl.col("symbol").is_in(syms))
    P = b.pivot(index="event_time", on="symbol", values="close").sort("event_time"); V = b.pivot(index="event_time", on="symbol", values="volume").sort("event_time")
    dates = P["event_time"].to_list(); Pm = P.select(syms).to_numpy(); Vm = V.select(syms).to_numpy()
    valid = np.isfinite(Pm)                                                   # pairs list at different dates: mask per date
    Pm = np.where(valid, Pm, np.nan); Vm = np.nan_to_num(Vm, nan=0.0)
    T, N = Pm.shape; R = np.where(valid[1:] & valid[:-1], Pm[1:] / Pm[:-1] - 1, 0.0)
    print(f"[panel] crypto {N} pairs x {T} days {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}")
    periods = 365
    dec = list(range(400, T - 61, args.every))
    sig = {k: np.full((T, N), np.nan) for k in THEMES}
    dvol = np.nan_to_num(Pm) * Vm
    for t in dec:
        Pw = Pm[:t + 1].copy()
        alive = valid[t] & valid[t - 365]
        med = np.median(dvol[t - 90:t + 1], axis=0); med[~alive] = -1
        top = np.argsort(-med)[:args.top_n]; U = np.zeros(N, bool); U[top] = med[top] > 0
        alive = alive & U                                                      # point-in-time top-K by trailing dollar volume
        Pw[:, ~alive] = np.nan
        th = price_themes(np.where(np.isfinite(Pw), Pw, np.nan), Vm[:t + 1], t, periods=periods)
        for k, v in th.items():
            sig[k][t] = np.where(alive, v, np.nan)
    scope = f"p4-realdata-crypto-N{N}"; Rfull = R if R.shape[0] == T else np.vstack([R, np.zeros((1, N))])
    def tilt(S):
        W = np.zeros((T, N)); cur = np.zeros(N)
        for t in range(T):
            if np.isfinite(S[t]).sum() >= 10:
                sc = S[t]; k = max(1, int(np.isfinite(sc).sum() * 0.2)); top = np.argsort(np.nan_to_num(sc, nan=-1e9))[-k:]
                cur = np.zeros(N); cur[top] = 1.0 / k
            W[t] = cur
        return portfolio_returns(W, Rfull, cost_bps=args.cost_bps)[0], W
    def long_short(S, target_vol=0.30):
        """Dollar-neutral top-minus-bottom quintile, scaled daily to a 30 % annualised vol target from trailing 60d vol (known at t)."""
        W = np.zeros((T, N)); cur = np.zeros(N)
        for t in range(T):
            if np.isfinite(S[t]).sum() >= 10:
                sc = S[t]; ok = np.isfinite(sc); k = max(1, int(ok.sum() * 0.2)); order = np.argsort(np.nan_to_num(sc, nan=np.nan))
                ranked = [i for i in np.argsort(np.where(ok, sc, np.nan)) if ok[i]]
                cur = np.zeros(N); cur[ranked[-k:]] = 1.0 / k; cur[ranked[:k]] = -1.0 / k
            W[t] = cur
        raw, _ = portfolio_returns(W, Rfull, cost_bps=0.0)
        vol = np.array([raw[max(0, t - 60):t].std() * np.sqrt(periods) if t > 20 else np.nan for t in range(T)])
        scale = np.clip(target_vol / np.where(np.isfinite(vol) & (vol > 0), vol, np.inf), 0.0, 3.0)
        return portfolio_returns(W * scale[:, None], Rfull, cost_bps=args.cost_bps)[0]
    nets = {k: tilt(S)[0] for k, S in sig.items()}
    if args.long_short:
        for k in list(sig):
            nets[f"{k}_ls"] = long_short(sig[k])
    pbo = pbo_cscv(np.stack([nets[k] for k in nets], axis=1), n_blocks=16)["pbo"]; print(f"[pbo] across {len(nets)} rungs: {pbo:.3f}")
    results = {}
    for name in THEMES:
        S = sig[name]
        for h in (30, 90):
            ics = []
            for t in dec:
                if t + h >= T or np.isfinite(S[t]).sum() < 10:
                    continue
                real = np.log(Pm[t + h] / Pm[t]); real = real - np.nanmean(np.where(np.isfinite(S[t]), real, np.nan))
                ics.append(rank_ic(S[t], real))
            ics = np.array([x for x in ics if np.isfinite(x)])
            if len(ics) < 5:
                continue
            r = {"n": int(len(ics)), "mean_ic": float(ics.mean()), "t": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), "hit": float((ics > 0).mean())}
            results[f"{name}_h{h}"] = r; print(f"[rankic] {name:12s} h={h:2d} n={r['n']:3d} meanIC={r['mean_ic']:+.4f} t={r['t']:+.2f} hit={r['hit']:.2f}")
        net, W = nets[name], tilt(S)[1]
        Wew = np.zeros((T, N)); cur = np.zeros(N)
        for t in range(T):
            if np.isfinite(S[t]).sum() >= 10:
                m = np.isfinite(S[t]); cur = m / m.sum()
            Wew[t] = cur
        ew, _ = portfolio_returns(Wew, Rfull, cost_bps=args.cost_bps)
        rep = certify(f"{name}_top20pct", T, lambda tr, te, net=net: net[te], net, ledger, scope, benchmark_returns=ew, periods=periods, B=1000, pbo=pbo, extra={"cost_bps": args.cost_bps})
        rel = rep["vs_benchmark_ann_excess_ci"]
        print(f"[certify] {name:12s} SR={rep['sharpe']:+.2f} mdd={rep['max_drawdown']*100:.1f}% DSR={rep['dsr']['dsr']:.3f}(N={rep['n_trials']}) vsEW={rel['lo']*100:+.1f}..{rel['hi']*100:+.1f}%/yr -> {'CERTIFIED' if rep['certified'] else 'not certified'}")
        results[f"{name}_certify"] = {k: rep[k] for k in ("sharpe", "max_drawdown", "certified")}
        if args.long_short:
            net_ls = nets[f"{name}_ls"]
            rep = certify(f"{name}_longshort_vt30", T, lambda tr, te, n_=net_ls: n_[te], net_ls, ledger, scope, benchmark_returns=None, periods=periods, B=1000, pbo=pbo, extra={"cost_bps": args.cost_bps, "vol_target": 0.30})
            print(f"[certify] {name+'_ls':12s} SR={rep['sharpe']:+.2f} mdd={rep['max_drawdown']*100:.1f}% CI[{rep['bootstrap_sharpe_ci']['lo']:+.2f},{rep['bootstrap_sharpe_ci']['hi']:+.2f}] DSR={rep['dsr']['dsr']:.3f}(N={rep['n_trials']}) -> {'CERTIFIED' if rep['certified'] else 'not certified'}")
    with open(os.path.join(args.root, f"p4-realdata-crypto-{now:%Y%m%d-%H%M}.json"), "w") as f:
        json.dump({"N": N, "T": T, "pbo": pbo, "results": results}, f, indent=1, default=str)
    assert ledger.verify_chain(); print("[done]")


if __name__ == "__main__":
    main()
