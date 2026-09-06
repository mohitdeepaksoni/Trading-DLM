"""P0 end-to-end: load point-in-time bars (real if reachable, synthetic otherwise)
into the bitemporal store, run the passive/classical baselines through the
certification stack with costs, write ledger rows, save the benchmark report.

    python scripts/p0_certify_baselines.py --market us   # nasdaq.com ETFs (keyless)
    python scripts/p0_certify_baselines.py --market crypto
    python scripts/p0_certify_baselines.py --market synthetic
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core import perf                                   # noqa: E402
from emporion.core.ledger import Ledger                           # noqa: E402
from emporion.core.registry import get                            # noqa: E402
from emporion.data.store import BitemporalStore                   # noqa: E402
import emporion.data.loaders                                      # noqa: E402,F401
from emporion.baselines import BASELINES                          # noqa: E402
from emporion.evaluate import portfolio_returns, pbo_cscv, certify, sharpe  # noqa: E402


def load_market(market: str, store: BitemporalStore) -> pl.DataFrame:
    t0 = time.perf_counter()
    if market == "us":
        bars = get("loader", "nasdaq")()   # stooq/yahoo are blocked from cloud IPs
    elif market == "crypto":
        bars = get("loader", "binance")()
    elif market == "nse":
        bars = get("loader", "nse")(start="2022-01-01", end=dt.date.today().isoformat())
    else:
        bars = get("loader", "synthetic")(n_assets=20, n_days=2520, seed=0)
    store.register_table("bars", ["symbol"])
    h = store.append("bars", bars)
    print(f"[data] {market}: {bars.height:,} bars, {bars['symbol'].n_unique()} symbols, "
          f"{bars['event_time'].min():%Y-%m-%d}..{bars['event_time'].max():%Y-%m-%d}  hash={h[:12]}  {time.perf_counter()-t0:.1f}s")
    return bars


def returns_panel(store: BitemporalStore, as_of: dt.datetime, min_history: int = 1500,
                  min_coverage: float = 0.97, top_n: int = 200, min_date_coverage: float = 0.98, start=None):
    """Point-in-time panel of simple returns.
    Universe: symbols with >= min_coverage of all sessions, then the top_n by median traded value.
    Dates: sessions where >= min_date_coverage of the universe has a bar; the few remaining gaps
    are forward-filled (flat return) — documented, and rare by construction."""
    bars = store.as_of("bars", as_of, columns=["close", "volume"], event_start=start)
    n_sessions = bars["event_time"].n_unique()
    cov = bars.group_by("symbol").agg(pl.len().alias("n"), (pl.col("close") * pl.col("volume")).median().alias("tv"))
    keep = cov.filter((pl.col("n") >= max(min_history, int(min_coverage * n_sessions)))).sort("tv", descending=True)
    keep = keep.head(top_n)["symbol"].to_list()
    wide = (bars.filter(pl.col("symbol").is_in(keep))
                .pivot(index="event_time", on="symbol", values="close").sort("event_time"))
    syms = [c for c in wide.columns if c != "event_time"]
    frac = wide.select(pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in syms]) / len(syms))
    wide = wide.filter(frac.to_series() >= min_date_coverage)
    wide = wide.with_columns([pl.col(c).fill_null(strategy="forward") for c in syms]).drop_nulls()
    px = wide.select(syms).to_numpy()
    R = px[1:] / px[:-1] - 1
    dates = wide["event_time"].to_list()[1:]
    return R, syms, dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="synthetic", choices=["us", "crypto", "crypto_full", "nse", "synthetic", "us_stocks", "us_sectors"])
    ap.add_argument("--root", default=os.environ.get("EMPORION_ROOT", "runs"))
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--periods", type=int, default=None)
    ap.add_argument("--top-n", type=int, default=200, help="max assets, by median traded value")
    ap.add_argument("--reuse-store", action="store_true", help="skip download if the store already has bars")
    ap.add_argument("--start", default=None, help="panel start date (YYYY-MM-DD) for staggered-listing universes")
    args = ap.parse_args()
    periods = args.periods or (365 if args.market.startswith("crypto") else 252)
    root = os.path.join(args.root, args.market)
    store = BitemporalStore(os.path.join(root, "store"))
    ledger = Ledger(os.path.join(root, "ledger"))

    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    have = args.reuse_store and "bars" in store.tables() and store.as_of("bars", now, columns=["close"]).height > 0
    if have:
        print("[data] reusing stored bars (--reuse-store)")
    else:
        try:
            load_market(args.market, store)
        except Exception as e:  # noqa: BLE001
            print(f"[data] {args.market} unreachable ({e}); falling back to synthetic")
            load_market("synthetic", store)

    start = dt.datetime.fromisoformat(args.start) if args.start else None
    R, syms, dates = returns_panel(store, now, min_history=1000 if args.market != "synthetic" else 100, top_n=args.top_n, start=start)
    T, N = R.shape
    scope = f"p0-baselines-{args.market}-N{N}"      # DSR deflates within one (market, universe) family
    print(f"[panel] T={T} days x N={N} assets  {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}")

    # baselines -> net returns with costs, executed next bar
    nets, tos = {}, {}
    for name, fn in BASELINES.items():
        W = fn(R)
        net, to = portfolio_returns(W, R, cost_bps=args.cost_bps)
        nets[name], tos[name] = net, to
    M = np.stack([nets[n] for n in BASELINES], axis=1)
    pbo = pbo_cscv(M, n_blocks=16)["pbo"]
    print(f"[pbo] across {len(BASELINES)} baselines: PBO={pbo:.3f}")

    bench = nets["equal_weight"]
    reports = {}
    for name in BASELINES:
        fn = BASELINES[name]

        def strat(tr, te, fn=fn):
            # refit on train (rolling estimators only see their own past anyway), evaluate on test
            W = fn(R)
            net, _ = portfolio_returns(W, R, cost_bps=args.cost_bps)
            return net[te]

        rep = certify(name, T, strat, nets[name], ledger, scope,
                      benchmark_returns=None if name == "equal_weight" else bench,
                      periods=periods, pbo=pbo, B=2000,
                      extra={"cost_bps": args.cost_bps, "avg_turnover": float(tos[name].mean()), "N": N})
        reports[name] = rep
        c = rep["cpcv"]
        print(f"[certify] {name:14s} SR={rep['sharpe']:+.2f} ret={rep['ann_return']*100:+.1f}% mdd={rep['max_drawdown']*100:.1f}% "
              f"CI[{rep['bootstrap_sharpe_ci']['lo']:+.2f},{rep['bootstrap_sharpe_ci']['hi']:+.2f}] "
              f"DSR={rep['dsr']['dsr']:.3f}(N={rep['n_trials']}) CPCV p5/p50/p95={c['sharpe_p5']:+.2f}/{c['sharpe_p50']:+.2f}/{c['sharpe_p95']:+.2f} "
              f"-> {'CERTIFIED' if rep['certified'] else 'not certified'}")

    assert ledger.verify_chain(), "ledger chain broken"
    os.makedirs("benchmarks", exist_ok=True)
    stamp = now.strftime("%Y%m%d-%H%M")
    out = {"market": args.market, "T": T, "N": N, "symbols": syms, "cost_bps": args.cost_bps, "periods": periods,
           "pbo": pbo, "reports": reports, "ledger_rows": ledger.frame().height, "chain_ok": True}
    with open(os.path.join(root, f"p0-report-{stamp}.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    perf.save_report(os.path.join("benchmarks", f"p0-{args.market}-{stamp}.json"),
                     extra={"market": args.market, "T": T, "N": N})
    print("[perf]", json.dumps({k: {"p50_s": round(v["p50_s"], 4), "rows/s": (round(v["best_rows_per_s"]) if v["best_rows_per_s"] else None)}
                                for k, v in perf.report().items()}, indent=0))
    print(f"[done] ledger rows={ledger.frame().height} chain_ok=True  report={root}/p0-report-{stamp}.json")


if __name__ == "__main__":
    main()
