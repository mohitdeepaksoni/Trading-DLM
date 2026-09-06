"""P1: volatility null models through the ledger — the bar for any learned vol forecaster.
    python scripts/p1_vol_nulls.py --market us
Evaluates next-day variance forecasts (EWMA, HAR level form, GARCH(1,1)-t) on the market panel,
QLIKE and MSE(log var) vs the r^2 proxy, walk-forward by construction, plus a constant-variance null."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core import perf                                     # noqa: E402
from emporion.core.ledger import Ledger                             # noqa: E402
from emporion.data.store import BitemporalStore                     # noqa: E402
from emporion.baselines.vol import VOL_BASELINES, vol_forecast_eval  # noqa: E402
from p0_certify_baselines import returns_panel                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="us")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--start", default=None)
    args = ap.parse_args()
    root = os.path.join(args.root, args.market)
    store = BitemporalStore(os.path.join(root, "store"))
    ledger = Ledger(os.path.join(root, "ledger"))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    R, syms, dates = returns_panel(store, now, min_history=1000 if args.market != "synthetic" else 100, top_n=args.top_n, start=(dt.datetime.fromisoformat(args.start) if args.start else None))
    T, N = R.shape
    scope = f"p1-volnulls-{args.market}-N{N}"
    print(f"[panel] {args.market}: T={T} N={N} {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}")
    # evaluation window: after every model has warmed up (GARCH needs 250, HAR 82)
    lo = 300
    res = {}
    const = np.full_like(R, R[:lo].var())
    res["constant"] = vol_forecast_eval(const[lo:], R[lo:])
    for name, fn in VOL_BASELINES.items():
        t0 = time.perf_counter()
        h = fn(R)
        res[name] = {**vol_forecast_eval(h[lo:], R[lo:]), "wall_s": time.perf_counter() - t0}
    best = min(res, key=lambda k: res[k]["qlike"])
    for name, r in res.items():
        ledger.append("trial", f"volnull:{name}", {"sharpe_is": -r["qlike"], **r}, scope=scope)
        print(f"[vol] {name:10s} QLIKE={r['qlike']:.4f} MSE(logvar)={r['mse_logvar']:.3f} cov={r['coverage']:.2f}"
              f"{'  <- best' if name == best else ''}")
    stamp = now.strftime("%Y%m%d-%H%M")
    with open(os.path.join(root, f"p1-volnulls-{stamp}.json"), "w") as f:
        json.dump({"market": args.market, "T": T, "N": N, "eval_from": lo, "results": res, "best": best}, f, indent=1)
    perf.save_report(os.path.join("benchmarks", f"p1-volnulls-{args.market}-{stamp}.json"), extra={"market": args.market, "T": T, "N": N})
    assert ledger.verify_chain()
    print(f"[done] best={best} ledger rows={ledger.frame().height}")


if __name__ == "__main__":
    main()
