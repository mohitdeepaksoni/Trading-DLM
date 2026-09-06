"""P2 encoder decision: does Kronos (zero-shot) rank next-h-day returns better than the
classical signals, on OUR panels, out of sample, with the ledger counting the trial?

    python scripts/p2_kronos_rankic.py --market us --model NeoQuasar/Kronos-small --horizon 5 --n-dates 60

Protocol: walk-forward decision dates (every `step` sessions over the last `n_dates*step`
sessions).  At each date d, every asset gets the last `lookback` bars (OHLCV+amount, all
known at d); Kronos predicts `horizon` bars; the signal is predicted close[h]/close[d]-1.
Realised = close[d+h]/close[d]-1.  RankIC = Spearman across assets per date; we report the
mean, its t-stat (Newey-West-free: dates are `step` apart >= horizon so non-overlapping),
and the same numbers for mom_12_1, rev_5, vol-scaled reversal, and a random control.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core import perf                       # noqa: E402
from emporion.core.ledger import Ledger               # noqa: E402
from emporion.data.store import BitemporalStore       # noqa: E402


def ohlcv_panel(store, as_of, top_n=200, min_coverage=0.97, min_date_coverage=0.98):
    bars = store.as_of("bars", as_of, columns=["open", "high", "low", "close", "volume"])
    n_sessions = bars["event_time"].n_unique()
    cov = bars.group_by("symbol").agg(pl.len().alias("n"), (pl.col("close") * pl.col("volume")).median().alias("tv"))
    keep = cov.filter(pl.col("n") >= int(min_coverage * n_sessions)).sort("tv", descending=True).head(top_n)["symbol"].to_list()
    bars = bars.filter(pl.col("symbol").is_in(keep))
    out = {}
    for c in ("open", "high", "low", "close", "volume"):
        w = bars.pivot(index="event_time", on="symbol", values=c).sort("event_time")
        out[c] = w
    syms = [c for c in out["close"].columns if c != "event_time"]
    frac = out["close"].select(pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in syms]) / len(syms)).to_series()
    mask = frac >= min_date_coverage
    arrs = {}
    for c, w in out.items():
        w = w.filter(mask).with_columns([pl.col(s).fill_null(strategy="forward") for s in syms])
        arrs[c] = w.select(syms).to_numpy().astype(float)
    dates = out["close"].filter(mask)["event_time"].to_list()
    # volume may be legitimately missing/zero on some venues: fill with 0; prices must be finite everywhere
    arrs["volume"] = np.nan_to_num(arrs["volume"], nan=0.0)
    ok = np.ones(arrs["close"].shape[0], bool)
    for c in ("open", "high", "low", "close"):
        ok &= np.isfinite(arrs[c]).all(axis=1) & (arrs[c] > 0).all(axis=1)
    for c in arrs:
        arrs[c] = arrs[c][ok]
    dates = [d for d, o in zip(dates, ok) if o]
    return arrs, syms, dates


def rank_ic(pred: np.ndarray, real: np.ndarray) -> float:
    ok = np.isfinite(pred) & np.isfinite(real)
    if ok.sum() < 5:
        return np.nan
    a = pd.Series(pred[ok]).rank().to_numpy(); b = pd.Series(real[ok]).rank().to_numpy()
    a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std()
    return float((a * b).mean())


def summarize(ics):
    ics = np.array([x for x in ics if np.isfinite(x)])
    return {"mean_ic": float(ics.mean()), "t_stat": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))),
            "hit": float((ics > 0).mean()), "n_dates": int(len(ics))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="us")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--model", default="NeoQuasar/Kronos-small")
    ap.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--kronos-path", default=os.path.expanduser("~/Kronos"))
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--n-dates", type=int, default=60)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--lookback", type=int, default=400)
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, args.kronos_path)
    import torch
    from model import Kronos, KronosTokenizer, KronosPredictor  # noqa: E402
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    root = os.path.join(args.root, args.market)
    store = BitemporalStore(os.path.join(root, "store"))
    ledger = Ledger(os.path.join(root, "ledger"))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    arrs, syms, dates = ohlcv_panel(store, now, top_n=args.top_n)
    C = arrs["close"]; T, N = C.shape
    h = args.horizon
    last = T - 1 - h
    dec = list(range(last - args.step * (args.n_dates - 1), last + 1, args.step))
    dec = [d for d in dec if d >= args.lookback + 260]
    print(f"[panel] {args.market}: T={T} N={N} {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}; {len(dec)} decision dates, horizon={h}")

    tok = KronosTokenizer.from_pretrained(args.tokenizer)
    mdl = Kronos.from_pretrained(args.model)
    pred = KronosPredictor(mdl, tok, device="cuda:0" if torch.cuda.is_available() else "cpu", max_context=512)
    n_params = sum(p.numel() for p in mdl.parameters())
    print(f"[model] {args.model} params={n_params/1e6:.1f}M device={pred.device}")

    realised = C[np.array(dec) + h] / C[dec] - 1                                  # (D,N)
    logC = np.log(C)
    sig = {
        "mom_12_1": np.array([logC[d - 21] - logC[d - 252] for d in dec]),
        "rev_5": np.array([-(logC[d] - logC[d - 5]) for d in dec]),
        "rev_1": np.array([-(logC[d] - logC[d - 1]) for d in dec]),
        "random": np.random.default_rng(args.seed).normal(size=(len(dec), N)),
    }
    kron = np.full((len(dec), N), np.nan)
    t0 = time.perf_counter()
    for di, d in enumerate(dec):
        x_ts = pd.to_datetime(pd.Series(dates[d - args.lookback + 1:d + 1]))
        y_ts = pd.to_datetime(pd.Series(pd.bdate_range(dates[d] + dt.timedelta(days=1), periods=h)))
        dfs, xts, yts = [], [], []
        for j in range(N):
            df = pd.DataFrame({c: arrs[c][d - args.lookback + 1:d + 1, j] for c in ("open", "high", "low", "close", "volume")})
            df["amount"] = df["volume"] * df["close"]
            dfs.append(df); xts.append(x_ts); yts.append(y_ts)
        outs = []
        for b0 in range(0, N, args.batch):
            outs += pred.predict_batch(dfs[b0:b0 + args.batch], xts[b0:b0 + args.batch], yts[b0:b0 + args.batch],
                                       pred_len=h, T=1.0, top_p=0.9, sample_count=args.samples, verbose=False)
        kron[di] = np.array([o["close"].iloc[-1] for o in outs]) / C[d] - 1
        if (di + 1) % 10 == 0 or di + 1 == len(dec):
            el = time.perf_counter() - t0
            print(f"[kronos] {di+1}/{len(dec)} dates  {el:.0f}s  ({el/(di+1)/N*1000:.1f} ms/asset/date)", flush=True)
    sig["kronos"] = kron

    results = {}
    for name, S in sig.items():
        ics = [rank_ic(S[i], realised[i]) for i in range(len(dec))]
        results[name] = summarize(ics)
        print(f"[rankic] {name:10s} mean={results[name]['mean_ic']:+.4f} t={results[name]['t_stat']:+.2f} hit={results[name]['hit']:.2f} n={results[name]['n_dates']}")
    scope = f"p2-encoder-{args.market}-N{N}-h{h}"
    for name, r in results.items():
        ledger.append("trial", f"rankic:{name}", {"sharpe_is": r["t_stat"] / np.sqrt(len(dec)) * np.sqrt(252 / args.step), **r,
                                                  "model": args.model if name == "kronos" else None, "horizon": h}, scope=scope)
    rep = {"market": args.market, "model": args.model, "params_M": n_params / 1e6, "T": T, "N": N, "horizon": h,
           "n_dates": len(dec), "lookback": args.lookback, "samples": args.samples, "results": results,
           "kronos_wall_s": time.perf_counter() - t0}
    stamp = now.strftime("%Y%m%d-%H%M")
    with open(os.path.join(root, f"p2-kronos-{args.model.split('/')[-1]}-h{h}-{stamp}.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    perf.save_report(os.path.join("benchmarks", f"p2-kronos-{args.market}-{stamp}.json"), extra=rep)
    assert ledger.verify_chain()
    print(f"[done] ledger rows={ledger.frame().height}")


if __name__ == "__main__":
    main()
