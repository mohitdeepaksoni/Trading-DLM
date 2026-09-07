"""P3c on real data — does a nowcast of per-stock institutional share between 13F filings beat carry-forward?
obs: institutional share (13F inst_shares / shares outstanding) placed at each quarter end; flows: (a) Z.1 aggregate
mandated-share drift interpolated within the quarter, (b) the stock's trailing 20-day return (demand chasing), (c) none.
Validation = `holdout_error`: hide every 4th filing per stock, MAE of the filter vs last-value carry-forward at the hidden
dates. A validation, not a trial: ledgered as a note. If the filter does not beat carry-forward, the per-stock state
between filings carries no information and P3c closes without a rung.
    python scripts/p3c_nowcast_validate.py --root runs/us_stocks_wide --top 500
"""
import argparse, datetime as dt, glob, json, os, sys
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from emporion.data.loaders.fed_z1 import mandated_share
from emporion.structure.nowcaster import holdout_error, fit_flow_loading
from p4_realdata_us_stocks import shares_asof

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs/us_stocks_wide"); ap.add_argument("--top", type=int, default=500); ap.add_argument("--start", default="2013-01-01")
args = ap.parse_args(); now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
store = BitemporalStore(os.path.join(args.root, "store")); ledger = Ledger(os.path.join(args.root, "ledger"))
bars = store.as_of("bars", now, columns=["close", "volume"], event_start=dt.datetime.fromisoformat(args.start))
wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time"); syms = [c for c in wide.columns if c != "event_time"]
wide = wide.with_columns([pl.col(c).fill_null(strategy="forward") for c in syms]); dates = wide["event_time"].to_list(); P = wide.select(syms).to_numpy().astype(float)
dv = np.nan_to_num(P) * np.nan_to_num(bars.pivot(index="event_time", on="symbol", values="volume").sort("event_time").select(syms).to_numpy().astype(float))
med = np.nanmedian(dv[-250:], axis=0); keep = np.argsort(-np.nan_to_num(med, nan=-1))[:args.top]; syms = [syms[i] for i in keep]; P = P[:, keep]; T, N = P.shape; si = {s: i for i, s in enumerate(syms)}
facts = store.all_versions("facts"); inst = store.all_versions("inst_holdings").filter(pl.col("symbol").is_in(syms))
# SEC's quarterly 13F structured datasets also carry LATE/AMENDED filings for earlier quarters as small partial batches;
# "latest revision wins" would replace a full aggregate (thousands of holders) with a late batch (research/03 §15).
# Rule: per (stock, quarter) keep the fullest known batch = max n_holders.
inst = inst.sort("n_holders").group_by(["symbol", "event_time"]).last()
qends = sorted(inst["event_time"].unique().to_list()); qends = [q for q in qends if q >= dates[0]]
di = {d.date(): i for i, d in enumerate(dates)}
obs = np.full((T, N), np.nan)
for q in qends:
    # place the observation at the first trading day on/after the quarter end
    cand = [i for d, i in di.items() if d >= q.date()]
    if not cand: continue
    t = min(cand); sh = shares_asof(facts, q + dt.timedelta(days=120), syms)                                   # shares as known ~after the quarter's 10-Q
    sub = inst.filter(pl.col("event_time") == q)
    for r in sub.iter_rows(named=True):
        j = si.get(r["symbol"])
        if j is not None and np.isfinite(sh[j]) and sh[j] > 0 and r["inst_shares"] is not None:
            obs[t, j] = min(r["inst_shares"] / sh[j], 1.5)
n_obs = int(np.isfinite(obs).sum()); print(f"[panel] {N} stocks x {T} days {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}; 13F observations {n_obs:,} over {len(qends)} quarter ends; median inst share {np.nanmedian(obs):.2f}")
# flows (a): Z.1 mandated-share drift, linearly interpolated within each quarter (uses the quarter's own value: an upper bound on what the aggregate could add)
z1 = mandated_share(store.as_of("z1", now)).select(["event_time", "phi_mandated"]).drop_nulls().sort("event_time")
zt = [d.date() for d in z1["event_time"].to_list()]; zv = z1["phi_mandated"].to_numpy()
zi = np.interp([d.date().toordinal() for d in dates], [d.toordinal() for d in zt], zv); fz = np.diff(zi, prepend=zi[0]); F_z1 = np.repeat(fz[:, None], N, axis=1)
# flows (b): trailing 20-day return per stock
R20 = np.full((T, N), 0.0); R20[20:] = np.nan_to_num(P[20:] / P[:-20] - 1.0)
res = {}
for name, F in (("none", np.zeros((T, N))), ("z1_drift", F_z1), ("trailing_20d_return", R20)):
    r = holdout_error(obs, F, every=4); res[name] = r
    print(f"[nowcast] flows={name:20s} b={r['b']:+.3g}  MAE filter {r['mae_filter']:.4f} vs last-value {r['mae_last_value']:.4f}  (hidden filings {r['n_hidden']:,})  -> {'BEATS' if r['mae_filter'] < r['mae_last_value'] * 0.99 else 'no better than'} carry-forward")
# scale: typical quarter-to-quarter change in the state
chg = []
for j in range(N):
    idx = np.flatnonzero(np.isfinite(obs[:, j])); chg += list(np.abs(np.diff(obs[idx, j])))
print(f"[scale] median |quarter-to-quarter change| in inst share = {np.median(chg):.4f}; mean {np.mean(chg):.4f}")
res["scale"] = {"median_abs_qoq_change": float(np.median(chg)), "mean_abs_qoq_change": float(np.mean(chg)), "n_obs": n_obs, "n_stocks": N}
ledger.append("note", "p3c_nowcast_validation_13f", {k: v for k, v in res.items() if k != "scale"} | res["scale"], scope="p3c-nowcast")
with open(os.path.join(args.root, f"p3c-nowcast-validate-{now:%Y%m%d-%H%M}.json"), "w") as f:
    json.dump(res, f, indent=1)
print("[done]")
