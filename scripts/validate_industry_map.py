"""Map validation (not a trial): synthetic ETF returns = ETF_EXPOSURE @ (stock-built ICIO sector portfolio returns).
If the industry->ICIO map is right, real ETF day-0 residual returns should rank-correlate with the synthetic ones on
tariff event days (and on ordinary days). Prints per-event rank correlation and the all-days average."""
import datetime as dt, json, glob, os, sys
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.data.store import BitemporalStore
from emporion.structure.industry_map import exposure_matrix, ICIO_CODES
from emporion.structure.sector_map import ETF_EXPOSURE, BENCHMARK
from scipy.stats import spearmanr

now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
st = BitemporalStore("runs/us_stocks_wide/store"); se = BitemporalStore("runs/us_sectors/store")
uni = st.as_of("universe", now, columns=["sector", "industry"]).group_by("symbol").last()
bars = st.as_of("bars", now, columns=["close"], event_start=dt.datetime(2024, 1, 1))
ws = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time")
syms = [c for c in ws.columns if c != "event_time"]; P = ws.select(syms).to_numpy().astype(float); dS = ws["event_time"].to_list()
RS = np.diff(np.log(P), axis=0); dS = dS[1:]
umap = {r["symbol"]: (r["industry"], r["sector"]) for r in uni.iter_rows(named=True)}
E = exposure_matrix([umap.get(s, ("", ""))[0] for s in syms], [umap.get(s, ("", ""))[1] for s in syms], ICIO_CODES)
eb = se.as_of("bars", now, columns=["close"], event_start=dt.datetime(2024, 1, 1))
etfs = sorted(ETF_EXPOSURE); we = eb.pivot(index="event_time", on="symbol", values="close").sort("event_time")
PE = we.select(etfs + [BENCHMARK]).to_numpy().astype(float); dE = we["event_time"].to_list()
RE = np.diff(np.log(PE), axis=0); dE = dE[1:]
common = sorted(set(dS) & set(dE)); iS = {d: i for i, d in enumerate(dS)}; iE = {d: i for i, d in enumerate(dE)}
X = np.zeros((len(etfs), len(ICIO_CODES)))
ci = {c: i for i, c in enumerate(ICIO_CODES)}
for j, e in enumerate(etfs):
    for c, w in ETF_EXPOSURE[e].items(): X[j, ci[c]] = w
ev_dates = {dt.date.fromisoformat(d) for d in json.load(open(sorted(glob.glob("runs/us_sectors/p3-event-harness-dlogPX-*.json"))[-1]))["sector_response"]}
rhos, rhos_ev = [], []
for d in common:
    rs = RS[iS[d]]; ok = np.isfinite(rs); mkt = np.nanmean(rs[ok])
    Wt = E[ok].T; cnt = (Wt > 0).sum(1); port = np.where(cnt >= 5, (Wt @ (rs[ok] - mkt)) / np.maximum(Wt.sum(1), 1e-9), np.nan)
    synth = np.array([np.nansum(X[j] * port) / max(np.nansum(X[j] * np.isfinite(port)), 1e-9) for j in range(len(etfs))])
    re = RE[iE[d]][:len(etfs)] - RE[iE[d]][len(etfs)]                                   # ETF minus SPY
    rho = spearmanr(synth, re).correlation; rhos.append(rho)
    if d.date() in ev_dates: rhos_ev.append((d.date().isoformat(), round(float(rho), 3))); 
print(f"[map] all days n={len(rhos)} mean rank-corr(synthetic ETF from stocks, real ETF) = {np.nanmean(rhos):+.3f}  (median {np.nanmedian(rhos):+.3f}; share>0.3: {np.mean(np.array(rhos)>0.3):.2f})")
print(f"[map] event days: {rhos_ev}")
print(f"[map] event-day mean = {np.mean([r for _, r in rhos_ev]):+.3f}")
