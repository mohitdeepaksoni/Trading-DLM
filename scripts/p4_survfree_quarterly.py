"""Measure survivorship bias directly (item 3): quarterly price-based rungs on the survivorship-free 13F panel
(all CUSIPs) versus the same rungs restricted to names still alive at the last quarter.  Universe at each quarter =
point-in-time top-K by number of 13F holders (a liquidity/size proxy that needs no look-ahead).  Signals from the
quarterly price series: momentum (4Q return skipping the last), short-term reversal (last quarter), low-risk
(negative std of the last 8 quarterly returns).  Scored by next-quarter RankIC; both panels on the ledger."""
import argparse, datetime as dt, json, os, sys
import numpy as np, polars as pl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from p3_event_harness import rank_ic

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=1000); args = ap.parse_args()
store = BitemporalStore("runs/us_survfree/store"); ledger = Ledger("runs/us_survfree/ledger"); now = dt.datetime.now()
q = store.as_of("qprices", now)
w = q.pivot(index="event_time", on="cusip", values="price").sort("event_time"); nq = q.pivot(index="event_time", on="cusip", values="n_managers").sort("event_time")
cus = [c for c in w.columns if c != "event_time"]; qs = w["event_time"].to_list()
P = w.select(cus).to_numpy().astype(float); NM = np.nan_to_num(nq.select(cus).to_numpy().astype(float))
T, N = P.shape; L = np.log(P)
alive_last = np.isfinite(P[-1])
print(f"[panel] {N:,} cusips x {T} quarters {qs[0]:%Y-%m-%d}..{qs[-1]:%Y-%m-%d}; alive at last quarter {int(alive_last.sum()):,}")
def run(label, allowed):
    res = {}
    for name in ("momentum", "st_reversal", "low_risk"):
        ics = []
        for t in range(9, T - 1):
            hist = np.isfinite(P[t - 8:t + 1]).all(0) & allowed
            U = np.zeros(N, bool); cand = np.where(hist, NM[t], -1); top = np.argsort(-cand)[:args.k]; U[top] = cand[top] > 0
            if U.sum() < 100:
                continue
            if name == "momentum": s = L[t - 1] - L[t - 4]
            elif name == "st_reversal": s = -(L[t] - L[t - 1])
            else: s = -np.diff(L[t - 8:t + 1], axis=0).std(0)
            s = np.where(U, s, np.nan); real = L[t + 1] - L[t]; ok = np.isfinite(s) & np.isfinite(real)
            real = np.where(ok, real - np.nanmean(np.where(ok, real, np.nan)), np.nan)
            ics.append(rank_ic(np.where(ok, s, np.nan), real))
        ics = np.array([x for x in ics if np.isfinite(x)])
        res[name] = {"mean_ic": float(ics.mean()), "t": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), "n": int(len(ics))}
        print(f"[{label:12s}] {name:12s} next-quarter RankIC {res[name]['mean_ic']:+.4f} (t {res[name]['t']:+.2f}, n={res[name]['n']})")
        ledger.append("trial", f"survfree:{label}:{name}", {"sharpe_is": res[name]["t"], **res[name]}, scope=f"survivorship-{label}")
    return res
full = run("all_cusips", np.ones(N, bool)); surv = run("survivors_only", alive_last)
delta = {k: surv[k]["mean_ic"] - full[k]["mean_ic"] for k in full}
print("[bias] survivors-only minus survivorship-free RankIC:", {k: round(v, 4) for k, v in delta.items()})
with open(f"runs/us_survfree/survivorship-{now:%Y%m%d-%H%M}.json", "w") as f:
    json.dump({"full": full, "survivors": surv, "delta": delta, "N": N, "T": T}, f, indent=1)
assert ledger.verify_chain(); print("[done]")
