"""Block C — first real-data Emporion signal (rung 5, minimal form): the structural clearing model
with Agora's price response as the fundamentals anchor, on the US sector-ETF panel around tariff events.

For event e (date d) and post-event day t = d+1..d+H:
  belief anchor   mu_i(t) = Rf * p_i(d-1) * exp(k * dlogPX_i(e))        (Agora shifts expected payoff; k fitted)
  equilibrium     p_hat(t) = closed-form clearing price given (mu, Sigma from 250 pre-event days, supply ~ cap weight, default state)
  mispricing      r_mkt_i(t) = log(p_i(t) / p_hat_i(t))  -> signal = -r_mkt (buy under-priced vs the model)
Walk-forward: k is fitted by least squares on day-0 reactions of the first half of events only; the second
half is out of sample. Scored as RankIC of the signal at t=d+1 against the residual return over the next
h in {5,20} days, permutation null, ledger rows in scope p4-realdata-us-events.
"""
from __future__ import annotations

import argparse, datetime as dt, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from emporion.structure.agora_bridge import Economy, sector_response
from emporion.structure.events_us_tariffs import events
from emporion.structure.sector_map import ETF_EXPOSURE, BENCHMARK
from emporion.models.investors import default_state
from emporion.models.clearing import closed_form_price
from p3_event_harness import rank_ic

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs")
    ap.add_argument("--perms", type=int, default=2000)
    args = ap.parse_args()
    root = os.path.join(args.root, "us_sectors")
    store = BitemporalStore(os.path.join(root, "store")); ledger = Ledger(os.path.join(root, "ledger"))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    etfs = sorted(ETF_EXPOSURE); syms = etfs + [BENCHMARK]
    bars = store.as_of("bars", now, columns=["close"])
    wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time").drop_nulls()
    dates = [d.date() for d in wide["event_time"].to_list()]; P = wide.select(syms).to_numpy(); L = np.log(P)
    R = np.diff(L, axis=0); ib = syms.index(BENCHMARK); T = R.shape[0]
    di = {d: i for i, d in enumerate(dates)}
    ev = events(); ec = Economy("USA")
    resp = sector_response("USA", [e["shock"] for e in ev], [e["te"] for e in ev], econ=ec)
    ci = {c: i for i, c in enumerate(resp["codes"])}
    E = np.zeros((len(etfs), len(resp["codes"])))
    for j, etf in enumerate(etfs):
        for c, w in ETF_EXPOSURE[etf].items():
            E[j, ci[c]] = w
    dpx = resp["dlogPX"] @ E.T                                              # (n_ev, n_etf) Agora price response per ETF
    ok = [k for k, e in enumerate(ev) if dt.date.fromisoformat(e["date"]) in di and di[dt.date.fromisoformat(e["date"])] > 260
          and di[dt.date.fromisoformat(e["date"])] + 22 < len(dates)]
    # day-0 residual reaction per event (for fitting k on the first half)
    def resid_ret(i0, i1):
        w = R[i0 - 250:i0]; b = w[:, ib]; beta = ((w - w.mean(0)) * (b - b.mean())[:, None]).mean(0) / b.var()
        rr = R[i0:i1] - beta * R[i0:i1, [ib]]
        return rr.sum(0)[:len(etfs)]
    half = len(ok) // 2
    X = np.concatenate([dpx[k] for k in ok[:half]]); Y = np.concatenate([resid_ret(di[dt.date.fromisoformat(ev[k]["date"])], di[dt.date.fromisoformat(ev[k]["date"])] + 1) for k in ok[:half]])
    k_hat = float((X * Y).sum() / (X * X).sum())
    print(f"[fit] k (day-0 price pass-through of Agora dlogPX) on first {half} events = {k_hat:+.3f}")
    Rf = torch.tensor([1.0002])
    rows, res = [], {}
    for h in (5, 20):
        ics, ics_naive, kept = [], [], []
        for k in ok[half:]:
            d = di[dt.date.fromisoformat(ev[k]["date"])]
            t = d + 1                                                           # decide at close of d+1 (post-event day 1)
            w = R[t - 250:t]; Sig = np.cov(w[:, :len(etfs)].T) + 1e-6 * np.eye(len(etfs))
            p_pre = P[d - 1 + 1, :len(etfs)]                                     # close of d-1 (P index = date index; R[t] = L[t+1]-L[t])
            p_now = P[t + 1, :len(etfs)]
            mu = Rf.item() * torch.tensor(p_pre) * torch.exp(k_hat * torch.tensor(dpx[k]))
            mkt = {"mu": mu[None], "mom": torch.zeros(1, len(etfs)), "Sig_inv": torch.linalg.inv(torch.tensor(Sig))[None],
                   "s": torch.ones(1, len(etfs)), "pbar": torch.tensor(p_pre)[None], "Rf": Rf}
            st = default_state(1, gen=torch.Generator().manual_seed(0)); st.eps = torch.zeros(1)
            # calibrate the supply so that the pre-event price is an equilibrium under mu=Rf*p_pre (no premium term mismatch)
            p_hat_pre = closed_form_price({**mkt, "mu": Rf.item() * torch.tensor(p_pre)[None]}, st)[0]
            p_hat = closed_form_price(mkt, st)[0] * (torch.tensor(p_pre) / p_hat_pre)      # equilibrium shifted by the belief change
            signal = -(np.log(p_now) - np.log(p_hat.numpy()))                               # buy what is below the model's equilibrium
            naive = -(np.log(p_now) - np.log(p_pre))                                          # pure reversal of the event move
            real = resid_ret(t + 1, t + 1 + h)
            ics.append(rank_ic(signal, real)); ics_naive.append(rank_ic(naive, real)); kept.append(k)
            rows.append({"event": ev[k]["name"][:60], "date": ev[k]["date"], "h": h, "rank_ic": ics[-1], "naive_reversal_ic": ics_naive[-1]})
        ics = np.array(ics); n = len(ics)
        rng = np.random.default_rng(0); null = np.empty(args.perms)
        sig_real = []
        for k in kept:
            d = di[dt.date.fromisoformat(ev[k]["date"])]; t = d + 1
            sig_real.append((rows[[r["date"] for r in rows if r["h"] == h].index(ev[k]["date"])] if False else None, resid_ret(t + 1, t + 1 + h)))
        # permutation null on the realised vectors with random signal permutations
        reals = [resid_ret(di[dt.date.fromisoformat(ev[k]["date"])] + 2, di[dt.date.fromisoformat(ev[k]["date"])] + 2 + h) for k in kept]
        for b in range(args.perms):
            null[b] = np.mean([rank_ic(rng.permutation(len(etfs)).astype(float), rr) for rr in reals])
        res[f"h{h}"] = {"n_events_oos": n, "mean_rank_ic": float(ics.mean()), "t": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(n))),
                        "perm_p": float((null >= ics.mean()).mean()), "hit": float((ics > 0).mean()),
                        "naive_reversal_ic": float(np.mean(ics_naive)), "k_hat": k_hat}
        r = res[f"h{h}"]
        print(f"[oos] h={h:2d} n={n}  Emporion signal IC={r['mean_rank_ic']:+.3f} t={r['t']:+.2f} perm-p={r['perm_p']:.3f} hit={r['hit']:.2f} | naive reversal IC={r['naive_reversal_ic']:+.3f}")
        ledger.append("trial", f"p4_realdata_us_events:h{h}", {"sharpe_is": r["t"], **r}, scope="p4-realdata-us-events")
    with open(os.path.join(root, f"p4-realdata-{now:%Y%m%d-%H%M}.json"), "w") as f:
        json.dump({"k_hat": k_hat, "results": res, "per_event": rows}, f, indent=1)
    assert ledger.verify_chain(); print("[done]")


if __name__ == "__main__":
    main()
