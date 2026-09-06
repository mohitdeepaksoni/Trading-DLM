"""P4 v1 channel 4 — POOLED training of Emporion v0 across markets (US large caps + NSE top-200), one law, one net.
Features common to both markets: [fund anchor, SUE, momentum, low-risk, log price, is_US, is_NSE]; supply = cap weights;
kappa from the panel estimator per date; Rf per market. CPCV (6 groups x 2 test, purged by horizon) over all dates in
time order; out-of-sample rank-IC of -r_mkt reported PER MARKET and pooled, against the value anchor alone.
    python scripts/p4_train_pooled.py --markets us,nse --h 20 --every 21 --steps 600
    python scripts/p4_train_pooled.py --markets us        (same code path, single market, for the comparison)
"""
import argparse, datetime as dt, json, os, sys
import numpy as np, polars as pl, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from emporion.models.emporion_v0 import EmporionV0
from emporion.solvers.residual_trainer import train_residual
from emporion.anchors import price_themes
from emporion.evaluate.cpcv import cpcv_splits
from p3_event_harness import rank_ic
from p4_realdata_us_stocks import ttm_eps_asof, shares_asof, sue_asof
from p4_realdata_nse import last_known_t, last_known, split_adjust

torch.set_default_dtype(torch.float64)
MARKET_ID = {"us": 0, "nse": 1}


def zs(x):
    m, s = np.nanmean(x), np.nanstd(x); return np.where(np.isfinite(x), (x - m) / (s + 1e-12), 0.0)


def _panel(store, now, start=None):
    bars = store.as_of("bars", now, columns=["close", "volume"], event_start=start)
    wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time"); syms = [c for c in wide.columns if c != "event_time"]
    wide = wide.with_columns([pl.col(c).fill_null(strategy="forward") for c in syms]); dates = wide["event_time"].to_list()
    P = wide.select(syms).to_numpy().astype(float)
    V = np.nan_to_num(bars.pivot(index="event_time", on="symbol", values="volume").sort("event_time").select(syms).to_numpy().astype(float))
    return syms, dates, P, V


def _sample(market, t, asof, P, V, listed, R, dv, ok_fund, eps, sue, sh, h, Rf, K, every):
    N = P.shape[1]
    hist = listed[max(0, t - 300):t + 1].all(0); med = np.median(dv[t - 250:t + 1], axis=0); med[~hist] = -1
    U = np.zeros(N, bool); top = np.argsort(-med)[:K]; U[top] = med[top] > 0
    p = P[t]; ok = U & ok_fund & np.isfinite(sh) & (sh > 0) & np.isfinite(p)
    if ok.sum() < 40:
        return None
    ey = eps[ok] / p[ok]; p_fund = eps[ok] / np.median(ey)
    Pu = np.where(U[None, :] & listed, P, np.nan); th = price_themes(Pu, V, t)
    fund = np.clip(np.log(p_fund / p[ok]), -1.0, 1.0) + h * np.log(Rf)
    feats = np.stack([fund, zs(sue[ok]), zs(th["momentum"][ok]), zs(th["low_risk"][ok]), zs(np.log(p[ok])),
                      np.full(ok.sum(), float(market == "us")), np.full(ok.sum(), float(market == "nse"))], axis=1)
    w = R[t - 250:t][:, ok]; Sig = np.cov(w.T) * h + 1e-6 * np.eye(ok.sum())
    cap = sh[ok] * p[ok]; s = cap / cap.sum()
    fwd = P[t - 250 + h:t + 1][:, ok] / P[t - 250:t + 1 - h][:, ok]; prem = fwd.mean(0) - Rf ** h
    x = Sig @ s; kappa = float(np.clip((prem * x).sum() / (x * x).sum(), 1e-4, 50.0))
    return {"market": market, "t": t, "date": asof, "feats": feats, "p_obs": np.ones(ok.sum()), "D": P[t + h][ok] / p[ok], "Sig": Sig, "s": s,
            "kappa": kappa, "idx": np.flatnonzero(ok), "Rf": Rf ** h, "P": P}


def build_us(root, h, every, K=200):
    store = BitemporalStore(os.path.join(root, "store")); now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    syms, dates, P, V = _panel(store, now); facts = store.all_versions("facts")
    listed = np.isfinite(P); T, N = P.shape; dv = np.nan_to_num(P) * V
    R = np.where(listed[1:] & listed[:-1], P[1:] / P[:-1] - 1, 0.0); Rf = 1.0 + 0.04 / 252
    out = []
    for t in range(300, T - h - 1, every):
        asof = dates[t]; eps = ttm_eps_asof(facts, asof, syms); sh = shares_asof(facts, asof, syms); sue = sue_asof(facts, asof, syms)
        s = _sample("us", t, asof, P, V, listed, R, dv, np.isfinite(eps) & (eps > 0), eps, np.nan_to_num(sue), sh, h, Rf, K, every)
        if s: out.append(s)
    return out, P


def build_nse(root, h, every, K=200):
    store = BitemporalStore(os.path.join(root, "store")); now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    syms, dates, P, V = _panel(store, now)
    res = store.all_versions("results").sort(["symbol", "event_time", "consolidated", "knowledge_time"])
    res_c = res.filter(pl.col("consolidated")); cons = set(res_c["symbol"].to_list())
    res = pl.concat([res_c, res.filter(~pl.col("consolidated") & ~pl.col("symbol").is_in(list(cons)))])
    sh_df = store.all_versions("shareholding")
    listed = np.isfinite(P); T, N = P.shape; dv = np.nan_to_num(P) * V
    R = np.where(listed[1:] & listed[:-1], P[1:] / P[:-1] - 1, 0.0); Rf = 1.0 + 0.065 / 252
    out = []
    for t in range(300, T - h - 1, every):
        asof = dates[t]
        shs = last_known_t(sh_df, asof, "shares_total", syms, n=40, days=100000, min_n=1)
        e4 = [split_adjust(a, b) for a, b in zip(last_known_t(res, asof, "eps_diluted", syms, n=4), shs)]
        e12 = [split_adjust(a, b) for a, b in zip(last_known_t(res, asof, "eps_diluted", syms, n=12, days=400, min_n=8), shs)]
        eps = np.array([sum(v) if v else np.nan for v in e4])
        sue = np.full(N, np.nan)
        for i, v in enumerate(e12):
            if v and len(v) >= 8:
                d = np.array(v[4:]) - np.array(v[:-4]); sd = d[:-1].std(ddof=1)
                if sd > 1e-9: sue[i] = np.clip(d[-1] / sd, -5, 5)
        sh = np.array([v[0][-1] if v else np.nan for v in shs])
        s = _sample("nse", t, asof, P, V, listed, R, dv, np.isfinite(eps) & (eps > 0), eps, np.nan_to_num(sue), sh, h, Rf, K, every)
        if s: out.append(s)
    return out, P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--markets", default="us,nse"); ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--every", type=int, default=21); ap.add_argument("--steps", type=int, default=600); ap.add_argument("--w-pen", type=float, default=1.0)
    ap.add_argument("--w-mu", type=float, default=1.0); ap.add_argument("--us-root", default="runs/us_stocks_wide"); ap.add_argument("--nse-root", default="runs/nse")
    ap.add_argument("--out-root", default="runs/p4_pooled"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--d-model", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pretrain-steps", type=int, default=0, help="synthetic pre-training steps on sample_panels before each split's real fine-tune")
    args = ap.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device); os.makedirs(args.out_root, exist_ok=True); ledger = Ledger(os.path.join(args.out_root, "ledger"))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None); h = args.h
    markets = args.markets.split(","); samples = []
    for m in markets:
        s, _ = (build_us(args.us_root, h, args.every) if m == "us" else build_nse(args.nse_root, h, args.every)); samples += s
        print(f"[panel] {m}: {len(s)} dates, {np.mean([len(x['idx']) for x in s]):.0f} assets/date, kappa median {np.median([x['kappa'] for x in s]):.3g}", flush=True)
    samples.sort(key=lambda x: x["date"])
    def pack(sub):
        n = max(len(x["idx"]) for x in sub); B = len(sub); F = sub[0]["feats"].shape[1]
        feats = torch.zeros(B, n, F); p = torch.ones(B, n); D = torch.ones(B, n); Sig_inv = torch.eye(n).repeat(B, 1, 1); s = torch.zeros(B, n)
        mask = torch.ones(B, n, dtype=torch.bool); kap = torch.zeros(B); Rf = torch.zeros(B)
        for b, x in enumerate(sub):
            m = len(x["idx"]); feats[b, :m] = torch.tensor(x["feats"]); p[b, :m] = torch.tensor(x["p_obs"]); D[b, :m] = torch.tensor(x["D"])
            Sig_inv[b, :m, :m] = torch.linalg.inv(torch.tensor(x["Sig"])); s[b, :m] = torch.tensor(x["s"]); mask[b, :m] = False; kap[b] = x["kappa"]; Rf[b] = x["Rf"]
        mkt = {"mom": torch.zeros(B, n), "Sig_inv": Sig_inv, "s": s, "pbar": p.clone(), "Rf": Rf, "kappa_in": kap, "phi_obs": torch.full((B,), 0.44)}
        G = torch.cat([torch.ones(B, n, 1), feats[..., 1:4]], dim=-1)
        return {"feats": feats.to(dev), "p_obs": p.to(dev), "D_real": D.to(dev), "G": G.to(dev), "mkt": {k: v.to(dev) for k, v in mkt.items()}, "mask": mask.to(dev)}
    def synthetic_pool(M=64, T=6, N=40, seed=0):
        """sample_panels adapted to the pooled feature layout [fund, SUE=0, momentum=0, low-risk, log p, is_US=0, is_NSE=0];
        kappa_in = true structural kappa, phi_obs = true phi (pre-training only; real data uses the panel estimator)."""
        from emporion.solvers import sample_panels
        g = torch.Generator().manual_seed(seed); Pn = sample_panels(M=M, T=T, N=N, gen=g); tr = Pn["truth"]
        f = Pn["feats"]; z = lambda x: (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-12)
        feats = torch.stack([f[..., 0], torch.zeros_like(f[..., 0]), torch.zeros_like(f[..., 0]), z(-f[..., 2]), z(f[..., 4]),
                             torch.zeros_like(f[..., 0]), torch.zeros_like(f[..., 0])], dim=-1)
        G = torch.cat([torch.ones(feats.shape[0], N, 1), feats[..., 1:4]], dim=-1)
        mkt = dict(Pn["mkt"]); mkt["kappa_in"] = tr["kappa_struct"].repeat_interleave(T); mkt["phi_obs"] = tr["phi"].repeat_interleave(T)
        return {"feats": feats.to(dev), "p_obs": Pn["p_obs"].to(dev), "D_real": Pn["D_real"].to(dev), "G": G.to(dev), "mkt": {k: v.to(dev) for k, v in mkt.items()}}
    n = len(samples); purge = max(1, int(np.ceil(h / args.every)))
    splits = cpcv_splits(n, n_groups=6, n_test=2, horizon=purge, embargo=1)
    ic_model, ic_value = {}, {}
    for k, (tr_idx, te_idx, combo) in enumerate(splits):
        cut = int(0.85 * len(tr_idx)); TR, VA, TE = pack([samples[i] for i in tr_idx[:cut]]), pack([samples[i] for i in tr_idx[cut:]]), pack([samples[i] for i in te_idx])
        model = EmporionV0(in_dim=TR["feats"].shape[-1], d_model=args.d_model, n_layers=2, n_heads=4, fund_col=0).double().to(dev)
        if args.pretrain_steps > 0:
            SP = synthetic_pool(seed=args.seed + k)
            train_residual(model, SP, steps=args.pretrain_steps, batch=128, lr=1e-3, log_every=100, verbose=False, balance_every=10 ** 9, w_r2_init=0.0, w_pen=args.w_pen, w_mu=args.w_mu)
        train_residual(model, {q: TR[q] for q in ("feats", "p_obs", "D_real", "G", "mkt")}, steps=args.steps, batch=min(32, TR["feats"].shape[0]), lr=1e-3, log_every=50,
                       verbose=False, balance_every=10 ** 9, val={q: VA[q] for q in ("feats", "p_obs", "D_real", "G", "mkt")}, patience=4, w_r2_init=0.0, w_pen=args.w_pen, w_mu=args.w_mu)
        with torch.no_grad():
            o = model.forward(TE["feats"], TE["p_obs"], TE["mkt"], mask=TE["mask"])
        for b, i in enumerate(te_idx):
            x = samples[i]; m_ = len(x["idx"]); real = np.log(x["P"][x["t"] + h][x["idx"]] / x["P"][x["t"]][x["idx"]]); real -= real.mean()
            ic_model.setdefault(i, []).append(rank_ic(-o["r_mkt"][b, :m_].cpu().numpy(), real)); ic_value[i] = rank_ic(x["feats"][:, 0], real)
        print(f"[cpcv] split {k+1}/{len(splits)} groups {combo}: test dates {len(te_idx)}", flush=True)
    res = {}
    for m in markets + ["pooled"]:
        idx = [i for i in ic_model if m == "pooled" or samples[i]["market"] == m]
        if not idx: continue
        a = np.array([np.mean(ic_model[i]) for i in idx]); v = np.array([ic_value[i] for i in idx])
        tt = lambda z: float(z.mean() / (z.std(ddof=1) / np.sqrt(len(z)))) if len(z) > 2 else float("nan")
        res[m] = {"n": len(idx), "model": {"mean_ic": float(a.mean()), "t": tt(a), "hit": float((a > 0).mean())}, "value_anchor": {"mean_ic": float(v.mean()), "t": tt(v)}}
        print(f"[cpcv-test] {m:6s} n={len(idx):3d}  Emporion -r_mkt RankIC {a.mean():+.4f} (t {tt(a):+.2f}, hit {(a>0).mean():.2f}) | value anchor {v.mean():+.4f} (t {tt(v):+.2f})", flush=True)
    name = f"emporion_v1_pooled_{'+'.join(markets)}_h{h}_every{args.every}_pen{args.w_pen}_d{args.d_model}" + (f"_pre{args.pretrain_steps}" if args.pretrain_steps else "")
    ledger.append("trial", name, {"sharpe_is": res["pooled"]["model"]["t"], **{f"{m}_ic": r["model"]["mean_ic"] for m, r in res.items()}, **{f"{m}_t": r["model"]["t"] for m, r in res.items()},
                                  "h": h, "every": args.every, "markets": markets, "cpcv": "6x2", "steps": args.steps, "pretrain_steps": args.pretrain_steps}, scope="p4-v1-pooled")
    with open(os.path.join(args.out_root, f"p4-v1-pooled-{'+'.join(markets)}-h{h}{'-pre' + str(args.pretrain_steps) if args.pretrain_steps else ''}-{now:%Y%m%d-%H%M}.json"), "w") as f:
        json.dump(res, f, indent=1)
    assert ledger.verify_chain(); print("[done]")


if __name__ == "__main__":
    main()
