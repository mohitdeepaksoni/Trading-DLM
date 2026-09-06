"""The deep-learning model on real US data, walk-forward (Block C, second pass).

Panel: monthly decision dates on the point-in-time top-200; per asset the anchor features known at t
  [log fair price (value anchor), SUE z, accruals z, momentum z, low-risk z, institutional level, log price]
Model: EmporionV0 (state_mode="kappa", law at the equilibrium price), beliefs anchored on the value level
(fund_col = log p_fund) and corrected by the net from the other features (penalised); kappa seeded from the
structural estimator on the trailing window; Sigma from trailing 250d; supply = shares outstanding (scaled).
Training: R1 on realised 20-day payoffs D = p_{t+20} (+ nothing else), validation early stopping.
Walk-forward: train < 2021-01, validate 2021, test 2022-01 → last; the test signal -r_mkt is scored by RankIC and
certify(), one ledger trial in scope p4-v0-us. Baseline in the same scope: the value anchor alone (kappa only).
"""
from __future__ import annotations

import argparse, datetime as dt, json, os, sys
import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from emporion.evaluate import portfolio_returns, certify
from emporion.models.emporion_v0 import EmporionV0
from emporion.solvers.residual_trainer import train_residual
from emporion.anchors import price_themes
from p3_event_harness import rank_ic
from p4_realdata_us_stocks import ttm_eps_asof, shares_asof, sue_asof, accruals_asof

torch.set_default_dtype(torch.float64)


def zs(x):
    m, s = np.nanmean(x), np.nanstd(x); return np.where(np.isfinite(x), (x - m) / (s + 1e-12), 0.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="runs/us_stocks_wide"); ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--steps", type=int, default=600); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train-end", default="2021-01-01"); ap.add_argument("--val-end", default="2022-01-01")
    ap.add_argument("--w-pen", type=float, default=1.0); ap.add_argument("--w-mu", type=float, default=1.0)
    ap.add_argument("--select", default="r1", choices=["r1", "ic"], help="checkpoint selection on validation: law residual, or rank-IC of -r_mkt (model selection only, never the loss)")
    ap.add_argument("--every", type=int, default=21, help="decision-date spacing in trading days (5 = ~20x the cross-sections, overlapping payoffs)")
    ap.add_argument("--cpcv", action="store_true", help="combinatorial purged CV over decision dates (6 groups, 2 test, purge = horizon) instead of a single split")
    args = ap.parse_args()
    dev = torch.device(args.device); store = BitemporalStore(os.path.join(args.root, "store")); ledger = Ledger(os.path.join(args.root, "ledger"))
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    bars = store.as_of("bars", now, columns=["close", "volume"]); facts = store.all_versions("facts")
    wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time"); syms = [c for c in wide.columns if c != "event_time"]
    wide = wide.with_columns([pl.col(c).fill_null(strategy="forward") for c in syms]); dates = wide["event_time"].to_list()
    P = wide.select(syms).to_numpy().astype(float); listed = np.isfinite(P); T, N = P.shape
    V = np.nan_to_num(bars.pivot(index="event_time", on="symbol", values="volume").sort("event_time").select(syms).to_numpy().astype(float)); dv = np.nan_to_num(P) * V
    R = np.where(listed[1:] & listed[:-1], P[1:] / P[:-1] - 1, 0.0)
    h = args.h; Rf = 1.0 + 0.04 / 252; K = 200
    dec = list(range(300, T - h - 1, args.every))
    samples = []                                                       # one per (date): feats (n,F), p_obs (n,), D_real (n,), Sig, s, kappa_in, idx
    for t in dec:
        hist = listed[max(0, t - 300):t + 1].all(0); med = np.median(dv[t - 250:t + 1], axis=0); med[~hist] = -1
        U = np.zeros(N, bool); top = np.argsort(-med)[:K]; U[top] = med[top] > 0
        asof = dates[t]; eps = ttm_eps_asof(facts, asof, syms); sh = shares_asof(facts, asof, syms); p = P[t]
        ok = U & np.isfinite(eps) & (eps > 0) & np.isfinite(sh) & (sh > 0) & np.isfinite(p)
        if ok.sum() < 40:
            continue
        ey = eps[ok] / p[ok]; p_fund = eps[ok] / np.median(ey)
        sue = sue_asof(facts, asof, syms)[ok]; acc = -accruals_asof(facts, asof, syms)[ok]
        Pu = np.where(U[None, :] & listed, P, np.nan); th = price_themes(Pu, V, t)
        feats = np.stack([np.log(p_fund), zs(sue), zs(acc), zs(th["momentum"][ok]), zs(th["low_risk"][ok]), np.log(p[ok])], axis=1)
        # RETURN UNITS: p_obs = 1, payoff D = gross h-day return, Sigma = covariance of h-day returns, supply = cap weights (sum 1)
        w = R[t - 250:t][:, ok]; Sig = np.cov(w.T) * h + 1e-6 * np.eye(ok.sum())
        cap = sh[ok] * p[ok]; s = cap / cap.sum()
        fwd = P[t - 250 + h:t + 1][:, ok] / P[t - 250:t + 1 - h][:, ok]; prem = fwd.mean(0) - Rf ** h            # trailing mean gross h-day excess return
        x = Sig @ s; kappa = float(np.clip((prem * x).sum() / (x * x).sum(), 1e-4, 50.0))      # winsorised
        D_real = P[t + h][ok] / p[ok]
        feats[:, 0] = np.clip(np.log(p_fund / p[ok]), -1.0, 1.0) + h * np.log(Rf)   # fund anchor (clipped: the value ratio has heavy tails)
        feats[:, 5] = zs(np.log(p[ok]))
        samples.append({"t": t, "date": asof, "feats": feats, "p_obs": np.ones(ok.sum()), "D": D_real, "Sig": Sig, "s": s, "kappa": kappa, "idx": np.flatnonzero(ok)})
    print(f"[panel] {len(samples)} monthly dates, {np.mean([len(x['idx']) for x in samples]):.0f} assets/date, features {samples[0]['feats'].shape[1]}; kappa median {np.median([x['kappa'] for x in samples]):.4g}")
    tr_end, va_end = dt.datetime.fromisoformat(args.train_end), dt.datetime.fromisoformat(args.val_end)
    split = {"train": [x for x in samples if x["date"] < tr_end], "val": [x for x in samples if tr_end <= x["date"] < va_end], "test": [x for x in samples if x["date"] >= va_end]}
    def pack(sub, n_fix=None):
        """Pad each date to a common asset count with a mask; the model handles padding via key-padding."""
        n = max(len(x["idx"]) for x in sub); B = len(sub); F = sub[0]["feats"].shape[1]
        feats = torch.zeros(B, n, F); p = torch.ones(B, n); D = torch.ones(B, n); Sig_inv = torch.eye(n).repeat(B, 1, 1); s = torch.zeros(B, n)
        mask = torch.ones(B, n, dtype=torch.bool); kap = torch.zeros(B)
        for b, x in enumerate(sub):
            m = len(x["idx"]); feats[b, :m] = torch.tensor(x["feats"]); p[b, :m] = torch.tensor(x["p_obs"]); D[b, :m] = torch.tensor(x["D"])
            Si = torch.linalg.inv(torch.tensor(x["Sig"])); Sig_inv[b, :m, :m] = Si; s[b, :m] = torch.tensor(x["s"]); mask[b, :m] = False; kap[b] = x["kappa"]
        mkt = {"mom": torch.zeros(B, n), "Sig_inv": Sig_inv, "s": s, "pbar": p.clone(), "Rf": torch.full((B,), Rf ** h), "kappa_in": kap, "phi_obs": torch.full((B,), 0.44)}
        G = torch.cat([torch.ones(B, n, 1), feats[..., 1:5]], dim=-1)
        return {"feats": feats.to(dev), "p_obs": p.to(dev), "D_real": D.to(dev), "G": G.to(dev), "mkt": {k: v.to(dev) for k, v in mkt.items()}, "mask": mask.to(dev)}
    print({k: len(v) for k, v in split.items()})
    TR, VA, TE = pack(split["train"]), pack(split["val"]), pack(split["test"])
    if args.cpcv:
        from emporion.evaluate.cpcv import cpcv_splits
        from emporion.core.ledger import Ledger as _L
        n = len(samples); purge = max(1, int(np.ceil(h / args.every)))
        splits = cpcv_splits(n, n_groups=6, n_test=2, horizon=purge, embargo=1)
        per_date_model = {}; per_date_value = {}
        def pack_idx(idx):
            return pack([samples[i] for i in idx])
        for k, (tr_idx, te_idx, combo) in enumerate(splits):
            cut = int(0.85 * len(tr_idx)); TRk, VAk = pack_idx(tr_idx[:cut]), pack_idx(tr_idx[cut:]); TEk = pack_idx(te_idx)
            model = EmporionV0(in_dim=TRk["feats"].shape[-1], d_model=64, n_layers=2, n_heads=4, fund_col=0).double().to(dev)
            train_residual(model, {q: TRk[q] for q in ("feats", "p_obs", "D_real", "G", "mkt")}, steps=args.steps, batch=min(32, TRk["feats"].shape[0]), lr=1e-3,
                           log_every=50, verbose=False, balance_every=10 ** 9, val={q: VAk[q] for q in ("feats", "p_obs", "D_real", "G", "mkt")}, patience=4, w_r2_init=0.0,
                           w_pen=args.w_pen, w_mu=args.w_mu)
            with torch.no_grad():
                o = model.forward(TEk["feats"], TEk["p_obs"], TEk["mkt"], mask=TEk["mask"])
            for b, i in enumerate(te_idx):
                x = samples[i]; m_ = len(x["idx"]); real = np.log(P[x["t"] + h][x["idx"]] / P[x["t"]][x["idx"]]); real -= real.mean()
                per_date_model.setdefault(i, []).append(rank_ic(-o["r_mkt"][b, :m_].cpu().numpy(), real))
                per_date_value[i] = rank_ic(x["feats"][:, 0], real)
            print(f"[cpcv] split {k+1}/{len(splits)} groups {combo}: test dates {len(te_idx)}", flush=True)
        icm = np.array([np.mean(v) for v in per_date_model.values()]); icv = np.array([per_date_value[i] for i in per_date_model])
        res = {"model": {"mean_ic": float(icm.mean()), "t": float(icm.mean() / (icm.std(ddof=1) / np.sqrt(len(icm)))), "hit": float((icm > 0).mean()), "n": int(len(icm))},
               "value_anchor": {"mean_ic": float(icv.mean()), "t": float(icv.mean() / (icv.std(ddof=1) / np.sqrt(len(icv)))), "hit": float((icv > 0).mean())}}
        print(f"[cpcv-test] Emporion v0 -r_mkt over ALL dates out-of-sample: RankIC {res['model']['mean_ic']:+.4f} (t {res['model']['t']:+.2f}, hit {res['model']['hit']:.2f}, n={res['model']['n']}) | value anchor: {res['value_anchor']['mean_ic']:+.4f} (t {res['value_anchor']['t']:+.2f})")
        ledger.append("trial", f"emporion_v0_us_cpcv_h{h}_every{args.every}_pen{args.w_pen}", {"sharpe_is": res["model"]["t"], **res["model"], "h": h, "cpcv": "6x2", "every": args.every, "w_pen": args.w_pen}, scope="p4-v0-us")
        with open(os.path.join(args.root, f"p4-v0-cpcv-h{h}-{now:%Y%m%d-%H%M}.json"), "w") as f:
            json.dump(res, f, indent=1)
        assert ledger.verify_chain(); print("[done]"); return
    model = EmporionV0(in_dim=TR["feats"].shape[-1], d_model=64, n_layers=2, n_heads=4, fund_col=0).double().to(dev)
    # padded assets must not enter the law: zero their supply, keep Sig_inv identity, mask in the encoder
    with torch.no_grad():
        _, p0, _, o0 = model.loss(VA["feats"], VA["p_obs"], VA["D_real"], VA["G"], VA["mkt"], mask=VA["mask"], w_pen=0.0, w_mu=0.0)
    print(f"[init] validation R1 = {float(p0['r1']):.4f}")
    with torch.no_grad():                                                     # kernel diagnostics at init
        M0 = model.kernel_at(VA["D_real"], o0, VA["mkt"])
        st = o0["state"]; A = (1 / st.a).sum(1); q = (1 - st.phi).unsqueeze(1) * VA["mkt"]["s"]
        wD = -(q / A.unsqueeze(1) * VA["D_real"]).sum(1)
        print(f"[diag] kappa_in min/median/max {float(VA['mkt']['kappa_in'].min()):.3g}/{float(VA['mkt']['kappa_in'].median()):.3g}/{float(VA['mkt']['kappa_in'].max()):.3g} | "
              f"w.D min/max {float(wD.min()):.3g}/{float(wD.max()):.3g} | M min/median/max {float(M0.min()):.3g}/{float(M0.median()):.3g}/{float(M0.max()):.3g} | "
              f"r_mkt std {float(o0['r_mkt'][~VA['mask']].std()):.3f} | fund feat min/max {float(VA['feats'][...,0].min()):.2f}/{float(VA['feats'][...,0].max()):.2f}")
    def val_ic(m):
        with torch.no_grad():
            o = m.forward(VA["feats"], VA["p_obs"], VA["mkt"], mask=VA["mask"])
        ics = []
        for b, x in enumerate(split["val"]):
            n_ = len(x["idx"]); sg = -o["r_mkt"][b, :n_].cpu().numpy(); real = np.log(P[x["t"] + h][x["idx"]] / P[x["t"]][x["idx"]]); ics.append(rank_ic(sg, real - real.mean()))
        return float(np.mean(ics))
    if args.select == "r1":
        hist = train_residual(model, {k: TR[k] for k in ("feats", "p_obs", "D_real", "G", "mkt")}, steps=args.steps, batch=min(32, TR["feats"].shape[0]), lr=1e-3,
                              log_every=50, verbose=True, balance_every=10 ** 9, val={k: VA[k] for k in ("feats", "p_obs", "D_real", "G", "mkt")}, patience=4, w_r2_init=0.0,
                              w_pen=args.w_pen, w_mu=args.w_mu)
    else:                                                                     # train on the law; keep the checkpoint with the best VALIDATION rank-IC
        best, best_state, hist = -1e9, None, []
        for chunk in range(args.steps // 50):
            hist += train_residual(model, {k: TR[k] for k in ("feats", "p_obs", "D_real", "G", "mkt")}, steps=50, batch=min(32, TR["feats"].shape[0]), lr=1e-3,
                                   log_every=50, verbose=False, balance_every=10 ** 9, w_r2_init=0.0, w_pen=args.w_pen, w_mu=args.w_mu)
            v = val_ic(model)
            if v > best:
                best, best_state = v, {k: t_.detach().clone() for k, t_ in model.state_dict().items()}
            print(f"[select] step {(chunk+1)*50}: validation rank-IC {v:+.4f} (best {best:+.4f})", flush=True)
        model.load_state_dict(best_state)
    with torch.no_grad():
        _, p1, _, o1 = model.loss(VA["feats"], VA["p_obs"], VA["D_real"], VA["G"], VA["mkt"], mask=VA["mask"], w_pen=0.0, w_mu=0.0)
        out_te = model.forward(TE["feats"], TE["p_obs"], TE["mkt"], mask=TE["mask"])
    print(f"[trained] validation R1 = {float(p1['r1']):.4f} (init {float(p0['r1']):.4f})")
    # test: signal = -r_mkt (buy below the model's equilibrium); baseline = value anchor alone (kappa-only, no net)
    ics, ics_base, rows = [], [], []
    for b, x in enumerate(split["test"]):
        m = len(x["idx"]); sig = -out_te["r_mkt"][b, :m].cpu().numpy()
        real = np.log(P[x["t"] + h][x["idx"]] / P[x["t"]][x["idx"]]); real = real - real.mean()
        base = x["feats"][:, 0]                                                   # value anchor alone = log(p_fund/p) + h log Rf
        ics.append(rank_ic(sig, real)); ics_base.append(rank_ic(base, real))
        rows.append({"date": x["date"].strftime("%Y-%m-%d"), "ic_model": ics[-1], "ic_value": ics_base[-1], "abstain": bool(np.isfinite(sig).sum() < 30)})
    ics, ics_base = np.array(ics), np.array(ics_base)
    res = {"model": {"mean_ic": float(ics.mean()), "t": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), "hit": float((ics > 0).mean()), "n": int(len(ics))},
           "value_anchor": {"mean_ic": float(ics_base.mean()), "t": float(ics_base.mean() / (ics_base.std(ddof=1) / np.sqrt(len(ics_base)))), "hit": float((ics_base > 0).mean())}}
    print(f"[test] Emporion v0 -r_mkt: RankIC {res['model']['mean_ic']:+.4f} (t {res['model']['t']:+.2f}, hit {res['model']['hit']:.2f}, n={res['model']['n']}) | value anchor alone: {res['value_anchor']['mean_ic']:+.4f} (t {res['value_anchor']['t']:+.2f})")
    ledger.append("trial", f"emporion_v0_us_walkforward_pen{args.w_pen}_sel{args.select}_every{args.every}", {"sharpe_is": res["model"]["t"], **res["model"], "h": h, "train_end": args.train_end, "val_end": args.val_end, "w_pen": args.w_pen, "select": args.select,
                                                         "val_r1_init": float(p0["r1"]), "val_r1_trained": float(p1["r1"]), "steps": len(hist) * 50}, scope="p4-v0-us")
    with open(os.path.join(args.root, f"p4-v0-walkforward-{now:%Y%m%d-%H%M}.json"), "w") as f:
        json.dump({"results": res, "per_date": rows, "history": hist}, f, indent=1, default=str)
    assert ledger.verify_chain(); print("[done]")


if __name__ == "__main__":
    main()
