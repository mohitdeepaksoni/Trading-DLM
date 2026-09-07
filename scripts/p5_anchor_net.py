"""Emporion v2 — a deep model OVER the anchors (Gu–Kelly–Xiu style), pooled US + NSE, judged against the no-fit composite.
Features per stock per decision date (cross-sectionally z-scored, clipped, missing -> 0 + indicator): 8 price themes,
value anchor, SUE, accruals, 13F holders change + institutional level (fullest-batch rule), Agora sector shock (industry
map x dlogPX of an event in the last 5 sessions, else 0), market token. Target: within-date rank of the forward
demeaned log return at h sessions. Models on identical CPCV splits (6 groups x 2 test, purged by horizon, dates from
both markets in time order): (0) composite_nofit (no fit), (1) ridge, (2) gradient boosting, (3) MLP ensemble (3 seeds,
early-stopped on validation rank-IC). Pre-declared exit line: OOS US rank-IC >= composite + 0.01 with t > 2 at h = 20 or 60.
Ledger scope p5-anchor-net, one trial per (model, h).
    python scripts/p5_anchor_net.py --h 20 --every 21
"""
import argparse, datetime as dt, json, os, sys, time
import numpy as np, polars as pl, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emporion.core.ledger import Ledger
from emporion.data.store import BitemporalStore
from emporion.anchors import price_themes
from emporion.anchors.library import THEMES
from emporion.evaluate.cpcv import cpcv_splits
from emporion.structure.industry_map import exposure_matrix
from p3_event_harness import rank_ic
from p4_realdata_us_stocks import ttm_eps_asof, shares_asof, sue_asof, accruals_asof
from emporion.anchors.nse import last_known_t, split_adjust

torch.set_default_dtype(torch.float32)
FEATS = list(THEMES) + ["value", "sue", "accruals", "holders_change", "inst_level", "agora_shock", "is_us", "is_nse"]
MISS = ["accruals", "holders_change", "inst_level", "sue", "value"]                          # get missing indicators


def zclip(x):
    ok = np.isfinite(x); z = np.zeros_like(x, dtype=float)
    if ok.sum() > 2:
        z[ok] = np.clip((x[ok] - x[ok].mean()) / (x[ok].std() + 1e-12), -3, 3)
    return z, (~ok).astype(float)


def _panel(store, now, K, every, market, h):
    bars = store.as_of("bars", now, columns=["close", "volume"])
    wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time"); syms = [c for c in wide.columns if c != "event_time"]
    wide = wide.with_columns([pl.col(c).fill_null(strategy="forward") for c in syms]); dates = wide["event_time"].to_list()
    P = wide.select(syms).to_numpy().astype(float); V = np.nan_to_num(bars.pivot(index="event_time", on="symbol", values="volume").sort("event_time").select(syms).to_numpy().astype(float))
    listed = np.isfinite(P); T, N = P.shape; dv = np.nan_to_num(P) * V
    def universe(t):
        hist = listed[max(0, t - 300):t + 1].all(0); med = np.median(dv[t - 250:t + 1], axis=0); med[~hist] = -1
        U = np.zeros(N, bool); top = np.argsort(-med)[:K]; U[top] = med[top] > 0; return U
    dec = list(range(300, T - h - 1, every))
    return syms, dates, P, V, listed, T, N, universe, dec


def build_us(root, h, every, K, agora_events):
    store = BitemporalStore(os.path.join(root, "store")); now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    syms, dates, P, V, listed, T, N, universe, dec = _panel(store, now, K, every, "us", h)
    facts = store.all_versions("facts"); inst = store.all_versions("inst_holdings").filter(pl.col("symbol").is_in(syms))
    uni = store.as_of("universe", now, columns=["sector", "industry"]).group_by("symbol").last(); umap = {r["symbol"]: (r["industry"], r["sector"]) for r in uni.iter_rows(named=True)}
    E = exposure_matrix([umap.get(s, ("", ""))[0] for s in syms], [umap.get(s, ("", ""))[1] for s in syms], agora_events["codes"]) if agora_events else None
    ev_dates = sorted(agora_events["by_date"]) if agora_events else []
    date_idx = {d.date(): i for i, d in enumerate(dates)}
    out = []
    for t in dec:
        asof = dates[t]; U = universe(t); p = P[t]
        eps = ttm_eps_asof(facts, asof, syms); sh = shares_asof(facts, asof, syms); sue = sue_asof(facts, asof, syms); acc = -accruals_asof(facts, asof, syms)
        ok = U & np.isfinite(p)
        if ok.sum() < 50: continue
        value = np.full(N, np.nan); okv = ok & np.isfinite(eps) & (eps > 0)
        if okv.sum() > 20:
            ey = eps[okv] / p[okv]; value[okv] = np.log((eps[okv] / np.median(ey)) / p[okv])
        # 13F fullest-batch rule (research/03 §15): per (symbol, quarter) the batch with max n_holders known by asof
        known = inst.filter(pl.col("knowledge_time") <= asof).sort("n_holders").group_by(["symbol", "event_time"]).last()
        g = known.sort("event_time").group_by("symbol").agg(pl.col("n_holders").tail(2).alias("nh"), pl.col("inst_shares").tail(1).alias("is_"), pl.col("event_time").tail(1).alias("t1"))
        hc = np.full(N, np.nan); lvl = np.full(N, np.nan); si = {s: i for i, s in enumerate(syms)}
        for r in g.iter_rows(named=True):
            i = si.get(r["symbol"])
            if i is None or (asof - r["t1"][0]).days > 200: continue
            if len(r["nh"]) == 2: hc[i] = np.log(max(r["nh"][1], 1) / max(r["nh"][0], 1))
            if np.isfinite(sh[i]) and sh[i] > 0: lvl[i] = min(r["is_"][0] / sh[i], 1.5)
        shock = np.zeros(N)
        if E is not None:
            recent = [d for d in ev_dates if d in date_idx and 0 <= t - date_idx[d] <= 5]
            if recent: shock = E @ agora_events["by_date"][recent[-1]]
        Pu = np.where(U[None, :] & listed, P, np.nan); th = price_themes(Pu, V, t)
        raw = {k: th.get(k, np.full(N, np.nan)) for k in THEMES}; raw.update({"value": value, "sue": sue, "accruals": acc, "holders_change": hc, "inst_level": lvl, "agora_shock": shock})
        out.append(_pack("us", t, asof, raw, ok, P, h, N))
    return out


def build_nse(root, h, every, K):
    store = BitemporalStore(os.path.join(root, "store")); now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    syms, dates, P, V, listed, T, N, universe, dec = _panel(store, now, K, every, "nse", h)
    res = store.all_versions("results").sort(["symbol", "event_time", "consolidated", "knowledge_time"]); res_c = res.filter(pl.col("consolidated")); cons = set(res_c["symbol"].to_list())
    res = pl.concat([res_c, res.filter(~pl.col("consolidated") & ~pl.col("symbol").is_in(list(cons)))]); sh_df = store.all_versions("shareholding")
    out = []
    for t in dec:
        asof = dates[t]; U = universe(t); p = P[t]; ok = U & np.isfinite(p)
        if ok.sum() < 50: continue
        shs = last_known_t(sh_df, asof, "shares_total", syms, n=40, days=100000, min_n=1)
        e4 = [split_adjust(a, b) for a, b in zip(last_known_t(res, asof, "eps_diluted", syms, n=4), shs)]
        e12 = [split_adjust(a, b) for a, b in zip(last_known_t(res, asof, "eps_diluted", syms, n=12, days=400, min_n=8), shs)]
        eps = np.array([sum(v) if v else np.nan for v in e4]); sue = np.full(N, np.nan)
        for i, v in enumerate(e12):
            if v and len(v) >= 8:
                d = np.array(v[4:]) - np.array(v[:-4]); sd = d[:-1].std(ddof=1)
                if sd > 1e-9: sue[i] = np.clip(d[-1] / sd, -5, 5)
        value = np.full(N, np.nan); okv = ok & np.isfinite(eps) & (eps > 0)
        if okv.sum() > 20:
            ey = eps[okv] / p[okv]; value[okv] = np.log((eps[okv] / np.median(ey)) / p[okv])
        Pu = np.where(U[None, :] & listed, P, np.nan); th = price_themes(Pu, V, t)
        raw = {k: th.get(k, np.full(N, np.nan)) for k in THEMES}; raw.update({"value": value, "sue": sue, "accruals": np.full(N, np.nan), "holders_change": np.full(N, np.nan), "inst_level": np.full(N, np.nan), "agora_shock": np.zeros(N)})
        out.append(_pack("nse", t, asof, raw, ok, P, h, N))
    return out


def _pack(market, t, asof, raw, ok, P, h, N):
    idx = np.flatnonzero(ok); X = []; names = []
    for k in FEATS:
        if k == "is_us": X.append(np.full(len(idx), float(market == "us"))); names.append(k); continue
        if k == "is_nse": X.append(np.full(len(idx), float(market == "nse"))); names.append(k); continue
        z, miss = zclip(np.asarray(raw[k], float)[idx]); X.append(z); names.append(k)
        if k in MISS: X.append(miss); names.append(k + "_missing")
    X = np.stack(X, 1)
    real = np.log(P[t + h][idx] / P[t][idx]); real = real - np.nanmean(real)
    y = (np.argsort(np.argsort(np.nan_to_num(real))) / max(len(idx) - 1, 1)) - 0.5
    comp = np.nanmean(np.stack([X[:, names.index("value")], X[:, names.index("sue")], X[:, names.index("accruals")], X[:, names.index("momentum")]]), axis=0)
    return {"market": market, "t": t, "date": asof, "X": X.astype(np.float32), "y": y.astype(np.float32), "real": real, "names": names, "composite": comp}


class MLP(torch.nn.Module):
    def __init__(self, d, w=64, p=0.2):
        super().__init__(); self.net = torch.nn.Sequential(torch.nn.Linear(d, w), torch.nn.GELU(), torch.nn.Dropout(p), torch.nn.Linear(w, w), torch.nn.GELU(), torch.nn.Dropout(p), torch.nn.Linear(w, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def train_mlp(train, val, d, seed, epochs=60, lr=1e-3, wd=1e-3):
    torch.manual_seed(seed); m = MLP(d); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Xs = [torch.tensor(s["X"]) for s in train]; ys = [torch.tensor(s["y"]) for s in train]
    best, best_state, bad = -1e9, None, 0
    for ep in range(epochs):
        m.train(); order = np.random.default_rng(seed + ep).permutation(len(Xs))
        for i in order:
            opt.zero_grad(); loss = ((m(Xs[i]) - ys[i]) ** 2).mean(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = np.mean([rank_ic(m(torch.tensor(s["X"])).numpy(), s["real"]) for s in val])
        if v > best: best, best_state, bad = v, {k: t_.clone() for k, t_ in m.state_dict().items()}, 0
        else: bad += 1
        if bad >= 8: break
    m.load_state_dict(best_state); m.eval(); return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--h", type=int, default=20); ap.add_argument("--every", type=int, default=21); ap.add_argument("--markets", default="us,nse")
    ap.add_argument("--us-root", default="runs/us_stocks_wide"); ap.add_argument("--nse-root", default="runs/nse"); ap.add_argument("--us-k", type=int, default=500); ap.add_argument("--nse-k", type=int, default=200)
    ap.add_argument("--out-root", default="runs/p5_anchor_net"); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--no-agora", action="store_true")
    args = ap.parse_args(); h = args.h; os.makedirs(args.out_root, exist_ok=True); ledger = Ledger(os.path.join(args.out_root, "ledger")); now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    agora = None
    if not args.no_agora:
        try:
            from emporion.structure.agora_bridge import sector_response
            from emporion.structure.events_us_tariffs import events
            ev = events(); resp = sector_response("USA", [e["shock"] for e in ev], [e["te"] for e in ev])
            agora = {"codes": resp["codes"], "by_date": {dt.date.fromisoformat(e["date"]): np.asarray(resp["dlogPX"][k]) for k, e in enumerate(ev)}}
            print(f"[agora] {len(ev)} events solved (max residual {resp['max_residual']:.1e})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[agora] unavailable ({str(e)[:60]}); shock feature = 0", flush=True)
    t0 = time.perf_counter(); samples = []
    for m in args.markets.split(","):
        s = build_us(args.us_root, h, args.every, args.us_k, agora) if m == "us" else build_nse(args.nse_root, h, args.every, args.nse_k); samples += s
        print(f"[panel] {m}: {len(s)} dates x ~{np.mean([len(x['y']) for x in s]):.0f} stocks, {s[0]['X'].shape[1]} features ({time.perf_counter()-t0:.0f}s)", flush=True)
    samples.sort(key=lambda x: x["date"]); names = samples[0]["names"]; d = len(names); n = len(samples)
    purge = max(1, int(np.ceil(h / args.every))); splits = cpcv_splits(n, n_groups=6, n_test=2, horizon=purge, embargo=1)
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    preds = {k: {} for k in ("composite_nofit", "ridge", "gbm", "mlp")}
    for k, (tr_idx, te_idx, combo) in enumerate(splits):
        cut = int(0.85 * len(tr_idx)); TR = [samples[i] for i in tr_idx[:cut]]; VA = [samples[i] for i in tr_idx[cut:]]; TE = [samples[i] for i in te_idx]
        Xtr = np.concatenate([s["X"] for s in TR]); ytr = np.concatenate([s["y"] for s in TR])
        ridge = Ridge(alpha=10.0).fit(Xtr, ytr)
        gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=0).fit(Xtr, ytr)
        mlps = [train_mlp(TR, VA, d, seed) for seed in range(args.seeds)]
        for b, i in enumerate(te_idx):
            s = samples[i]; X = s["X"]
            with torch.no_grad(): pm = np.mean([m(torch.tensor(X)).numpy() for m in mlps], axis=0)
            for key, pr in (("composite_nofit", s["composite"]), ("ridge", ridge.predict(X)), ("gbm", gbm.predict(X)), ("mlp", pm)):
                preds[key].setdefault(i, []).append(rank_ic(pr, s["real"]))
        print(f"[cpcv] split {k+1}/{len(splits)} groups {combo}: {len(TE)} test dates ({time.perf_counter()-t0:.0f}s)", flush=True)
    res = {}
    for key in preds:
        res[key] = {}
        for m in args.markets.split(",") + ["pooled"]:
            idx = [i for i in preds[key] if m == "pooled" or samples[i]["market"] == m]
            if not idx: continue
            a = np.array([np.mean(preds[key][i]) for i in idx]); tt = float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 2 else float("nan")
            res[key][m] = {"n": len(idx), "mean_ic": float(a.mean()), "t": tt, "hit": float((a > 0).mean())}
        print(f"[oos h={h}] {key:16s} " + " | ".join(f"{m}: {r['mean_ic']:+.4f} (t {r['t']:+.2f}, n {r['n']})" for m, r in res[key].items()), flush=True)
    us = res.get("mlp", {}).get("us"); comp = res["composite_nofit"].get("us")
    if us and comp:
        passed = us["mean_ic"] >= comp["mean_ic"] + 0.01 and us["t"] > 2.0
        print(f"[exit-line] MLP US {us['mean_ic']:+.4f} (t {us['t']:+.2f}) vs composite {comp['mean_ic']:+.4f}: {'PASSED' if passed else 'NOT met'}", flush=True)
    for key in ("ridge", "gbm", "mlp"):
        r = res[key]; ledger.append("trial", f"v2_anchor_{key}_h{h}_every{args.every}_k{args.us_k}_{'+'.join(args.markets.split(','))}", {"sharpe_is": r.get("pooled", {}).get("t", 0.0), **{f"{m}_ic": v["mean_ic"] for m, v in r.items()}, **{f"{m}_t": v["t"] for m, v in r.items()}, "h": h, "every": args.every, "features": names, "cpcv": "6x2"}, scope="p5-anchor-net")
    # per-date out-of-sample ICs (for stability / overlap-adjusted t) and ridge coefficients on the full sample (attribution only, in-sample)
    per_date = [{"date": samples[i]["date"].strftime("%Y-%m-%d"), "market": samples[i]["market"], **{k: float(np.mean(preds[k][i])) for k in preds if i in preds[k]}} for i in range(n) if i in preds["mlp"]]
    Xall = np.concatenate([s["X"] for s in samples]); yall = np.concatenate([s["y"] for s in samples]); coef = dict(zip(names, map(float, Ridge(alpha=10.0).fit(Xall, yall).coef_)))
    Xn = np.concatenate([s["X"] for s in samples if s["market"] == "nse"]); yn = np.concatenate([s["y"] for s in samples if s["market"] == "nse"]); coef_nse = dict(zip(names, map(float, Ridge(alpha=10.0).fit(Xn, yn).coef_))) if len(Xn) else {}
    with open(os.path.join(args.out_root, f"p5-anchor-net-h{h}-every{args.every}-{'+'.join(args.markets.split(','))}-{now:%Y%m%d-%H%M}.json"), "w") as f:
        json.dump({"results": res, "features": names, "n_dates": n, "per_date": per_date, "ridge_coef_full_sample": coef, "ridge_coef_nse_only": coef_nse}, f, indent=1)
    assert ledger.verify_chain(); print("[done]")


if __name__ == "__main__":
    main()
