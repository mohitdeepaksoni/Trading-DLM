"""Daily pre-registered signals (doc 07 B.1): one file per market per day, hashed, chained, ledgered, and
externally timestamped BEFORE the next session. The file carries the point-in-time universe, every anchor score,
the pre-registered strategy's target weights (accruals top-quintile on US; low-risk on crypto; value/holdings on
NSE when the tables exist), and the abstention flags.  Nothing here is fitted on the day it is published."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from typing import Dict, Optional

import numpy as np
import polars as pl

from ..core.ledger import Ledger
from ..data.store import BitemporalStore
from ..anchors import price_themes
from ..anchors.fundamentals import ttm_eps, sue as sue_asof, accruals as accruals_asof, value_anchor


def _pit_universe(P, listed, dv, t, K, min_days=300):
    hist = listed[max(0, t - min_days):t + 1].all(0)
    med = np.median(dv[max(0, t - 250):t + 1], axis=0); med[~hist] = -1
    keep = np.zeros(P.shape[1], bool); top = np.argsort(-med)[:K]; keep[top] = med[top] > 0
    return keep


def _accruals_us(facts_all: pl.DataFrame, asof: dt.datetime, syms):
    """-(NI_ttm - CFO_ttm)/Assets from all revisions resolved at asof."""
    f = facts_all.filter(pl.col("knowledge_time") <= asof).sort("knowledge_time").group_by(["symbol", "concept", "event_time"]).last()
    def last4(concept):
        g = f.filter(pl.col("concept") == concept).sort("event_time").group_by("symbol").agg(pl.col("value").tail(4).alias("v"), pl.col("event_time").tail(4).alias("t"))
        return {r["symbol"]: sum(r["v"]) for r in g.iter_rows(named=True) if len(r["v"]) == 4 and (asof - r["t"][-1]).days < 400}
    ni, cfo = last4("net_income"), last4("cfo")
    a = f.filter(pl.col("concept") == "assets").sort("knowledge_time").group_by("symbol").last()
    assets = {r["symbol"]: r["value"] for r in a.iter_rows(named=True)}
    return np.array([-(ni[s] - cfo[s]) / assets[s] if (s in ni and s in cfo and assets.get(s)) else np.nan for s in syms])


def build_signals(market: str, root: str, K: int = 200, asof: Optional[dt.datetime] = None, strategy_key: str = "primary") -> Dict:
    store = BitemporalStore(os.path.join(root, "store"))
    asof = asof or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    bars = store.as_of("bars", asof, columns=["close", "volume"])
    wide = bars.pivot(index="event_time", on="symbol", values="close").sort("event_time")
    syms = [c for c in wide.columns if c != "event_time"]
    wide = wide.with_columns([pl.col(c).fill_null(strategy="forward") for c in syms])
    P = wide.select(syms).to_numpy().astype(float); listed = np.isfinite(P); dates = wide["event_time"].to_list()
    V = np.nan_to_num(bars.pivot(index="event_time", on="symbol", values="volume").sort("event_time").select(syms).to_numpy().astype(float))
    t = P.shape[0] - 1
    U = _pit_universe(P, listed, np.nan_to_num(P) * V, t, K, min_days=300 if market != "crypto_full" else 365)
    Pu = np.where(U[None, :] & listed, P, np.nan)
    th = price_themes(Pu, V, t, periods=365 if market.startswith("crypto") else 252)
    out = {"market": market if strategy_key == "primary" else f"{market}-{strategy_key}", "as_of": asof.isoformat(timespec="seconds"), "last_bar": dates[-1].strftime("%Y-%m-%d"),
           "universe_size": int(U.sum()), "n_pool": len(syms), "rule": "point-in-time top-K by trailing 250d median dollar volume, listed >= 300d"}
    scores = {k: np.where(U, v, np.nan) for k, v in th.items()}
    if market in ("us_stocks_wide", "us_stocks") and "facts" in store.tables():
        facts = store.all_versions("facts")
        eps = ttm_eps(facts, asof, syms); scores["accruals"] = np.where(U, -accruals_asof(facts, asof, syms), np.nan)
        scores["value"] = np.where(U, value_anchor(eps, P[t]), np.nan); scores["sue"] = np.where(U, sue_asof(facts, asof, syms), np.nan)
        zc = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)
        # PRE-REGISTERED 2026-09-06: no-fit composite = equal-weight mean of z(value), z(SUE), z(accruals), z(momentum); weights fixed forever
        parts = [zc(scores["value"]), zc(scores["sue"]), zc(scores["accruals"]), zc(scores.get("momentum", np.full(len(syms), np.nan)))]
        scores["composite_nofit"] = np.where(U, np.nanmean(np.stack(parts), axis=0), np.nan)
        if strategy_key == "composite":
            strategy, primary = "composite_nofit_top20pct (pre-registered 2026-09-06; equal z-weights value/SUE/accruals/momentum)", "composite_nofit"
        else:
            strategy, primary = "accruals_top20pct (pre-registered 2026-09-06)", "accruals"
    elif market.startswith("crypto"):
        strategy, primary = "low_risk_top20pct (pre-registered 2026-09-06, research track)", "low_risk"
    elif market == "nse" and "results" in store.tables():
        from .. import anchors as _a  # noqa: F401
        from ..anchors.nse import consolidated_first, nse_eps_sue_asof
        res = consolidated_first(store.all_versions("results")); shh = store.all_versions("shareholding") if "shareholding" in store.tables() else pl.DataFrame({"symbol": [], "event_time": [], "knowledge_time": []})
        eps, sue_n = nse_eps_sue_asof(res, shh, asof, syms)
        scores["sue"] = np.where(U, sue_n, np.nan); scores["value"] = np.where(U, value_anchor(eps, P[t]), np.nan)
        # PRE-REGISTERED 2026-09-07: NSE earnings surprise, top quintile of SUE within the point-in-time top-200; replaces the momentum placeholder
        strategy, primary = "sue_top20pct (pre-registered 2026-09-07; seasonal EPS surprise / 8-quarter std, split-adjusted, consolidated-first)", "sue"
    else:
        strategy, primary = "value_top20pct (placeholder until NSE fundamentals land)", "momentum"
    s = scores.get(primary, np.full(len(syms), np.nan)); ok = np.isfinite(s)
    k = max(1, int(ok.sum() * 0.2)); top = np.argsort(np.nan_to_num(s, nan=-1e9))[-k:] if ok.sum() >= 30 else np.array([], int)
    w = np.zeros(len(syms)); w[top] = 1.0 / k if len(top) else 0.0
    abstain = bool(ok.sum() < 30)
    rows = []
    for i, sym in enumerate(syms):
        if not U[i]:
            continue
        rows.append({"symbol": sym, "close": float(P[t, i]), **{kk: (None if not np.isfinite(vv[i]) else round(float(vv[i]), 6)) for kk, vv in scores.items()},
                     "target_weight": round(float(w[i]), 6)})
    out.update({"strategy": strategy, "primary_score": primary, "abstain": abstain, "n_scored": int(ok.sum()), "n_positions": int(len(top)),
                "rows": sorted(rows, key=lambda r: -r["target_weight"])})
    return out


def publish(sig: Dict, out_dir: str, ledger: Ledger) -> Dict:
    """Write the signal file, hash it, chain the hash, ledger it. Returns the chain entry."""
    os.makedirs(out_dir, exist_ok=True)
    day = sig["as_of"][:10]; path = os.path.join(out_dir, f"{sig['market']}-{day}.json")
    body = json.dumps(sig, sort_keys=True, separators=(",", ":")).encode()
    h = hashlib.sha256(body).hexdigest()
    with open(path, "wb") as f:
        f.write(body)
    chain_path = os.path.join(out_dir, "chain.txt")
    prev = "0" * 64
    if os.path.exists(chain_path):
        with open(chain_path) as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        if lines:
            prev = lines[-1].split()[-1]
    link = hashlib.sha256((prev + h).encode()).hexdigest()
    with open(chain_path, "a") as f:
        f.write(f"{sig['as_of']} {sig['market']} {h} {link}\n")
    row = ledger.append("signal", f"{sig['market']}:{day}", {"file": os.path.basename(path), "sha256": h, "chain": link, "prev": prev,
                                                             "strategy": sig["strategy"], "n_positions": sig["n_positions"], "abstain": sig["abstain"],
                                                             "universe_size": sig["universe_size"], "last_bar": sig["last_bar"]}, scope="shadow-signals")
    return {"path": path, "sha256": h, "chain": link, "ledger_seq": row["seq"]}
