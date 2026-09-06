"""Agora bridge — policy shock -> general-equilibrium sector response (doc 04 §3, doc 06 §5).

Uses Agora's *exact* open-economy CGE solver on the calibrated ICIO-2018 anchors
(45 ISIC sectors, 70 economies). No trained weights are needed for a few dozen
scenarios; the 540M surrogate is only a speed-up. The Agora package is located via
$AGORA_PKG (a directory containing `agora/`), default ~/agora-pkg or the sibling
Mathemetics-DLM/agora checkout.

sector_response(code, dtm, dte) returns, per sector, the log change in real output
(QX), producer price (PX), nominal revenue (PX*QX) and composite price (PQ) between
the baseline tariff schedule and the shocked one, solved simultaneously as a batch.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import torch

from ..core.perf import timed


def _find_agora_pkg() -> str:
    cands = [os.environ.get("AGORA_PKG", ""), os.path.expanduser("~/agora-pkg"),
             os.path.join(os.path.dirname(__file__), "..", "..", "..", "Mathemetics-DLM", "agora")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "agora")):
            return os.path.abspath(c)
    raise FileNotFoundError("Agora package not found; set AGORA_PKG")


def _import_agora():
    pkg = _find_agora_pkg()
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    import agora  # noqa: F401
    from agora.models.country_general import build_open_batch
    from agora.models.open_cge import solve_open_cge, aggregates
    return pkg, build_open_batch, solve_open_cge, aggregates


# ISIC-group elasticities (Agora icio45 convention): sigma_q (Armington), omega_t (CET), sigma_va, income elasticity
_ELAST = {"A": (1.14, 0.92, 0.78, 0.40), "I": (2.50, 1.50, 0.90, 1.00), "S": (2.15, 0.90, 1.20, 1.20)}
# baseline applied import tariff by group (US MFN trade-weighted, pre-2025): agriculture ~5%, goods ~2%, services 0
_BASE_TM = {"A": 0.05, "I": 0.02, "S": 0.0}


@lru_cache(maxsize=None)
def anchor(code: str) -> dict:
    pkg = _find_agora_pkg()
    d = json.load(open(os.path.join(pkg, "agora", "data", "icio_anchors_2018.json")))
    if code not in d:
        raise KeyError(f"{code} not in anchors; have {sorted(d)[:10]}...")
    return d[code]


def sectors(code: str = "USA") -> List[str]:
    return list(anchor(code)["codes"])


def groups(code: str = "USA") -> List[str]:
    return list(anchor(code)["groups"])


def goods_mask(code: str = "USA") -> np.ndarray:
    return np.array([g in ("A", "I") for g in groups(code)])


class Economy:
    """One calibrated economy (from its anchor) with a baseline tariff vector."""

    def __init__(self, code: str = "USA", dtype=torch.float64):
        self.code, self.dtype = code, dtype
        self.pkg, self._build, self._solve, self._agg = _import_agora()
        a = anchor(code)
        Z = torch.tensor(a["Z"], dtype=dtype)
        VA = torch.tensor(a["VA"], dtype=dtype).clamp_min(1e-6)
        QX = Z.sum(0) + VA
        self.S = QX.shape[0]
        self.ica = (Z / QX.unsqueeze(0))
        cs = self.ica.sum(0, keepdim=True).clamp_min(1e-6)
        self.ica = self.ica / cs * cs.clamp(max=0.7)
        ex = torch.tensor(a["exports"], dtype=dtype).clamp_min(0)
        im = torch.tensor(a["imports"], dtype=dtype).clamp_min(0)
        self.e = (ex / QX).clamp(0, 0.85)
        self.m = (im / (im + (QX - ex).clamp_min(1e-6))).clamp(0, 0.85)
        g = a["groups"]
        ls = torch.tensor([{"A": 0.60, "I": 0.42, "S": 0.55}[x] for x in g], dtype=dtype)
        self.delta = torch.stack([ls, 1 - ls])                       # (F,S)
        self.QX0 = QX
        el = np.array([_ELAST[x] for x in g])
        self.sigma_q = torch.tensor(el[:, 0], dtype=dtype); self.omega_t = torch.tensor(el[:, 1], dtype=dtype)
        self.sigma_va = torch.tensor(el[:, 2], dtype=dtype); self.income_elast = torch.tensor(el[:, 3], dtype=dtype)
        self.tm_base = torch.tensor([_BASE_TM[x] for x in g], dtype=dtype)
        self.codes, self.groups = list(a["codes"]), list(g)

    def batch(self, tm_list: List[torch.Tensor], te_list: Optional[List[torch.Tensor]] = None):
        E = len(tm_list)
        rep = lambda t: t.unsqueeze(0).expand(E, *t.shape).clone()
        tm = torch.stack(tm_list)
        te = torch.stack(te_list) if te_list is not None else torch.zeros(E, self.S, dtype=self.dtype)
        ta = torch.zeros(E, self.S, dtype=self.dtype)
        ty = torch.full((E,), 0.2, dtype=self.dtype); mps = torch.full((E,), 0.25, dtype=self.dtype)
        gov = torch.full((E,), 0.12, dtype=self.dtype)
        cal = self._build(rep(self.ica), rep(self.QX0), rep(self.e), rep(self.m), rep(self.delta),
                          rep(self.sigma_q), rep(self.omega_t), tm, te, ta, ty, mps, gov,
                          rep(self.sigma_va), rep(self.income_elast))
        return cal, ty, mps

    def _solve_one(self, tm: torch.Tensor, te: torch.Tensor, tol=1e-9):
        """One economy: calibrate at the baseline schedule, overwrite tariffs, Newton; homotopy fallback."""
        cal, ty, mps = self.batch([self.tm_base.clone()], [torch.zeros(self.S, dtype=self.dtype)])
        cal.tm[0] = tm; cal.te[0] = te
        res, resid, iters = self._solve(cal, ty, mps, tol=tol, max_iter=100)
        r = float(torch.as_tensor(resid).max())
        if r > 1e-8:                                              # continuation in the policy, warm-started
            u0, iters = None, 0
            for f in (0.25, 0.5, 0.75, 1.0):
                cal.tm[0] = self.tm_base + f * (tm - self.tm_base); cal.te[0] = f * te
                res, resid, it = self._solve(cal, ty, mps, tol=tol, max_iter=100, u0=u0)
                u0 = torch.cat([res["PD"].log(), res["WF"][:, 1:].log(), res["EXR"].log().unsqueeze(1)], dim=1)
                iters += int(it)
            r = float(torch.as_tensor(resid).max())
        return res, r, int(iters)

    @timed("agora.solve", rows=lambda self, tms, *a, **k: int(len(tms)))
    def solve(self, tms: List[torch.Tensor], tes: List[torch.Tensor]):
        """Solve each (tm, te) economy separately (the batched Newton in this build stalls for E>1;
        singles converge in 3–4 iterations). Returns stacked result dict, per-economy residuals, total iters."""
        outs, resids, iters = [], [], 0
        for tm, te in zip(tms, tes):
            res, r, it = self._solve_one(tm, te)
            outs.append(res); resids.append(r); iters += it
        stacked = {k: torch.cat([o[k] for o in outs], dim=0) for k in outs[0]}
        return stacked, np.array(resids), iters


def sector_response(code: str, shocks: List[Dict[str, float]], te_shocks: Optional[List[Dict[str, float]]] = None,
                    econ: Optional[Economy] = None) -> dict:
    """shocks[i] = {sector_code: delta_tm}. Returns dict with arrays (n_shocks, S) of log changes
    vs. the baseline: dlogQX, dlogPX, dlogREV (PX*QX), dlogPQ; plus macro (real GDP, CPI) changes."""
    ec = econ or Economy(code)
    S = ec.S; idx = {c: i for i, c in enumerate(ec.codes)}
    tms = [ec.tm_base.clone()]
    tes = [torch.zeros(S, dtype=ec.dtype)]
    for k, sh in enumerate(shocks):
        t = ec.tm_base.clone()
        for c, d in sh.items():
            t[idx[c]] = (t[idx[c]] + d).clamp(0.0, 2.0)
        tms.append(t)
        u = torch.zeros(S, dtype=ec.dtype)
        if te_shocks and te_shocks[k]:
            for c, d in te_shocks[k].items():
                u[idx[c]] = float(np.clip(d, 0.0, 0.9))
        tes.append(u)
    res, resid, iters = ec.solve(tms, tes)
    resid = np.asarray(resid)[1:]                               # drop the baseline economy
    QX, PX, PQ = res["QX"], res["PX"], res["PQ"]
    base = 0
    out = {
        "codes": ec.codes, "groups": ec.groups, "residual": resid, "max_residual": float(np.max(resid)), "iters": iters,
        "dlogQX": (QX[1:].log() - QX[base].log()).numpy(),
        "dlogPX": (PX[1:].log() - PX[base].log()).numpy(),
        "dlogREV": ((PX[1:] * QX[1:]).log() - (PX[base] * QX[base]).log()).numpy(),
        "dlogPQ": (PQ[1:].log() - PQ[base].log()).numpy(),
        "dlogGDP": (res["realGDP"][1:].log() - res["realGDP"][base].log()).numpy(),
        "dlogCPI": (res["CPI"][1:].log() - res["CPI"][base].log()).numpy(),
    }
    return out
