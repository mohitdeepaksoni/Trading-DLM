"""Residual trainer — self-supervised on the law, with the tail machinery from Agora
(dl-equilibrium-tail-methods): residual-adaptive resampling (RAD) over a candidate pool,
self-adaptive per-date weights, LogSumExp tail term (inside the model loss), gradient-magnitude
balancing between R1 and the soft R2 term. No P&L."""
from __future__ import annotations

import time

import torch

from ..core.perf import timed


def _sub(d, idx):
    if isinstance(d, dict):
        return {k: _sub(v, idx) for k, v in d.items()}
    if hasattr(d, "a"):                                   # InvestorState
        from ..models.investors import InvestorState
        return InvestorState(*[getattr(d, f)[idx] for f in ("a", "kappa", "phi", "eps", "lev", "wealth")])
    return d[idx]


@timed("p4.train_residual", rows=lambda model, pool, steps=200, **k: int(steps))
def train_residual(model, pool: dict, steps: int = 200, batch: int = 64, lr: float = 1e-3, rad_k: float = 1.0,
                   rad_c: float = 0.5, balance_every: int = 25, log_every: int = 50, gen=None, verbose=True, w_pen: float = 1.0, w_mu: float = 1.0,
                   val: dict = None, patience: int = 3, w_r2_init: float = 0.0):
    """pool: a dict from sample_markets (a large candidate pool of dates). Returns history list."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    P = pool["p_obs"].shape[0]
    resid_est = torch.ones(P, dtype=pool["p_obs"].dtype, device=pool["p_obs"].device)
    w_r2 = w_r2_init
    hist, t0 = [], time.perf_counter()
    best, best_state, bad = float("inf"), None, 0
    def _val():
        with torch.no_grad():
            _, pv, _, _ = model.loss(val["feats"], val["p_obs"], val["D_real"], val["G"], val["mkt"], w_pen=0.0, w_mu=0.0, gen=gen)
        return float(pv["r1"])
    if val is not None:
        best = _val(); best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for step in range(1, steps + 1):
        # RAD: draw the batch from density ∝ r^k / E[r^k] + c  (Wu et al. 2023)
        prob = resid_est.pow(rad_k); prob = prob / prob.mean() + rad_c
        idx = torch.multinomial(prob / prob.sum(), batch, replacement=True, generator=gen)
        sub = _sub({k: pool[k] for k in ("feats", "p_obs", "D_real", "G", "mkt")}, idx)
        total, parts, per_date, out = model.loss(sub["feats"], sub["p_obs"], sub["D_real"], sub["G"], sub["mkt"], w_r2=w_r2, w_pen=w_pen, w_mu=w_mu, gen=gen)
        opt.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        resid_est[idx] = 0.7 * resid_est[idx] + 0.3 * per_date                  # self-adaptive per-date estimate
        if step % balance_every == 0:                                             # gradient-magnitude balancing (Wang et al. 2021)
            _, pb, _, _ = model.loss(sub["feats"], sub["p_obs"], sub["D_real"], sub["G"], sub["mkt"], w_r2=1.0, gen=gen)
            params = [q for q in model.parameters() if q.requires_grad]
            g1 = torch.autograd.grad(pb["r1"], params, allow_unused=True, retain_graph=True)
            g2 = torch.autograd.grad(pb["r2"], params, allow_unused=True)
            n1 = torch.sqrt(sum((g ** 2).sum() for g in g1 if g is not None)).clamp_min(1e-12)
            n2 = torch.sqrt(sum((g ** 2).sum() for g in g2 if g is not None)).clamp_min(1e-12)
            w_r2 = float(0.9 * w_r2 + 0.1 * float((n1 / n2).clamp(1e-3, 1e3)))
        if val is not None and (step % log_every == 0 or step == steps):     # early stopping on validation R1
            v = _val()
            if v < best:
                best, bad = v, 0; best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    model.load_state_dict(best_state)
                    if verbose:
                        print(f"[p4] early stop at step {step}: best val R1={best:.4e}", flush=True)
                    break
        if step % log_every == 0 or step == steps:
            rec = {"step": step, "total": float(total.detach()), "r1": float(parts["r1"].detach()), "r2": float(parts["r2"].detach()),
                   "mom_max": float(parts["mom_max"]), "w_r2": w_r2, "wall_s": time.perf_counter() - t0}
            hist.append(rec)
            if verbose:
                print(f"[p4] step {step:5d} total={rec['total']:.3e} R1={rec['r1']:.3e} R2soft={rec['r2']:.3e} "
                      f"mom_max={rec['mom_max']:.3e} w_r2={w_r2:.2f} {rec['wall_s']:.0f}s", flush=True)
    if val is not None and best_state is not None:
        model.load_state_dict(best_state)                                          # always return the best validated weights
    return hist
