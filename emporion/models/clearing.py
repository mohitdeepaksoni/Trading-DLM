"""Market clearing (R2): z(p) = sum_h x_h(p) - s, its Walras identity, and its solution.

Written for ONE market (functions take un-batched tensors) so they compose under vmap and
jacrev exactly like Agora's `excess_demand_single`; the batched API wraps them.
"""
from __future__ import annotations

import torch
from torch.func import vmap, jacrev


def _cara_demand(a: torch.Tensor, mu: torch.Tensor, Sig_inv: torch.Tensor, p: torch.Tensor, Rf: torch.Tensor):
    return (Sig_inv @ (mu - Rf * p)) / a


def demands_single(p, mu, mom, Sig_inv, s, pbar, Rf, a, kappa, phi, eps, lev, wealth):
    """Per-investor share demands for one market. Returns (H+1, N): [active, leveraged, retail, index]."""
    x_act = _cara_demand(a[0], mu, Sig_inv, p, Rf)
    x_lev_star = _cara_demand(a[1], mu, Sig_inv, p, Rf)
    cost = (p * x_lev_star).sum().abs().clamp_min(1e-12)
    scale = torch.clamp(lev * wealth[1] / cost, max=1.0)              # binding cap scales the position
    x_lev = x_lev_star * scale
    x_ret = _cara_demand(a[2], mu + kappa * mom, Sig_inv, p, Rf)       # extrapolative beliefs
    x_idx = phi * s * (1.0 - eps * (p / pbar - 1.0))                    # inelastic mandate
    return torch.stack([x_act, x_lev, x_ret, x_idx])


def excess_demand_single(p, mu, mom, Sig_inv, s, pbar, Rf, a, kappa, phi, eps, lev, wealth):
    return demands_single(p, mu, mom, Sig_inv, s, pbar, Rf, a, kappa, phi, eps, lev, wealth).sum(0) - s


def cash_excess_demand_single(p, mu, mom, Sig_inv, s, pbar, Rf, a, kappa, phi, eps, lev, wealth, endow, cash_supply):
    """Walras' law with a numeraire: investors fund risky demand from wealth = cash endowment +
    value of share endowment; excess demand for cash = sum_h (W_h - p.x_h) - cash_supply = -p.z(p)."""
    X = demands_single(p, mu, mom, Sig_inv, s, pbar, Rf, a, kappa, phi, eps, lev, wealth)
    W = (endow * p.unsqueeze(0)).sum(1) + cash_supply                  # endow (H+1, N) shares, cash_supply (H+1,)
    return (W - (X * p.unsqueeze(0)).sum(1)).sum() - cash_supply.sum()


_ARGS = ("mu", "mom", "Sig_inv", "s", "pbar", "Rf", "a", "kappa", "phi", "eps", "lev", "wealth")


def excess_demand(p, mkt: dict, st) -> torch.Tensor:
    """Batched: p (B,N) -> z (B,N). mkt holds mu, mom, Sig_inv, s, pbar, Rf (batched); st = InvestorState."""
    return vmap(excess_demand_single)(p, mkt["mu"], mkt["mom"], mkt["Sig_inv"], mkt["s"], mkt["pbar"], mkt["Rf"],
                                      st.a, st.kappa, st.phi, st.eps, st.lev, st.wealth)


def cash_excess_demand(p, mkt, st, endow, cash_supply):
    return vmap(cash_excess_demand_single)(p, mkt["mu"], mkt["mom"], mkt["Sig_inv"], mkt["s"], mkt["pbar"], mkt["Rf"],
                                           st.a, st.kappa, st.phi, st.eps, st.lev, st.wealth, endow, cash_supply)


def closed_form_price(mkt, st) -> torch.Tensor:
    """Exact equilibrium when the leverage cap does not bind and the index is perfectly inelastic
    (eps = 0): sum_h (1/a_h) Sigma^{-1}(mu_h - Rf p) = (1 - phi) s  =>
    p = [ (sum_h mu_h/a_h)/A - Sigma (1-phi) s / A ] / Rf,   A = sum_h 1/a_h  (CARA aggregation)."""
    inv_a = 1.0 / st.a                                                  # (B,3)
    A = inv_a.sum(1)                                                    # (B,)
    mu_ret = mkt["mu"] + st.kappa.unsqueeze(1) * mkt["mom"]
    mu_bar = (inv_a[:, 0:1] * mkt["mu"] + inv_a[:, 1:2] * mkt["mu"] + inv_a[:, 2:3] * mu_ret) / A.unsqueeze(1)
    Sig = torch.linalg.inv(mkt["Sig_inv"])
    resid_supply = ((1 - st.phi).unsqueeze(1) * mkt["s"]).unsqueeze(-1)
    return (mu_bar - (Sig @ resid_supply).squeeze(-1) / A.unsqueeze(1)) / mkt["Rf"].unsqueeze(1)


def solve_clearing(mkt, st, p0=None, tol=1e-10, max_iter=50, damping=1e-10):
    """Batched damped Newton on z(p)=0 with per-market backtracking (Agora's recipe).
    Returns (p*, |z| inf-norm per market, iters)."""
    p = (closed_form_price(mkt, st) if p0 is None else p0).clone()
    args = tuple(mkt[k] if k in mkt else getattr(st, k) for k in _ARGS)
    z_fn = vmap(excess_demand_single); J_fn = vmap(jacrev(excess_demand_single, argnums=0))
    B, N = p.shape
    eye = torch.eye(N, dtype=p.dtype, device=p.device).expand(B, N, N)
    it = 0
    for it in range(1, max_iter + 1):
        z = z_fn(p, *args)
        if z.abs().amax(1).max() < tol:
            break
        J = J_fn(p, *args)
        step = -torch.linalg.solve(J + damping * eye, z.unsqueeze(-1)).squeeze(-1)
        c0 = (z ** 2).sum(1); alpha = torch.ones(B, 1, dtype=p.dtype, device=p.device); p_new = p.clone()
        acc = torch.zeros(B, dtype=torch.bool, device=p.device)
        for _ in range(20):
            trial = p + alpha * step
            ct = (z_fn(trial, *args) ** 2).sum(1)
            take = (ct < c0) & ~acc
            p_new = torch.where(take.unsqueeze(1), trial, p_new); acc |= (ct < c0)
            if bool(acc.all()):
                break
            alpha = torch.where((ct < c0).unsqueeze(1), alpha, alpha * 0.5)
        p = p_new
    return p, z_fn(p, *args).abs().amax(1), it
