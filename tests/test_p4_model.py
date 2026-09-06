import torch
import pytest

from emporion.models import default_state, excess_demand, cash_excess_demand, solve_clearing, closed_form_price
from emporion.models.kernel import structural_kernel, project_kernel, moment_residual
from emporion.models.investors import InvestorState
from emporion.nets import SetEncoder

torch.set_default_dtype(torch.float64)


def _market(B=4, N=6, seed=0, eps_zero=False, lev_big=True):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(B, N, N, generator=g) * 0.05
    Sig = A @ A.transpose(1, 2) + 0.02 * torch.eye(N)
    mkt = {"mu": 1.0 + 0.1 * torch.rand(B, N, generator=g), "mom": torch.randn(B, N, generator=g) * 0.05,
           "Sig_inv": torch.linalg.inv(Sig), "s": 0.5 + torch.rand(B, N, generator=g), "pbar": torch.ones(B, N),
           "Rf": torch.full((B,), 1.02)}
    st = default_state(B, gen=g)
    if eps_zero:
        st.eps = torch.zeros(B)
    if lev_big:
        st.lev = torch.full((B,), 1e6)
    return mkt, st, Sig


def test_closed_form_is_an_equilibrium_and_newton_agrees():
    mkt, st, _ = _market(eps_zero=True)
    p_cf = closed_form_price(mkt, st)
    z = excess_demand(p_cf, mkt, st)
    assert z.abs().max() < 1e-9                       # closed form clears the market exactly
    mkt2, st2, _ = _market(eps_zero=False)            # inelastic index with elasticity -> needs Newton
    p, r, it = solve_clearing(mkt2, st2)
    assert r.max() < 1e-9 and it <= 10


def test_walras_law_with_numeraire():
    mkt, st, _ = _market()
    B, N = mkt["s"].shape
    p = 0.8 + 0.4 * torch.rand(B, N)                  # arbitrary, off-equilibrium prices
    endow = torch.rand(B, 4, N); endow = endow / endow.sum(1, keepdim=True) * mkt["s"].unsqueeze(1)   # shares sum to supply
    cash = torch.rand(B, 4)
    zc = cash_excess_demand(p, mkt, st, endow, cash)
    z = excess_demand(p, mkt, st)
    assert torch.allclose(zc, -(p * z).sum(1), atol=1e-9)   # cash excess demand = -p.z(p)


def test_scale_invariance_of_demands():
    """Scaling payoffs, prices and covariance by k with risk aversion a/k leaves share demands unchanged."""
    mkt, st, Sig = _market(eps_zero=True)
    p = closed_form_price(mkt, st); z0 = excess_demand(p, mkt, st)
    k = 3.0
    mkt2 = dict(mkt); mkt2["mu"] = k * mkt["mu"]; mkt2["mom"] = k * mkt["mom"]; mkt2["Sig_inv"] = mkt["Sig_inv"] / k ** 2
    mkt2["pbar"] = k * mkt["pbar"]
    st2 = InvestorState(st.a / k, st.kappa, st.phi, st.eps, st.lev, st.wealth * k)
    z1 = excess_demand(k * p, mkt2, st2)
    assert torch.allclose(z0, z1, atol=1e-9)


def test_structural_kernel_prices_the_equilibrium():
    """R1/R2 consistency: the kernel derived from the investors prices the clearing price exactly (Gaussian payoffs)."""
    mkt, st, Sig = _market(B=2, N=4, eps_zero=True)
    st.kappa = torch.zeros(2)                         # homogeneous beliefs -> representative CARA
    p = closed_form_price(mkt, st)
    g = torch.Generator().manual_seed(1)
    K = 200_000
    L = torch.linalg.cholesky(Sig)
    D = mkt["mu"].unsqueeze(1) + (torch.randn(2, K, 4, generator=g) @ L.transpose(1, 2))
    M = structural_kernel(D, mkt["s"], st.phi, st.a, mkt["Rf"])
    assert torch.allclose(M.mean(1), 1 / mkt["Rf"], atol=1e-12)
    p_implied = (M.unsqueeze(-1) * D).mean(1)
    assert (p_implied - p).abs().max() < 0.01          # Monte-Carlo error only


def test_projection_and_moments():
    mkt, st, Sig = _market(B=2, N=4, eps_zero=True)
    D = mkt["mu"].unsqueeze(1) + 0.1 * torch.randn(2, 1000, 4)
    M = structural_kernel(D, mkt["s"], st.phi, st.a, mkt["Rf"])
    Mp = project_kernel(M, torch.randn(2, 1000) * 0.1, mkt["Rf"])
    assert torch.allclose(Mp.mean(1), 1 / mkt["Rf"], atol=1e-12)
    Re = D / closed_form_price(mkt, st).unsqueeze(1) - mkt["Rf"].view(2, 1, 1)
    G = torch.ones(2, 1000, 4, 1)
    r = moment_residual(M, Re, G)
    assert r.shape == (2, 4, 1)


def test_set_encoder_is_permutation_equivariant_and_handles_padding():
    torch.manual_seed(0)
    enc = SetEncoder(in_dim=5, d_model=32, n_heads=4, n_layers=2).double()
    x = torch.randn(2, 7, 5); mask = torch.zeros(2, 7, dtype=torch.bool); mask[1, 5:] = True
    out = enc(x, mask)
    perm = torch.randperm(5)
    x2 = x.clone(); x2[1, :5] = x[1, perm]
    out2 = enc(x2, mask)
    assert torch.allclose(out["mu_hat"][1, perm], out2["mu_hat"][1, :5], atol=1e-8)   # equivariant
    assert torch.allclose(out["state"][1], out2["state"][1], atol=1e-8)                 # invariant pooled state
    assert torch.isfinite(out["score"]).all() and out["score"].min() >= 0
