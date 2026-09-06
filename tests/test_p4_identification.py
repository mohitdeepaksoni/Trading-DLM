"""Pins the identification result of research/09-IDENTIFICATION.md numerically."""
import torch

from emporion.solvers import sample_panels
from emporion.models import estimate_kappa, state_from_kappa

torch.set_default_dtype(torch.float64)


def _corr(a, b):
    a = a.flatten() - a.mean(); b = b.flatten() - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def test_returns_identify_mu_and_kappa_but_not_A_or_phi():
    g = torch.Generator().manual_seed(0)
    M, T, N = 200, 2000, 8
    P = sample_panels(M=M, T=T, N=N, gen=g); tr = P["truth"]
    k_hat, mu_hat = estimate_kappa(P["D_real"], P["p_obs"], tr["Sig"], P["mkt"]["s"][::T], P["mkt"]["Rf"][::T], P["market"], M)
    assert _corr(mu_hat, tr["mu"]) > 0.99                        # beliefs: identified by time-series means
    assert _corr(k_hat, tr["kappa_struct"]) > 0.85               # kappa = (1-phi)/A: identified by the cross-section
    # phi and A are not separately identified from returns: kappa carries no information about phi beyond kappa itself
    resid = k_hat - tr["kappa_struct"]
    assert abs(_corr(resid, tr["phi"])) < 0.15                    # estimation error uncorrelated with phi
    # with phi from holdings, A = (1-phi)/kappa is recoverable (up to kappa noise); without it, a guessed phi does no better
    A_h = state_from_kappa(k_hat, tr["phi"]); A_g = (1 - 0.4) / k_hat
    assert _corr(A_h, tr["A"]) > _corr(A_g, tr["A"]) - 0.1


def test_kappa_convergence_is_slow():
    """One year of daily data pins kappa only loosely (~0.6); four years ~0.85 — kappa is slow-layer state."""
    g = torch.Generator().manual_seed(1)
    out = {}
    for T in (250, 1000):
        P = sample_panels(M=150, T=T, N=8, gen=g); tr = P["truth"]
        k_hat, _ = estimate_kappa(P["D_real"], P["p_obs"], tr["Sig"], P["mkt"]["s"][::T], P["mkt"]["Rf"][::T], P["market"], 150)
        out[T] = _corr(k_hat, tr["kappa_struct"])
    assert 0.4 < out[250] < 0.8 and out[1000] > out[250] + 0.1, out
