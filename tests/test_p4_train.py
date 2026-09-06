import torch
import pytest

from emporion.models.emporion_v0 import EmporionV0
from emporion.solvers import sample_markets, train_residual, gate

torch.set_default_dtype(torch.float64)


def _corr(a, b):
    a = a.flatten() - a.mean(); b = b.flatten() - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def test_law_at_equilibrium_price_preserves_mispricing_out_of_sample():
    """Finding 2026-09-06 (research/09): with the moment condition imposed at the model's equilibrium price
    (not the observed price), kappa seeded by the structural estimator, and validation early stopping,
    training does not destroy the mispricing signal out of sample and does not worsen validation R1."""
    from emporion.solvers import sample_panels
    from emporion.models import estimate_kappa
    g = torch.Generator().manual_seed(0)
    M, T, N = 128, 200, 8
    P = sample_panels(M=M, T=T, N=N, gen=g); tr = P["truth"]
    t_idx = torch.arange(M * T) % T; trm = t_idx < int(0.6 * T); tem = ~trm
    k_hat, _ = estimate_kappa(P["D_real"][trm], P["p_obs"][trm], tr["Sig"], P["mkt"]["s"][::T], P["mkt"]["Rf"][::T], P["market"][trm], M)
    def sub(m):
        d = {k: P[k][m] for k in ("feats", "p_obs", "D_real", "G")}; d["mkt"] = {k: v[m] for k, v in P["mkt"].items()}
        d["mkt"]["kappa_in"] = k_hat.repeat_interleave(T)[m]; d["mkt"]["phi_obs"] = tr["phi"].repeat_interleave(T)[m]; return d
    TR, TE = sub(trm), sub(tem)
    model = EmporionV0(in_dim=5, d_model=48, n_layers=2, n_heads=4).double()
    with torch.no_grad():
        _, p0, _, o0 = model.loss(TE["feats"], TE["p_obs"], TE["D_real"], TE["G"], TE["mkt"], w_pen=0.0, w_mu=0.0)
    c0 = _corr(o0["r_mkt"], tr["eta"][tem]); assert c0 > 0.9, c0
    train_residual(model, TR, steps=300, batch=512, lr=1e-3, gen=g, log_every=50, verbose=False, balance_every=10 ** 9, val=TE)
    with torch.no_grad():
        _, p1, _, o1 = model.loss(TE["feats"], TE["p_obs"], TE["D_real"], TE["G"], TE["mkt"], w_pen=0.0, w_mu=0.0)
    c1 = _corr(o1["r_mkt"], tr["eta"][tem])
    assert c1 > 0.9, (c0, c1)
    assert float(p1["r1"]) <= float(p0["r1"]) * 1.01               # early stopping guarantees no worse validation law


def test_gate_polish_and_abstention():
    g = torch.Generator().manual_seed(1)
    pool = sample_markets(B=32, N=6, gen=g)
    models = [EmporionV0(in_dim=5, d_model=32, n_layers=1, n_heads=4, n_draws=64, state_mode="bands").double() for _ in range(2)]
    mk = {k: v for k, v in pool["mkt"].items()}
    res = gate(models, pool["feats"], pool["p_obs"], mk)
    assert res["r_self"].max() < 1e-8               # Newton polish reaches tolerance (index elastic in bands)
    assert 0.0 <= res["abstain_rate"] <= 1.0 and res["r_mkt"].shape == (32, 6)
