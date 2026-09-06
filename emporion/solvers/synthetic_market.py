"""Synthetic market sampler — a distribution of markets with KNOWN investor state, beliefs and
injected mispricing, so the model can be checked against the truth (Agora's collocation trick).

Each sample = one 'date' of one market: N assets, covariance Sigma, true beliefs mu*, true state,
equilibrium price p* (closed form), observed price p_obs = p* * exp(eta) with eta ~ N(0, s_mis)
(the mispricing to be detected), realised payoff D ~ N(mu*, Sigma), and features that carry noisy
information about mu* plus momentum/vol-style columns."""
from __future__ import annotations

import torch

from ..models.investors import InvestorState
from ..models.clearing import closed_form_price


def sample_markets(B: int, N: int, gen=None, dtype=torch.float64, device=None, s_mis: float = 0.05,
                   feat_noise: float = 0.5):
    u = lambda lo, hi, *sh: lo + (hi - lo) * torch.rand(*sh, generator=gen, dtype=dtype, device=device)
    n = lambda *sh: torch.randn(*sh, generator=gen, dtype=dtype, device=device)
    A = n(B, N, N) * 0.03
    Sig = A @ A.transpose(1, 2) + 0.01 * torch.eye(N, dtype=dtype, device=device)
    Rf = torch.full((B,), 1.02, dtype=dtype, device=device)
    s = u(0.5, 1.5, B, N)
    mom = n(B, N) * 0.05
    # true state (mirrors the bands in EmporionV0.state_from)
    st = InvestorState(a=u(1.0, 5.0, B, 3), kappa=u(0.0, 0.5, B), phi=u(0.2, 0.6, B), eps=torch.zeros(B, dtype=dtype, device=device),
                       lev=torch.full((B,), 1e6, dtype=dtype, device=device), wealth=torch.ones(B, 3, dtype=dtype, device=device))
    mu_true = 1.0 + u(0.02, 0.15, B, N)                                   # expected payoff per share
    mkt = {"mu": mu_true, "mom": mom, "Sig_inv": torch.linalg.inv(Sig), "s": s, "pbar": torch.ones(B, N, dtype=dtype, device=device), "Rf": Rf}
    p_star = closed_form_price(mkt, st)
    eta = n(B, N) * s_mis
    p_obs = p_star * torch.exp(eta)
    L = torch.linalg.cholesky(Sig)
    D_real = mu_true + (n(B, N).unsqueeze(1) @ L.transpose(1, 2)).squeeze(1)
    sd = torch.sqrt(torch.diagonal(Sig, dim1=1, dim2=2))
    # features known at t: noisy signal about the premium, momentum, vol, supply, a size-like column
    # fundamentals anchor: a noisy LEVEL signal about expected payoff (earnings/dividend-like), not price-relative
    fund = torch.log(mu_true) + feat_noise * 0.05 * n(B, N)
    feats = torch.stack([fund, mom / 0.05, sd / sd.mean(), s, torch.log(p_obs)], dim=-1)
    G = torch.cat([torch.ones(B, N, 1, dtype=dtype, device=device), feats], dim=-1)   # instruments (B,N,6): 1 + all features
    mkt_obs = {k: v for k, v in mkt.items() if k != "mu"}                       # the model must infer mu
    return {"feats": feats, "p_obs": p_obs, "D_real": D_real, "G": G, "mkt": mkt_obs,
            "truth": {"mu": mu_true, "state": st, "p_star": p_star, "eta": eta}}


def sample_panels(M: int, T: int, N: int, gen=None, dtype=torch.float64, device=None, s_mis: float = 0.05,
                  feat_noise: float = 0.3):
    """M markets observed over T dates each with PERSISTENT beliefs mu*, state and covariance
    (the structure real panels have): per-date realised payoffs D_t ~ N(mu*, Sigma), per-date mispricing
    eta_t, per-date features. Returns flat tensors of B = M*T samples plus `market` ids (B,)."""
    u = lambda lo, hi, *sh: lo + (hi - lo) * torch.rand(*sh, generator=gen, dtype=dtype, device=device)
    n = lambda *sh: torch.randn(*sh, generator=gen, dtype=dtype, device=device)
    A = n(M, N, N) * 0.03
    Sig = A @ A.transpose(1, 2) + 0.01 * torch.eye(N, dtype=dtype, device=device)
    Rf = torch.full((M,), 1.02, dtype=dtype, device=device)
    s = u(0.5, 1.5, M, N)
    st = InvestorState(a=u(1.0, 5.0, M, 3), kappa=torch.zeros(M, dtype=dtype, device=device), phi=u(0.2, 0.6, M),
                       eps=torch.zeros(M, dtype=dtype, device=device), lev=torch.full((M,), 1e6, dtype=dtype, device=device),
                       wealth=torch.ones(M, 3, dtype=dtype, device=device))
    mu_true = 1.0 + u(0.02, 0.15, M, N)
    mkt_m = {"mu": mu_true, "mom": torch.zeros(M, N, dtype=dtype, device=device), "Sig_inv": torch.linalg.inv(Sig), "s": s,
             "pbar": torch.ones(M, N, dtype=dtype, device=device), "Rf": Rf}
    p_star = closed_form_price(mkt_m, st)                                      # (M,N) persistent equilibrium
    rep = lambda x: x.repeat_interleave(T, dim=0)
    market = torch.arange(M, device=device).repeat_interleave(T)
    eta = n(M * T, N) * s_mis
    p_obs = rep(p_star) * torch.exp(eta)
    L = torch.linalg.cholesky(Sig)
    D_real = rep(mu_true) + torch.einsum("bn,bnk->bk", n(M * T, N), rep(L.transpose(1, 2)))
    sd = torch.sqrt(torch.diagonal(Sig, dim1=1, dim2=2))
    fund = rep(torch.log(mu_true)) + feat_noise * 0.05 * n(M * T, N)
    feats = torch.stack([fund, torch.zeros(M * T, N, dtype=dtype, device=device), rep(sd / sd.mean(1, keepdim=True)), rep(s), torch.log(p_obs)], dim=-1)
    G = torch.cat([torch.ones(M * T, N, 1, dtype=dtype, device=device), feats], dim=-1)
    mkt = {"mom": rep(mkt_m["mom"]), "Sig_inv": rep(mkt_m["Sig_inv"]), "s": rep(s), "pbar": rep(mkt_m["pbar"]), "Rf": rep(Rf)}
    truth = {"mu": mu_true, "state": st, "p_star": p_star, "eta": eta, "kappa_struct": (1 - st.phi) / (1.0 / st.a).sum(1),
             "phi": st.phi, "A": (1.0 / st.a).sum(1), "Sig": Sig}
    return {"feats": feats, "p_obs": p_obs, "D_real": D_real, "G": G, "mkt": mkt, "market": market, "M": M, "T": T, "truth": truth}
