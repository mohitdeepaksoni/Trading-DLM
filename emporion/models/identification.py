"""What prices and returns identify in the clearing model — and what they cannot (2026-09-06).

CARA investors with aggregate risk tolerance A holding residual supply q = (1-phi) s, Gaussian payoffs:
    R_f p = mu - Sigma q / A = mu - kappa * Sigma s,        kappa := (1 - phi) / A.
Time-series means of realised payoffs identify mu (per asset); the cross-section of expected excess payoffs
    E[D_i] - R_f p_i = kappa * (Sigma s)_i
identifies the single scalar kappa by regression on (Sigma s). Nothing in prices or returns separates phi from A:
they enter only through kappa. phi is observable from holdings (the index/mandated share of supply), and then
A = (1 - phi)/kappa. Hence: holdings + returns identify the full state; returns alone identify (mu, kappa) only.
"""
from __future__ import annotations

import torch


def estimate_kappa(D_real: torch.Tensor, p_obs: torch.Tensor, Sig: torch.Tensor, s: torch.Tensor, Rf: torch.Tensor,
                   market: torch.Tensor, M: int):
    """Per-market GMM/OLS estimate of kappa from a panel of dates.
    D_real, p_obs (B,N) flat over dates; Sig (M,N,N); s (M,N); Rf (M,); market (B,) ids.
    Returns kappa_hat (M,), mu_hat (M,N)."""
    N = D_real.shape[1]
    mu_hat = torch.zeros(M, N, dtype=D_real.dtype, device=D_real.device).index_add_(0, market, D_real)
    cnt = torch.zeros(M, dtype=D_real.dtype, device=D_real.device).index_add_(0, market, torch.ones_like(market, dtype=D_real.dtype))
    mu_hat = mu_hat / cnt.unsqueeze(1)
    p_bar = torch.zeros(M, N, dtype=D_real.dtype, device=D_real.device).index_add_(0, market, p_obs) / cnt.unsqueeze(1)
    prem = mu_hat - Rf.unsqueeze(1) * p_bar                                   # expected excess payoff per asset (M,N)
    x = torch.einsum("mij,mj->mi", Sig, s)                                     # (Sigma s)_i
    kappa = (prem * x).sum(1) / (x * x).sum(1)                                  # OLS through the origin, one scalar per market
    return kappa, mu_hat


def state_from_kappa(kappa: torch.Tensor, phi_obs: torch.Tensor):
    """Holdings give phi; returns give kappa; together A = (1-phi)/kappa."""
    return (1.0 - phi_obs) / kappa.clamp_min(1e-6)
