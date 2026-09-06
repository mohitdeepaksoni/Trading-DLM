"""The pricing kernel (R1), derived from the clearing model's investors (doc 07 A.1).

For CARA investors with aggregate risk tolerance A = sum_h 1/a_h holding the residual supply
q = (1-phi) s, the equilibrium stochastic discount factor is
    M(D) = exp(-(1/A) q.D) / (Rf * E[exp(-(1/A) q.D)])
so that E[M] = 1/Rf and p_i = E[M D_i] for every asset i (the no-arbitrage condition).
`structural_kernel` evaluates M on realised payoff draws; `project_kernel` applies a neural
correction exp(delta) and re-imposes E[M] = 1/Rf as a hard normalisation (a projection, not a
penalty); `moment_residual` is the bank of test-function moments E[M R^e_i g_k] that must vanish.
"""
from __future__ import annotations

import torch


def structural_kernel(D: torch.Tensor, s: torch.Tensor, phi: torch.Tensor, a: torch.Tensor, Rf: torch.Tensor):
    """D (B,K,N) payoff draws; s (B,N); phi (B,); a (B,H). Returns M (B,K) with E_K[M] = 1/Rf."""
    A = (1.0 / a).sum(1)                                                # (B,)
    q = (1 - phi).unsqueeze(1) * s                                      # residual supply held by CARA investors
    expo = -(D * q.unsqueeze(1)).sum(-1) / A.unsqueeze(1)               # (B,K)
    m = torch.exp(expo - expo.max(dim=1, keepdim=True).values)          # stable
    return m / (m.mean(dim=1, keepdim=True) * Rf.unsqueeze(1))


def project_kernel(M_struct: torch.Tensor, delta: torch.Tensor, Rf: torch.Tensor):
    """M = M_struct * exp(delta) renormalised so that E[M] = 1/Rf exactly (per market)."""
    m = M_struct * torch.exp(delta - delta.max(dim=1, keepdim=True).values)
    return m / (m.mean(dim=1, keepdim=True) * Rf.unsqueeze(1))


def moment_residual(M: torch.Tensor, Re: torch.Tensor, G: torch.Tensor):
    """M (B,K); Re (B,K,N) excess returns per draw; G (B,K,N,T) test functions (known at t).
    Returns residuals (B,N,T): E_K[ M * Re_i * g_{i,t} ]."""
    w = (M.unsqueeze(-1) * Re).unsqueeze(-1)                            # (B,K,N,1)
    return (w * G).mean(dim=1)
