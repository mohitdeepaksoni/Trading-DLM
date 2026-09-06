"""Investor types — the heterogeneous agents whose demands clear the asset market (doc 07 A.1/A.2).

One-period market: N risky assets with supply s (shares), price p (numeraire = riskless asset,
gross rate R_f), next-period payoff D ~ (mu_h under investor h's beliefs, covariance Sigma).

Type          demand (shares)                                        latent state
active CARA   x = (1/a) Sigma^{-1} (mu_h - R_f p)                     a (absolute risk aversion), belief tilt
mandated      x = phi * s * (1 - eps*(p/pbar - 1))                    phi (index share of supply), eps ~ 0.2 (inelastic)
leveraged     CARA demand scaled so that p.x <= L * W_h               L (leverage cap), binding => slack = 0
retail        CARA with momentum-tilted beliefs mu_h = mu + kappa*m   kappa (extrapolation)

CARA + Gaussian payoffs gives demands linear in p, so the clearing equilibrium has a closed form
that the Newton solver and the net are tested against — the same discipline Agora applies with
Cobb-Douglas/CES economies of known equilibrium.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InvestorState:
    """Batched investor state, B markets. All tensors float64 by default."""
    a: torch.Tensor        # (B, H)   absolute risk aversion (>0), H = 3 CARA-type investors: active, leveraged, retail
    kappa: torch.Tensor    # (B,)     retail momentum extrapolation (belief tilt per unit of momentum feature)
    phi: torch.Tensor      # (B,)     mandated/index share of supply in [0, 0.9]
    eps: torch.Tensor      # (B,)     index demand elasticity (small, ~0.2)
    lev: torch.Tensor      # (B,)     leverage cap L for the leveraged investor (p.x <= L * W)
    wealth: torch.Tensor   # (B, H)   initial wealth of the three CARA investors (numeraire units)

    @property
    def n_markets(self): return self.a.shape[0]

    def to(self, device=None, dtype=None):
        f = lambda t: t.to(device=device, dtype=dtype)
        return InvestorState(f(self.a), f(self.kappa), f(self.phi), f(self.eps), f(self.lev), f(self.wealth))


def default_state(B: int, dtype=torch.float64, device=None, gen=None) -> InvestorState:
    """Literature-ish draws: risk aversion 1–5, index share 20–60 %, leverage cap 1–3."""
    u = lambda lo, hi, *shape: lo + (hi - lo) * torch.rand(*shape, generator=gen, dtype=dtype, device=device)
    return InvestorState(a=u(1.0, 5.0, B, 3), kappa=u(0.0, 0.5, B), phi=u(0.2, 0.6, B), eps=u(0.1, 0.3, B),
                         lev=u(1.0, 3.0, B), wealth=u(0.5, 2.0, B, 3))
