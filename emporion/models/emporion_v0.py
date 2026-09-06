"""Emporion v0 — the equilibrium-anchored model (doc 04 §2, doc 07 A).

forward(features, p_obs, mkt) :
  encoder -> beliefs mu_hat (per asset), investor state (bands), kernel-correction loadings
  p_hat  = clearing price implied by (mu_hat, state)         [closed form; Newton polish at inference]
  r_mkt  = log(p_obs / p_hat)                                 the mispricing signal
  M      = structural kernel at the REALISED payoff, corrected and projected to E[M] = 1/Rf
loss():
  R1  : moment residuals  mean_b[ M_b * Re_{b,i} * g_{b,i,k} ] over a batch of dates (GMM-style), MSE + LogSumExp tail
  R2s : soft market-consistency  mean r_mkt^2  (observed prices should be near the model's equilibrium on average)
  pen : ||delta loadings||^2
No P&L anywhere.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..nets.set_encoder import SetEncoder
from .investors import InvestorState
from .clearing import closed_form_price, solve_clearing
from .kernel import moment_residual


def _band(x, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(x)


class EmporionV0(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, n_layers: int = 3, n_heads: int = 4, n_draws: int = 256,
                 mu_mode: str = "level", fund_col: int = 0, state_mode: str = "kappa"):
        super().__init__()
        self.mu_mode, self.fund_col, self.state_mode = mu_mode, fund_col, state_mode
        self.enc = SetEncoder(in_dim, d_model=d_model, n_heads=n_heads, n_layers=n_layers, n_state=6)
        self.head_dload = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head_dload.weight); nn.init.zeros_(self.head_dload.bias)
        self.n_draws = n_draws

    def state_from(self, z):                                   # z (B,6) -> InvestorState in bands
        B = z.shape[0]
        return InvestorState(a=_band(z[:, 0:3], 1.0, 5.0), kappa=_band(z[:, 3], 0.0, 0.5), phi=_band(z[:, 4], 0.2, 0.6),
                             eps=_band(z[:, 5], 0.1, 0.3), lev=torch.full((B,), 1e6, dtype=z.dtype, device=z.device),
                             wealth=torch.ones(B, 3, dtype=z.dtype, device=z.device))

    def forward(self, feats, p_obs, mkt, mask=None):
        out = self.enc(feats, mask)
        # beliefs: expected payoff relative to the riskless-grown observed price; zero-init => mu = Rf p (no premium)
        if self.mu_mode == "level":      # beliefs anchored to the fundamental-level feature, corrected by the net
            mu_hat = torch.exp(feats[..., self.fund_col] + out["mu_hat"])
        else:                            # price-anchored (kept for the ablation; loses identification of mispricing)
            mu_hat = mkt["Rf"].unsqueeze(1) * p_obs * torch.exp(out["mu_hat"])
        m = dict(mkt); m["mu"] = mu_hat
        if self.state_mode == "kappa":
            # identified parametrisation: kappa = (1-phi)/A seeded by the structural estimator (mkt["kappa_in"]),
            # refined by a penalised log-correction; p_hat = (mu - kappa * Sigma s)/Rf  (A, phi never needed for pricing)
            kappa = mkt["kappa_in"] * torch.exp(0.5 * torch.tanh(out["state"][:, 0]))
            Sig = torch.linalg.inv(mkt["Sig_inv"])
            p_hat = (mu_hat - kappa.unsqueeze(1) * torch.einsum("bij,bj->bi", Sig, mkt["s"])) / mkt["Rf"].unsqueeze(1)
            phi = mkt.get("phi_obs", torch.full_like(kappa, 0.4))
            A = (1 - phi) / kappa
            st = InvestorState(a=(3.0 / A).unsqueeze(1).expand(-1, 3).clone(), kappa=torch.zeros_like(kappa), phi=phi,
                               eps=torch.zeros_like(kappa), lev=torch.full_like(kappa, 1e6), wealth=torch.ones(kappa.shape[0], 3, dtype=kappa.dtype, device=kappa.device))
            st_corr = out["state"][:, 0]
        else:
            st = self.state_from(out["state"]); st_corr = torch.zeros(mu_hat.shape[0], dtype=mu_hat.dtype, device=mu_hat.device)
            p_hat = closed_form_price(m, st)
        r_mkt = torch.log(p_obs.clamp_min(1e-9) / p_hat.clamp_min(1e-9))
        dload = 0.1 * torch.tanh(self.head_dload(out["tokens"]).squeeze(-1))   # (B,N) kernel-correction loadings
        return {"mu_hat": mu_hat, "mu_corr": out["mu_hat"], "state": st, "state_corr": st_corr, "p_hat": p_hat, "r_mkt": r_mkt, "dload": dload, "mask": mask,
                "score": out["score"], "mkt": m}

    def kernel_at(self, D_real, out, mkt, gen=None):
        """Kernel at the realised payoff, normalised ANALYTICALLY under the model's Gaussian beliefs
        (no Monte-Carlo noise): M = exp(w.D) / (Rf * E[exp(w.D)]),  E[exp(w.D)] = exp(w.mu + 0.5 w'Sigma w),
        with w = -q/A + dload/sd  (structural CARA term + penalised neural correction)."""
        st, mu = out["state"], out["mu_hat"]
        A = (1.0 / st.a).sum(1)                                                   # (B,)
        q = (1 - st.phi).unsqueeze(1) * mkt["s"]                                 # (B,N)
        Sig = torch.linalg.inv(mkt["Sig_inv"])
        sd = torch.sqrt(torch.diagonal(Sig, dim1=1, dim2=2))
        w = -q / A.unsqueeze(1) + out["dload"] / sd                              # (B,N)
        if out.get("mask") is not None:
            w = w.masked_fill(out["mask"], 0.0)                                   # padded assets carry no kernel weight
        log_norm = (w * mu).sum(1) + 0.5 * torch.einsum("bi,bij,bj->b", w, Sig, w)
        return torch.exp((w * D_real).sum(1) - log_norm) / mkt["Rf"]

    def loss(self, feats, p_obs, D_real, G, mkt, mask=None, w_r2=0.0, w_pen=1.0, w_mu=1.0, tau=20.0, gen=None,
             law_on: str = "p_hat"):
        """G (B,N,T) test functions known at t. Returns total, parts, per-date residual weights (for RAD)."""
        out = self.forward(feats, p_obs, mkt, mask)
        M = self.kernel_at(D_real, out, mkt, gen)                                # (B,)
        # THE LAW HOLDS AT THE EQUILIBRIUM PRICE, NOT THE OBSERVED ONE (finding 2026-09-06): measured against p_obs,
        # E[M (D/p_obs - Rf)] = 0 is the statement that there is no mispricing, and minimising it teaches the model that
        # the market is always right. Measured against p_hat it says beliefs and kappa must be consistent with realised
        # payoffs, and leaves the deviation r_mkt = log(p_obs/p_hat) as a separate, tradeable object.
        base = out["p_hat"] if law_on == "p_hat" else p_obs
        Re = D_real / base.detach().clamp_min(1e-9) if law_on == "p_obs" else D_real / base.clamp_min(1e-9)
        Re = Re - mkt["Rf"].unsqueeze(1)                                        # realised excess payoff return (B,N)
        Gm = G if mask is None else G * (~mask).unsqueeze(-1)
        mom = (M[:, None, None] * Re[:, :, None] * Gm).mean(0)                  # (N,T): GMM average over the batch of dates
        r1 = (mom ** 2) * Re.shape[0]                                            # GMM scaling: B * gbar^2 has an O(1) noise floor
        L_r1 = r1.mean() + (1.0 / tau) * torch.logsumexp(tau * r1.flatten(), 0) / r1.numel()
        rm = out["r_mkt"] if mask is None else out["r_mkt"].masked_fill(mask, 0.0)
        L_r2 = (rm ** 2).mean()
        L_pen = (out["dload"] ** 2).mean() + w_mu * ((out["mu_corr"] ** 2).mean() + (out["state_corr"] ** 2).mean())   # corrections stay near zero unless the law demands
        per_date = (M[:, None] * Re * Gm[..., 0]).abs().mean(1).detach()         # residual contribution per date (RAD)
        total = L_r1 + w_r2 * L_r2 + w_pen * L_pen
        return total, {"r1": L_r1, "r2": L_r2, "pen": L_pen, "mom_max": r1.max().sqrt().detach()}, per_date, out

    @torch.no_grad()
    def polish(self, out, mkt, mask=None, tol=1e-10):
        """Newton polish of the clearing condition with the net's state/beliefs (exact when the index is elastic)."""
        m = dict(mkt); m["mu"] = out["mu_hat"]
        p, r_self, it = solve_clearing(m, out["state"], p0=out["p_hat"], tol=tol)
        return p, r_self, it
