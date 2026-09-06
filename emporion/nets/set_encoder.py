"""Set encoder: assets are a set, not a vector (doc 07 A.3).

features (B, N, F) -> per-asset embedding (MLP + Fourier features) -> Transformer encoder over
the N assets (permutation-equivariant, variable N, key-padding mask) with an optional additive
attention bias from a sector-adjacency matrix -> heads:
  mu_hat   (B, N)   belief about next-period excess payoff per asset (feeds the clearing model)
  delta    (B,)     log neural correction to the structural kernel (penalised)
  state    (B, S)   investor-state latent, squashed to bands by the caller
  score    (B, N)   conformal non-conformity score (per asset)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int, n_freq: int = 8, scale: float = 1.0):
        super().__init__()
        self.register_buffer("Bmat", torch.randn(in_dim, n_freq) * scale)

    def forward(self, x):
        proj = 2 * math.pi * x @ self.Bmat
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)


class SetEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 3, n_state: int = 8,
                 n_freq: int = 8, dropout: float = 0.0):
        super().__init__()
        self.ff = FourierFeatures(in_dim, n_freq)
        self.in_proj = nn.Sequential(nn.Linear(in_dim + 2 * n_freq, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head_mu = nn.Linear(d_model, 1)
        self.head_score = nn.Linear(d_model, 1)
        self.head_state = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, n_state))
        self.head_delta = nn.Linear(d_model, 1)
        nn.init.zeros_(self.head_mu.weight); nn.init.zeros_(self.head_mu.bias)
        nn.init.zeros_(self.head_delta.weight); nn.init.zeros_(self.head_delta.bias)

    def forward(self, x, mask=None, sector_bias=None):
        """x (B,N,F); mask (B,N) True = padded; sector_bias (N,N) additive attention bias or None."""
        h = self.in_proj(self.ff(x))
        h = self.enc(h, src_key_padding_mask=mask, mask=sector_bias)
        h = self.norm(h)
        pooled = h.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(1, keepdim=True) if mask is not None else h.mean(1)
        return {"mu_hat": self.head_mu(h).squeeze(-1), "score": nn.functional.softplus(self.head_score(h).squeeze(-1)),
                "state": self.head_state(pooled), "delta_global": self.head_delta(pooled).squeeze(-1), "tokens": h}
