"""Asset embeddings from holdings (Gabaix, Koijen, Richmond, Yogo 2025) — v0.

Input: a holdings panel H (n_assets x n_investors) of portfolio shares (or ownership fractions) at one date.
Method (v0, linear): X = log(H + eps) centred by asset and investor; truncated SVD; asset embedding = U_k S_k^{1/2},
investor embedding = V_k S_k^{1/2}.  The paper shows holdings recover pricing-relevant latent characteristics and
that similar assets (in embedding space) have similar valuations and comove.  Uses:
  * exposure map: cosine similarity in embedding space (replaces the hand-built sector map)
  * peer-implied valuation rung: an asset's fair earnings yield = similarity-weighted average of its neighbours'
"""
from __future__ import annotations

import numpy as np


def asset_embeddings(H: np.ndarray, k: int = 8, eps: float = 1e-4):
    """H (A, I) non-negative holdings. Returns (asset_emb (A,k), investor_emb (I,k), explained_var (k,))."""
    X = np.log(np.maximum(H, 0) + eps)
    X = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    k = min(k, S.shape[0])
    ev = S[:k] ** 2 / (S ** 2).sum()
    return U[:, :k] * np.sqrt(S[:k]), Vt[:k].T * np.sqrt(S[:k]), ev


def similarity(E: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    return (E / n) @ (E / n).T


def peer_implied_yield(E: np.ndarray, ey: np.ndarray, n_peers: int = 10):
    """For each asset, the similarity-weighted mean earnings yield of its n nearest peers (excluding itself);
    the peer-implied mispricing signal is log(peer_ey / own_ey): higher = cheaper than peers."""
    S = similarity(E); np.fill_diagonal(S, -np.inf)
    out = np.full(len(ey), np.nan)
    for i in range(len(ey)):
        ok = np.isfinite(ey); ok[i] = False
        idx = np.argsort(-S[i] * ok)[:n_peers]
        w = np.maximum(S[i, idx], 0); w = w * ok[idx]
        if w.sum() > 0 and np.isfinite(ey[i]) and ey[i] > 0:
            out[i] = np.log((w @ ey[idx]) / w.sum() / ey[i]) if (w @ ey[idx]) > 0 else np.nan
    return out
