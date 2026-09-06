"""Regime detection and conformal intervals (doc 01 §3.5, doc 04 §2.5, doc 06 §4).

  GaussianHMM  : K-state hidden Markov model on a return series (Hamilton 1989), EM in numpy,
                 with *filtered* probabilities P(s_t | r_{1:t}) — no smoothing, so no look-ahead.
  conformal_quantile : the simple rolling-quantile split-conformal interval that won in the
                 2026 "Conformal Kelly" study — width from the empirical |residual| quantile over
                 the last `window` observations, updated daily.
"""
from __future__ import annotations

import numpy as np

from ..core.perf import timed


def _norm_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


class GaussianHMM:
    def __init__(self, n_states: int = 2, n_iter: int = 100, tol: float = 1e-6, seed: int = 0):
        self.K, self.n_iter, self.tol, self.seed = n_states, n_iter, tol, seed
        self.A = self.pi = self.mu = self.sd = None

    def _init(self, r):
        rng = np.random.default_rng(self.seed)
        q = np.quantile(np.abs(r), np.linspace(0.2, 0.8, self.K))
        self.sd = np.sort(q) + 1e-8
        self.mu = np.zeros(self.K) + rng.normal(0, 1e-4, self.K)
        self.A = np.full((self.K, self.K), 0.05 / (self.K - 1)) + np.eye(self.K) * (0.95 - 0.05 / (self.K - 1))
        self.pi = np.full(self.K, 1 / self.K)

    def _forward_backward(self, r):
        T = r.shape[0]
        B = np.stack([_norm_pdf(r, self.mu[k], self.sd[k]) for k in range(self.K)], axis=1) + 1e-300
        alpha = np.zeros((T, self.K)); c = np.zeros(T)
        alpha[0] = self.pi * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]; c[t] = alpha[t].sum(); alpha[t] /= c[t]
        beta = np.ones((T, self.K))
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta; gamma /= gamma.sum(1, keepdims=True)
        xi = np.zeros((self.K, self.K))
        for t in range(T - 1):
            x = alpha[t][:, None] * self.A * (B[t + 1] * beta[t + 1])[None, :] / c[t + 1]
            xi += x
        return alpha, gamma, xi, np.log(c).sum()

    @timed("hmm.fit", rows=lambda *a, **k: int(len(a[1])))
    def fit(self, r: np.ndarray):
        r = np.asarray(r, float); self._init(r); ll_old = -np.inf
        for _ in range(self.n_iter):
            _, g, xi, ll = self._forward_backward(r)
            self.pi = g[0] / g[0].sum()
            self.A = xi / xi.sum(1, keepdims=True)
            w = g.sum(0)
            self.mu = (g * r[:, None]).sum(0) / w
            self.sd = np.sqrt((g * (r[:, None] - self.mu) ** 2).sum(0) / w) + 1e-8
            if ll - ll_old < self.tol:
                break
            ll_old = ll
        order = np.argsort(self.sd)                      # state 0 = calmest
        self.mu, self.sd = self.mu[order], self.sd[order]
        self.A = self.A[np.ix_(order, order)]; self.pi = self.pi[order]
        return self

    def filter(self, r: np.ndarray) -> np.ndarray:
        """Filtered P(s_t | r_{1:t}) — uses only the past; safe for decisions at t."""
        alpha, _, _, _ = self._forward_backward(np.asarray(r, float))
        return alpha

    def rolling_filter(self, r: np.ndarray, refit_every: int = 63, min_train: int = 500) -> np.ndarray:
        """Walk-forward: refit on r[:t] every `refit_every` days, filter forward with current params."""
        r = np.asarray(r, float); T = r.shape[0]
        P = np.full((T, self.K), np.nan)
        for t0 in range(min_train, T, refit_every):
            self.fit(r[:t0])
            P[t0:t0 + refit_every] = self.filter(r[:min(T, t0 + refit_every)])[t0:t0 + refit_every]
        return P


def conformal_quantile(residuals: np.ndarray, alpha: float = 0.1, window: int = 250) -> np.ndarray:
    """Rolling split-conformal half-width q_t = quantile_{1-alpha}(|resid| over the previous `window`)
    (finite-sample corrected), aligned so q_t uses residuals strictly before t."""
    e = np.abs(np.asarray(residuals, float))
    T = e.shape[0]
    q = np.full(T, np.nan)
    for t in range(window, T):
        w = e[t - window:t]
        k = int(np.ceil((window + 1) * (1 - alpha)))
        q[t] = np.sort(w)[min(k, window) - 1]
    return q


def coverage(actual_abs_resid: np.ndarray, q: np.ndarray) -> float:
    ok = np.isfinite(q)
    return float((np.abs(actual_abs_resid[ok]) <= q[ok]).mean())
