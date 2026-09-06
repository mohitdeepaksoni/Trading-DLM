"""Deflated Sharpe Ratio (Bailey & López de Prado 2014) and friends.
Sharpe ratios here are PER PERIOD (not annualised), T = number of periods."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """E[max SR] of n_trials null strategies with Sharpe variance var_sharpe."""
    if n_trials <= 1 or var_sharpe <= 0:
        return 0.0
    n = float(n_trials)
    return np.sqrt(var_sharpe) * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n) + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e)))


def probabilistic_sharpe(sr: float, sr_ref: float, T: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr_ref) * np.sqrt(T - 1) / denom))


def deflated_sharpe(sr: float, T: int, n_trials: int, var_sharpe: float, skew: float = 0.0,
                    kurt: float = 3.0) -> dict:
    """Returns dict with sr_star (expected max under the null) and dsr = P(SR > sr_star)."""
    sr_star = expected_max_sharpe(n_trials, var_sharpe)
    return {"sr": float(sr), "sr_star": float(sr_star), "n_trials": int(n_trials), "T": int(T),
            "dsr": probabilistic_sharpe(sr, sr_star, T, skew, kurt)}


def min_backtest_length(n_trials: int, target_sr_per_period: float, periods: int = 252) -> float:
    """Years of history needed so that a per-period SR is not explained by n_trials of luck."""
    if n_trials <= 1:
        return 0.0
    n = float(n_trials)
    num = ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n) + EULER_GAMMA * norm.ppf(1 - 1 / (n * np.e))) ** 2
    return float(num / target_sr_per_period ** 2 / periods)
