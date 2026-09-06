from math import comb
import numpy as np
import pytest

from emporion.evaluate import (sharpe, max_drawdown, portfolio_returns, cpcv_splits, cpcv_paths, n_paths,
                               expected_max_sharpe, deflated_sharpe, min_backtest_length, pbo_cscv,
                               bootstrap_ci)
from emporion.evaluate.metrics import skew_kurt


def test_metrics_vectorised_and_sane():
    rng = np.random.default_rng(0)
    R = rng.normal(0.0005, 0.01, (2520, 4))
    s = sharpe(R)
    assert s.shape == (4,) and np.all(np.abs(s - 0.05 * np.sqrt(252)) < 0.5)   # true daily SR 0.05 -> ~0.79/yr
    assert np.all(max_drawdown(R) <= 0)
    sk, ku = skew_kurt(R)
    assert np.all(np.abs(ku - 3) < 0.5)


def test_no_same_bar_execution_and_costs():
    # asset 0 has a +10% return on day 3; weights decided on day 3 must NOT capture it
    R = np.zeros((6, 2)); R[3, 0] = 0.10
    W = np.zeros((6, 2)); W[3, 0] = 1.0; W[4, 0] = 1.0
    net, to = portfolio_returns(W, R, cost_bps=0)
    assert net[3] == 0.0 and net[4] == 0.0            # weight applies from day 4; return on day 4 is 0
    net10, to10 = portfolio_returns(W, R, cost_bps=10)
    assert to10[4] == pytest.approx(1.0)               # buys full position -> 1.0 turnover
    assert net10[4] == pytest.approx(-10 / 1e4)


def test_cpcv_counts_and_purge():
    T, n, k, h, e = 1200, 6, 2, 5, 3
    splits = cpcv_splits(T, n, k, h, e)
    assert len(splits) == comb(n, k)
    for tr, te, combo in splits:
        assert len(set(tr) & set(te)) == 0
        # purge: no train index within h of a test index (before) or h+e (after)
        assert np.min(np.abs(tr[:, None] - te[None, :])) > 0
        assert not np.any(np.isin(te.min() - np.arange(1, h + 1), tr))
        assert not np.any(np.isin(te.max() + np.arange(1, h + e + 1), tr))
    paths = cpcv_paths(T, lambda tr, te: np.ones(len(te)), n, k, h, e)
    assert paths.shape[0] == n_paths(n, k) == comb(n, k) * k // n
    assert not np.isnan(paths).any()                   # every path fully covered


def test_dsr_monotone_and_known_values():
    assert expected_max_sharpe(1, 0.01) == 0.0
    e10, e100, e1000 = (expected_max_sharpe(n, 0.01) for n in (10, 100, 1000))
    assert 0 < e10 < e100 < e1000
    d = deflated_sharpe(sr=0.1, T=2520, n_trials=1, var_sharpe=0.0)
    assert d["dsr"] > 0.999                            # SR 0.1/day over 10y, single trial: certain
    d2 = deflated_sharpe(sr=0.05, T=252, n_trials=500, var_sharpe=0.004)
    assert d2["dsr"] < 0.95                            # one year, 500 trials: not certifiable
    assert min_backtest_length(100, 0.05) > min_backtest_length(10, 0.05)


def test_pbo_near_half_for_noise_and_low_for_true_skill():
    """A single noise panel has high PBO variance (one lucky series wins both halves),
    so average over independent panels; a genuinely skilled strategy drives PBO to ~0."""
    pn, ps = [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 0.01, (2000, 30))
        pn.append(pbo_cscv(noise, n_blocks=12)["pbo"])
        skill = noise.copy(); skill[:, 0] += 0.003
        ps.append(pbo_cscv(skill, n_blocks=12)["pbo"])
    assert 0.3 < np.mean(pn) < 0.7, pn
    assert np.mean(ps) < 0.1, ps


def test_bootstrap_ci_covers_truth():
    """CI must bracket the sample statistic, be positive for a strongly positive series,
    and have a width consistent with the analytic Sharpe standard error sqrt(periods/T)."""
    rng = np.random.default_rng(2)
    T = 6000
    r = rng.normal(0.001, 0.01, T)                      # true annual SR ~1.59
    ci = bootstrap_ci(r, lambda x: sharpe(x), B=500)
    se = np.sqrt(252 / T)
    assert ci["lo"] < ci["point"] < ci["hi"] and ci["lo"] > 0
    assert abs(ci["point"] - 1.59) < 3 * se
    assert 0.5 * se < (ci["hi"] - ci["lo"]) / 2 < 3 * se


def test_costs_handle_dollar_neutral_books():
    """Long-short weights must not explode turnover: a static book with zero returns has zero turnover after entry."""
    R = np.zeros((6, 4)); W = np.zeros((6, 4)); W[1:, 0] = 0.5; W[1:, 1] = 0.5; W[1:, 2] = -0.5; W[1:, 3] = -0.5
    net, to = portfolio_returns(W, R, cost_bps=10)
    assert to[2] == pytest.approx(2.0)                # entering 2.0 gross exposure
    assert np.allclose(to[3:], 0.0) and np.allclose(net[3:], 0.0)
    # long-only unchanged: drift then rebalance to equal weight costs the drift only
    R2 = np.zeros((4, 2)); R2[1, 0] = 0.1; W2 = np.full((4, 2), 0.5)
    _, to2 = portfolio_returns(W2, R2, cost_bps=0)
    assert to2[2] == pytest.approx(abs(0.5 - 0.55 / 1.05) * 2, rel=1e-6)
