import numpy as np
import pytest

from emporion.baselines import BASELINES, hrp_weights
from emporion.core import perf
from emporion.evaluate import portfolio_returns, pbo_cscv, cpcv_paths, bootstrap_ci, sharpe


def _R(T=2520, N=20, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_t(4, (T, N)) * 0.01 + 0.0003


def test_baselines_are_valid_weights():
    R = _R()
    for name, fn in BASELINES.items():
        W = fn(R)
        assert W.shape == R.shape
        s = W.sum(axis=1)
        assert np.allclose(s[s > 0], 1.0, atol=1e-9), name
        assert (W >= -1e-12).all(), name


def test_hrp_weights_sum_and_prefer_low_var():
    cov = np.diag([0.01, 0.04, 0.09, 0.16])
    w = hrp_weights(cov)
    assert w.sum() == pytest.approx(1.0) and w[0] > w[3]


def test_no_lookahead_in_rolling_baselines():
    """Perturbing the future must not change today's weights."""
    R = _R(T=800, N=8)
    R2 = R.copy(); R2[600:] += 0.05
    for name in ("inverse_vol", "hrp", "momentum_top"):
        W1, W2 = BASELINES[name](R), BASELINES[name](R2)
        assert np.allclose(W1[:600], W2[:600]), name


def test_perf_budgets_p0():
    """Measured before optimised: the certification stack on a 10y x 20-asset panel."""
    perf.clear()
    R = _R()
    M = np.stack([portfolio_returns(fn(R), R, 10.0)[0] for fn in BASELINES.values()], axis=1)
    pbo_cscv(M, n_blocks=16)
    cpcv_paths(R.shape[0], lambda tr, te: M[te, 0], 6, 2, 1, 5)
    bootstrap_ci(M[:, 0], lambda x: sharpe(x), B=2000)
    rep = perf.report()
    perf.budget("pbo.cscv", 2.0)               # 12,870 combos x 5 strategies
    perf.budget("cpcv.paths", 0.5)
    perf.budget("bootstrap.indices", 1.5)      # 2000 x 2520 indices
    perf.budget("costs.portfolio_returns", 0.05)
    perf.budget("baseline.hrp", 3.0)
    assert rep["pbo.cscv"]["best_rows_per_s"] > 1e5
