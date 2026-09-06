import numpy as np
from emporion.structure.nowcaster import kalman_nowcast, fit_flow_loading, holdout_error


def _panel(T=1000, N=20, seed=0, b_true=0.02, every=63):
    rng = np.random.default_rng(seed)
    flows = rng.normal(0, 1, (T, N))
    x = np.zeros((T, N)); x[0] = rng.uniform(0.3, 0.7, N)
    for t in range(1, T):
        x[t] = x[t - 1] + b_true * flows[t] + rng.normal(0, 0.001, N)
    obs = np.full((T, N), np.nan); obs[::every] = x[::every]           # quarterly filings
    return x, obs, flows


def test_filter_tracks_state_and_flows_help():
    x, obs, flows = _panel()
    b = fit_flow_loading(obs, flows)
    assert abs(b - 0.02) < 0.006, b
    m, v = kalman_nowcast(obs, flows, q=1e-6, b=b)
    ok = np.isfinite(m)
    assert np.nanmean(np.abs(m[ok] - x[ok])) < 0.02
    assert (v[np.isfinite(obs)] < 1e-6).all()                            # exact at filing dates
    res = holdout_error(obs, flows, every=4)
    assert res["mae_filter"] < 0.7 * res["mae_last_value"], res          # flows carry information -> filter beats carry-forward


def test_filter_no_lookahead_and_bounded_variance():
    x, obs, flows = _panel(seed=1)
    m1, v1 = kalman_nowcast(obs, flows, q=1e-6, b=0.02)
    obs2 = obs.copy(); obs2[600:] = np.nan; flows2 = flows.copy(); flows2[600:] = 99.0
    m2, v2 = kalman_nowcast(obs2, flows2, q=1e-6, b=0.02)
    assert np.allclose(m1[:600], m2[:600], equal_nan=True)               # the future never changes the past
    assert v1[500:].max() <= 63 * 1e-6 + 1e-12                            # innovation bounded between filings
