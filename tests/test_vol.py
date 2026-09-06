import numpy as np
from emporion.baselines.vol import ewma, har_rv, garch11_t, vol_forecast_eval, fit_garch_t


def _garch_sim(T=1500, N=2, seed=0, omega=2e-6, alpha=0.08, beta=0.90):
    rng = np.random.default_rng(seed)
    R = np.zeros((T, N)); h = np.full(N, omega / (1 - alpha - beta))
    for t in range(1, T):
        h = omega + alpha * R[t - 1] ** 2 + beta * h
        R[t] = np.sqrt(h) * rng.standard_t(6, N) / np.sqrt(6 / 4)
    return R


def test_garch_fit_recovers_persistence():
    R = _garch_sim(T=3000, N=1)
    omega, alpha, beta, nu = fit_garch_t(R[:, 0])
    assert 0.9 < alpha + beta < 0.999 and 3 < nu < 15


def test_vol_models_beat_constant_and_no_lookahead():
    R = _garch_sim()
    const = np.full_like(R, R.var())
    base = vol_forecast_eval(const, R)["qlike"]
    for fn in (ewma, har_rv, garch11_t):
        h = fn(R)
        ev = vol_forecast_eval(h, R)
        assert ev["qlike"] < base, fn.__name__
        # no look-ahead: perturbing returns after t must not change forecasts up to t
        R2 = R.copy(); R2[1000:] *= 3
        h2 = fn(R2)
        m = np.isfinite(h[:1000]) & np.isfinite(h2[:1000])
        assert np.allclose(h[:1000][m], h2[:1000][m]), fn.__name__
