import numpy as np
from emporion.baselines.regime import GaussianHMM, conformal_quantile, coverage


def _regime_series(T=3000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros(T, int)
    for t in range(1, T):
        s[t] = rng.choice(2, p=[[0.99, 0.01], [0.04, 0.96]][s[t - 1]])
    r = rng.normal(0, np.where(s == 0, 0.008, 0.03))
    return r, s


def test_hmm_recovers_regimes():
    r, s = _regime_series()
    m = GaussianHMM(2).fit(r)
    assert m.sd[0] < 0.012 and m.sd[1] > 0.02
    P = m.filter(r)
    pred = P.argmax(1)
    acc = (pred == s).mean()
    assert acc > 0.85, acc


def test_rolling_filter_has_no_lookahead():
    r, _ = _regime_series()
    P1 = GaussianHMM(2).rolling_filter(r, refit_every=200, min_train=800)
    r2 = r.copy(); r2[2000:] *= 5
    P2 = GaussianHMM(2).rolling_filter(r2, refit_every=200, min_train=800)
    ok = np.isfinite(P1[:2000, 0])
    assert np.allclose(P1[:2000][ok], P2[:2000][ok])


def test_conformal_coverage():
    rng = np.random.default_rng(1)
    e = rng.standard_t(4, 3000) * 0.01
    q = conformal_quantile(e, alpha=0.1, window=250)
    c = coverage(e, q)
    assert 0.87 < c < 0.93, c
