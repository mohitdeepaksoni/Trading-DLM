"""Validation of the validators (research/03): the certification gate must NOT certify noise, the permutation
p-value must be uniform under the null, and the Deflated Sharpe must deflate more as the ledger fills."""
import os
import numpy as np
import pytest

from emporion.core.ledger import Ledger
from emporion.evaluate import certify, deflated_sharpe, expected_max_sharpe, bootstrap_ci, sharpe, pbo_cscv


def test_gate_false_positive_rate_on_pure_noise(tmp_path):
    """20 independent noise strategies with the same ledger scope: the fraction certified must be ~0
    (each has to beat a bootstrap CI, a DSR that counts the 20 trials, and an equal-weight benchmark)."""
    rng = np.random.default_rng(0)
    L = Ledger(str(tmp_path / "l"))
    T = 1500
    bench = rng.normal(0.0003, 0.01, T)
    certified = 0
    for k in range(20):
        r = rng.normal(0.0, 0.01, T)
        rep = certify(f"noise{k}", T, lambda tr, te, r=r: r[te], r, L, "noise", benchmark_returns=bench, B=300, pbo=0.5)
        certified += int(rep["certified"])
    assert certified == 0, certified


def test_gate_certifies_a_real_edge(tmp_path):
    """A strategy with true annual Sharpe ~2 over 6 years must pass (power check), even after 5 prior trials."""
    rng = np.random.default_rng(1)
    L = Ledger(str(tmp_path / "l"))
    T = 2500
    for k in range(5):
        L.append("trial", f"prior{k}", {"sharpe_is": float(rng.normal(0, 0.5))}, scope="edge")
    r = rng.normal(0.00126, 0.01, T)                     # daily SR 0.126 -> annual 2.0, ten years
    bench = rng.normal(0.0, 0.01, T)
    rep = certify("edge", T, lambda tr, te: r[te], r, L, "edge", benchmark_returns=bench, B=300, pbo=0.05)
    assert rep["certified"], rep["checks"]


def test_dsr_deflates_with_trial_count():
    d = [deflated_sharpe(sr=0.06, T=1500, n_trials=n, var_sharpe=0.004)["dsr"] for n in (1, 10, 100, 1000)]
    assert d[0] > d[1] > d[2] > d[3]
    assert expected_max_sharpe(1000, 0.004) > expected_max_sharpe(10, 0.004)


def test_permutation_pvalue_is_uniform_under_null():
    """Rank-IC permutation p-values on pure noise must be ~Uniform(0,1): mean ~0.5, ~10 % below 0.1."""
    rng = np.random.default_rng(2)
    def rank_ic(a, b):
        ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
        ra = (ra - ra.mean()) / ra.std(); rb = (rb - rb.mean()) / rb.std(); return float((ra * rb).mean())
    pvals = []
    for rep in range(200):
        preds = [rng.normal(size=30) for _ in range(15)]; reals = [rng.normal(size=30) for _ in range(15)]
        obs = np.mean([rank_ic(p, r) for p, r in zip(preds, reals)])
        null = np.array([np.mean([rank_ic(rng.permutation(p), r) for p, r in zip(preds, reals)]) for _ in range(200)])
        pvals.append((null >= obs).mean())
    pvals = np.array(pvals)
    assert 0.4 < pvals.mean() < 0.6 and 0.04 < (pvals < 0.1).mean() < 0.18, (pvals.mean(), (pvals < 0.1).mean())


def test_bootstrap_ci_coverage_under_null():
    """95 % stationary-bootstrap CI on the Sharpe of noise must contain 0 about 95 % of the time."""
    rng = np.random.default_rng(3)
    inside = 0
    for _ in range(60):
        r = rng.normal(0, 0.01, 1000)
        ci = bootstrap_ci(r, lambda x: sharpe(x), B=300, seed=int(rng.integers(1e9)))
        inside += int(ci["lo"] <= 0 <= ci["hi"])
    assert inside >= 52, inside                          # >= ~87 % (binomial slack on 60 draws)


def test_pbo_flags_selection_among_many_noise_strategies():
    """PBO of the IS-best among many noise strategies ~ 0.5 in expectation; average over panels (single draws vary)."""
    rng = np.random.default_rng(4)
    p = [pbo_cscv(rng.normal(0, 0.01, (2000, 100)), n_blocks=12)["pbo"] for _ in range(5)]
    assert np.mean(p) > 0.3, p


def test_rerunning_the_same_procedure_is_not_a_new_trial(tmp_path):
    """Distinct-procedure counting: three identical evaluations of one rung count once; a different rung counts."""
    rng = np.random.default_rng(7)
    L = Ledger(str(tmp_path / "l")); T = 800; r = rng.normal(0.0005, 0.01, T)
    for _ in range(3):
        rep = certify("same_rung", T, lambda tr, te: r[te], r, L, "s", B=100, pbo=0.1)
    assert rep["n_trials"] == 1 and rep["n_evaluations"] == 3
    rep2 = certify("other_rung", T, lambda tr, te: r[te], r, L, "s", B=100, pbo=0.1)
    assert rep2["n_trials"] == 2 and rep2["n_evaluations"] == 4
    assert rep2["dsr"]["dsr"] >= rep2["dsr_strict"]                   # strict count deflates at least as much
