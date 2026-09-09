"""Indian delivery cost model: statutory components, buy-side asymmetry, and the vectorised cost path matches the scalar one when flat."""
import numpy as np
from emporion.evaluate.costs import portfolio_returns
from emporion.evaluate.costs_in import cost_vectors, one_way_cost_bps, round_trip_bps, STT_BPS, STAMP_BUY_BPS


def test_components_and_round_trip():
    assert one_way_cost_bps(slippage_bps=0.0) > STT_BPS and abs(round_trip_bps() - (2 * one_way_cost_bps() + STAMP_BUY_BPS)) < 1e-12
    c, tax = cost_vectors(3, slippage_bps=np.array([5.0, 10.0, 20.0]))
    assert abs(c[2] - c[0] - 15.0) < 1e-9 and (tax == STAMP_BUY_BPS).all()


def test_vectorised_costs_reduce_to_scalar_and_buys_cost_more():
    import importlib.util, os, sys
    spec = importlib.util.spec_from_file_location("nse_ladder", os.path.join(os.path.dirname(__file__), "..", "scripts", "p4_realdata_nse.py"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    rng = np.random.default_rng(0); T, N = 60, 5; R = rng.normal(0, 0.01, (T, N)); W = np.zeros((T, N)); W[::10] = 1.0 / N; W = np.maximum.accumulate(W, axis=0)
    a, _ = portfolio_returns(W, R, cost_bps=15.0); b, _ = m.portfolio_returns_vec(W, R, np.full(N, 15.0), np.zeros(N))
    assert np.allclose(a, b)
    c, _ = m.portfolio_returns_vec(W, R, np.full(N, 15.0), np.full(N, 1.5))
    assert (c <= b + 1e-12).all() and c.sum() < b.sum()                       # stamp duty on purchases only lowers returns
