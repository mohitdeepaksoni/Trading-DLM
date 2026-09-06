import numpy as np
from emporion.anchors import price_themes, THEMES
from emporion.structure.embeddings import asset_embeddings, similarity, peer_implied_yield


def test_price_themes_no_lookahead_and_oriented():
    rng = np.random.default_rng(0)
    T, N = 1200, 30
    r = rng.normal(0, 0.01, (T, N)); r[:, 0] += 0.002              # asset 0 trends up
    P = 100 * np.exp(np.cumsum(r, 0)); V = rng.lognormal(10, 0.3, (T, N))
    th = price_themes(P, V, t=800)
    assert set(th) == set(THEMES)
    assert th["momentum"][0] == np.max(th["momentum"])              # strongest momentum = the trending asset
    P2 = P.copy(); P2[801:] *= 3
    th2 = price_themes(P2, V, t=800)
    for k in th:
        assert np.allclose(th[k], th2[k], equal_nan=True), k       # future data does not change the signal


def test_embeddings_recover_latent_clusters():
    rng = np.random.default_rng(1)
    A, I, k = 60, 12, 2
    z = rng.normal(size=(A, k)); w = rng.normal(size=(I, k))
    H = np.exp(z @ w.T + rng.normal(0, 0.1, (A, I)))               # holdings driven by 2 latent characteristics
    E, Einv, ev = asset_embeddings(H, k=4)
    assert ev[:2].sum() > 0.8                                        # two dominant dimensions
    S = similarity(E[:, :2])
    # nearest neighbours in embedding space share the latent characteristic
    close = np.argsort(-S[0])[1:6]
    assert np.linalg.norm(z[close] - z[0], axis=1).mean() < np.linalg.norm(z - z[0], axis=1).mean()
    ey = np.exp(z[:, 0] * 0.3 + rng.normal(0, 0.05, A)) * 0.05     # earnings yield tied to latent 1
    sig = peer_implied_yield(E[:, :2], ey)
    assert np.isfinite(sig).mean() > 0.9 and abs(np.nanmean(sig)) < 0.2
