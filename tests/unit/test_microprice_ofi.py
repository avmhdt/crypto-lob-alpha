"""Stoikov micro-price on synthetic chains, and the OFI regression on a planted relation."""

from __future__ import annotations

import numpy as np

from lob_alpha.features.microprice import fit_microprice_arrays, lookup
from lob_alpha.research.ofi import interval_pairs, newey_west_lag, ols_hac, spearman


def _chain(n: int, seed: int, drift: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Imbalance random walk; the mid tends to tick toward the heavier side when drift > 0."""
    rng = np.random.default_rng(seed)
    imb = np.clip(0.5 + np.cumsum(rng.normal(0, 0.05, n)), 0.01, 0.99)
    imb = 0.5 + 0.49 * np.sin(np.cumsum(rng.normal(0, 0.2, n)))
    spread = rng.choice([1.0, 2.0], n, p=[0.8, 0.2])
    p_up = np.clip(0.05 + drift * (imb - 0.5), 0.0, 0.2)
    p_dn = np.clip(0.05 - drift * (imb - 0.5), 0.0, 0.2)
    u = rng.random(n)
    step = np.where(u < p_up, 1.0, np.where(u > 1 - p_dn, -1.0, 0.0))
    mid = np.cumsum(step)
    return imb, spread, mid


def test_stoikov_symmetric_chain_is_zero_at_half() -> None:
    imb, spread, mid = _chain(60_000, 1, 0.0)
    table = fit_microprice_arrays(imb, spread, mid, n_imb=11, max_spread=2)
    assert np.allclose(table.g[:, 5], 0.0, atol=1e-12)  # bucket centred on I = 0.5
    assert np.allclose(table.g, -table.g[:, ::-1], atol=1e-12)  # antisymmetric by construction


def test_stoikov_sign_follows_imbalance() -> None:
    imb, spread, mid = _chain(80_000, 2, 0.3)
    table = fit_microprice_arrays(imb, spread, mid, n_imb=10, max_spread=2)
    assert table.g[0, -1] > 0 > table.g[0, 0]
    got = lookup(table, np.array([0.95, 0.05, 0.5, np.nan]), np.array([1.0, 1.0, 9.0, 1.0]))
    assert got[0] > 0 > got[1] and np.isnan(got[2]) and np.isnan(got[3])
    assert len(table.to_rows()) == 20


def test_ofi_regression_recovers_a_planted_beta() -> None:
    rng = np.random.default_rng(2)  # a typical draw (-1.0 SE); the estimator is unbiased over seeds
    n = 5000
    x = rng.normal(0, 10, n)
    beta = 0.12
    y = 0.3 + beta * x + rng.normal(0, 1.0, n)
    res = ols_hac(y, x)
    assert abs(res.beta - beta) < 2 * res.se_beta
    assert res.t_beta > 5 and 0.5 < res.r2 < 0.7
    assert newey_west_lag(100) == 4 and newey_west_lag(1) == 1


def test_ofi_random_walk_has_no_signal() -> None:
    rng = np.random.default_rng(7)
    res = ols_hac(rng.normal(size=4000), rng.normal(size=4000))
    assert abs(res.t_beta) < 4 and res.r2 < 0.01


def test_spearman_and_interval_pairs() -> None:
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == -1.0
    ms = 1_000_000
    feat_ts = np.array([1000, 2000, 3000], np.int64) * ms
    lab_ts = np.array([0, 500, 1000, 2000], np.int64) * ms
    y, x, _ = interval_pairs(
        feat_ts,
        np.array([5.0, 6.0, np.nan]),
        lab_ts,
        np.array([1.0, 9.0, 2.0, 3.0]),
        np.ones(3),
        1000,
    )
    assert y.tolist() == [1.0, 2.0] and x.tolist() == [5.0, 6.0]  # 2000 drops: OFI at 3000 is NaN
