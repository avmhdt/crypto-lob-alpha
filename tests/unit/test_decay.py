"""Half-life interpolation and the day-block bootstrap (FR-21)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lob_alpha.research.decay import (
    DayStat,
    bootstrap,
    day_stat,
    half_life,
    half_life_ci,
    pooled_r2,
)


def test_half_life_interpolates_on_log_latency() -> None:
    lat = [0, 1, 10, 100]
    # R2 halves (0.2 -> 0.1) between 10 ms (0.15) and 100 ms (0.05): midpoint on log scale
    hl = half_life(lat, [0.2, 0.18, 0.15, 0.05])
    assert hl == pytest.approx(math.sqrt(10 * 100))
    # between 0 and the first positive latency: linear in L
    assert half_life(lat, [0.2, 0.0, 0.0, 0.0]) == pytest.approx(0.5)
    assert half_life(lat, [0.2, 0.19, 0.18, 0.17]) is None
    assert half_life(lat, [-0.1, -0.2, -0.3, -0.4]) is None


def test_pooled_r2_and_bootstrap() -> None:
    a = day_stat("d1", np.array([1.0, -1.0, 2.0]), np.array([0.5, -0.5, 1.0]), np.zeros(3))
    assert (a.sse, a.sst, a.hits, a.nonzero) == (1.5, 6.0, 3, 3)
    b = DayStat("d2", 3, 3.0, 6.0, 0.1, 1, 2)
    assert pooled_r2([a, b]) == pytest.approx(1 - 4.5 / 12)
    lo, hi = bootstrap([a, b], resamples=500)
    assert 0.5 - 1e-12 <= lo <= hi <= 0.75 + 1e-12
    assert all(math.isnan(x) for x in bootstrap([a]))


def test_half_life_ci_brackets_point() -> None:
    rng = np.random.default_rng(1)
    lats = [0, 1, 10, 100, 1000]
    stats: dict[int, list[DayStat]] = {}
    for L in lats:
        stats[L] = []
        for d in range(8):
            sst = 100.0
            r2 = 0.2 * math.exp(-L / 50) + rng.normal(0, 0.01)
            stats[L].append(DayStat(f"d{d}", 10, sst * (1 - r2), sst, 0.0, 0, 0))
    hl = half_life_ci(stats, resamples=300)
    assert hl.half_life_ms is not None and hl.lo is not None and hl.hi is not None
    assert hl.lo <= hl.half_life_ms <= hl.hi
    assert 10 < hl.half_life_ms < 100


def test_event_horizon_match_on_log_scale() -> None:
    from lob_alpha.research.event_clock import match

    durations = {1: 4.0, 10: 60.0, 50: 400.0, 100: 900.0, 250: 2600.0}
    assert match(1000, durations) == 100  # |log 900 - log 1000| < |log 2600 - log 1000|
    assert match(5000, durations) == 250
    assert match(100, {5: float("nan"), 10: 60.0}) == 10
