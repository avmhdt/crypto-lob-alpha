"""REST checkpoint cadence: fixed period, no drift, no catch-up burst (AC-W-03 needs 144/day)."""

from __future__ import annotations

import itertools

from lob_alpha.record.supervisor import checkpoint_due


def test_fetch_time_does_not_stretch_the_period() -> None:
    pause, fetch = 200.0, 0.4
    due = now = 0.0
    times = []
    for _ in range(3 * 144):  # three symbols, one day
        due = checkpoint_due(due, now, pause)
        now = max(now, due) + fetch  # the fetch finishes after its due time
        times.append(due)
    assert times[-1] == 3 * 144 * pause  # 144 per symbol fit in exactly one day
    assert all(b - a == pause for a, b in itertools.pairwise(times))


def test_a_stall_restarts_the_schedule_instead_of_bursting() -> None:
    assert checkpoint_due(0.0, 1000.0, 200.0) == 1000.0
    assert checkpoint_due(0.0, 350.0, 200.0) == 200.0  # late by less than a pause: keep phase
