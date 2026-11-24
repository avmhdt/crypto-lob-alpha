"""Walk-forward splits never put a test date at or before a train date (AC-W-05)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from lob_alpha.models.splits import day_start_ns, needs_purge, purge_keep, walk_forward

BASE = date(2026, 11, 1)
day_sets = st.sets(st.integers(0, 60), min_size=0, max_size=40).map(
    lambda s: [BASE + timedelta(days=k) for k in sorted(s)]
)


@given(day_sets, st.integers(1, 6), st.integers(1, 2), st.integers(1, 2), st.integers(1, 3))
def test_folds_are_ordered_disjoint_and_complete(
    days: list[date], tr: int, va: int, te: int, step: int
) -> None:
    folds = walk_forward(days, tr, va, te, step)
    width = tr + va + te
    expected = 0 if len(days) < width else (len(days) - width) // step + 1
    assert len(folds) == expected
    for k, f in enumerate(folds):
        assert len(f.train) == tr and len(f.val) == va and len(f.test) == te
        assert max(f.train) < min(f.val) <= max(f.val) < min(f.test)
        assert set(f.all_days()) <= set(days)
        # consecutive complete days: no complete day is skipped inside a fold
        i = days.index(f.train[0])
        assert list(f.all_days()) == days[i : i + width]
        assert i == k * step


@given(day_sets.filter(lambda d: len(d) >= 7))
def test_purge_only_at_set_boundaries(days: list[date]) -> None:
    for f in walk_forward(days):
        purged = [d for d in f.all_days() if needs_purge(f, d)]
        assert purged == [f.train[-1], f.val[-1]]


def test_purge_keeps_rows_before_cutoff() -> None:
    d = date(2026, 11, 5)
    end = day_start_ns(d) + 86_400 * 10**9
    ts = np.array([end - 40 * 10**9, end - 31 * 10**9 - 1, end - 31 * 10**9, end - 1])
    assert purge_keep(ts, d, 31).tolist() == [True, True, False, False]


def test_unsorted_or_duplicate_days_rejected() -> None:
    d = [BASE, BASE]
    try:
        walk_forward(d)
    except ValueError:
        return
    raise AssertionError("duplicates must raise")
