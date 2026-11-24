"""Walk-forward splits by UTC day, never shuffled (FR-20, AC-W-05).

The days are the complete days (DQ-10) of one stream, in date order. A fold takes
`train_days` consecutive complete days for training, the next `val_days` for validation
(ridge alpha, LightGBM early stopping) and the next `test_days` for out-of-sample testing,
then moves on by `step_days`. "Consecutive" counts complete days, so a fold can span a
calendar gap such as an outage: the gap only removes days, it never moves a later day
before an earlier one.

Purge: within each set, rows whose label window could reach into the next set are dropped.
A sample at t with horizon h and latency L looks at data up to t + L + h. Every sample of a
training day whose next complete day is not a training day loses the last `purge_s`
seconds before 00:00 UTC of the following day; the same holds for the validation day before
the test day. Labels are already null across 00:00 UTC (labels.py), so this purge is a
second, explicit guard, and it keeps working if a label definition ever changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import numpy.typing as npt

NS_PER_S = 1_000_000_000
NS_PER_DAY = 86_400 * NS_PER_S


@dataclass(frozen=True)
class Fold:
    index: int
    train: tuple[date, ...]
    val: tuple[date, ...]
    test: tuple[date, ...]

    def all_days(self) -> tuple[date, ...]:
        return self.train + self.val + self.test


def walk_forward(
    days: Sequence[date],
    train_days: int = 5,
    val_days: int = 1,
    test_days: int = 1,
    step_days: int = 1,
) -> list[Fold]:
    """Folds over sorted unique days; an empty list if there are too few days."""
    ordered = sorted(set(days))
    if len(ordered) != len(days):
        raise ValueError("days must be unique")
    width = train_days + val_days + test_days
    folds = []
    for k, i in enumerate(range(0, len(ordered) - width + 1, step_days)):
        block = ordered[i : i + width]
        folds.append(
            Fold(
                k,
                tuple(block[:train_days]),
                tuple(block[train_days : train_days + val_days]),
                tuple(block[train_days + val_days :]),
            )
        )
    return folds


def day_start_ns(day: date) -> int:
    return (day - date(1970, 1, 1)).days * NS_PER_DAY


def purge_keep(ts: npt.NDArray[np.int64], day: date, purge_s: float) -> npt.NDArray[np.bool_]:
    """Rows of `day` that are at least `purge_s` seconds before the next 00:00 UTC."""
    cutoff = day_start_ns(day) + NS_PER_DAY - round(purge_s * NS_PER_S)
    return np.asarray(ts < cutoff)


def needs_purge(fold: Fold, day: date) -> bool:
    """True when the fold's next day after `day` belongs to a different set."""
    days = fold.all_days()
    pos = days.index(day)
    if pos + 1 >= len(days):
        return False
    sets = [fold.train, fold.val, fold.test]
    here = next(i for i, s in enumerate(sets) if day in s)
    there = next(i for i, s in enumerate(sets) if days[pos + 1] in s)
    return here != there
