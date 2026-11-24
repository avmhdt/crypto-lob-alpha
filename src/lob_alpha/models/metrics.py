"""Out-of-sample prediction metrics (FR-20, FR-21).

* `oos_r2` = 1 - sum (y - yhat)^2 / sum (y - ybar_train)^2: the benchmark is the training
  mean of the label, a forecast known before the test day (Campbell and Thompson 2008).
  A model that predicts the training mean scores 0; worse than that is negative.
* `ic_spearman`: Spearman rank correlation of yhat and y.
* `hit_rate_nonzero`: share of rows with y != 0 where sign(yhat) == sign(y) (yhat = 0
  counts as a miss).
* `mae_ticks`: mean |y - yhat|.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass
class Scores:
    n: int
    oos_r2: float
    ic_spearman: float
    hit_rate_nonzero: float
    mae_ticks: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _rank(x: F64) -> F64:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    # average ranks for ties
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    ends = np.r_[starts[1:], len(xs)]
    avg = (starts + ends - 1) / 2.0
    ranks[order] = np.repeat(avg, ends - starts)
    return ranks


def spearman(a: F64, b: F64) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def oos_r2(y: F64, yhat: F64, train_mean: float) -> float:
    sst = float(np.sum((y - train_mean) ** 2))
    if sst == 0:
        return float("nan")
    return 1.0 - float(np.sum((y - yhat) ** 2)) / sst


def score(y: F64, yhat: F64, train_mean: float) -> Scores:
    nz = y != 0
    hit = float(np.mean(np.sign(yhat[nz]) == np.sign(y[nz]))) if nz.any() else float("nan")
    return Scores(
        len(y),
        oos_r2(y, yhat, train_mean),
        spearman(yhat, y),
        hit,
        float(np.mean(np.abs(y - yhat))) if len(y) else float("nan"),
    )
