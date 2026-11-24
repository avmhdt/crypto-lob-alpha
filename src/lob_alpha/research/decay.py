"""Alpha decay against horizon and added latency, and the signal half-life (FR-21, AC-H-01).

* Horizon curve: one walk-forward run per horizon (same model, features and folds), scored
  on its own test days.
* Latency curve: the L = 0 run's predictions, made at t, scored against the label
  shifted by L: mid(t + L + h) - mid(t + L). That is how much of the forecast is left if the
  order can only act L ms after the signal. The model is not retrained per L: a model
  retrained on shifted labels answers a different question (what is predictable L ms
  ahead), and the trading question is the decay of the signal you have.
* Statistics per test day: SSE, the benchmark sum of squares against the fold's
  training mean, Spearman IC and the hit rate on non-zero moves. Pooled OOS R2 =
  1 - sum SSE / sum SST. 95% CIs come from a day-block bootstrap (2,000 resamples, fixed
  seed): whole test days are resampled with replacement.
* Half-life: the latency at which OOS R2 falls to half its L = 0 value, by linear
  interpolation on log latency between the two grid latencies that bracket it (linear in
  L between 0 and the first positive latency). None when R2 never falls that far on the
  grid, or when R2 at L = 0 is not positive.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from lob_alpha.models.data import load_day
from lob_alpha.models.metrics import spearman

F64 = npt.NDArray[np.float64]
SEED = 20261102


@dataclass
class DayStat:
    day: str
    n: int
    sse: float
    sst: float
    ic: float
    hits: int
    nonzero: int


def day_stat(day: str, y: F64, yhat: F64, train_mean: F64) -> DayStat:
    ok = ~(np.isnan(y) | np.isnan(yhat))
    y, yhat, mu = y[ok], yhat[ok], train_mean[ok]
    nz = y != 0
    return DayStat(
        day,
        len(y),
        float(np.sum((y - yhat) ** 2)),
        float(np.sum((y - mu) ** 2)),
        spearman(yhat, y),
        int(np.sum(np.sign(yhat[nz]) == np.sign(y[nz]))),
        int(np.sum(nz)),
    )


def pooled_r2(stats: list[DayStat]) -> float:
    sst = sum(s.sst for s in stats)
    return 1.0 - sum(s.sse for s in stats) / sst if sst > 0 else float("nan")


def bootstrap(
    stats: list[DayStat], resamples: int = 2000, seed: int = SEED, ci: float = 0.95
) -> tuple[float, float]:
    """Day-block bootstrap CI of the pooled OOS R2."""
    if len(stats) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sse = np.array([s.sse for s in stats])
    sst = np.array([s.sst for s in stats])
    k = rng.integers(0, len(stats), size=(resamples, len(stats)))
    r2 = 1.0 - sse[k].sum(axis=1) / sst[k].sum(axis=1)
    a = (1 - ci) / 2
    return float(np.quantile(r2, a)), float(np.quantile(r2, 1 - a))


def half_life(lat: list[int], r2: list[float]) -> float | None:
    if not lat or lat[0] != 0 or not r2[0] > 0:
        return None
    half = r2[0] / 2
    for i in range(1, len(lat)):
        if r2[i] <= half:
            l0, l1, a, b = lat[i - 1], lat[i], r2[i - 1], r2[i]
            frac = (a - half) / (a - b) if a != b else 0.0
            if l0 == 0:
                return l0 + frac * (l1 - l0)
            return math.exp(math.log(l0) + frac * (math.log(l1) - math.log(l0)))
    return None


@dataclass
class CurveRow:
    venue: str
    symbol: str
    clock: str
    model: str
    run_id: str
    horizon: int
    latency_ms: int
    n_days: int
    n: int
    oos_r2: float
    r2_lo: float
    r2_hi: float
    ic_mean: float
    hit_rate_nonzero: float


def curve_row(
    meta: dict[str, str | int], horizon: int, latency: int, stats: list[DayStat]
) -> CurveRow:
    lo, hi = bootstrap(stats)
    nz = sum(s.nonzero for s in stats)
    return CurveRow(
        str(meta["venue"]),
        str(meta["symbol"]),
        str(meta["clock"]),
        str(meta["model"]),
        str(meta["run_id"]),
        horizon,
        latency,
        len(stats),
        sum(s.n for s in stats),
        pooled_r2(stats),
        lo,
        hi,
        float(np.nanmean([s.ic for s in stats])) if stats else float("nan"),
        sum(s.hits for s in stats) / nz if nz else float("nan"),
    )


def run_day_stats(data_root: Path, run_id: str) -> list[DayStat]:
    """Per-test-day statistics of a run's own predictions."""
    t = pq.read_table(data_root / "runs" / run_id / "predictions.parquet")
    days = np.array([d.isoformat() for d in t.column("day").to_pylist()])
    y = t.column("y").to_numpy()
    yhat = t.column("yhat").to_numpy()
    mu = t.column("train_mean").to_numpy()
    return [day_stat(d, y[days == d], yhat[days == d], mu[days == d]) for d in sorted(set(days))]


def latency_day_stats(
    data_root: Path, run_id: str, venue: str, symbol: str, horizon_ms: int, latency_ms: int
) -> list[DayStat]:
    """The run's L = 0 predictions scored against the label shifted by `latency_ms`."""
    if latency_ms == 0:
        return run_day_stats(data_root, run_id)
    label = f"dmid_ticks_time_{horizon_ms}_lat{latency_ms}"
    t = pq.read_table(data_root / "runs" / run_id / "predictions.parquet")
    days = np.array([d.isoformat() for d in t.column("day").to_pylist()])
    ts = t.column("ts").to_numpy()
    yhat = t.column("yhat").to_numpy()
    mu = t.column("train_mean").to_numpy()
    out = []
    for d in sorted(set(days)):
        m = days == d
        panel = load_day(data_root, venue, symbol, date.fromisoformat(d), "time", [], [label])
        pos = np.searchsorted(panel.ts, ts[m])
        pos_c = np.minimum(pos, panel.n - 1)
        hit = (pos < panel.n) & (panel.ts[pos_c] == ts[m])
        y = np.where(hit, panel.labels[label][pos_c], np.nan)
        out.append(day_stat(d, y, yhat[m], mu[m]))
    return out


def write_rows(path: Path, rows: list[CurveRow], key: tuple[str, ...]) -> None:
    """Merge rows into a CSV, replacing earlier rows with the same key."""
    old: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            old = list(csv.DictReader(fh))
    new = [{k: _fmt(v) for k, v in asdict(r).items()} for r in rows]
    keys = {tuple(r[k] for k in key) for r in new}
    kept = [r for r in old if tuple(r[k] for k in key) not in keys]
    merged = sorted(
        kept + new,
        key=lambda r: (
            r["venue"],
            r["symbol"],
            r["clock"],
            r["model"],
            int(r["horizon"]),
            int(r["latency_ms"]),
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])), lineterminator="\n")
        w.writeheader()
        w.writerows(merged)


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return "" if not math.isfinite(v) else f"{v:.6g}"
    return str(v)


@dataclass
class HalfLife:
    half_life_ms: float | None
    lo: float | None
    hi: float | None
    r2_l0: float
    n_days: int
    share_undefined: float  # bootstrap resamples where the half-life is off the grid


def half_life_ci(
    lat_stats: dict[int, list[DayStat]], resamples: int = 2000, seed: int = SEED
) -> HalfLife:
    """Half-life and its day-block bootstrap 95% CI (resamples off the grid are left out
    of the quantiles and counted)."""
    lats = sorted(lat_stats)
    days = [s.day for s in lat_stats[lats[0]]]
    sse = np.array([[next(s.sse for s in lat_stats[L] if s.day == d) for d in days] for L in lats])
    sst = np.array([[next(s.sst for s in lat_stats[L] if s.day == d) for d in days] for L in lats])
    r2 = 1.0 - sse.sum(axis=1) / sst.sum(axis=1)
    point = half_life(lats, [float(x) for x in r2])
    if len(days) < 2:
        return HalfLife(point, None, None, float(r2[0]), len(days), float("nan"))
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(days), size=(resamples, len(days)))
    vals = []
    for row in k:
        rb = 1.0 - sse[:, row].sum(axis=1) / sst[:, row].sum(axis=1)
        vals.append(half_life(lats, [float(x) for x in rb]))
    good = np.array([v for v in vals if v is not None], dtype=np.float64)
    lo = float(np.quantile(good, 0.025)) if len(good) else None
    hi = float(np.quantile(good, 0.975)) if len(good) else None
    return HalfLife(point, lo, hi, float(r2[0]), len(days), 1.0 - len(good) / resamples)


def write_half_life(path: Path, meta: dict[str, str | int], horizon: int, hl: HalfLife) -> None:
    cols = [
        "venue",
        "symbol",
        "clock",
        "model",
        "run_id",
        "horizon",
        "r2_l0",
        "half_life_ms",
        "lo",
        "hi",
        "n_days",
        "share_undefined",
    ]
    row = {
        "venue": str(meta["venue"]),
        "symbol": str(meta["symbol"]),
        "clock": str(meta["clock"]),
        "model": str(meta["model"]),
        "run_id": str(meta["run_id"]),
        "horizon": str(horizon),
        "r2_l0": _fmt(hl.r2_l0),
        "half_life_ms": _fmt(hl.half_life_ms) if hl.half_life_ms is not None else "",
        "lo": _fmt(hl.lo) if hl.lo is not None else "",
        "hi": _fmt(hl.hi) if hl.hi is not None else "",
        "n_days": str(hl.n_days),
        "share_undefined": _fmt(hl.share_undefined),
    }
    old: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            old = list(csv.DictReader(fh))
    key = ("venue", "symbol", "clock", "model", "horizon")
    kept = [r for r in old if tuple(r[k] for k in key) != tuple(row[k] for k in key)]
    merged = sorted([*kept, row], key=lambda r: tuple(r[k] for k in key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(merged)
