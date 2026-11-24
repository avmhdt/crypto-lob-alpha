"""Event time against clock time at matched horizons (FR-27, AC-H-05).

For a stream and a clock horizon H ms, the matched event horizon is the label horizon
h (in book batches, from `labels.horizons_events`) whose median clock duration on the
training days of the event run's first fold is closest to H on a log scale. Durations are
ts[i + h] - ts[i] on the event grid, within LIVE stretches. The comparison reports both
runs' pooled OOS R2 with day-block 95% CIs (decay.bootstrap) and the median duration that
was matched, so a reader can see how close the match is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from lob_alpha.models.data import load_day


def median_event_duration_ms(
    data_root: Path, venue: str, symbol: str, days: list[date], horizons: list[int]
) -> dict[int, float]:
    """Median clock duration of h book batches, per h, over `days` (event grid)."""
    out: dict[int, list[float]] = {h: [] for h in horizons}
    for d in days:
        p = load_day(data_root, venue, symbol, d, "event", [], [])
        for h in horizons:
            if p.n > h:
                dur = (p.ts[h:] - p.ts[:-h]) / 1e6
                out[h].append(float(np.median(dur)))
    return {h: float(np.median(v)) if v else float("nan") for h, v in out.items()}


def match(clock_ms: int, durations: dict[int, float]) -> int:
    ok = {h: v for h, v in durations.items() if v > 0 and math.isfinite(v)}
    return min(ok, key=lambda h: abs(math.log(ok[h]) - math.log(clock_ms)))


@dataclass
class Comparison:
    venue: str
    symbol: str
    clock_horizon_ms: int
    event_horizon: int
    event_median_ms: float
    clock_run: str
    event_run: str
    clock_r2: float
    clock_lo: float
    clock_hi: float
    event_r2: float
    event_lo: float
    event_hi: float
    n_days_clock: int
    n_days_event: int


def write_table(path: Path, rows: list[Comparison]) -> None:
    import csv  # noqa: PLC0415
    from dataclasses import asdict  # noqa: PLC0415

    from lob_alpha.research.decay import _fmt  # noqa: PLC0415

    key = ("venue", "symbol", "clock_horizon_ms")
    new = [{k: _fmt(v) for k, v in asdict(r).items()} for r in rows]
    keys = {tuple(r[k] for k in key) for r in new}
    old: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            old = [r for r in csv.DictReader(fh) if tuple(r[k] for k in key) not in keys]
    merged = sorted(old + new, key=lambda r: (r["venue"], r["symbol"], int(r["clock_horizon_ms"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])), lineterminator="\n")
        w.writeheader()
        w.writerows(merged)
