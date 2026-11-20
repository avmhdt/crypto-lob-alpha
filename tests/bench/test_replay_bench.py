"""Replay kernel throughput (NFR-03: >= 1,000,000 level updates/s on one core).

Run with `pytest tests/bench --benchmark-only`. The workload is Binance-like: batches of 20
level updates spread around a moving mid, a tenth of them removals, top 10 recorded after
every batch. The result line is appended to reports/tables/benchmarks.csv.
"""

from __future__ import annotations

import csv
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from lob_alpha.book.replay import CRC_TABLE, TOP_N, BookState, empty_probes, run_batches

N_BATCHES = 50_000
PER_BATCH = 20
BENCH_CSV = Path("reports/tables/benchmarks.csv")


def _workload(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = N_BATCHES * PER_BATCH
    mid = 50_000 + np.cumsum(rng.integers(-2, 3, N_BATCHES)).repeat(PER_BATCH)
    offset = rng.integers(1, 200, n)
    sides = rng.choice(np.array([1, -1], dtype=np.int8), n)
    ticks = np.where(sides == 1, mid - offset, mid + offset).astype(np.int64)
    qtys = np.where(rng.random(n) < 0.1, 0.0, rng.random(n) * 5 + 0.01)
    bstart = np.arange(0, n + 1, PER_BATCH, dtype=np.int64)
    bsnap = np.zeros(N_BATCHES, dtype=np.bool_)
    bsnap[0] = True
    return {"sides": sides, "ticks": ticks, "qtys": qtys, "bstart": bstart, "bsnap": bsnap}


def _run(w: dict[str, np.ndarray]) -> None:
    state = BookState(40_000, 20_000, raw=False)
    nb = N_BATCHES
    empty = np.zeros((0, 1), np.uint8)
    pr = empty_probes()
    run_batches(
        state.qty, state.meta, state.px_raw, state.px_len, state.qty_raw, state.qty_len,
        state.lo, 0, w["sides"], w["ticks"], w["qtys"], empty, np.zeros(0, np.uint8), empty,
        np.zeros(0, np.uint8), w["bstart"], w["bsnap"], 0, nb,
        np.empty((nb, TOP_N), np.int64), np.empty((nb, TOP_N)), np.empty((nb, TOP_N), np.int64),
        np.empty((nb, TOP_N)), np.zeros(nb, np.int8), np.zeros(nb, np.bool_),
        np.full(nb, -1, np.int64), CRC_TABLE, pr["pr_batch"], pr["pr_snap_off"], pr["snap_tick"],
        pr["snap_qty"], pr["pr_ex_off"], pr["ex_tick"], pr["pr_out"],
    )  # fmt: skip


def test_replay_kernel_updates_per_second(benchmark: pytest.FixtureRequest) -> None:
    w = _workload()
    _run(w)  # compile outside the timing
    t0 = time.perf_counter()
    benchmark(_run, w)
    elapsed = benchmark.stats.stats.mean if benchmark.stats else time.perf_counter() - t0
    rate = N_BATCHES * PER_BATCH / elapsed
    BENCH_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not BENCH_CSV.exists()
    with BENCH_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["utc", "benchmark", "value", "unit", "target", "machine", "python"])
        writer.writerow(
            [datetime.now(tz=UTC).isoformat(timespec="seconds"), "replay_kernel_level_updates",
             round(rate), "updates/s", 1_000_000,
             f"{platform.processor() or platform.machine()} ({os.cpu_count()} cpus)",
             platform.python_version()]
        )  # fmt: skip
    assert rate > 0
