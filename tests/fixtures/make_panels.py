"""Synthetic processed feature/label panels for model, export and leakage tests.

Each day has 12:00 to 12:00 + `minutes` UTC on the 100 ms grid, every stored column the
model feature sets read, and a label with a planted relation to depth-normalised OFI:

    dmid_ticks_time_1000 = 0.8 * ofi_l1_ms1000 / depth_l1 + 0.3 * (qi_l1) + noise
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lob_alpha.models import featureset
from lob_alpha.models.splits import day_start_ns
from lob_alpha.validate.rebuild import partition

VENUE, SYMBOL = "binance-usdm", "SYNUSDT"
TARGET = "dmid_ticks_time_1000"
GRID_NS = 100_000_000
FEATURES = [
    "ofi_l1_ms100",
    "ofi_l1_ms1000",
    "mlofi_l2_ms1000",
    "integrated_ofi_ms1000",
    "qi_l1",
    "qi_l5",
    "microprice_adj_ticks",
    "wmid_adj_ticks",
    "spread_ticks",
    "tsi_vol_ms1000",
    "tsi_cnt_tr50",
    "midchg_ms1000",
    "rv_ms1000",
]


def make_day(root: Path, day: date, seed: int, minutes: int = 20, beta: float = 0.8) -> None:
    rng = np.random.default_rng(seed)
    n = minutes * 600
    t0 = day_start_ns(day) + 12 * 3_600_000_000_000
    ts = t0 + np.arange(n, dtype=np.int64) * GRID_NS
    cols: dict[str, np.ndarray] = {}
    depth = rng.uniform(5.0, 15.0, n)
    cols["depth_l1"] = depth
    imb = np.clip(rng.beta(2, 2, n), 0.0, 1.0)
    cols["imb_l1"] = imb
    cols["qi_l1"] = 2 * imb - 1
    cols["qi_l5"] = np.clip(cols["qi_l1"] * 0.5 + rng.normal(0, 0.2, n), -1, 1)
    spread = rng.choice([1.0, 2.0, 3.0], n, p=[0.85, 0.1, 0.05])
    cols["spread_ticks"] = spread
    cols["mid_ticks"] = 500_000 + np.cumsum(rng.choice([-0.5, 0.0, 0.5], n, p=[0.1, 0.8, 0.1]))
    cols["wmid_adj_ticks"] = (imb - 0.5) * spread
    for m in range(1, 11):
        cols[f"mlofi_l{m}_ms1000"] = rng.normal(0, 10.0 / m, n)
    cols["ofi_l1_ms1000"] = cols["mlofi_l1_ms1000"]
    cols["ofi_l1_ms100"] = cols["ofi_l1_ms1000"] * 0.3 + rng.normal(0, 2, n)
    cols["tsi_vol_ms1000"] = rng.uniform(-1, 1, n)
    cols["tsi_cnt_tr50"] = rng.uniform(-1, 1, n)
    cols["midchg_ms1000"] = rng.poisson(1.0, n).astype(np.float64)
    cols["rv_ms1000"] = rng.gamma(2.0, 0.5, n)
    cols["ofi_l1_ms100"][:5] = np.nan  # warm-up rows
    needed = featureset.stored_columns(FEATURES)
    assert set(needed) <= set(cols), set(needed) - set(cols)
    y = beta * cols["ofi_l1_ms1000"] / depth + 0.3 * cols["qi_l1"] + rng.normal(0, 1.0, n)
    y[-10:] = np.nan  # label runs past the data
    feat = pa.table({"ts": pa.array(ts, pa.int64())} | {k: pa.array(v) for k, v in cols.items()})
    lab = pa.table({"ts": pa.array(ts, pa.int64()), TARGET: pa.array(y)})
    for name, table in (("features", feat), ("labels", lab)):
        out = partition(root / "processed", name, VENUE, SYMBOL, day) / "clock=time"
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out / "part-12.parquet")
    topn = partition(root / "interim", "topn", VENUE, SYMBOL, day)
    topn.mkdir(parents=True, exist_ok=True)
    bt = np.sort(rng.integers(t0 - 1_000_000_000, ts[-1], 4 * n)).astype(np.int64)
    pq.write_table(
        pa.table({"ts_recv": pa.array(bt), "ts_exchange": pa.array(bt - 3_000_000)}),
        topn / "part-12.parquet",
    )


def make_days(root: Path, first: date, n_days: int, skip: tuple[int, ...] = ()) -> list[date]:
    days = []
    for k in range(n_days + len(skip)):
        if k in skip:
            continue
        d = first + timedelta(days=k)
        make_day(root, d, seed=100 + k)
        days.append(d)
    return days
