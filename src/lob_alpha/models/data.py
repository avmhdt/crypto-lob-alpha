"""Read one (venue, symbol, day, clock) panel of stored features and labels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from lob_alpha.validate.rebuild import partition

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass
class DayPanel:
    day: date
    ts: I64
    cols: dict[str, F64]
    labels: dict[str, F64]

    @property
    def n(self) -> int:
        return len(self.ts)

    def take(self, keep: npt.NDArray[np.bool_]) -> DayPanel:
        return DayPanel(
            self.day,
            self.ts[keep],
            {k: v[keep] for k, v in self.cols.items()},
            {k: v[keep] for k, v in self.labels.items()},
        )


def _read(folder: Path, columns: list[str]) -> pa.Table:
    files = sorted(folder.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {folder}")
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in files])


def _f64(table: pa.Table, name: str) -> F64:
    col = table.column(name).cast(pa.float64())
    return np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float64)


def load_day(
    root: Path,
    venue: str,
    symbol: str,
    day: date,
    clock: str,
    columns: list[str],
    labels: list[str],
) -> DayPanel:
    """Feature columns and label columns joined on `ts` (rows in time order)."""
    proc = root / "processed"
    feats = _read(
        partition(proc, "features", venue, symbol, day) / f"clock={clock}", ["ts", *columns]
    )
    labs = _read(partition(proc, "labels", venue, symbol, day) / f"clock={clock}", ["ts", *labels])
    ts = np.asarray(feats.column("ts").to_numpy(), dtype=np.int64)
    lts = np.asarray(labs.column("ts").to_numpy(), dtype=np.int64)
    cols = {c: _f64(feats, c) for c in columns}
    lab = {c: _f64(labs, c) for c in labels}
    if not np.array_equal(ts, lts):
        pos = np.searchsorted(lts, ts)
        pos_c = np.minimum(pos, len(lts) - 1)
        hit = (pos < len(lts)) & (lts[pos_c] == ts)
        lab = {c: np.where(hit, v[pos_c], np.nan) for c, v in lab.items()}
    if np.any(np.diff(ts) < 0):
        raise ValueError(f"{venue} {symbol} {day} {clock}: ts not sorted")
    return DayPanel(day, ts, cols, lab)
