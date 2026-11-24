"""Seeded synthetic capture for golden tests: interim/topn, trades and gaps across an hour.

Two venues, one symbol each, from 10:59:00 to 11:01:30 UTC on 2026-11-09. Each venue has a
5 s book gap at 11:00:20 followed by a new sync epoch. Run as a script to write into a
directory, or call `write_capture(root)`.
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lob_alpha import schemas
from lob_alpha.validate.rebuild import TOPN, partition

DAY = date(2026, 11, 9)
NS = 1_000_000_000
VENUES = (("binance-usdm", "BTCUSDT", 11), ("kraken", "BTC/USD", 12))
GAP = (11, 0, 20, 25)  # hour, minute, second start, second end


def _t(h: int, m: int, s: float) -> int:
    return int(datetime(DAY.year, DAY.month, DAY.day, h, m, tzinfo=UTC).timestamp()) * NS + int(
        s * NS
    )


def _book(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t0, t1 = _t(10, 59, 0), _t(11, 1, 30)
    ts = np.arange(t0, t1, 150_000_000, dtype=np.int64) + rng.integers(
        0, 40_000_000, (t1 - t0) // 150_000_000
    )
    ts = np.sort(ts)
    gs, ge = _t(GAP[0], GAP[1], GAP[2]), _t(GAP[0], GAP[1], GAP[3])
    ts = ts[(ts < gs) | (ts > ge)]
    n = len(ts)
    mid = 1_000_000 + np.cumsum(rng.choice([-1, 0, 0, 0, 1], n))
    spread = rng.choice([1, 1, 1, 2], n)
    bid1 = mid - spread // 2 - 1 + 1
    ask1 = bid1 + spread
    cols: dict[str, np.ndarray] = {
        "ts_recv": ts,
        "ts_exchange": ts - 50_000_000,
        "batch_id": ts.copy(),
    }
    cols["event_idx"] = np.arange(n, dtype=np.int64)
    for k in range(1, 11):
        cols[f"bid_px_{k}"] = bid1 - (k - 1)
        cols[f"bid_qty_{k}"] = np.round(rng.uniform(0.001, 5.0, n), 6)
        cols[f"ask_px_{k}"] = ask1 + (k - 1)
        cols[f"ask_qty_{k}"] = np.round(rng.uniform(0.001, 5.0, n), 6)
    cols["sync_epoch"] = np.where(ts > ge, 2, 1).astype(np.int32)
    return cols


def write_capture(root: Path) -> None:
    gs, ge = _t(GAP[0], GAP[1], GAP[2]), _t(GAP[0], GAP[1], GAP[3])
    for venue, symbol, seed in VENUES:
        cols = _book(seed)
        hours = (cols["ts_recv"] // (3600 * NS)) % 24
        for h in np.unique(hours):
            sel = hours == h
            table = pa.Table.from_pydict(
                {k: pa.array(v[sel]) for k, v in cols.items()}, schema=TOPN
            )
            folder = partition(root / "interim", "topn", venue, symbol, DAY)
            folder.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, folder / f"part-{int(h):02d}.parquet")
        rng = np.random.default_rng(seed + 100)
        tt = np.sort(rng.integers(_t(10, 59, 0), _t(11, 1, 30), 600)).astype(np.int64)
        n = len(tt)
        trades = {
            "ts_exchange": tt - 1000, "ts_recv": tt, "venue": [venue] * n, "symbol": [symbol] * n,
            "trade_id": [str(i) for i in range(n)], "price": np.full(n, 100000.0),
            "qty": np.round(rng.uniform(0.001, 2.0, n), 6),
            "side": rng.choice([1, -1], n).astype(np.int8),
            "batch_id": np.arange(n), "is_aggregated": [False] * n, "seq": [None] * n,
        }  # fmt: skip
        tf = partition(root, "trades", venue, symbol, DAY)
        tf.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pydict(trades, schema=schemas.TRADES), tf / "part-10.parquet")
        gap = {"gap_id": [f"g-{venue}"], "venue": [venue], "symbol": [symbol],
               "stream": ["book"], "start_ns": [gs], "end_ns": [ge],
               "duration_ms": [(ge - gs) / 1e6], "reason": ["disconnect"], "seq_before": [None],
               "seq_after": [None], "host": ["h"], "recorder_version": ["v"]}  # fmt: skip
        gfold = partition(root, "gaps", venue, symbol, DAY)
        gfold.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pydict(gap, schema=schemas.GAPS), gfold / "part-11.parquet")


if __name__ == "__main__":
    write_capture(Path(sys.argv[1]))
