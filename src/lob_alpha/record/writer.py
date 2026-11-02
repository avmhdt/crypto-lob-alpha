"""Hourly zstd Parquet writers with atomic rename (FR-08, DQ-13).

Rows are buffered in lists on the event loop and handed to one writer thread, so a slow disk
can delay a file but never the moment `ts_recv` is taken. D: is a spinning disk shared with
other projects, so this matters.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq

from lob_alpha import schemas
from lob_alpha.record.durable import durable_replace
from lob_alpha.record.sync import GapRow
from lob_alpha.record.venues.base import BboEvent, Level, TradeEvent
from lob_alpha.settings import WriterConfig

NS_PER_HOUR = 3_600_000_000_000
SOURCE_FEED = "feed"


class PartNamer(Protocol):
    """Picks the final path of a new hourly part."""

    def __call__(self, directory: Path, hour_stem: str, run: str) -> Path | None: ...


def free_part(directory: Path, hour_stem: str, run: str) -> Path:
    """First `part-HH[-rN].parquet` with no closed file and no leftover `.tmp`."""
    n = 0
    while True:
        name = f"{hour_stem}.parquet" if n == 0 else f"{hour_stem}-r{n}.parquet"
        path = directory / name
        if not path.exists() and not path.with_name(name + ".tmp").exists():
            return path
        n += 1


def partition_dir(root: Path, dataset: str, venue: str, symbol: str | None, hour: int) -> Path:
    """Hive path for one hour. The symbol is percent-encoded (DQ-12)."""
    day = datetime.fromtimestamp(hour * 3600, tz=UTC).strftime("%Y-%m-%d")
    path = root / dataset / f"venue={venue}"
    if symbol is not None:
        path = path / f"symbol={quote(symbol, safe='')}"
    return path / f"date={day}"


def hour_stem(hour: int) -> str:
    return "part-" + datetime.fromtimestamp(hour * 3600, tz=UTC).strftime("%H")


@dataclass(slots=True)
class _Buffer:
    dataset: str
    venue: str
    symbol: str | None
    hour: int
    run: str
    columns: dict[str, list[Any]]
    rows: int = 0
    first_ns: int = field(default_factory=time.monotonic_ns)


@dataclass(slots=True)
class _Open:
    writer: pq.ParquetWriter
    tmp: Path
    final: Path


_Key = tuple[str, str, str | None, int, str]
_Item = tuple[str, Any, Any]


class ParquetSink:
    """Implements `sync.Sink` over hourly partitioned Parquet files."""

    def __init__(
        self,
        root: Path,
        cfg: WriterConfig,
        *,
        run: str = "live",
        namer: PartNamer = free_part,
        write_gaps: bool = True,
        hours: set[int] | None = None,
    ) -> None:
        self.root = root
        self.cfg = cfg
        self.run = run
        self.namer = namer
        self.write_gaps = write_gaps
        self.hours = hours
        self.schemas = {k: v for k, v in schemas.DATASETS.items() if k != "gaps"}
        self._names = {name: schema.names for name, schema in self.schemas.items()}
        self._buffers: dict[_Key, _Buffer] = {}
        self._queue: queue.Queue[_Item | None] = queue.Queue()
        self._open: dict[_Key, _Open] = {}
        self.rows_written = 0
        self.files_closed = 0
        self.flush_errors = 0
        self.last_error = ""
        self._thread = threading.Thread(target=self._drain, name="parquet-writer", daemon=True)
        self._thread.start()

    # ---- rows, on the caller's thread -----------------------------------------------------

    def _buffer(self, dataset: str, venue: str, symbol: str | None, ts_recv: int) -> _Buffer | None:
        hour = ts_recv // NS_PER_HOUR
        if self.hours is not None and hour not in self.hours:
            return None
        key = (dataset, venue, symbol, hour, self.run)
        buffer = self._buffers.get(key)
        if buffer is None:
            columns: dict[str, list[Any]] = {name: [] for name in self._names[dataset]}
            buffer = _Buffer(dataset, venue, symbol, hour, self.run, columns)
            self._buffers[key] = buffer
        return buffer

    def book_batch(
        self,
        venue: str,
        symbol: str,
        ts_exchange: int,
        ts_recv: int,
        seq: int,
        batch_id: int,
        is_snapshot: bool,
        seq_first: int | None,
        seq_prev: int | None,
        sync_epoch: int,
        bids: Sequence[Level],
        asks: Sequence[Level],
    ) -> None:
        buffer = self._buffer("book_l2", venue, symbol, ts_recv)
        if buffer is None:
            return
        c = buffer.columns
        for side, levels in ((1, bids), (-1, asks)):
            n = len(levels)
            if not n:
                continue
            c["ts_exchange"].extend([ts_exchange] * n)
            c["ts_recv"].extend([ts_recv] * n)
            c["venue"].extend([venue] * n)
            c["symbol"].extend([symbol] * n)
            c["seq"].extend([seq] * n)
            c["side"].extend([side] * n)
            c["is_snapshot"].extend([is_snapshot] * n)
            c["batch_id"].extend([batch_id] * n)
            c["seq_first"].extend([seq_first] * n)
            c["seq_prev"].extend([seq_prev] * n)
            c["sync_epoch"].extend([sync_epoch] * n)
            for px, qty, px_str, qty_str in levels:
                c["price"].append(px)
                c["qty"].append(qty)
                c["px_str"].append(px_str)
                c["qty_str"].append(qty_str)
            buffer.rows += n

    def trade(self, venue: str, ev: TradeEvent, batch_id: int) -> None:
        buffer = self._buffer("trades", venue, ev.symbol, ev.ts_recv)
        if buffer is None:
            return
        c = buffer.columns
        c["ts_exchange"].append(ev.ts_exchange)
        c["ts_recv"].append(ev.ts_recv)
        c["venue"].append(venue)
        c["symbol"].append(ev.symbol)
        c["trade_id"].append(ev.trade_id)
        c["price"].append(ev.price)
        c["qty"].append(ev.qty)
        c["side"].append(ev.side)
        c["batch_id"].append(batch_id)
        c["is_aggregated"].append(ev.is_aggregated)
        c["seq"].append(ev.seq)
        buffer.rows += 1

    def bbo(self, venue: str, ev: BboEvent) -> None:
        buffer = self._buffer("bbo", venue, ev.symbol, ev.ts_recv)
        if buffer is None:
            return
        c = buffer.columns
        c["ts"].append(ev.ts)
        c["venue"].append(venue)
        c["symbol"].append(ev.symbol)
        c["bid_px"].append(ev.bid_px)
        c["bid_qty"].append(ev.bid_qty)
        c["ask_px"].append(ev.ask_px)
        c["ask_qty"].append(ev.ask_qty)
        c["ts_recv"].append(ev.ts_recv)
        c["seq"].append(ev.seq)
        c["source"].append(SOURCE_FEED)
        buffer.rows += 1

    def checkpoint(
        self,
        venue: str,
        symbol: str,
        checkpoint_id: str,
        ts_recv: int,
        kind: str,
        seq: int,
        batch_id: int,
        expected: int | None,
        computed: int | None,
        ok: bool,
        n_levels: int,
    ) -> None:
        buffer = self._buffer("checkpoints", venue, symbol, ts_recv)
        if buffer is None:
            return
        c = buffer.columns
        c["checkpoint_id"].append(checkpoint_id)
        c["ts_recv"].append(ts_recv)
        c["venue"].append(venue)
        c["symbol"].append(symbol)
        c["kind"].append(kind)
        c["seq"].append(seq)
        c["batch_id"].append(batch_id)
        c["expected"].append(expected)
        c["computed_live"].append(computed)
        c["ok_live"].append(ok)
        c["n_levels"].append(n_levels)
        buffer.rows += 1

    def checkpoint_levels(
        self,
        venue: str,
        symbol: str,
        ts_recv: int,
        checkpoint_id: str,
        bids: Sequence[Level],
        asks: Sequence[Level],
    ) -> None:
        buffer = self._buffer("checkpoint_levels", venue, symbol, ts_recv)
        if buffer is None:
            return
        c = buffer.columns
        for side, levels in ((1, bids), (-1, asks)):
            for px, qty, _, _ in levels:
                c["checkpoint_id"].append(checkpoint_id)
                c["side"].append(side)
                c["price"].append(px)
                c["qty"].append(qty)
            buffer.rows += len(levels)

    def clock(
        self, venue: str, ts_recv: int, server_time_ns: int, rtt_ns: int, offset_ns: int
    ) -> None:
        buffer = self._buffer("clock", venue, None, ts_recv)
        if buffer is None:
            return
        c = buffer.columns
        c["ts_recv"].append(ts_recv)
        c["venue"].append(venue)
        c["server_time_ns"].append(server_time_ns)
        c["rtt_ns"].append(rtt_ns)
        c["offset_ns"].append(offset_ns)
        buffer.rows += 1

    def gap(self, row: GapRow) -> None:
        """Gap rows go to disk at once, so a later crash cannot lose them."""
        if self.write_gaps:
            self._queue.put(("gap", row, None))

    def set_run(self, run: str) -> None:
        """Start a new run: its rows go to their own parts, as they did live."""
        self.tick(force=True)
        self.run = run

    def tick(self, force: bool = False) -> None:
        """Hand full or old buffers to the writer thread and close finished hours."""
        now = time.monotonic_ns()
        current_hour = time.time_ns() // NS_PER_HOUR
        interval = int(self.cfg.flush_interval_s * 1e9)
        for key, buffer in list(self._buffers.items()):
            hour_done = force or buffer.hour < current_hour or buffer.run != self.run
            if (
                hour_done
                or buffer.rows >= self.cfg.row_group_rows
                or now - buffer.first_ns >= interval
            ):
                del self._buffers[key]
                self._queue.put(("write", key, buffer))
                if hour_done:
                    self._queue.put(("close", key, None))
        for key in list(self._open):
            if key[3] < current_hour:
                self._queue.put(("close", key, None))

    def close(self) -> None:
        """Flush everything, close every open file and stop the thread."""
        self.tick(force=True)
        self._queue.put(("close_all", None, None))
        self._queue.put(None)
        self._thread.join()

    def wait_idle(self) -> None:
        """Block until the writer thread has caught up. For tests and recovery."""
        self._queue.join()

    # ---- writer thread --------------------------------------------------------------------

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                action, key, payload = item
                if action == "write":
                    self._write(key, payload)
                elif action == "close":
                    self._close(key)
                elif action == "close_all":
                    for open_key in list(self._open):
                        self._close(open_key)
                elif action == "gap":
                    self._write_gap(key)
            except Exception as exc:  # the recorder keeps running; the raw archive still has it
                self.flush_errors += 1
                self.last_error = repr(exc)
            finally:
                self._queue.task_done()

    def _write(self, key: _Key, buffer: _Buffer) -> None:
        if not buffer.rows:
            return
        schema = self.schemas[buffer.dataset]
        arrays = [pa.array(buffer.columns[f.name], type=f.type) for f in schema]
        table = pa.Table.from_arrays(arrays, schema=schema)
        handle = self._open.get(key)
        if handle is None:
            directory = partition_dir(
                self.root, buffer.dataset, buffer.venue, buffer.symbol, buffer.hour
            )
            directory.mkdir(parents=True, exist_ok=True)
            final = self.namer(directory, hour_stem(buffer.hour), buffer.run)
            if final is None:
                return
            tmp = final.with_name(final.name + ".tmp")
            writer = pq.ParquetWriter(
                tmp,
                schema,
                compression=self.cfg.compression,
                compression_level=self.cfg.compression_level,
            )
            handle = _Open(writer, tmp, final)
            self._open[key] = handle
        handle.writer.write_table(table, row_group_size=self.cfg.row_group_rows)
        self.rows_written += buffer.rows

    def _close(self, key: _Key) -> None:
        handle = self._open.pop(key, None)
        if handle is None:
            return
        handle.writer.close()
        durable_replace(handle.tmp, handle.final)
        self.files_closed += 1

    def _write_gap(self, row: GapRow) -> None:
        hour = row.start_ns // NS_PER_HOUR
        directory = partition_dir(self.root, "gaps", row.venue, row.symbol, hour)
        directory.mkdir(parents=True, exist_ok=True)
        final = free_part(directory, hour_stem(hour), self.run)
        tmp = final.with_name(final.name + ".tmp")
        data = {f.name: [getattr(row, f.name)] for f in schemas.GAPS}
        table = pa.Table.from_pydict(data, schema=schemas.GAPS)
        pq.write_table(table, tmp, compression=self.cfg.compression)
        durable_replace(tmp, final)
