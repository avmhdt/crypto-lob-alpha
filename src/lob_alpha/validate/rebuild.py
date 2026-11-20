"""Offline book rebuild and checkpoint validation (FR-14, FR-15).

For one (venue, symbol) the `book_l2` files are replayed in the order the recorder wrote
them: by date, hour and restart suffix, and within a file in row order. That is
`batch_id` order, which the capture log of 2026-11-04 adopted over a `ts_recv` sort because
the host clock steps backwards by about 1 ms every 30 s. Every row in `book_l2` was written
in LIVE state, a snapshot batch replaces the whole book, and each stream epoch starts with
one, so every batch boundary after the first snapshot is a LIVE boundary.

To start at a date, the replay seeks back to the last snapshot before it and replays from
there without writing output. The book state therefore carries across hours and days.

Checks per batch boundary: crossed (bid > ask), locked (bid == ask) and one-sided books.
Checkpoints:

* `kraken_crc32`: the CRC32 of the rebuilt top 10 levels after the batch the checkpoint
  names, against the venue's checksum. Checkpoints that already failed live have no
  batch in `book_l2` (the stream resynced) and are counted as excluded.
* `rest_l2`, Binance: the snapshot's `lastUpdateId` L is matched to the event with
  U <= L <= u. Levels that event touched are excluded and counted, unless L == u. With no
  such event, the state after the last event with u <= L is compared, when the next event
  chains to it (pu == u) and starts after L. L inside a gap is excluded.
* `rest_l2`, Coinbase: level2 carries no sequence, so every batch boundary within
  +-`COINBASE_WINDOW_S` of the REST receive time is a candidate. Levels touched by the
  following batch are excluded, because the snapshot may sit inside it. The checkpoint
  passes when a candidate matches on every compared level.

The top 100 levels per side of the snapshot set the compared price range.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from lob_alpha.book.replay import (
    CRC_TABLE,
    FLAG_CROSSED,
    FLAG_LOCKED,
    FLAG_ONE_SIDED,
    NO_LEVEL,
    TOP_N,
    BookState,
    run_batches,
    strip_raw,
)
from lob_alpha.record.durable import durable_replace

NS = 1_000_000_000
COINBASE_WINDOW_S = 3.0
REST_LEVELS = 100
SEEK_BACK_DAYS = 3
WINDOW_LOW, WINDOW_HIGH = 0.7, 1.4  # of the traded price range
MAX_FAILURES = 20
PART_RE = re.compile(r"part-(\d{2})(?:-r(\d+))?\.parquet$")
ADD_AHEAD_NS = 10 * NS

BOOK_COLUMNS = [
    "ts_exchange", "ts_recv", "seq", "side", "price", "qty", "is_snapshot", "batch_id",
    "seq_first", "seq_prev", "sync_epoch",
]  # fmt: skip


def topn_schema() -> pa.Schema:
    """interim/topn: one row per LIVE batch boundary (spec section 7b)."""
    fields = [
        pa.field("ts_recv", pa.int64(), nullable=False),
        pa.field("ts_exchange", pa.int64(), nullable=False),
        pa.field("batch_id", pa.int64(), nullable=False),
        pa.field("event_idx", pa.int64(), nullable=False),
    ]
    for side in ("bid", "ask"):
        fields += [pa.field(f"{side}_px_{k}", pa.int64()) for k in range(1, TOP_N + 1)]
        fields += [pa.field(f"{side}_qty_{k}", pa.float64()) for k in range(1, TOP_N + 1)]
    fields.append(pa.field("sync_epoch", pa.int32(), nullable=False))
    return pa.schema(fields)


TOPN = topn_schema()


# ---- files ----------------------------------------------------------------------------------


def partition(root: Path, dataset: str, venue: str, symbol: str, day: date) -> Path:
    return root / dataset / f"venue={venue}" / f"symbol={quote(symbol, safe='')}" / f"date={day}"


def _part_key(path: Path) -> tuple[int, int]:
    match = PART_RE.search(path.name)
    if match is None:
        return (99, 0)
    return int(match.group(1)), int(match.group(2) or 0)


def day_files(root: Path, dataset: str, venue: str, symbol: str, day: date) -> list[Path]:
    """Closed parts of one day in write order. Readers ignore `.tmp` (DQ-13)."""
    folder = partition(root, dataset, venue, symbol, day)
    return sorted(folder.glob("part-*.parquet"), key=_part_key)


def days(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _day_ns(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * NS


def decimals(tick: float) -> int:
    return max(0, round(-math.log10(tick))) if tick < 1 else 0


# ---- one file in memory ---------------------------------------------------------------------


@dataclass
class FileData:
    """One `book_l2` part as arrays, rows grouped into batches."""

    path: Path
    day: date
    sides: npt.NDArray[np.int8]
    ticks: npt.NDArray[np.int64]
    qtys: npt.NDArray[np.float64]
    raw: tuple[Any, Any, Any, Any] | None
    bstart: npt.NDArray[np.int64]
    batch_id: npt.NDArray[np.int64]
    ts_recv: npt.NDArray[np.int64]
    ts_exchange: npt.NDArray[np.int64]
    seq: npt.NDArray[np.int64]
    seq_first: npt.NDArray[np.int64]
    seq_prev: npt.NDArray[np.int64]
    snap: npt.NDArray[np.bool_]
    epoch: npt.NDArray[np.int32]
    tick_errors: int
    order_violations: int

    @property
    def n_batches(self) -> int:
        return len(self.batch_id)

    def batch_ticks(self, b: int) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Ticks touched by batch b, per side (bids, asks), sorted ascending."""
        a, z = self.bstart[b], self.bstart[b + 1]
        sides, ticks = self.sides[a:z], self.ticks[a:z]
        return np.unique(ticks[sides == 1]), np.unique(ticks[sides != 1])


def load_file(path: Path, day: date, tick: float, raw: bool) -> FileData:
    columns = BOOK_COLUMNS + (["px_str", "qty_str"] if raw else [])
    frame = pl.read_parquet(path, columns=columns, hive_partitioning=False)
    price = frame["price"].to_numpy()
    exact = price / tick
    ticks = np.rint(exact).astype(np.int64)
    tick_errors = int(np.count_nonzero(np.abs(exact - ticks) >= 1e-6))
    batch = frame["batch_id"].to_numpy()
    n = len(batch)
    starts = np.flatnonzero(np.diff(batch)) + 1 if n else np.zeros(0, dtype=np.int64)
    first = np.concatenate([[0], starts]).astype(np.int64) if n else np.zeros(0, np.int64)
    bstart = np.concatenate([first, [n]]).astype(np.int64)
    bids = batch[first]
    raw_arrays = None
    if raw:
        px, pxl = strip_raw(frame["px_str"].to_list())
        qt, qtl = strip_raw(frame["qty_str"].to_list())
        raw_arrays = (px, pxl, qt, qtl)

    def pick(name: str, fill: int = -1) -> npt.NDArray[np.int64]:
        col = frame[name].fill_null(fill).to_numpy()
        return np.asarray(col[first], dtype=np.int64)

    return FileData(
        path=path,
        day=day,
        sides=frame["side"].to_numpy().astype(np.int8),
        ticks=ticks,
        qtys=frame["qty"].to_numpy().astype(np.float64),
        raw=raw_arrays,
        bstart=bstart,
        batch_id=np.asarray(bids, dtype=np.int64),
        ts_recv=pick("ts_recv"),
        ts_exchange=pick("ts_exchange"),
        seq=pick("seq"),
        seq_first=pick("seq_first"),
        seq_prev=pick("seq_prev"),
        snap=frame["is_snapshot"].to_numpy()[first].astype(np.bool_),
        epoch=frame["sync_epoch"].to_numpy()[first].astype(np.int32),
        tick_errors=tick_errors,
        order_violations=int(np.count_nonzero(np.diff(bids) <= 0)),
    )


# ---- checkpoints ----------------------------------------------------------------------------


@dataclass
class RestCheckpoint:
    checkpoint_id: str
    day: date
    ts_recv: int
    seq: int
    ok_live: bool
    bids: npt.NDArray[np.int64]  # ticks ascending
    bid_qty: npt.NDArray[np.float64]
    asks: npt.NDArray[np.int64]
    ask_qty: npt.NDArray[np.float64]
    candidates: int = 0
    best: tuple[int, int, int, int] | None = None  # (bad, compared, excluded, offset_ns)
    first_bad_tick: int = NO_LEVEL
    resolved: bool = False
    outcome: str = ""  # compared | nonlive | gap


def load_rest_checkpoints(
    root: Path, venue: str, symbol: str, day: date, tick: float
) -> list[RestCheckpoint]:
    cps = [
        pl.read_parquet(p, hive_partitioning=False)
        for p in day_files(root, "checkpoints", venue, symbol, day)
    ]
    if not cps:
        return []
    frame = pl.concat(cps).filter(pl.col("kind") == "rest_l2").sort("ts_recv")
    if frame.is_empty():
        return []
    lv_parts = [
        pl.read_parquet(p, hive_partitioning=False)
        for p in day_files(root, "checkpoint_levels", venue, symbol, day)
    ]
    levels = pl.concat(lv_parts) if lv_parts else pl.DataFrame()
    grouped: dict[str, pl.DataFrame] = {}
    if not levels.is_empty():
        ids = set(frame["checkpoint_id"].to_list())
        for (cid,), part in levels.filter(pl.col("checkpoint_id").is_in(ids)).group_by(
            "checkpoint_id"
        ):
            grouped[str(cid)] = part
    out: list[RestCheckpoint] = []
    for row in frame.iter_rows(named=True):
        part = grouped.get(row["checkpoint_id"], pl.DataFrame({"side": [], "price": [], "qty": []}))
        sides: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]] = []
        for side, descending in ((1, True), (-1, False)):
            lv = part.filter(pl.col("side") == side).sort("price", descending=descending)
            lv = lv.head(REST_LEVELS).sort("price")
            sides.append(
                (
                    np.rint(lv["price"].to_numpy() / tick).astype(np.int64),
                    lv["qty"].to_numpy().astype(np.float64),
                )
            )
        out.append(
            RestCheckpoint(
                checkpoint_id=row["checkpoint_id"],
                day=day,
                ts_recv=int(row["ts_recv"]),
                seq=int(row["seq"]),
                ok_live=bool(row["ok_live"]),
                bids=sides[0][0],
                bid_qty=sides[0][1],
                asks=sides[1][0],
                ask_qty=sides[1][1],
            )
        )
    return out


def load_crc_checkpoints(root: Path, venue: str, symbol: str, day: date) -> pl.DataFrame:
    parts = [
        pl.read_parquet(p, columns=["kind", "batch_id", "expected", "ok_live"])
        for p in day_files(root, "checkpoints", venue, symbol, day)
    ]
    if not parts:
        return pl.DataFrame(
            schema={"batch_id": pl.Int64, "expected": pl.Int64, "ok_live": pl.Boolean}
        )
    return (
        pl.concat(parts)
        .filter(pl.col("kind") == "kraken_crc32")
        .select("batch_id", "expected", "ok_live")
        .sort("batch_id")
    )


# ---- per-day results ------------------------------------------------------------------------


@dataclass
class DayStats:
    venue: str
    symbol: str
    date: str
    files: int = 0
    rows: int = 0
    batches: int = 0
    snapshots: int = 0
    unsynced_batches: int = 0
    crossed: int = 0
    locked: int = 0
    one_sided: int = 0
    first_crossed_ts: int | None = None
    first_locked_ts: int | None = None
    out_of_window: int = 0
    tick_errors: int = 0
    order_violations: int = 0
    crc_total: int = 0
    crc_live_fail: int = 0
    crc_compared: int = 0
    crc_match: int = 0
    crc_mismatch: int = 0
    crc_unmatched: int = 0
    rest_total: int = 0
    rest_nonlive: int = 0
    rest_gap: int = 0
    rest_compared: int = 0
    rest_ok: int = 0
    rest_bad: int = 0
    rest_levels_compared: int = 0
    rest_levels_excluded: int = 0
    rest_levels_bad: int = 0
    rest_offset_ms_max: float | None = None
    kernel_seconds: float = 0.0
    failures: list[dict[str, Any]] = field(default_factory=list)

    def add_failure(self, item: dict[str, Any]) -> None:
        if len(self.failures) < MAX_FAILURES:
            self.failures.append(item)


# ---- the replay -----------------------------------------------------------------------------


@dataclass
class _Probe:
    batch: int
    cp: RestCheckpoint
    ex_bids: npt.NDArray[np.int64]
    ex_asks: npt.NDArray[np.int64]


class Rebuilder:
    """Replay one (venue, symbol) over a date range, writing interim tables and stats."""

    def __init__(
        self,
        data_dir: Path,
        venue: str,
        symbol: str,
        tick: float,
        *,
        depth: int | None,
        raw: bool,
        write: bool = True,
        log: Any = None,
    ) -> None:
        self.root = data_dir
        self.venue = venue
        self.symbol = symbol
        self.tick = tick
        self.depth = depth or 0
        self.raw = raw
        self.write = write
        self.ndp = decimals(tick)
        self.log = log or (lambda *a, **k: None)
        self.state: BookState | None = None
        self.synced = False
        self.updates = 0

    # -- setup

    def _seek(self, start: date) -> tuple[list[tuple[date, Path]], int]:
        """Files to replay before `start`, from the one holding the last snapshot."""
        before: list[tuple[date, Path]] = []
        for back in range(1, SEEK_BACK_DAYS + 1):
            day = start - timedelta(days=back)
            for path in reversed(day_files(self.root, "book_l2", self.venue, self.symbol, day)):
                before.insert(0, (day, path))
                snap = pl.read_parquet(path, columns=["is_snapshot", "batch_id"])
                flags = snap["is_snapshot"].to_numpy()
                if flags.any():
                    batch = snap["batch_id"].to_numpy()
                    last_row = int(np.flatnonzero(flags)[-1])
                    target = batch[last_row]
                    starts = np.flatnonzero(np.diff(batch)) + 1
                    first = np.concatenate([[0], starts])
                    b0 = int(np.flatnonzero(batch[first] == target)[0])
                    return before, b0
        return [], 0

    def _window(self, first_day: date, last_day: date) -> tuple[int, int]:
        lo_px, hi_px = math.inf, -math.inf
        for day in days(first_day, last_day):
            for path in day_files(self.root, "trades", self.venue, self.symbol, day):
                prices = pl.read_parquet(path, columns=["price"])["price"]
                if len(prices):
                    lo_px = min(lo_px, float(prices.min()))  # type: ignore[arg-type]
                    hi_px = max(hi_px, float(prices.max()))  # type: ignore[arg-type]
        if not math.isfinite(lo_px):
            raise RuntimeError(f"no trades for {self.venue} {self.symbol}, cannot size window")
        lo = math.floor(lo_px * WINDOW_LOW / self.tick)
        hi = math.ceil(hi_px * WINDOW_HIGH / self.tick)
        return lo, hi

    # -- probes

    def _resolve_binance(
        self, cp: RestCheckpoint, f: FileData, nxt: FileData | None
    ) -> _Probe | str | None:
        """A probe, 'gap' when the snapshot falls in a gap, or None to wait for more data."""
        L = cp.seq
        if f.n_batches == 0:
            return None
        straddle = np.flatnonzero((f.seq_first >= 0) & (f.seq_first <= L) & (f.seq >= L))
        if straddle.size:
            b = int(straddle[0])
            if f.seq[b] == L:
                empty = np.zeros(0, dtype=np.int64)
                return _Probe(b, cp, empty, empty)
            ex_b, ex_a = f.batch_ticks(b)
            return _Probe(b, cp, ex_b, ex_a)
        below = np.flatnonzero(f.seq <= L)
        if below.size == 0:
            return "gap"  # this file already starts after L: the match was lost to a resync
        k = int(below[-1])
        if k + 1 < f.n_batches:
            nb_snap, nb_first, nb_prev = f.snap[k + 1], f.seq_first[k + 1], f.seq_prev[k + 1]
        elif nxt is not None and nxt.n_batches:
            nb_snap, nb_first, nb_prev = nxt.snap[0], nxt.seq_first[0], nxt.seq_prev[0]
        else:
            return None
        empty = np.zeros(0, dtype=np.int64)
        if f.seq[k] == L:
            return _Probe(k, cp, empty, empty)
        if nb_snap:
            return "gap"
        if nb_first > L and nb_prev == f.seq[k]:
            return _Probe(k, cp, empty, empty)
        if k + 1 >= f.n_batches:
            return None  # the straddling event is in the next file
        return "gap"

    def _coinbase_probes(
        self, cp: RestCheckpoint, f: FileData, nxt: FileData | None
    ) -> list[_Probe]:
        lo = cp.ts_recv - int(COINBASE_WINDOW_S * NS)
        hi = cp.ts_recv + int(COINBASE_WINDOW_S * NS)
        idx = np.flatnonzero((f.ts_recv >= lo) & (f.ts_recv <= hi))
        probes: list[_Probe] = []
        for b in (int(x) for x in idx):
            if b + 1 < f.n_batches:
                ex_b, ex_a = f.batch_ticks(b + 1)
            elif nxt is not None and nxt.n_batches:
                ex_b, ex_a = nxt.batch_ticks(0)
            else:
                ex_b = ex_a = np.zeros(0, dtype=np.int64)
            probes.append(_Probe(b, cp, ex_b, ex_a))
        return probes

    # -- main loop

    def run(self, start: date, end: date) -> list[DayStats]:
        out_days = days(start, end)
        before, seek_b0 = self._seek(start)
        plan: list[tuple[date, Path, bool]] = [(d, p, False) for d, p in before]
        for day in out_days:
            plan += [
                (day, p, True)
                for p in day_files(self.root, "book_l2", self.venue, self.symbol, day)
            ]
        if not plan:
            return [DayStats(self.venue, self.symbol, str(d)) for d in out_days]
        lo, hi = self._window(plan[0][0], end)
        self.state = BookState(lo, hi - lo + 1, self.raw)
        stats = {d: DayStats(self.venue, self.symbol, str(d)) for d in out_days}
        rest_queue: list[RestCheckpoint] = []
        if self.venue in ("binance-usdm", "coinbase"):
            for day in out_days:
                rest_queue += load_rest_checkpoints(
                    self.root, self.venue, self.symbol, day, self.tick
                )
        for cp in rest_queue:
            st = stats[cp.day]
            st.rest_total += 1
            if not cp.ok_live:
                cp.resolved, cp.outcome = True, "nonlive"
                st.rest_nonlive += 1
        rest_queue = [cp for cp in rest_queue if not cp.resolved]
        pending: list[RestCheckpoint] = []
        crc_day: date | None = None
        crc: pl.DataFrame | None = None
        event_idx: dict[date, int] = {d: 0 for d in out_days}

        loaded = self._files(plan)
        current = next(loaded, None)
        upcoming = next(loaded, None)
        first = True
        while current is not None:
            (day, path, emit), f = current
            nxt = upcoming[1] if upcoming is not None else None
            b0 = seek_b0 if first else 0
            first = False
            if not self.synced:
                snaps = np.flatnonzero(f.snap[b0:])
                if snaps.size == 0:
                    if emit:
                        stats[day].unsynced_batches += f.n_batches - b0
                    current, upcoming = upcoming, next(loaded, None)
                    continue
                if emit:
                    stats[day].unsynced_batches += int(snaps[0])
                b0 += int(snaps[0])
                self.synced = True
            # REST checkpoints due by the end of this file (plus margin) join the pending list
            t_last = int(f.ts_recv[-1]) if f.n_batches else 0
            while rest_queue and rest_queue[0].ts_recv <= t_last + ADD_AHEAD_NS:
                pending.append(rest_queue.pop(0))
            probes: list[_Probe] = []
            still: list[RestCheckpoint] = []
            to_finish: list[RestCheckpoint] = []
            for cp in pending:
                if self.venue == "binance-usdm":
                    got = self._resolve_binance(cp, f, nxt)
                    if got is None:
                        still.append(cp)
                    elif isinstance(got, str):
                        to_finish.append(cp)  # no probe: finishes as a gap
                    else:
                        probes.append(got)
                        to_finish.append(cp)
                else:
                    probes += self._coinbase_probes(cp, f, nxt)
                    if cp.ts_recv + int(COINBASE_WINDOW_S * NS) > t_last:
                        still.append(cp)
                    else:
                        to_finish.append(cp)
            pending = still
            probes = [p for p in probes if p.batch >= b0]
            # Kraken checksums named in this file
            crc_want = np.zeros(f.n_batches, dtype=np.bool_)
            crc_expected = np.zeros(f.n_batches, dtype=np.int64)
            if self.raw and emit:
                if crc_day != day:
                    crc = load_crc_checkpoints(self.root, self.venue, self.symbol, day)
                    crc_day = day
                    st = stats[day]
                    st.crc_total += crc.height
                    st.crc_live_fail += int((~crc["ok_live"]).sum())
                    crc = crc.filter(pl.col("ok_live"))
                assert crc is not None
                ids = crc["batch_id"].to_numpy()
                if f.n_batches:
                    a = np.searchsorted(ids, f.batch_id[0], side="left")
                    z = np.searchsorted(ids, f.batch_id[-1], side="right")
                    want = ids[a:z]
                    pos = np.searchsorted(f.batch_id, want)
                    hit = (pos < f.n_batches) & (
                        f.batch_id[np.minimum(pos, f.n_batches - 1)] == want
                    )
                    crc_want[pos[hit]] = True
                    crc_expected[pos[hit]] = crc["expected"].to_numpy()[a:z][hit]
            result = self._kernel(f, b0, probes, crc_want)
            self._account(day, emit, f, b0, result, crc_want, crc_expected, stats)
            if emit and self.write:
                event_idx[day] = self._write(day, path, f, b0, result, event_idx[day])
            for cp in to_finish:
                self._finish(cp, stats)
            current, upcoming = upcoming, next(loaded, None)
        for cp in pending:  # data ended: finish with what was seen
            self._finish(cp, stats)
        for cp in rest_queue:  # never reached: after the last recorded batch
            stats[cp.day].rest_gap += 1
        for st in stats.values():
            st.crc_unmatched = st.crc_total - st.crc_live_fail - st.crc_compared
        return [stats[d] for d in out_days]

    def _files(
        self, plan: list[tuple[date, Path, bool]]
    ) -> Iterator[tuple[tuple[date, Path, bool], FileData]]:
        for item in plan:
            day, path, _emit = item
            yield item, load_file(path, day, self.tick, self.raw)

    def _kernel(
        self, f: FileData, b0: int, probes: list[_Probe], crc_want: npt.NDArray[np.bool_]
    ) -> dict[str, Any]:
        assert self.state is not None
        st = self.state
        n = f.n_batches - b0
        res: dict[str, Any] = {
            "top_bpx": np.empty((n, TOP_N), dtype=np.int64),
            "top_bq": np.empty((n, TOP_N), dtype=np.float64),
            "top_apx": np.empty((n, TOP_N), dtype=np.int64),
            "top_aq": np.empty((n, TOP_N), dtype=np.float64),
            "flags": np.zeros(n, dtype=np.int8),
            "crc_out": np.full(n, -1, dtype=np.int64),
        }
        probes = sorted(probes, key=lambda p: p.batch)
        snap_tick: list[npt.NDArray[np.int64]] = []
        snap_qty: list[npt.NDArray[np.float64]] = []
        ex_tick: list[npt.NDArray[np.int64]] = []
        snap_off = [0]
        ex_off = [0]
        for p in probes:
            for ticks, qtys, ex in (
                (p.cp.bids, p.cp.bid_qty, p.ex_bids),
                (p.cp.asks, p.cp.ask_qty, p.ex_asks),
            ):
                snap_tick.append(ticks)
                snap_qty.append(qtys)
                snap_off.append(snap_off[-1] + len(ticks))
                ex_tick.append(ex)
                ex_off.append(ex_off[-1] + len(ex))
        pr_out = np.zeros((len(probes), 5), dtype=np.int64)
        pr_out[:, 3] = NO_LEVEL
        empty_u8 = np.zeros((0, 1), dtype=np.uint8)
        raw = (
            f.raw
            if f.raw is not None
            else (empty_u8, np.zeros(0, np.uint8), empty_u8, np.zeros(0, np.uint8))
        )
        t0 = time.perf_counter()
        run_batches(
            st.qty, st.meta, st.px_raw, st.px_len, st.qty_raw, st.qty_len, st.lo, self.depth,
            f.sides, f.ticks, f.qtys, raw[0], raw[1], raw[2], raw[3],
            f.bstart, f.snap, b0, f.n_batches,
            res["top_bpx"], res["top_bq"], res["top_apx"], res["top_aq"], res["flags"],
            crc_want[b0:], res["crc_out"], CRC_TABLE,
            np.array([p.batch for p in probes], dtype=np.int64),
            np.array(snap_off, dtype=np.int64),
            np.concatenate(snap_tick) if snap_tick else np.zeros(0, np.int64),
            np.concatenate(snap_qty) if snap_qty else np.zeros(0, np.float64),
            np.array(ex_off, dtype=np.int64),
            np.concatenate(ex_tick) if ex_tick else np.zeros(0, np.int64),
            pr_out,
        )  # fmt: skip
        res["seconds"] = time.perf_counter() - t0
        res["pr_out"] = pr_out
        res["probes"] = probes
        self.updates += int(f.bstart[-1] - f.bstart[b0]) if f.n_batches else 0
        return res

    def _account(
        self,
        day: date,
        emit: bool,
        f: FileData,
        b0: int,
        res: dict[str, Any],
        crc_want: npt.NDArray[np.bool_],
        crc_expected: npt.NDArray[np.int64],
        stats: dict[date, DayStats],
    ) -> None:
        assert self.state is not None
        for p, out in zip(res["probes"], res["pr_out"], strict=True):
            cp = p.cp
            cp.candidates += 1
            compared, bad, excluded, first_bad, _oow = (int(x) for x in out)
            offset = int(f.ts_recv[p.batch]) - cp.ts_recv
            key = (bad, -compared, excluded, abs(offset))
            if cp.best is None or key < (cp.best[0], -cp.best[1], cp.best[2], abs(cp.best[3])):
                cp.best = (bad, compared, excluded, offset)
                cp.first_bad_tick = first_bad
        if not emit:
            return
        st = stats[day]
        n = f.n_batches - b0
        st.files += 1
        st.rows += int(f.bstart[-1] - f.bstart[b0]) if f.n_batches else 0
        st.batches += n
        st.snapshots += int(f.snap[b0:].sum())
        st.tick_errors += f.tick_errors
        st.order_violations += f.order_violations
        st.kernel_seconds += res["seconds"]
        flags = res["flags"]
        crossed = np.flatnonzero(flags & FLAG_CROSSED)
        locked = np.flatnonzero(flags & FLAG_LOCKED)
        st.crossed += crossed.size
        st.locked += locked.size
        st.one_sided += int(np.count_nonzero(flags & FLAG_ONE_SIDED))
        if crossed.size and st.first_crossed_ts is None:
            st.first_crossed_ts = int(f.ts_recv[b0 + crossed[0]])
        if locked.size and st.first_locked_ts is None:
            st.first_locked_ts = int(f.ts_recv[b0 + locked[0]])
        st.out_of_window = self.state.out_of_window
        if self.raw:
            want = crc_want[b0:]
            got = res["crc_out"][want]
            exp = crc_expected[b0:][want]
            ok = got == exp
            st.crc_compared += int(want.sum())
            st.crc_match += int(ok.sum())
            st.crc_mismatch += int((~ok).sum())
            for j in np.flatnonzero(~ok)[: MAX_FAILURES - len(st.failures)]:
                b = b0 + int(np.flatnonzero(want)[j])
                st.add_failure(
                    {"kind": "kraken_crc32", "batch_id": int(f.batch_id[b]),
                     "ts_recv": int(f.ts_recv[b]),
                     "expected": int(exp[j]), "computed": int(got[j]), "file": f.path.name}
                )  # fmt: skip

    def _finish(self, cp: RestCheckpoint, stats: dict[date, DayStats]) -> None:
        st = stats[cp.day]
        if cp.best is None:
            cp.outcome = "gap"
            st.rest_gap += 1
            return
        cp.outcome = "compared"
        bad, compared, excluded, offset = cp.best
        st.rest_compared += 1
        st.rest_levels_compared += compared
        st.rest_levels_excluded += excluded
        st.rest_levels_bad += bad
        ms = abs(offset) / 1e6
        st.rest_offset_ms_max = (
            ms if st.rest_offset_ms_max is None else max(st.rest_offset_ms_max, ms)
        )
        if bad == 0 and compared > 0:
            st.rest_ok += 1
        else:
            st.rest_bad += 1
            st.add_failure(
                {"kind": "rest_l2", "checkpoint_id": cp.checkpoint_id, "seq": cp.seq,
                 "ts_recv": cp.ts_recv, "compared": compared, "mismatched": bad,
                 "excluded": excluded, "first_bad_tick": cp.first_bad_tick,
                 "candidates": cp.candidates, "offset_ms": offset / 1e6}
            )  # fmt: skip

    # -- outputs

    def _write(
        self, day: date, path: Path, f: FileData, b0: int, res: dict[str, Any], idx0: int
    ) -> int:
        n = f.n_batches - b0
        if n <= 0:
            return idx0
        sl = slice(b0, f.n_batches)
        cols: dict[str, pa.Array] = {
            "ts_recv": pa.array(f.ts_recv[sl]),
            "ts_exchange": pa.array(f.ts_exchange[sl]),
            "batch_id": pa.array(f.batch_id[sl]),
            "event_idx": pa.array(np.arange(idx0, idx0 + n, dtype=np.int64)),
        }
        for side, pxk, qk in (("bid", "top_bpx", "top_bq"), ("ask", "top_apx", "top_aq")):
            px, q = res[pxk], res[qk]
            missing = px == NO_LEVEL
            for k in range(TOP_N):
                cols[f"{side}_px_{k + 1}"] = pa.array(px[:, k], mask=missing[:, k])
            for k in range(TOP_N):
                cols[f"{side}_qty_{k + 1}"] = pa.array(q[:, k], mask=missing[:, k])
        cols["sync_epoch"] = pa.array(f.epoch[sl])
        table = pa.Table.from_pydict(cols, schema=TOPN)
        self._atomic(
            partition(self.root / "interim", "topn", self.venue, self.symbol, day) / path.name,
            table,
        )

        both = (res["top_bpx"][:, 0] != NO_LEVEL) & (res["top_apx"][:, 0] != NO_LEVEL)
        from lob_alpha import schemas  # noqa: PLC0415

        bbo = pa.Table.from_pydict(
            {
                "ts": pa.array(f.ts_exchange[sl][both]),
                "venue": pa.array([self.venue] * int(both.sum()), pa.string()),
                "symbol": pa.array([self.symbol] * int(both.sum()), pa.string()),
                "bid_px": pa.array(np.round(res["top_bpx"][both, 0] * self.tick, self.ndp)),
                "bid_qty": pa.array(res["top_bq"][both, 0]),
                "ask_px": pa.array(np.round(res["top_apx"][both, 0] * self.tick, self.ndp)),
                "ask_qty": pa.array(res["top_aq"][both, 0]),
                "ts_recv": pa.array(f.ts_recv[sl][both]),
                "seq": pa.array(f.seq[sl][both]),
                "source": pa.array(["rebuilt"] * int(both.sum()), pa.string()),
            },
            schema=schemas.BBO,
        )
        self._atomic(
            partition(self.root / "interim", "bbo_rebuilt", self.venue, self.symbol, day)
            / path.name,
            bbo,
        )
        return idx0 + n

    @staticmethod
    def _atomic(final: Path, table: pa.Table) -> None:
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = final.with_name(final.name + ".tmp")
        pq.write_table(table, tmp, compression="zstd", compression_level=3)
        durable_replace(tmp, final)


def stats_path(root: Path, venue: str, symbol: str, day: date) -> Path:
    return (
        root / "interim" / "validation" / f"venue={venue}" / f"symbol={quote(symbol, safe='')}"
        / f"date={day}.json"
    )  # fmt: skip


def save_stats(root: Path, items: list[DayStats]) -> None:
    for st in items:
        path = stats_path(root, st.venue, st.symbol, date.fromisoformat(st.date))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(asdict(st), indent=1), encoding="utf-8")
        durable_replace(tmp, path)


def load_stats(root: Path) -> list[dict[str, Any]]:
    folder = root / "interim" / "validation"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.rglob("date=*.json"))]
