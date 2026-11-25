"""`lob features build`: features and labels per (venue, symbol, day), hour by hour (FR-17, FR-18).

Each hour is computed from the rebuilt top 10 of that hour, plus a lookback taken from the
previous hour (enough for the longest feature window) and a look-ahead from the next hour
(enough for the longest label). Windows and labels are therefore exact without holding a
whole day in memory; a Kraken BTC/USD day has about 19 million batches.

A UTC day is the unit (DQ-10): windows do not reach into the previous day, and labels whose
horizon runs past midnight are null.

Grids (spec section 7b):
* time: every 100 ms on the clock, kept where the book is LIVE: after the first batch of the
  current LIVE stretch and outside every logged book gap (`clock` rows are not gaps).
* event: every LIVE book batch.

Output: `processed/features/.../date=D/clock={time,event}/part-HH.parquet` and the same under
`processed/labels`. Writes are paced (`MAX_WRITE_BYTES_PER_S`) so a build never bursts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from lob_alpha.features import core
from lob_alpha.labels import clock_labels, event_labels
from lob_alpha.reports.gaplog import load_gaps
from lob_alpha.validate.rebuild import day_files, partition

NS = 1_000_000_000
MS = 1_000_000
HOUR = 3600 * NS
DAY = 24 * HOUR
MAX_WRITE_BYTES_PER_S = 20_000_000
LEVELS = 10


@dataclass
class Params:
    """The `features` and `labels` sections of config/research.yaml, with spec defaults."""

    grid_ms: int = 100
    ofi_events: list[int] = field(default_factory=lambda: [10, 50, 200])
    ofi_ms: list[int] = field(default_factory=lambda: [100, 1000, 10000])
    mlofi_events: list[int] = field(default_factory=lambda: [50])
    mlofi_ms: list[int] = field(default_factory=lambda: [1000])
    qi_levels: list[int] = field(default_factory=lambda: [1, 5])
    tsi_trades: list[int] = field(default_factory=lambda: [50, 200])
    tsi_ms: list[int] = field(default_factory=lambda: [1000, 10000])
    rv_ms: list[int] = field(default_factory=lambda: [1000, 10000, 60000])
    midchg_ms: list[int] = field(default_factory=lambda: [1000, 10000])
    horizons_ms: list[int] = field(
        default_factory=lambda: [100, 250, 500, 1000, 2000, 5000, 10000, 30000]
    )
    horizons_events: list[int] = field(default_factory=lambda: [1, 5, 10, 25, 50, 100, 250, 500])
    latencies_ms: list[int] = field(
        default_factory=lambda: [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000]
    )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Params:
        f, lab = cfg.get("features", {}), cfg.get("labels", {})
        p = cls()
        p.grid_ms = int(cfg.get("grids", {}).get("time_ms", p.grid_ms))
        p.ofi_events = list(f.get("ofi", {}).get("windows_events", p.ofi_events))
        p.ofi_ms = list(f.get("ofi", {}).get("windows_ms", p.ofi_ms))
        p.mlofi_events = list(f.get("mlofi", {}).get("windows_events", p.mlofi_events))
        p.mlofi_ms = list(f.get("mlofi", {}).get("windows_ms", p.mlofi_ms))
        p.qi_levels = list(f.get("queue_imbalance", {}).get("levels", p.qi_levels))
        p.tsi_trades = list(f.get("trade_sign_imbalance", {}).get("windows_events", p.tsi_trades))
        p.tsi_ms = list(f.get("trade_sign_imbalance", {}).get("windows_ms", p.tsi_ms))
        p.rv_ms = list(f.get("realized_vol", {}).get("windows_ms", p.rv_ms))
        p.horizons_ms = list(lab.get("horizons_ms", p.horizons_ms))
        p.horizons_events = list(lab.get("horizons_events", p.horizons_events))
        p.latencies_ms = list(lab.get("latencies_ms", p.latencies_ms))
        return p

    @property
    def lookback_ns(self) -> int:
        return max([*self.ofi_ms, *self.mlofi_ms, *self.tsi_ms, *self.rv_ms, *self.midchg_ms]) * MS

    @property
    def lookback_events(self) -> int:
        return max([*self.ofi_events, *self.mlofi_events])

    @property
    def lookahead_ns(self) -> int:
        return (max(self.horizons_ms) + max(self.latencies_ms)) * MS

    @property
    def lookahead_events(self) -> int:
        return max(self.horizons_events)


# ---- inputs ----------------------------------------------------------------------------------


TOPN_COLS = ["ts_recv", "ts_exchange", "batch_id", "sync_epoch"]
TOPN_COLS += [
    f"{s}_{k}_{i}" for s in ("bid", "ask") for k in ("px", "qty") for i in range(1, LEVELS + 1)
]


@dataclass
class Rows:
    """Top-10 rows of a stretch of batches."""

    ts_recv: npt.NDArray[np.int64]
    ts_exchange: npt.NDArray[np.int64]
    batch_id: npt.NDArray[np.int64]
    epoch: npt.NDArray[np.int64]
    bpx: npt.NDArray[np.int64]
    bq: npt.NDArray[np.float64]
    apx: npt.NDArray[np.int64]
    aq: npt.NDArray[np.float64]

    @property
    def n(self) -> int:
        return len(self.ts_recv)

    def slice(self, a: int, b: int) -> Rows:
        return Rows(
            *(
                getattr(self, k)[a:b]
                for k in ("ts_recv", "ts_exchange", "batch_id", "epoch", "bpx", "bq", "apx", "aq")
            )
        )

    @staticmethod
    def concat(parts: list[Rows]) -> Rows:
        keys = ("ts_recv", "ts_exchange", "batch_id", "epoch", "bpx", "bq", "apx", "aq")
        return Rows(*(np.concatenate([getattr(p, k) for p in parts]) for k in keys))


def _empty_rows() -> Rows:
    z = np.zeros(0, np.int64)
    m = np.zeros((0, LEVELS), np.int64)
    return Rows(z, z, z, z, m, m.astype(np.float64), m, m.astype(np.float64))


def read_topn(path: Path) -> Rows:
    table = pq.read_table(path, columns=TOPN_COLS)

    def col(name: str, fill: float) -> npt.NDArray[Any]:
        return np.asarray(table.column(name).fill_null(fill).to_numpy())

    def side(s: str, k: str, fill: float, dtype: Any) -> npt.NDArray[Any]:
        return np.stack([col(f"{s}_{k}_{i}", fill) for i in range(1, LEVELS + 1)], axis=1).astype(
            dtype
        )

    return Rows(
        col("ts_recv", 0).astype(np.int64), col("ts_exchange", 0).astype(np.int64),
        col("batch_id", 0).astype(np.int64), col("sync_epoch", 0).astype(np.int64),
        side("bid", "px", -1, np.int64), side("bid", "qty", 0.0, np.float64),
        side("ask", "px", -1, np.int64), side("ask", "qty", 0.0, np.float64),
    )  # fmt: skip


def day_data_end(by_hour: dict[int, list[Path]]) -> int | None:
    """Known-at time of the day's last batch, from the last hour's files."""
    if not by_hour:
        return None
    last = max(by_hour)
    ts = [pq.read_table(f, columns=["ts_recv"]).column("ts_recv").to_numpy() for f in by_hour[last]]
    vals = [int(x.max()) for x in ts if len(x)]
    return max(vals) if vals else None


def hour_of(path: Path) -> int:
    return int(path.name[5:7])


def read_trades(root: Path, venue: str, symbol: str, day: date, lot: float = 1e-8) -> core.Trades:
    parts = [
        pq.read_table(p, columns=["ts_recv", "side", "qty"])
        for p in day_files(root, "trades", venue, symbol, day)
    ]
    if not parts:
        return core.make_trades(np.zeros(0, np.int64), np.zeros(0, np.int8), np.zeros(0), lot)
    t = pa.concat_tables(parts)
    return core.make_trades(
        np.asarray(t.column("ts_recv").to_numpy(), np.int64),
        np.asarray(t.column("side").to_numpy(), np.int8),
        np.asarray(t.column("qty").to_numpy(), np.float64),
        lot,
    )


def book_gaps(root: Path, venue: str, symbol: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """Book gaps overlapping [lo, hi), `clock` rows aside (they are not missing data)."""
    gaps = load_gaps(root)
    out = []
    for v, s, stream, reason, a, b in gaps.select(
        "venue", "symbol", "stream", "reason", "start_ns", "end_ns"
    ).iter_rows():
        if (
            v == venue
            and s in (symbol, "*")
            and stream == "book"
            and reason != "clock"
            and b > lo
            and a < hi
        ):
            out.append((int(a), int(b)))
    return sorted(out)


# ---- feature evaluation on a stretch --------------------------------------------------------


def live_mask(
    t: npt.NDArray[np.int64],
    book: core.Book,
    gaps: list[tuple[int, int]],
    data_end: int | None = None,
) -> npt.NDArray[np.bool_]:
    """True where t is LIVE: a batch is known, t is not inside a gap, and that batch is after it.

    `data_end` is the day's last recorded batch: past it nothing is known (a capture that
    stopped before its gap row was written), so the grid stops there.
    """
    i = core.asof_index(book.ts, t)
    ok = i >= 0
    if data_end is not None:
        ok &= t <= data_end
    if gaps:
        starts = np.array([a for a, _ in gaps], np.int64)
        ends = np.array([b for _, b in gaps], np.int64)
        k = np.searchsorted(starts, t, side="right") - 1
        has = k >= 0
        kk = np.maximum(k, 0)
        inside = has & (t <= ends[kk])
        last_end = np.where(has, ends[kk], np.iinfo(np.int64).min)
        known = np.where(i >= 0, book.ts[np.maximum(i, 0)], np.iinfo(np.int64).min)
        ok &= ~inside & (known >= last_end)
    return ok


def evaluate(
    book: core.Book, trades: core.Trades, t: npt.NDArray[np.int64], at: npt.NDArray[np.int64],
    p: Params, day_start: int, grid: str,
) -> dict[str, npt.NDArray[Any]]:  # fmt: skip
    """All feature columns at sample times t with as-of batch indices `at` (all valid)."""
    seg = book.segment_start[at]
    lower_ts = np.maximum(book.ts[seg], day_start)
    cols: dict[str, npt.NDArray[Any]] = {}
    mid = core.mid_ticks(book)
    cols["mid_ticks"] = mid[at]
    cols["spread_ticks"] = core.spread_ticks(book)[at]
    cols["imb_l1"] = core.imbalance_l1(book)[at]
    for lv in p.qi_levels:
        cols[f"qi_l{lv}"] = core.queue_imbalance(book, lv)[at]
    cols["wmid_adj_ticks"] = core.weighted_mid_adj(book)[at]
    cols["depth_l1"] = core.depth_l1(book)[at]
    e1 = core.cks_e(
        book.bpx[:, 0], book.bq[:, 0], book.apx[:, 0], book.aq[:, 0], book.segment_start
    )
    lot = book.lot
    for n in p.ofi_events:
        cols[f"ofi_l1_ev{n}"] = core.event_window_sum(e1, at, n, seg, lot)
    for w in p.ofi_ms:
        cols[f"ofi_l1_ms{w}"] = core.time_window_sum(e1, book.ts, t, w * MS, lower_ts, lot)
    for m in range(LEVELS):
        em = core.cks_e(
            book.bpx[:, m], book.bq[:, m], book.apx[:, m], book.aq[:, m], book.segment_start
        )
        for n in p.mlofi_events:
            cols[f"mlofi_l{m + 1}_ev{n}"] = core.event_window_sum(em, at, n, seg, lot)
        for w in p.mlofi_ms:
            cols[f"mlofi_l{m + 1}_ms{w}"] = core.time_window_sum(
                em, book.ts, t, w * MS, lower_ts, lot
            )
    for n in p.tsi_trades:
        v, c = core.trade_imbalance(trades, t, last_n=n, lower_ts=np.full(len(t), day_start))
        cols[f"tsi_vol_tr{n}"], cols[f"tsi_cnt_tr{n}"] = v, c
    for w in p.tsi_ms:
        v, c = core.trade_imbalance(trades, t, w_ns=w * MS, lower_ts=np.full(len(t), day_start))
        cols[f"tsi_vol_ms{w}"], cols[f"tsi_cnt_ms{w}"] = v, c
    changed = np.zeros(book.n, dtype=np.int64)
    if book.n > 1:
        d = mid[1:] != mid[:-1]
        changed[1:] = np.where(np.isnan(mid[1:]) | np.isnan(mid[:-1]), 0, d.astype(np.int64))
    changed[book.segment_start == np.arange(book.n)] = 0
    for w in p.midchg_ms:
        cols[f"midchg_ms{w}"] = core.time_window_sum(changed, book.ts, t, w * MS, lower_ts)
    del grid
    return cols


def rv_on_grid(
    book: core.Book, gaps: list[tuple[int, int]], t_from: int, t_to: int, p: Params,
    data_end: int | None = None,
) -> tuple[npt.NDArray[np.int64], dict[int, npt.NDArray[np.float64]]]:  # fmt: skip
    """Realised volatility on the full 100 ms grid over [t_from, t_to); NaN where not LIVE."""
    step = p.grid_ms * MS
    g = np.arange(t_from, t_to, step, dtype=np.int64)
    ok = live_mask(g, book, gaps, data_end)
    at = core.asof_index(book.ts, g)
    mid = np.where(ok, core.mid_ticks(book)[np.maximum(at, 0)], np.nan)
    key = np.where(ok, book.segment_start[np.maximum(at, 0)], -1)
    new = np.ones(len(g), bool)
    if len(g) > 1:
        new[1:] = (key[1:] != key[:-1]) | ~ok[1:] | ~ok[:-1]
    starts = np.flatnonzero(new)
    gseg = (
        starts[np.searchsorted(starts, np.arange(len(g)), side="right") - 1]
        if len(g)
        else np.zeros(0, np.int64)
    )
    out = core.realized_vol(mid, gseg.astype(np.int64), [w // p.grid_ms for w in p.rv_ms])
    return g, {w: np.where(ok, out[w // p.grid_ms], np.nan) for w in p.rv_ms}


# ---- the day loop ----------------------------------------------------------------------------


@dataclass
class DayResult:
    venue: str
    symbol: str
    date: str
    time_rows: int = 0
    event_rows: int = 0
    bytes_written: int = 0
    seconds: float = 0.0


def _write(path: Path, cols: dict[str, npt.NDArray[Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        k: pa.array(v) if v.dtype.kind != "f" else pa.array(v, from_pandas=True)
        for k, v in cols.items()
    }
    table = pa.table(arrays)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=3)
    tmp.replace(path)
    size = path.stat().st_size
    time.sleep(size / MAX_WRITE_BYTES_PER_S)
    return size


def build_day(
    root: Path, venue: str, symbol: str, day: date, p: Params, *,
    grids: tuple[str, ...] = ("time", "event"),
    write: bool = True,
    lot: float | None = None,
) -> tuple[DayResult, dict[str, list[dict[str, npt.NDArray[Any]]]]]:  # fmt: skip
    """Features and labels for one day. Returns the result and, when not writing, the tables."""
    t0 = time.perf_counter()
    res = DayResult(venue, symbol, str(day))
    day_start = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * NS
    day_end = day_start + DAY
    files = day_files(root / "interim", "topn", venue, symbol, day)
    by_hour: dict[int, list[Path]] = {}
    for f in files:
        by_hour.setdefault(hour_of(f), []).append(f)
    hours = sorted(by_hour)
    if lot is None:
        from lob_alpha.instruments import load_instruments  # noqa: PLC0415

        lot = load_instruments().get(venue, symbol, day).lot_size
    trades = read_trades(root, venue, symbol, day, lot)
    gaps = book_gaps(root, venue, symbol, day_start, day_end)
    data_end = day_data_end(by_hour)
    kept: dict[str, list[dict[str, npt.NDArray[Any]]]] = {g: [] for g in grids}

    def load(h: int) -> Rows:
        return Rows.concat([read_topn(f) for f in by_hour[h]]) if h in by_hour else _empty_rows()

    prev_tail = _empty_rows()
    cur = load(hours[0]) if hours else _empty_rows()
    for k, h in enumerate(hours):
        nxt = load(hours[k + 1]) if k + 1 < len(hours) else _empty_rows()
        # look-ahead: the next hour's first rows up to the longest label span
        nt = nxt.ts_recv
        head_n = int(
            np.searchsorted(
                np.maximum.accumulate(nt) if len(nt) else nt,
                (cur.ts_recv.max() if cur.n else day_start) + p.lookahead_ns,
                side="right",
            )
        )
        head_n = min(max(head_n, p.lookahead_events), nxt.n)
        rows = Rows.concat([prev_tail, cur, nxt.slice(0, head_n)])
        n_prev, n_cur = prev_tail.n, cur.n
        book = core.make_book(
            rows.ts_recv,
            rows.ts_exchange,
            rows.bpx,
            rows.bq,
            rows.apx,
            rows.aq,
            rows.epoch,
            lot,
        )
        hour_lo = day_start + h * HOUR
        hour_hi = hour_lo + HOUR
        if "time" in grids:
            g = np.arange(hour_lo, hour_hi, p.grid_ms * MS, dtype=np.int64)
            ok = live_mask(g, book, gaps, data_end)
            t = g[ok]
            at = core.asof_index(book.ts, t)
            cols: dict[str, npt.NDArray[Any]] = {"ts": t, "event_idx": rows.batch_id[at]}
            cols |= evaluate(book, trades, t, at, p, day_start, "time")
            rv_lo = max(day_start, hour_lo - p.lookback_ns)
            gg, rv = rv_on_grid(book, gaps, rv_lo, hour_hi, p, data_end)
            pos = np.searchsorted(gg, t)
            for w, vals in rv.items():
                cols[f"rv_ms{w}"] = vals[pos]
            lab = clock_labels(
                book.ts,
                core.mid_ticks(book),
                book.segment_start,
                t,
                p.horizons_ms,
                p.latencies_ms,
                gaps,
                day_end,
            )
            res.time_rows += len(t)
            if write:
                res.bytes_written += _write(
                    _out(root, "features", venue, symbol, day, "time", h), cols
                )
                res.bytes_written += _write(
                    _out(root, "labels", venue, symbol, day, "time", h), {"ts": t, **lab}
                )
            else:
                kept["time"].append(cols | lab)
        if "event" in grids and n_cur:
            at = np.arange(n_prev, n_prev + n_cur, dtype=np.int64)
            t = book.ts[at]
            cols = {"ts": t, "event_idx": rows.batch_id[at]}
            cols |= evaluate(book, trades, t, at, p, day_start, "event")
            gg, rv = rv_on_grid(
                book,
                gaps,
                max(day_start, int(t.min()) - p.lookback_ns),
                int(t.max()) + 1,
                p,
                data_end,
            )
            pos = np.clip(np.searchsorted(gg, t, side="right") - 1, 0, max(len(gg) - 1, 0))
            for w, vals in rv.items():
                cols[f"rv_ms{w}"] = vals[pos] if len(gg) else np.full(len(t), np.nan)
            evl = event_labels(core.mid_ticks(book), book.segment_start, p.horizons_events)
            lab = {k: v[at] for k, v in evl.items()}
            last_day = rows.ts_recv < day_end
            for k2 in lab:  # an event label may not use a batch from the next day
                hz = int(k2.rsplit("_", 1)[1])
                j = np.minimum(at + hz, rows.n - 1)
                lab[k2] = np.where(last_day[j] & (at + hz < rows.n), lab[k2], np.nan)
            res.event_rows += len(t)
            if write:
                res.bytes_written += _write(
                    _out(root, "features", venue, symbol, day, "event", h), cols
                )
                res.bytes_written += _write(
                    _out(root, "labels", venue, symbol, day, "event", h), {"ts": t, **lab}
                )
            else:
                kept["event"].append(cols | lab)
        # lookback for the next hour
        cur_ts = np.maximum.accumulate(cur.ts_recv) if cur.n else cur.ts_recv
        a = (
            int(np.searchsorted(cur_ts, (cur_ts[-1] if cur.n else 0) - p.lookback_ns, side="left"))
            if cur.n
            else 0
        )
        a = max(0, min(a, cur.n - p.lookback_events))
        prev_tail = cur.slice(a, cur.n)
        cur = nxt
    res.seconds = time.perf_counter() - t0
    return res, kept


def _out(
    root: Path, dataset: str, venue: str, symbol: str, day: date, grid: str, hour: int
) -> Path:
    return (
        partition(root / "processed", dataset, venue, symbol, day)
        / f"clock={grid}"
        / f"part-{hour:02d}.parquet"
    )
