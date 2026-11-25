"""Causal microstructure features as NumPy kernels on one (venue, symbol, day) (FR-17).

Inputs are the rebuilt top 10 (`interim/topn`): one row per LIVE batch boundary, prices in
integer ticks (DQ-04), sizes in base units. Every feature at sample time t uses only rows
with known-at time <= t (DQ-02). The known-at time of a batch is its `ts_recv` made
non-decreasing (a running maximum), because the host clock steps back about 1 ms every
30 s and a later batch cannot be known before an earlier one.

Windows never reach across a book reset (a snapshot batch, which starts a new sync epoch)
or the start of the day: the sum then covers only the part of the window since the reset.
Missing levels have price -1 and size 0 in the arrays.

Exact arithmetic (parity with an independent online engine): book and trade sizes are held
as int64 counts of the instrument's lot size, so every window sum, every e_n and every
level sum is an exact integer. A quantity in base units is the integer count times
`lot_size`, one float multiplication at the end; a ratio is one division of two exact
integers. Mid changes are summed as integer changes of bid + ask (half ticks).

Definitions (all in ticks or base units, as named):

* CKS e_n (Cont, Kukanov and Stoikov 2014), from batch n-1 to n at level 1:
  e_n = 1[Pb_n >= Pb_{n-1}] qb_n - 1[Pb_n <= Pb_{n-1}] qb_{n-1}
        - 1[Pa_n <= Pa_{n-1}] qa_n + 1[Pa_n >= Pa_{n-1}] qa_{n-1}.
  e_n is 0 at the first batch of a segment. OFI over a window is the sum of e_n.
* Multi-level OFI: the same e_n with level-m prices and sizes, m = 1..10.
* Queue imbalance at L levels: (sum qb_1..L - sum qa_1..L) / (sum qb + sum qa).
* Level-1 imbalance I = qb_1 / (qb_1 + qa_1), the micro-price state variable.
* Weighted mid minus mid, ticks: (Pb qa + Pa qb) / (qb + qa) - (Pb + Pa) / 2.
* Spread, ticks: Pa_1 - Pb_1. Mid, ticks: (Pa_1 + Pb_1) / 2.
* Realised volatility over W ms: sqrt of the sum of squared 100 ms mid changes (ticks) in
  (t - W, t], on the 100 ms grid.
* Mid-change count over W ms: batches in (t - W, t] whose mid differs from the batch before.
* Trade-sign imbalance: volume (sum s q / sum q) and count (sum s / n) over the last N
  trades or over (t - W, t], s = +1 buyer-initiated, -1 seller-initiated (DQ-06).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
B = npt.NDArray[np.bool_]

NO_LEVEL = -1


@dataclass
class Book:
    """Top 10 per batch boundary of one day, in replay order."""

    ts: I64  # known-at time: running max of ts_recv
    ts_exchange: I64
    bpx: I64  # (n, 10) ticks, -1 when absent
    bq: I64  # (n, 10) sizes in lots, 0 when absent
    apx: I64
    aq: I64
    segment_start: I64  # index of the first batch of each batch's segment
    lot: float = 1e-8  # lot size, base units per lot

    @property
    def n(self) -> int:
        return len(self.ts)


LOT_TOLERANCE = 1e-3  # largest accepted |size / lot - round(size / lot)|


def to_lots(q: F64, lot: float) -> I64:
    """Sizes in base units to int64 lot counts; refuses sizes that are not whole lots."""
    r = np.asarray(q, dtype=np.float64) / lot
    r = np.where(np.isfinite(r), r, 0.0)
    lots = np.rint(r)
    if r.size and float(np.max(np.abs(r - lots))) > LOT_TOLERANCE:
        raise ValueError(f"sizes are not whole multiples of lot size {lot}")
    return lots.astype(np.int64)


def make_book(
    ts_recv: I64, ts_exchange: I64, bpx: I64, bq: F64, apx: I64, aq: F64, epoch: I64,
    lot: float = 1e-8,
) -> Book:  # fmt: skip
    """Build a `Book` (sizes to lots), deriving known-at times and segment starts."""
    n = len(ts_recv)
    ts = np.maximum.accumulate(ts_recv) if n else ts_recv.astype(np.int64)
    new = np.ones(n, dtype=bool)
    if n > 1:
        new[1:] = epoch[1:] != epoch[:-1]
    starts = np.flatnonzero(new)
    seg = np.zeros(n, dtype=np.int64)
    if n:
        seg = starts[np.searchsorted(starts, np.arange(n), side="right") - 1]
    return Book(
        ts.astype(np.int64),
        ts_exchange.astype(np.int64),
        bpx,
        to_lots(bq, lot),
        apx,
        to_lots(aq, lot),
        seg,
        lot,
    )


# ---- per-batch quantities --------------------------------------------------------------------


def cks_e(bp: I64, bq: I64, ap: I64, aq: I64, segment_start: I64) -> I64:
    """CKS e_n for one level in lots; 0 at a segment's first batch or where a side is missing."""
    n = len(bp)
    e = np.zeros(n, dtype=np.int64)
    if n < 2:
        return e
    pb0, pb1 = bp[:-1], bp[1:]
    pa0, pa1 = ap[:-1], ap[1:]
    qb0, qb1 = bq[:-1], bq[1:]
    qa0, qa1 = aq[:-1], aq[1:]
    val = (
        np.where(pb1 >= pb0, qb1, 0)
        - np.where(pb1 <= pb0, qb0, 0)
        - np.where(pa1 <= pa0, qa1, 0)
        + np.where(pa1 >= pa0, qa0, 0)
    )
    ok = (pb0 != NO_LEVEL) & (pb1 != NO_LEVEL) & (pa0 != NO_LEVEL) & (pa1 != NO_LEVEL)
    e[1:] = np.where(ok, val, 0)
    e[segment_start == np.arange(n)] = 0
    return e


def mid_ticks(book: Book) -> F64:
    b, a = book.bpx[:, 0], book.apx[:, 0]
    out = (b + a) / 2.0
    return np.where((b == NO_LEVEL) | (a == NO_LEVEL), np.nan, out)


def spread_ticks(book: Book) -> F64:
    b, a = book.bpx[:, 0], book.apx[:, 0]
    return np.where((b == NO_LEVEL) | (a == NO_LEVEL), np.nan, (a - b).astype(np.float64))


def _ratio(num: I64, den: I64) -> F64:
    """num / den as one float64 division of exact integers; NaN where den <= 0."""
    n, d = num.astype(np.float64), den.astype(np.float64)
    out: F64 = np.divide(n, d, out=np.full(len(n), np.nan), where=den > 0)
    return out


def imbalance_l1(book: Book) -> F64:
    """I = qb_1 / (qb_1 + qa_1), sizes in lots."""
    qb, qa = book.bq[:, 0], book.aq[:, 0]
    return _ratio(qb, qb + qa)


def queue_imbalance(book: Book, levels: int) -> F64:
    qb = book.bq[:, :levels].sum(axis=1)
    qa = book.aq[:, :levels].sum(axis=1)
    return _ratio(qb - qa, qb + qa)


def weighted_mid_adj(book: Book) -> F64:
    """Weighted mid minus mid in ticks: (qb - qa) S / (2 (qb + qa)), S = Pa - Pb.

    Equal to (Pb qa + Pa qb) / (qb + qa) - (Pb + Pa) / 2 and to (I - 1/2) S, computed as one
    division of exact integers. NaN when a side of level 1 is missing.
    """
    b, a = book.bpx[:, 0], book.apx[:, 0]
    qb, qa = book.bq[:, 0], book.aq[:, 0]
    out = _ratio((qb - qa) * (a - b), 2 * (qb + qa))
    return np.where((b == NO_LEVEL) | (a == NO_LEVEL), np.nan, out)


def depth_l1(book: Book) -> F64:
    """(qb_1 + qa_1) / 2 in base units."""
    out: F64 = (book.bq[:, 0] + book.aq[:, 0]).astype(np.float64) * book.lot / 2.0
    return out


# ---- windows ---------------------------------------------------------------------------------


def _csum(x: I64) -> I64:
    """Exact int64 prefix sums (callers keep |sums| below 2^63)."""
    out = np.zeros(len(x) + 1, dtype=np.int64)
    np.cumsum(np.asarray(x, dtype=np.int64), out=out[1:])
    return out


def event_window_sum(x: I64, at: I64, n: int, lower: I64, scale: float = 1.0) -> F64:
    """Exact integer sum of x over the last n batches ending at `at` (never before index
    `lower`), times `scale` in one multiplication."""
    c = _csum(x)
    lo = np.maximum(at - n + 1, lower)
    res = (c[at + 1] - c[np.maximum(lo, 0)]).astype(np.float64) * scale
    return np.where(at >= 0, res, np.nan)


def time_window_sum(x: I64, ts: I64, t: I64, w_ns: int, lower_ts: I64, scale: float = 1.0) -> F64:
    """Exact integer sum of x over batches with known-at time in (t - w, t], never before
    `lower_ts`, times `scale` in one multiplication."""
    c = _csum(x)
    hi = np.searchsorted(ts, t, side="right")
    lo = np.searchsorted(ts, np.maximum(t - w_ns, lower_ts - 1), side="right")
    lo = np.minimum(lo, hi)
    out: F64 = (c[hi] - c[lo]).astype(np.float64) * scale
    return out


def asof_index(ts: I64, t: I64) -> I64:
    """Index of the last batch known at t, or -1."""
    return np.searchsorted(ts, t, side="right").astype(np.int64) - 1


# ---- trades ----------------------------------------------------------------------------------


@dataclass
class Trades:
    ts: I64  # known-at, non-decreasing
    side: I64  # +1 / -1 / 0
    qty: I64  # lots

    @property
    def n(self) -> int:
        return len(self.ts)


def make_trades(ts_recv: I64, side: npt.NDArray[np.int8], qty: F64, lot: float = 1e-8) -> Trades:
    ts = np.maximum.accumulate(ts_recv) if len(ts_recv) else ts_recv
    return Trades(ts.astype(np.int64), side.astype(np.int64), to_lots(qty, lot))


def trade_imbalance(
    tr: Trades,
    t: I64,
    *,
    last_n: int | None = None,
    w_ns: int | None = None,
    lower_ts: I64 | None = None,
) -> tuple[F64, F64]:
    """(volume imbalance, count imbalance) of trades known at t, over a count or a time window."""
    sv = _csum(tr.side * tr.qty)
    v = _csum(tr.qty)
    s = _csum(tr.side)
    hi = np.searchsorted(tr.ts, t, side="right")
    if last_n is not None:
        lo = np.maximum(hi - last_n, 0)
        if lower_ts is not None:
            lo = np.maximum(lo, np.searchsorted(tr.ts, lower_ts, side="left"))
    else:
        assert w_ns is not None
        start = t - w_ns if lower_ts is None else np.maximum(t - w_ns, lower_ts - 1)
        lo = np.searchsorted(tr.ts, start, side="right")
    lo = np.minimum(lo, hi)
    vol = v[hi] - v[lo]
    cnt = (hi - lo).astype(np.int64)
    vimb = np.where(vol > 0, _ratio(sv[hi] - sv[lo], vol), 0.0)
    cimb = np.where(cnt > 0, _ratio(s[hi] - s[lo], cnt), 0.0)
    return vimb, cimb


# ---- realised volatility on the 100 ms grid ---------------------------------------------------


def realized_vol(mid_grid: F64, grid_seg: I64, windows_steps: list[int]) -> dict[int, F64]:
    """sqrt(sum of squared grid-to-grid mid changes over the last k steps), within a segment.

    `grid_seg` is the index of the first grid point of each point's LIVE stretch.
    """
    n = len(mid_grid)
    r2 = np.zeros(n, dtype=np.int64)
    if n > 1:
        # mid changes as integer changes of bid + ask (half ticks), squared exactly
        two = np.where(np.isnan(mid_grid), 0.0, np.rint(2.0 * mid_grid)).astype(np.int64)
        d = np.diff(two)
        missing = np.isnan(mid_grid[1:]) | np.isnan(mid_grid[:-1])
        r2[1:] = np.where(missing, 0, d * d)
    r2[grid_seg == np.arange(n)] = 0
    c = _csum(r2)
    idx = np.arange(n)
    out = {}
    for k in windows_steps:
        lo = np.maximum(idx - k + 1, grid_seg)
        out[k] = np.sqrt((c[idx + 1] - c[lo]).astype(np.float64)) / 2.0
    return out
