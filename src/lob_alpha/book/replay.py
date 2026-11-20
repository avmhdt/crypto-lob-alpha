"""Numba replay kernel: L2 books on a dense tick window (FR-14, NFR-03).

The book for one (venue, symbol) is two float64 arrays indexed by `tick - lo`, one per
side, plus the best and worst occupied index and a level count per side. Applying a level
update is O(1); removing the best or worst level scans to the next occupied tick, which
stays short because books are dense near the top. Updates outside the window are counted
and ignored: the window is set from the traded price range with a wide margin, so they are
levels far from the touch that never reach the top 100.

Kraken checksums need the raw price and size strings (NFR-06). For a checksum venue the
kernel also keeps, per tick, the stripped digits of the last string written there, and
computes the CRC32 of the top 10 levels in the venue's documented order.

Side index 0 is bids and 1 is asks. Exact tick indices only, never float equality on
prices (DQ-04).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numba
import numpy as np
import numpy.typing as npt

TOP_N = 10
RAW_WIDTH = 24  # bytes kept per stripped price or size string
NO_LEVEL = -1
QTY_RTOL = 1e-9  # REST and websocket sizes are the same decimals parsed by the same float()

# meta layout, per book
BEST_BID, BEST_ASK, WORST_BID, WORST_ASK, COUNT_BID, COUNT_ASK, OUT_OF_WINDOW = range(7)
META_SIZE = 7

# flags per batch boundary
FLAG_CROSSED = 1  # best bid > best ask
FLAG_LOCKED = 2  # best bid == best ask
FLAG_ONE_SIDED = 4  # a side is empty


def _jit[F: Callable[..., Any]](fn: F) -> F:
    return cast(F, numba.njit(cache=True, nogil=True)(fn))


def _crc_table() -> npt.NDArray[np.uint32]:
    table = np.zeros(256, dtype=np.uint32)
    for i in range(256):
        c = i
        for _ in range(8):
            c = (0xEDB88320 ^ (c >> 1)) if c & 1 else (c >> 1)
        table[i] = c
    return table


CRC_TABLE = _crc_table()

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
ByteArray = npt.NDArray[np.uint8]


class BookState:
    """Arrays the kernel mutates in place. One instance per (venue, symbol) replay."""

    def __init__(self, lo: int, size: int, raw: bool) -> None:
        if size <= 0:
            raise ValueError("window size must be positive")
        self.lo = lo
        self.size = size
        self.raw = raw
        self.qty: FloatArray = np.zeros((2, size), dtype=np.float64)
        n_raw = size if raw else 0
        self.px_raw: ByteArray = np.zeros((2, n_raw, RAW_WIDTH), dtype=np.uint8)
        self.px_len: ByteArray = np.zeros((2, n_raw), dtype=np.uint8)
        self.qty_raw: ByteArray = np.zeros((2, n_raw, RAW_WIDTH), dtype=np.uint8)
        self.qty_len: ByteArray = np.zeros((2, n_raw), dtype=np.uint8)
        self.meta: IntArray = np.array([-1, -1, -1, -1, 0, 0, 0], dtype=np.int64)

    @property
    def out_of_window(self) -> int:
        return int(self.meta[OUT_OF_WINDOW])

    def levels(self, side: int) -> list[tuple[int, float]]:
        """Every occupied level on one side as (tick, qty), best first. For tests."""
        s = 0 if side == 1 else 1
        idx = np.flatnonzero(self.qty[s])
        ticks = [(int(i) + self.lo, float(self.qty[s, i])) for i in idx]
        return sorted(ticks, key=lambda item: -item[0] if s == 0 else item[0])


# ---- kernel pieces --------------------------------------------------------------------------


@_jit
def _remove(qty: FloatArray, meta: IntArray, s: int, i: int) -> None:  # pragma: no cover
    qty[s, i] = 0.0
    meta[COUNT_BID + s] -= 1
    if meta[COUNT_BID + s] == 0:
        meta[BEST_BID + s] = -1
        meta[WORST_BID + s] = -1
        return
    if s == 0:
        if i == meta[BEST_BID]:
            j = i - 1
            while qty[0, j] == 0.0:
                j -= 1
            meta[BEST_BID] = j
        if i == meta[WORST_BID]:
            j = i + 1
            while qty[0, j] == 0.0:
                j += 1
            meta[WORST_BID] = j
    else:
        if i == meta[BEST_ASK]:
            j = i + 1
            while qty[1, j] == 0.0:
                j += 1
            meta[BEST_ASK] = j
        if i == meta[WORST_ASK]:
            j = i - 1
            while qty[1, j] == 0.0:
                j -= 1
            meta[WORST_ASK] = j


@_jit
def _set(qty: FloatArray, meta: IntArray, s: int, i: int, q: float) -> None:  # pragma: no cover
    if qty[s, i] == 0.0:
        meta[COUNT_BID + s] += 1
        if meta[BEST_BID + s] == -1:
            meta[BEST_BID + s] = i
            meta[WORST_BID + s] = i
        elif s == 0:
            meta[BEST_BID] = max(meta[BEST_BID], i)
            meta[WORST_BID] = min(meta[WORST_BID], i)
        else:
            meta[BEST_ASK] = min(meta[BEST_ASK], i)
            meta[WORST_ASK] = max(meta[WORST_ASK], i)
    qty[s, i] = q


@_jit
def _clear(qty: FloatArray, meta: IntArray) -> None:  # pragma: no cover
    for s in range(2):
        if meta[COUNT_BID + s] > 0:
            a = min(meta[BEST_BID + s], meta[WORST_BID + s])
            b = max(meta[BEST_BID + s], meta[WORST_BID + s])
            qty[s, a : b + 1] = 0.0
        meta[BEST_BID + s] = -1
        meta[WORST_BID + s] = -1
        meta[COUNT_BID + s] = 0


@_jit
def _truncate(qty: FloatArray, meta: IntArray, depth: int) -> None:  # pragma: no cover
    for s in range(2):
        while meta[COUNT_BID + s] > depth:
            _remove(qty, meta, s, meta[WORST_BID + s])


@_jit
def _crc_bytes(crc: int, table: Any, buf: Any, n: int) -> int:  # pragma: no cover
    for k in range(n):
        crc = table[(crc ^ buf[k]) & 0xFF] ^ (crc >> 8)
    return crc


@_jit
def _checksum(  # pragma: no cover
    qty: FloatArray,
    meta: IntArray,
    px_raw: ByteArray,
    px_len: ByteArray,
    qty_raw: ByteArray,
    qty_len: ByteArray,
    table: Any,
    levels: int,
) -> int:
    """Kraken v2: asks best first, then bids best first, price digits then size digits."""
    crc = 0xFFFFFFFF
    i = meta[BEST_ASK]
    taken = 0
    if i >= 0:
        last = meta[WORST_ASK]
        while taken < levels and i <= last:
            if qty[1, i] != 0.0:
                crc = _crc_bytes(crc, table, px_raw[1, i], px_len[1, i])
                crc = _crc_bytes(crc, table, qty_raw[1, i], qty_len[1, i])
                taken += 1
            i += 1
    i = meta[BEST_BID]
    taken = 0
    if i >= 0:
        last = meta[WORST_BID]
        while taken < levels and i >= last:
            if qty[0, i] != 0.0:
                crc = _crc_bytes(crc, table, px_raw[0, i], px_len[0, i])
                crc = _crc_bytes(crc, table, qty_raw[0, i], qty_len[0, i])
                taken += 1
            i -= 1
    return crc ^ 0xFFFFFFFF


@_jit
def _top(  # pragma: no cover
    qty: FloatArray, meta: IntArray, lo: int, s: int, out_px: IntArray, out_q: FloatArray, row: int
) -> None:
    n = out_px.shape[1]
    taken = 0
    i = meta[BEST_BID + s]
    if i >= 0:
        last = meta[WORST_BID + s]
        step = -1 if s == 0 else 1
        while taken < n:
            if qty[s, i] != 0.0:
                out_px[row, taken] = lo + i
                out_q[row, taken] = qty[s, i]
                taken += 1
            if i == last:
                break
            i += step
    while taken < n:
        out_px[row, taken] = NO_LEVEL
        out_q[row, taken] = 0.0
        taken += 1


@_jit
def _compare_side(  # pragma: no cover
    qty: FloatArray,
    lo: int,
    s: int,
    snap_tick: IntArray,
    snap_qty: FloatArray,
    a: int,
    b: int,
    ex_tick: IntArray,
    ea: int,
    eb: int,
    out: IntArray,
) -> None:
    """Compare one side over the snapshot's own price range.

    `snap_tick[a:b]` is sorted ascending, `ex_tick[ea:eb]` too. A tick counts when either
    source has a level there; an excluded tick (touched by the straddling batch) is counted
    separately and not compared. out = [compared, mismatched, excluded, first_bad_tick, oow].
    """
    if b <= a:
        return
    size = qty.shape[1]
    j = a
    e = ea
    for t in range(snap_tick[a], snap_tick[b - 1] + 1):
        sq = 0.0
        if j < b and snap_tick[j] == t:
            sq = snap_qty[j]
            j += 1
        i = t - lo
        if 0 <= i < size:
            bq = qty[s, i]
        else:
            if sq != 0.0:
                out[4] += 1
            continue
        if sq == 0.0 and bq == 0.0:
            continue
        while e < eb and ex_tick[e] < t:
            e += 1
        if e < eb and ex_tick[e] == t:
            out[2] += 1
            continue
        out[0] += 1
        if abs(bq - sq) > QTY_RTOL * max(abs(sq), 1e-12):
            out[1] += 1
            if out[3] == NO_LEVEL:
                out[3] = t


@_jit
def run_batches(  # pragma: no cover
    qty: FloatArray,
    meta: IntArray,
    px_raw: ByteArray,
    px_len: ByteArray,
    qty_raw: ByteArray,
    qty_len: ByteArray,
    lo: int,
    depth: int,
    sides: npt.NDArray[np.int8],
    ticks: IntArray,
    qtys: FloatArray,
    row_px: ByteArray,
    row_pxl: ByteArray,
    row_qt: ByteArray,
    row_qtl: ByteArray,
    bstart: IntArray,
    bsnap: npt.NDArray[np.bool_],
    b0: int,
    b1: int,
    top_bpx: IntArray,
    top_bq: FloatArray,
    top_apx: IntArray,
    top_aq: FloatArray,
    flags: npt.NDArray[np.int8],
    crc_want: npt.NDArray[np.bool_],
    crc_out: IntArray,
    table: Any,
    pr_batch: IntArray,
    pr_snap_off: IntArray,
    snap_tick: IntArray,
    snap_qty: FloatArray,
    pr_ex_off: IntArray,
    ex_tick: IntArray,
    pr_out: IntArray,
) -> None:
    """Apply batches [b0, b1) and record the state at every batch boundary.

    Output rows are indexed `batch - b0`. Probe p is evaluated after batch `pr_batch[p]`
    (probes sorted by batch); its snapshot levels are slices 2p (bids) and 2p+1 (asks) of
    `pr_snap_off`, and its excluded ticks slices 2p and 2p+1 of `pr_ex_off`.
    """
    size = qty.shape[1]
    raw = px_raw.shape[1] > 0
    p = 0
    n_probes = pr_batch.shape[0]
    while p < n_probes and pr_batch[p] < b0:
        p += 1
    for bi in range(b0, b1):
        if bsnap[bi]:
            _clear(qty, meta)
        for r in range(bstart[bi], bstart[bi + 1]):
            s = 0 if sides[r] == 1 else 1
            i = ticks[r] - lo
            if i < 0 or i >= size:
                meta[OUT_OF_WINDOW] += 1
                continue
            q = qtys[r]
            if q <= 0.0:
                if qty[s, i] != 0.0:
                    _remove(qty, meta, s, i)
            else:
                _set(qty, meta, s, i, q)
                if raw:
                    n = row_pxl[r]
                    px_raw[s, i, :n] = row_px[r, :n]
                    px_len[s, i] = n
                    n = row_qtl[r]
                    qty_raw[s, i, :n] = row_qt[r, :n]
                    qty_len[s, i] = n
        if depth > 0:
            _truncate(qty, meta, depth)
        row = bi - b0
        _top(qty, meta, lo, 0, top_bpx, top_bq, row)
        _top(qty, meta, lo, 1, top_apx, top_aq, row)
        bb = meta[BEST_BID]
        ba = meta[BEST_ASK]
        f = 0
        if bb < 0 or ba < 0:
            f |= FLAG_ONE_SIDED
        elif bb > ba:
            f |= FLAG_CROSSED
        elif bb == ba:
            f |= FLAG_LOCKED
        flags[row] = f
        if raw and crc_want[row]:
            crc_out[row] = _checksum(qty, meta, px_raw, px_len, qty_raw, qty_len, table, 10)
        while p < n_probes and pr_batch[p] == bi:
            for side in range(2):
                _compare_side(
                    qty, lo, side, snap_tick, snap_qty,
                    pr_snap_off[2 * p + side], pr_snap_off[2 * p + side + 1],
                    ex_tick, pr_ex_off[2 * p + side], pr_ex_off[2 * p + side + 1],
                    pr_out[p],
                )  # fmt: skip
            p += 1


def empty_probes() -> dict[str, npt.NDArray[Any]]:
    """Probe arrays for a call with no checkpoint comparisons."""
    return {
        "pr_batch": np.zeros(0, dtype=np.int64),
        "pr_snap_off": np.zeros(1, dtype=np.int64),
        "snap_tick": np.zeros(0, dtype=np.int64),
        "snap_qty": np.zeros(0, dtype=np.float64),
        "pr_ex_off": np.zeros(1, dtype=np.int64),
        "ex_tick": np.zeros(0, dtype=np.int64),
        "pr_out": np.zeros((0, 5), dtype=np.int64),
    }


def strip_raw(values: list[str | None]) -> tuple[ByteArray, ByteArray]:
    """Kraken's checksum form of each string: no decimal point, no leading zeros."""
    stripped = [(v or "").replace(".", "").lstrip("0") for v in values]
    longest = max((len(v) for v in stripped), default=0)
    if longest > RAW_WIDTH:
        raise ValueError(f"raw string longer than {RAW_WIDTH} bytes")
    buf = np.zeros((len(stripped), RAW_WIDTH), dtype=np.uint8)
    if stripped:
        packed = np.array(stripped, dtype=f"S{RAW_WIDTH}")
        buf[:] = packed.view(np.uint8).reshape(len(stripped), RAW_WIDTH)
    lengths = np.array([len(v) for v in stripped], dtype=np.uint8)
    return buf, lengths
