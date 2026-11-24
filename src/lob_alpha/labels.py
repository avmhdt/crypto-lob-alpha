"""Mid-price change labels in ticks (FR-18).

Clock labels on the 100 ms grid:

    dmid_ticks_time_{h}          = mid(t + h) - mid(t)
    dmid_ticks_time_{h}_lat{L}   = mid(t + L + h) - mid(t + L)     for L > 0

Event labels on the batch grid (one row per LIVE book batch):

    dmid_ticks_event_{h}         = mid(n + h) - mid(n)

mid(x) is the mid of the last batch known at x (DQ-02). A label is null when its span
reaches a gap or leaves the LIVE stretch it starts in (DQ-07), or runs past the end of the
day; nothing is forward-filled across a gap. A stretch ends at the start of the next logged
book gap (any reason but `clock`, which marks timestamps, not missing data) or at a book
reset.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def stretch_end(t: I64, gaps: list[tuple[int, int]], day_end: int) -> I64:
    """End of the LIVE stretch containing each time: the next gap start after t, or day end."""
    starts = np.array(sorted(a for a, _ in gaps), dtype=np.int64)
    if starts.size == 0:
        return np.full(len(t), day_end, dtype=np.int64)
    k = np.searchsorted(starts, t, side="right")
    nxt = np.where(k < len(starts), starts[np.minimum(k, len(starts) - 1)], day_end)
    return np.minimum(nxt, day_end)


def clock_labels(
    book_ts: I64,
    book_mid: F64,
    book_seg: I64,
    t: I64,
    horizons_ms: Sequence[int],
    latencies_ms: Sequence[int],
    gaps: list[tuple[int, int]],
    day_end: int,
) -> dict[str, F64]:
    """All clock labels for sample times t (known-at times of the book, non-decreasing)."""
    ms = 1_000_000
    i0 = np.searchsorted(book_ts, t, side="right") - 1
    end = stretch_end(t, gaps, day_end)
    seg0 = np.where(i0 >= 0, book_seg[np.maximum(i0, 0)], -1)
    out: dict[str, F64] = {}

    def mid_at(x: I64) -> tuple[F64, I64]:
        i = np.searchsorted(book_ts, x, side="right") - 1
        m = np.where(i >= 0, book_mid[np.maximum(i, 0)], np.nan)
        s = np.where(i >= 0, book_seg[np.maximum(i, 0)], -2)
        return m, s

    for lat in latencies_ms:
        base_t = t + lat * ms
        base_mid, base_seg = mid_at(base_t)
        for h in horizons_ms:
            end_t = base_t + h * ms
            m1, s1 = mid_at(end_t)
            ok = (i0 >= 0) & (end_t <= end) & (s1 == seg0) & (base_seg == seg0)
            val = np.where(ok, m1 - base_mid, np.nan)
            name = f"dmid_ticks_time_{h}" if lat == 0 else f"dmid_ticks_time_{h}_lat{lat}"
            out[name] = val
    return out


def event_labels(book_mid: F64, book_seg: I64, horizons: Sequence[int]) -> dict[str, F64]:
    """Event labels on the batch grid; null when n + h leaves the segment or the day."""
    n = len(book_mid)
    idx = np.arange(n)
    out: dict[str, F64] = {}
    for h in horizons:
        j = idx + h
        inside = j < n
        jj = np.minimum(j, n - 1)
        ok = inside & (book_seg[jj] == book_seg)
        out[f"dmid_ticks_event_{h}"] = np.where(ok, book_mid[jj] - book_mid, np.nan)
    return out
