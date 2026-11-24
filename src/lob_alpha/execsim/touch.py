"""Touch-to-touch taker check, the daily return series for M5's trial-log rows.

M6 builds the full execution check (FR-22: depth walk for the order size, maker fills with
a queue model, fee scenarios). Until then each model run is judged by this simpler taker:

* signal: |yhat| >= threshold, where the threshold is the 0.9 quantile of |yhat| on that
  fold's training rows (the spec's `train_quantile` method; 0.9 is this check's default);
* one position at a time, of `size_usd` notional, in the direction of yhat;
* entry at the prevailing touch L ms after the signal (buy at the ask, sell at the bid),
  exit at the touch h later (the label's span), both as taker, `taker_fee_bps` per side;
* only signals whose label is not null, so entry and exit lie in one LIVE stretch.

Per trade, in bps of the entry mid:

    net = 1e4 * (side * dmid - spread_entry / 2 - spread_exit / 2) / mid_entry - 2 * fee

with dmid, spreads and mid in ticks. The daily return is the sum of the day's trade
returns (net bps x 1e-4) on a capital equal to the position size. Days with no trade
return 0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numba
import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
DEFAULT_QUANTILE = 0.9


def _jit[F: Callable[..., Any]](fn: F) -> F:
    return cast(F, numba.njit(cache=True, nogil=True)(fn))


@_jit
def _greedy(ts: I64, cand: I64, span_ns: I64) -> I64:  # pragma: no cover - numba
    out = np.empty(len(cand), np.int64)
    k = 0
    free_at = np.int64(-(2**62))
    for i in cand:
        if ts[i] >= free_at:
            out[k] = i
            k += 1
            free_at = ts[i] + span_ns[i]
    return out[:k]


@dataclass
class TouchTrades:
    idx: I64
    side: F64
    net_bps: F64
    gross_bps: F64


def touch_taker(
    ts: I64,
    yhat: F64,
    y: F64,
    mid_entry: F64,
    spread_entry: F64,
    spread_exit: F64,
    span_ns: I64,
    threshold: float,
    fee_bps: float,
) -> TouchTrades:
    """Trades on one day's rows (in time order); entry/exit book values per row."""
    ok = (
        ~np.isnan(y)
        & ~np.isnan(yhat)
        & (np.abs(yhat) >= threshold)
        & (yhat != 0)
        & np.isfinite(mid_entry)
        & np.isfinite(spread_entry)
        & np.isfinite(spread_exit)
        & (mid_entry > 0)
    )
    cand = np.flatnonzero(ok).astype(np.int64)
    idx = _greedy(ts, cand, span_ns.astype(np.int64))
    side = np.sign(yhat[idx])
    gross_ticks = side * y[idx]
    gross = 1e4 * gross_ticks / mid_entry[idx]
    cost_ticks = spread_entry[idx] / 2 + spread_exit[idx] / 2
    net = 1e4 * (gross_ticks - cost_ticks) / mid_entry[idx] - 2 * fee_bps
    return TouchTrades(idx, side, net, gross)


def threshold_from_train(yhat_train: F64, quantile: float = DEFAULT_QUANTILE) -> float:
    vals = np.abs(yhat_train[~np.isnan(yhat_train)])
    return float(np.quantile(vals, quantile)) if len(vals) else float("inf")
