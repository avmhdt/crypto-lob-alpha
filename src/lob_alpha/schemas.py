"""Arrow schemas for every dataset, the single source of truth (spec section 7b).

Canonical fields come first and never change (NFR-15); added fields follow.
`venue` and `symbol` stay plain strings here: Parquet dictionary-encodes them on disk
and readers get strings back, which is what the downstream projects expect.
"""

from __future__ import annotations

import pyarrow as pa

TRADES_CANONICAL = ("ts_exchange", "ts_recv", "venue", "symbol", "trade_id", "price", "qty", "side")
BOOK_L2_CANONICAL = (
    "ts_exchange",
    "ts_recv",
    "venue",
    "symbol",
    "seq",
    "side",
    "price",
    "qty",
    "is_snapshot",
    "batch_id",
)
BOOK_L3_CANONICAL = (
    "ts_exchange",
    "ts_recv",
    "venue",
    "symbol",
    "order_id",
    "event",
    "side",
    "price_ticks",
    "qty",
    "new_order_id",
    "match_id",
)
BBO_CANONICAL = ("ts", "venue", "symbol", "bid_px", "bid_qty", "ask_px", "ask_qty")

TRADES = pa.schema(
    [
        pa.field("ts_exchange", pa.int64(), nullable=False),
        pa.field("ts_recv", pa.int64(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("trade_id", pa.string(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("qty", pa.float64(), nullable=False),
        pa.field("side", pa.int8(), nullable=False),
        pa.field("batch_id", pa.int64(), nullable=False),
        pa.field("is_aggregated", pa.bool_(), nullable=False),
        pa.field("seq", pa.int64()),
    ]
)

BOOK_L2 = pa.schema(
    [
        pa.field("ts_exchange", pa.int64(), nullable=False),
        pa.field("ts_recv", pa.int64(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("seq", pa.int64(), nullable=False),
        pa.field("side", pa.int8(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("qty", pa.float64(), nullable=False),
        pa.field("is_snapshot", pa.bool_(), nullable=False),
        pa.field("batch_id", pa.int64(), nullable=False),
        pa.field("seq_first", pa.int64()),
        pa.field("seq_prev", pa.int64()),
        pa.field("px_str", pa.string()),
        pa.field("qty_str", pa.string()),
        pa.field("sync_epoch", pa.int32(), nullable=False),
    ]
)

BOOK_L3 = pa.schema(
    [
        pa.field("ts_exchange", pa.int64(), nullable=False),
        pa.field("ts_recv", pa.int64(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("order_id", pa.uint64(), nullable=False),
        pa.field("event", pa.string(), nullable=False),
        pa.field("side", pa.int8(), nullable=False),
        pa.field("price_ticks", pa.int64(), nullable=False),
        pa.field("qty", pa.int64(), nullable=False),
        pa.field("new_order_id", pa.uint64()),
        pa.field("match_id", pa.uint64()),
        pa.field("order_uuid", pa.string()),
        pa.field("seq", pa.int64()),
    ]
)

BBO = pa.schema(
    [
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("bid_px", pa.float64(), nullable=False),
        pa.field("bid_qty", pa.float64(), nullable=False),
        pa.field("ask_px", pa.float64(), nullable=False),
        pa.field("ask_qty", pa.float64(), nullable=False),
        pa.field("ts_recv", pa.int64(), nullable=False),
        pa.field("seq", pa.int64()),
        pa.field("source", pa.string(), nullable=False),
    ]
)

GAPS = pa.schema(
    [
        pa.field("gap_id", pa.string(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("stream", pa.string(), nullable=False),
        pa.field("start_ns", pa.int64(), nullable=False),
        pa.field("end_ns", pa.int64(), nullable=False),
        pa.field("duration_ms", pa.float64(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
        pa.field("seq_before", pa.int64()),
        pa.field("seq_after", pa.int64()),
        pa.field("host", pa.string(), nullable=False),
        pa.field("recorder_version", pa.string(), nullable=False),
    ]
)

CHECKPOINTS = pa.schema(
    [
        pa.field("checkpoint_id", pa.string(), nullable=False),
        pa.field("ts_recv", pa.int64(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("seq", pa.int64(), nullable=False),
        pa.field("batch_id", pa.int64(), nullable=False),
        pa.field("expected", pa.int64()),
        pa.field("computed_live", pa.int64()),
        pa.field("ok_live", pa.bool_(), nullable=False),
        pa.field("n_levels", pa.int32(), nullable=False),
    ]
)

CHECKPOINT_LEVELS = pa.schema(
    [
        pa.field("checkpoint_id", pa.string(), nullable=False),
        pa.field("side", pa.int8(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("qty", pa.float64(), nullable=False),
    ]
)

CLOCK = pa.schema(
    [
        pa.field("ts_recv", pa.int64(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("server_time_ns", pa.int64(), nullable=False),
        pa.field("rtt_ns", pa.int64(), nullable=False),
        pa.field("offset_ns", pa.int64(), nullable=False),
    ]
)

DATASETS: dict[str, pa.Schema] = {
    "trades": TRADES,
    "book_l2": BOOK_L2,
    "book_l3": BOOK_L3,
    "bbo": BBO,
    "gaps": GAPS,
    "checkpoints": CHECKPOINTS,
    "checkpoint_levels": CHECKPOINT_LEVELS,
    "clock": CLOCK,
}
