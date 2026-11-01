"""The sequence checks flag exactly the injected gaps."""

from __future__ import annotations

import orjson
from hypothesis import given
from hypothesis import strategies as st

from conftest import MemorySink, binance_cfg
from lob_alpha.record.sync import VenuePipeline
from lob_alpha.record.venues.base import SyncState
from lob_alpha.record.venues.binance_usdm import BinanceUsdm

T0 = 1_793_566_970_000_000_000
CONN = "public.g0"


def _event(index: int) -> bytes:
    first = index * 10 + 1
    last = index * 10 + 10
    data = {
        "e": "depthUpdate", "E": 1, "T": 1, "s": "BTCUSDT", "U": first, "u": last,
        "pu": first - 1, "b": [["100.0", str(index + 1)]], "a": [],
    }  # fmt: skip
    return orjson.dumps({"stream": "x", "data": data})


def _snapshot(seq: int) -> bytes:
    return orjson.dumps({"lastUpdateId": seq, "T": 1, "bids": [], "asks": [["101.0", "1"]]})


@given(st.sets(st.integers(min_value=2, max_value=39), max_size=4))
def test_every_dropped_event_is_a_gap_and_nothing_else_is(dropped: set[int]) -> None:
    sink = MemorySink()
    pipeline = VenuePipeline(BinanceUsdm(binance_cfg()), sink, {"BTCUSDT": 0.1})
    pipeline.ingest_control(T0, {"k": "conn_open", "conn": CONN, "logical": "public"})
    book = pipeline.books["BTCUSDT"]
    pipeline.ingest_frame(CONN, T0 + 1, _event(0))
    pipeline.ingest_control(T0 + 2, {"k": "snapshot", "symbol": "BTCUSDT"}, _snapshot(5))
    assert book.state is SyncState.LIVE
    resyncs = 0
    for index in range(1, 40):
        if index in dropped:
            continue
        ts = T0 + 1000 * index
        previous_ok = book.state is SyncState.LIVE
        pipeline.ingest_frame(CONN, ts, _event(index))
        if previous_ok and book.state is not SyncState.LIVE:
            resyncs += 1
            pipeline.ingest_control(
                ts + 1, {"k": "snapshot", "symbol": "BTCUSDT"}, _snapshot(index * 10 + 5)
            )
            assert book.state is SyncState.LIVE
    expected = 0
    last = 0
    for index in range(1, 40):
        if index in dropped:
            continue
        if index != last + 1:
            expected += 1
        last = index
    assert resyncs == expected
    assert len(sink.gaps) == expected
    assert all(g.reason == "seq_gap" for g in sink.gaps)
    seqs = [b.seq for b in sink.batches if not b.is_snapshot]
    assert seqs == sorted(seqs)
