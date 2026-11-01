"""Recorder per-message cost (NFR-01). Run with `pytest tests/bench --benchmark-only`."""

from __future__ import annotations

import orjson
import pytest

from conftest import MemorySink, binance_cfg, frames, kraken_cfg
from lob_alpha.record.sync import VenuePipeline
from lob_alpha.record.venues.binance_usdm import BinanceUsdm
from lob_alpha.record.venues.kraken import Kraken

T0 = 1_793_566_970_000_000_000


def _depth(first: int, last: int, prev: int) -> bytes:
    side = [[f"{100 + i * 0.1:.1f}", "1.000"] for i in range(40)]
    data = {
        "e": "depthUpdate", "E": 1, "T": 1, "s": "BTCUSDT", "U": first, "u": last, "pu": prev,
        "b": side, "a": side,
    }  # fmt: skip
    return orjson.dumps({"stream": "btcusdt@depth@100ms", "data": data})


def test_binance_depth_parse_and_check(benchmark: pytest.FixtureRequest) -> None:
    pipeline = VenuePipeline(BinanceUsdm(binance_cfg()), MemorySink(), {"BTCUSDT": 0.1})
    pipeline.ingest_control(T0, {"k": "conn_open", "conn": "c", "logical": "public"})
    pipeline.ingest_frame("c", T0, _depth(1, 10, 0))
    snapshot = orjson.dumps({"lastUpdateId": 5, "T": 1, "bids": [], "asks": []})
    pipeline.ingest_control(T0, {"k": "snapshot", "symbol": "BTCUSDT"}, snapshot)
    state = {"seq": 10}

    def step() -> None:
        last = state["seq"] + 10
        pipeline.ingest_frame("c", T0, _depth(state["seq"] + 1, last, state["seq"]))
        state["seq"] = last

    benchmark(step)
    assert pipeline.books["BTCUSDT"].resyncs == 0


def test_kraken_update_with_checksum(benchmark: pytest.FixtureRequest) -> None:
    pipeline = VenuePipeline(Kraken(kraken_cfg()), MemorySink(), {"BTC/USD": 0.1})
    pipeline.ingest_control(T0, {"k": "conn_open", "conn": "c", "logical": "ws"})
    lines = frames("kraken.jsonl")
    for raw in lines:
        pipeline.ingest_frame("c", T0, raw)
    update = next(r for r in lines if b'"type":"update"' in r and b'"book"' in r)
    adapter = pipeline.adapter

    def step() -> None:
        events = adapter.parse(update, T0)
        adapter.check(None, events[0], pipeline.books["BTC/USD"].book)

    benchmark(step)
