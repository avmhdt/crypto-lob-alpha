"""State machine, gaps and resync on the sans-IO pipeline (Fig. 2, FR-06, FR-07)."""

from __future__ import annotations

import orjson

from conftest import MemorySink, binance_cfg, coinbase_cfg, frames, kraken_cfg
from lob_alpha.record.sync import CloseConn, NeedSnapshot, Resubscribe, VenuePipeline
from lob_alpha.record.venues.base import SyncState
from lob_alpha.record.venues.binance_usdm import BinanceUsdm
from lob_alpha.record.venues.coinbase import Coinbase
from lob_alpha.record.venues.kraken import Kraken

SEC = 1_000_000_000
T0 = 1_793_566_970 * SEC
PUBLIC = "public.g0"
MARKET = "market.g0"
TICKER = "ticker.g0"
TICKS = {"BTCUSDT": 0.1, "ETHUSDT": 0.01}


def depth(first: int, last: int, prev: int, symbol: str = "BTCUSDT", bid: str = "100.0") -> bytes:
    data = {
        "e": "depthUpdate", "E": 1, "T": 1, "s": symbol, "U": first, "u": last, "pu": prev,
        "b": [[bid, "1.0"]], "a": [["101.0", "2.0"]],
    }  # fmt: skip
    return orjson.dumps({"stream": "x", "data": data})


def rest(last_update_id: int) -> bytes:
    return orjson.dumps(
        {
            "lastUpdateId": last_update_id,
            "T": 1,
            "E": 1,
            "bids": [["99.0", "1.0"]],
            "asks": [["102.0", "1.0"]],
        }
    )


def trade(trade_id: int, symbol: str = "BTCUSDT") -> bytes:
    data = {"e": "aggTrade", "T": 1, "a": trade_id, "s": symbol, "p": "1", "q": "1", "m": False}
    return orjson.dumps({"stream": "x", "data": data})


def ticker(update_id: int, symbol: str = "BTCUSDT") -> bytes:
    data = {
        "e": "bookTicker", "T": 1, "u": update_id, "s": symbol,
        "b": "1", "B": "1", "a": "2", "A": "1",
    }  # fmt: skip
    return orjson.dumps({"stream": "x", "data": data})


def make(sink: MemorySink) -> VenuePipeline:
    pipeline = VenuePipeline(BinanceUsdm(binance_cfg()), sink, TICKS, host="h", version="v")
    pipeline.ingest_control(T0, {"k": "conn_open", "conn": PUBLIC, "logical": "public"})
    pipeline.ingest_control(T0, {"k": "conn_open", "conn": MARKET, "logical": "market"})
    pipeline.ingest_control(T0, {"k": "conn_open", "conn": TICKER, "logical": "ticker"})
    return pipeline


def to_live(pipeline: VenuePipeline, ts: int = T0 + 1) -> None:
    assert pipeline.ingest_frame(PUBLIC, ts, depth(10, 20, 9)) == [NeedSnapshot("BTCUSDT")]
    pipeline.ingest_control(ts + 1, {"k": "snapshot", "symbol": "BTCUSDT"}, rest(15))
    assert pipeline.books["BTCUSDT"].state is SyncState.LIVE


def test_snapshot_is_requested_once_and_bridges_to_live(sink: MemorySink) -> None:
    pipeline = make(sink)
    book = pipeline.books["BTCUSDT"]
    assert book.state is SyncState.SNAPSHOT
    assert pipeline.ingest_frame(PUBLIC, T0 + 1, depth(10, 20, 9)) == [NeedSnapshot("BTCUSDT")]
    assert pipeline.ingest_frame(PUBLIC, T0 + 2, depth(21, 30, 20)) == []
    pipeline.ingest_control(T0 + 3, {"k": "snapshot", "symbol": "BTCUSDT"}, rest(15))
    assert book.state is SyncState.LIVE
    assert book.epoch == 1
    assert [b.is_snapshot for b in sink.batches] == [True, False, False]
    assert sink.batches[0].seq == 15
    assert [b.seq for b in sink.batches[1:]] == [20, 30]


def test_bridged_rows_are_never_stamped_before_the_snapshot(sink: MemorySink) -> None:
    pipeline = make(sink)
    pipeline.ingest_frame(PUBLIC, T0 + 1, depth(10, 20, 9))
    pipeline.ingest_control(T0 + 100, {"k": "snapshot", "symbol": "BTCUSDT"}, rest(15))
    pipeline.ingest_frame(PUBLIC, T0 + 101, depth(21, 30, 20))
    times = [b.ts_recv for b in sink.batches]
    ids = [b.batch_id for b in sink.batches]
    assert times == [T0 + 100, T0 + 100, T0 + 101]
    assert ids == sorted(ids)
    assert len(set(ids)) == 3


def test_seq_gap_goes_through_gap_and_resync_and_records_one_row(sink: MemorySink) -> None:
    pipeline = make(sink)
    to_live(pipeline)
    book = pipeline.books["BTCUSDT"]
    actions = pipeline.ingest_frame(PUBLIC, T0 + SEC, depth(40, 50, 39))
    assert actions == [NeedSnapshot("BTCUSDT")]
    assert book.state is SyncState.RESYNC
    assert sink.gaps == []  # the row is written when the stream is back in LIVE
    pipeline.ingest_control(T0 + 2 * SEC, {"k": "snapshot", "symbol": "BTCUSDT"}, rest(45))
    assert book.state is SyncState.LIVE
    assert book.epoch == 2
    assert len(sink.gaps) == 1
    gap = sink.gaps[0]
    assert (gap.reason, gap.stream, gap.symbol) == ("seq_gap", "book", "BTCUSDT")
    assert gap.seq_before == 20
    assert gap.seq_after == 50
    assert gap.host == "h"
    assert gap.recorder_version == "v"
    assert gap.duration_ms > 0


def test_stale_snapshot_requests_another_one(sink: MemorySink) -> None:
    pipeline = make(sink)
    pipeline.ingest_frame(PUBLIC, T0 + 1, depth(100, 110, 99))
    actions = pipeline.ingest_control(T0 + 2, {"k": "snapshot", "symbol": "BTCUSDT"}, rest(50))
    assert actions == [NeedSnapshot("BTCUSDT")]
    book = pipeline.books["BTCUSDT"]
    assert book.state is SyncState.RESYNC
    pipeline.ingest_control(T0 + 3, {"k": "snapshot", "symbol": "BTCUSDT"}, rest(105))
    assert book.state is SyncState.LIVE


def test_stale_gap_is_labelled_after_a_failed_resync(sink: MemorySink) -> None:
    pipeline = make(sink)
    to_live(pipeline)
    pipeline.ingest_frame(PUBLIC, T0 + SEC, depth(40, 50, 39))
    pipeline.ingest_control(T0 + 2 * SEC, {"k": "snapshot", "symbol": "BTCUSDT"}, rest(10))
    pipeline.ingest_control(T0 + 3 * SEC, {"k": "snapshot", "symbol": "BTCUSDT"}, rest(45))
    assert [g.reason for g in sink.gaps] == ["seq_gap"]  # the first cause names the interval


def test_snapshot_ahead_of_buffer_waits_for_the_stream(sink: MemorySink) -> None:
    pipeline = make(sink)
    pipeline.ingest_frame(PUBLIC, T0 + 1, depth(10, 20, 9))
    pipeline.ingest_control(T0 + 2, {"k": "snapshot", "symbol": "BTCUSDT"}, rest(25))
    book = pipeline.books["BTCUSDT"]
    assert book.state is SyncState.SYNCING
    pipeline.ingest_frame(PUBLIC, T0 + 3, depth(21, 30, 20))
    assert book.state is SyncState.LIVE
    assert [b.seq for b in sink.batches] == [25, 30]


def test_broken_chain_inside_the_buffer_fails_the_bridge(sink: MemorySink) -> None:
    pipeline = make(sink)
    pipeline.ingest_frame(PUBLIC, T0 + 1, depth(10, 20, 9))
    pipeline.ingest_frame(PUBLIC, T0 + 2, depth(25, 30, 24))
    actions = pipeline.ingest_control(T0 + 3, {"k": "snapshot", "symbol": "BTCUSDT"}, rest(15))
    assert actions == [NeedSnapshot("BTCUSDT")]
    assert pipeline.books["BTCUSDT"].state is SyncState.RESYNC


def test_failure_on_one_symbol_leaves_the_others_live(sink: MemorySink) -> None:
    pipeline = make(sink)
    to_live(pipeline)
    pipeline.ingest_frame(PUBLIC, T0 + 2, depth(10, 20, 9, "ETHUSDT"))
    pipeline.ingest_control(T0 + 3, {"k": "snapshot", "symbol": "ETHUSDT"}, rest(15))
    pipeline.ingest_frame(PUBLIC, T0 + SEC, depth(40, 50, 39))
    assert pipeline.books["BTCUSDT"].state is SyncState.RESYNC
    assert pipeline.books["ETHUSDT"].state is SyncState.LIVE


def test_disconnect_opens_gaps_on_every_stream_of_the_connection(sink: MemorySink) -> None:
    pipeline = make(sink)
    to_live(pipeline)
    pipeline.ingest_frame(MARKET, T0 + 5, trade(1))
    pipeline.ingest_frame(TICKER, T0 + 6, ticker(1))
    pipeline.ingest_control(T0 + SEC, {"k": "conn_close", "conn": PUBLIC, "reason": "disconnect"})
    book = pipeline.books["BTCUSDT"]
    assert book.state is SyncState.CONNECTING
    assert pipeline.bbos["BTCUSDT"].state is SyncState.LIVE  # bookTicker has its own connection
    assert pipeline.trades["BTCUSDT"].state is SyncState.LIVE  # the market connection is fine
    next_conn = "public.g1"
    pipeline.ingest_control(
        T0 + 2 * SEC, {"k": "conn_open", "conn": next_conn, "logical": "public"}
    )
    pipeline.ingest_frame(next_conn, T0 + 2 * SEC + 1, depth(40, 50, 39))
    pipeline.ingest_control(T0 + 3 * SEC, {"k": "snapshot", "symbol": "BTCUSDT"}, rest(45))
    assert [(g.stream, g.symbol, g.reason) for g in sink.gaps] == [
        ("book", "BTCUSDT", "disconnect")
    ]


def test_a_dropped_ticker_connection_leaves_the_book_live(sink: MemorySink) -> None:
    pipeline = make(sink)
    to_live(pipeline)
    pipeline.ingest_frame(TICKER, T0 + 6, ticker(1))
    pipeline.ingest_control(T0 + SEC, {"k": "conn_close", "conn": TICKER, "reason": "disconnect"})
    assert pipeline.books["BTCUSDT"].state is SyncState.LIVE
    assert pipeline.bbos["BTCUSDT"].state is SyncState.CONNECTING
    pipeline.ingest_control(
        T0 + 2 * SEC, {"k": "conn_open", "conn": "ticker.g1", "logical": "ticker"}
    )
    pipeline.ingest_frame("ticker.g1", T0 + 3 * SEC, ticker(2))
    assert pipeline.bbos["BTCUSDT"].state is SyncState.LIVE
    assert sorted((g.stream, g.symbol) for g in sink.gaps) == [
        ("bbo", "BTCUSDT"),
        ("bbo", "ETHUSDT"),
    ]


def test_archives_from_the_shared_public_connection_still_replay(sink: MemorySink) -> None:
    """Hours recorded before the split carried bookTicker on `public` and must still recover."""
    pipeline = VenuePipeline(BinanceUsdm(binance_cfg()), sink, TICKS)
    pipeline.ingest_control(T0, {"k": "conn_open", "conn": PUBLIC, "logical": "public"})
    pipeline.ingest_frame(PUBLIC, T0 + 1, depth(10, 20, 9))
    pipeline.ingest_control(T0 + 2, {"k": "snapshot", "symbol": "BTCUSDT"}, rest(15))
    pipeline.ingest_frame(PUBLIC, T0 + 3, ticker(1))
    pipeline.ingest_frame(PUBLIC, T0 + 4, depth(21, 30, 20))
    pipeline.ingest_frame(PUBLIC, T0 + 5, ticker(2))
    assert [b.seq for b in sink.bbos] == [1, 2]
    assert [b.seq for b in sink.batches] == [15, 20, 30]
    assert pipeline.parse_errors == 0


def test_trades_dedupe_and_flag_id_jumps(sink: MemorySink) -> None:
    pipeline = make(sink)
    pipeline.ingest_frame(MARKET, T0 + 1, trade(100))
    pipeline.ingest_frame(MARKET, T0 + 2, trade(100))
    pipeline.ingest_frame(MARKET, T0 + 3, trade(101))
    pipeline.ingest_frame(MARKET, T0 + 4, trade(105))
    assert [ev.trade_id for ev, _ in sink.trades] == ["100", "101", "105"]
    assert pipeline.trades["BTCUSDT"].duplicates == 1
    assert len(sink.gaps) == 1
    assert sink.gaps[0].reason == "seq_gap"
    assert (sink.gaps[0].seq_before, sink.gaps[0].seq_after) == (101, 105)


def test_batch_ids_strictly_increase_even_for_equal_timestamps(sink: MemorySink) -> None:
    pipeline = make(sink)
    for trade_id in (1, 2, 3):
        pipeline.ingest_frame(MARKET, T0 + 1, trade(trade_id))
    ids = [batch for _, batch in sink.trades]
    assert ids == sorted(set(ids))


def test_checkpoints_record_levels_and_never_touch_the_book(sink: MemorySink) -> None:
    pipeline = make(sink)
    to_live(pipeline)
    before = pipeline.books["BTCUSDT"].book.best_bid()
    ctrl = {"k": "checkpoint", "symbol": "BTCUSDT", "checkpoint": "rest_l2"}
    pipeline.ingest_control(T0 + SEC, ctrl, rest(16))
    assert sink.checkpoints == [("rest_l2", True, None, None)]
    assert len(sink.levels) == 1
    assert pipeline.books["BTCUSDT"].book.best_bid() == before


def test_timers_flag_stale_book_and_silent_connection(sink: MemorySink) -> None:
    pipeline = make(sink)
    to_live(pipeline)
    book = pipeline.books["BTCUSDT"]
    now = T0 + 31 * SEC
    pipeline.conns[PUBLIC].last_frame_ns = now
    pipeline.conns[MARKET].last_frame_ns = now
    pipeline.conns[TICKER].last_frame_ns = now
    controls, actions = pipeline.timers(now)
    assert {"k": "stale", "symbol": "BTCUSDT"} in controls
    assert actions == [NeedSnapshot("ETHUSDT")]  # ETHUSDT never reached LIVE
    assert pipeline.ingest_control(now, {"k": "stale", "symbol": "BTCUSDT"}) == [
        NeedSnapshot("BTCUSDT")
    ]
    assert book.state is SyncState.RESYNC
    assert book.gap is not None
    assert book.gap.reason == "stale"
    pipeline.conns[PUBLIC].last_frame_ns = T0
    _, actions = pipeline.timers(now)
    assert CloseConn(PUBLIC, "stale") in actions


def test_snapshot_error_keeps_the_stream_out_of_live_and_retries(sink: MemorySink) -> None:
    pipeline = make(sink)
    to_live(pipeline)
    pipeline.ingest_frame(PUBLIC, T0 + SEC, depth(40, 50, 39))
    pipeline.ingest_control(
        T0 + 2 * SEC, {"k": "snapshot_error", "symbol": "BTCUSDT", "error": "HTTP 500"}
    )
    book = pipeline.books["BTCUSDT"]
    assert book.state is SyncState.RESYNC
    assert book.snapshot_requested is False


def test_clock_offset_past_the_limit_five_times_logs_a_clock_gap(sink: MemorySink) -> None:
    pipeline = make(sink)
    for i in range(5):
        ts = T0 + i * 60 * SEC
        pipeline.ingest_control(
            ts, {"k": "clock", "server_time_ns": ts + 100_000_000, "sent_ns": ts}
        )
    assert sink.gaps == []
    ts = T0 + 300 * SEC
    pipeline.ingest_control(ts, {"k": "clock", "server_time_ns": ts, "sent_ns": ts})
    assert [g.reason for g in sink.gaps] == ["clock"]
    assert sink.gaps[0].symbol == "*"
    assert len(sink.clocks) == 6


def test_a_clock_interval_open_at_shutdown_is_still_written(sink: MemorySink) -> None:
    pipeline = make(sink)
    for i in range(5):
        ts = T0 + i * 60 * SEC
        pipeline.ingest_control(
            ts, {"k": "clock", "server_time_ns": ts + 100_000_000, "sent_ns": ts}
        )
    pipeline.flush_gaps(T0 + 400 * SEC)
    assert [(g.reason, g.start_ns) for g in sink.gaps] == [("clock", T0)]


def test_four_bad_clock_probes_are_tolerated(sink: MemorySink) -> None:
    pipeline = make(sink)
    for i in range(4):
        ts = T0 + i
        pipeline.ingest_control(
            ts, {"k": "clock", "server_time_ns": ts + 100_000_000, "sent_ns": ts}
        )
    ts = T0 + 10
    pipeline.ingest_control(ts, {"k": "clock", "server_time_ns": ts, "sent_ns": ts})
    assert sink.gaps == []


def test_restart_gap_is_seeded_from_the_previous_run(sink: MemorySink) -> None:
    pipeline = VenuePipeline(BinanceUsdm(binance_cfg()), sink, TICKS)
    saved = {"BTCUSDT|book": {"epoch": 7, "last_good_ns": T0, "last_seq": 123}}
    pipeline.ingest_control(
        T0 + 60 * SEC, {"k": "run_start", "streams": saved, "reason": "host_down"}
    )
    pipeline.ingest_control(T0 + 60 * SEC, {"k": "conn_open", "conn": PUBLIC, "logical": "public"})
    pipeline.ingest_frame(PUBLIC, T0 + 61 * SEC, depth(10, 20, 9))
    pipeline.ingest_control(T0 + 62 * SEC, {"k": "snapshot", "symbol": "BTCUSDT"}, rest(15))
    assert pipeline.books["BTCUSDT"].epoch == 8
    gap = sink.gaps[0]
    assert (gap.reason, gap.start_ns, gap.seq_before) == ("host_down", T0, 123)


def test_first_ever_start_is_not_a_gap(sink: MemorySink) -> None:
    pipeline = make(sink)
    to_live(pipeline)
    assert sink.gaps == []


def test_parse_error_on_a_data_message_reconnects_and_counts(sink: MemorySink) -> None:
    pipeline = make(sink)
    actions = pipeline.ingest_frame(PUBLIC, T0 + 1, b'{"stream":"x","data":{"e":"mystery"}}')
    assert actions == [CloseConn(PUBLIC, "snapshot_error")]
    assert pipeline.parse_errors == 1


def test_undecodable_frame_is_counted_and_survives(sink: MemorySink) -> None:
    pipeline = make(sink)
    assert pipeline.ingest_frame(PUBLIC, T0 + 1, b"not json") == []
    assert pipeline.parse_errors == 1


def test_frames_from_unknown_connections_are_dropped(sink: MemorySink) -> None:
    pipeline = make(sink)
    assert pipeline.ingest_frame("nobody", T0, depth(1, 2, 0)) == []
    assert sink.batches == []


def test_timestamp_regressions_are_counted(sink: MemorySink) -> None:
    pipeline = make(sink)
    pipeline.ingest_frame(MARKET, T0 + 5, trade(1))
    pipeline.ingest_frame(MARKET, T0 + 4, trade(2))
    assert pipeline.ts_regressions == 1


def kraken_pipeline(sink: MemorySink) -> VenuePipeline:
    pipeline = VenuePipeline(Kraken(kraken_cfg()), sink, {"BTC/USD": 0.1})
    pipeline.ingest_control(T0, {"k": "conn_open", "conn": "ws.g0", "logical": "ws"})
    return pipeline


def test_kraken_recorded_stream_passes_every_checksum(sink: MemorySink) -> None:
    pipeline = kraken_pipeline(sink)
    for i, raw in enumerate(frames("kraken.jsonl")):
        pipeline.ingest_frame("ws.g0", T0 + i, raw)
    book = pipeline.books["BTC/USD"]
    assert book.state is SyncState.LIVE
    assert book.resyncs == 0
    assert len(sink.checkpoints) >= 50
    assert all(ok and expected == computed for _, ok, expected, computed in sink.checkpoints)
    assert sink.batches[0].is_snapshot
    assert sink.clocks
    assert sink.trades


def test_kraken_corrupted_update_fails_the_checksum_and_resubscribes(sink: MemorySink) -> None:
    pipeline = kraken_pipeline(sink)
    lines = frames("kraken.jsonl")
    index = next(i for i, raw in enumerate(lines) if b'"type":"update"' in raw and b"book" in raw)
    for i, raw in enumerate(lines[:index]):
        pipeline.ingest_frame("ws.g0", T0 + i, raw)
    message = orjson.loads(lines[index])
    message["data"][0]["checksum"] += 1
    actions = pipeline.ingest_frame("ws.g0", T0 + 1000, orjson.dumps(message))
    assert actions == [Resubscribe("BTC/USD", "ws.g0")]
    assert pipeline.books["BTC/USD"].state is SyncState.RESYNC
    assert sink.checkpoints[-1][1] is False
    for raw in lines:
        if b'"type":"snapshot"' in raw:
            pipeline.ingest_frame("ws.g0", T0 + 2000, raw)
    assert pipeline.books["BTC/USD"].state is SyncState.LIVE
    assert [g.reason for g in sink.gaps] == ["checksum_fail"]


def test_kraken_truncates_to_subscribed_depth(sink: MemorySink) -> None:
    pipeline = kraken_pipeline(sink)
    for i, raw in enumerate(frames("kraken.jsonl")):
        pipeline.ingest_frame("ws.g0", T0 + i, raw)
    book = pipeline.books["BTC/USD"].book
    assert book.depth(1) <= 10
    assert book.depth(-1) <= 10


def test_coinbase_goes_live_on_the_ws_snapshot(sink: MemorySink) -> None:
    pipeline = VenuePipeline(Coinbase(coinbase_cfg()), sink, {"BTC-USD": 0.01})
    pipeline.ingest_control(T0, {"k": "conn_open", "conn": "ws.g0", "logical": "ws"})
    for i, raw in enumerate(frames("coinbase.jsonl")):
        pipeline.ingest_frame("ws.g0", T0 + i, raw)
    assert pipeline.books["BTC-USD"].state is SyncState.LIVE
    assert sink.batches[0].is_snapshot
    assert sink.batches[1].seq == sink.batches[1].batch_id  # local monotone counter
    assert len(sink.trades) == 2
    assert pipeline.status(T0)["streams"][0]["state"] == "LIVE"


def test_coinbase_stale_book_resubscribes(sink: MemorySink) -> None:
    pipeline = VenuePipeline(Coinbase(coinbase_cfg()), sink, {"BTC-USD": 0.01})
    pipeline.ingest_control(T0, {"k": "conn_open", "conn": "ws.g0", "logical": "ws"})
    for i, raw in enumerate(frames("coinbase.jsonl")):
        pipeline.ingest_frame("ws.g0", T0 + i, raw)
    actions = pipeline.ingest_control(T0 + 40 * SEC, {"k": "stale", "symbol": "BTC-USD"})
    assert actions == [Resubscribe("BTC-USD", "ws.g0")]


def test_ws_venue_stuck_outside_live_reconnects(sink: MemorySink) -> None:
    pipeline = VenuePipeline(Coinbase(coinbase_cfg()), sink, {"BTC-USD": 0.01})
    pipeline.ingest_control(T0, {"k": "conn_open", "conn": "ws.g0", "logical": "ws"})
    now = T0 + 31 * SEC
    pipeline.conns["ws.g0"].last_frame_ns = now
    _, actions = pipeline.timers(now)
    assert actions == [CloseConn("ws.g0", "stale")]
