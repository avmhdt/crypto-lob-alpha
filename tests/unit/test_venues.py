"""Per-venue parsers on recorded sample messages, side mapping (DQ-06) and sync rules."""

from __future__ import annotations

import orjson
import pytest

from conftest import binance_cfg, coinbase_cfg, frames, kraken_cfg
from lob_alpha.book.l2 import BookL2
from lob_alpha.record.venues.base import (
    BboEvent,
    ClockSample,
    DepthEvent,
    Heartbeat,
    ParseError,
    Snapshot,
    SnapshotEvent,
    TradeEvent,
    rfc3339_ns,
)
from lob_alpha.record.venues.binance_usdm import BinanceUsdm
from lob_alpha.record.venues.coinbase import Coinbase
from lob_alpha.record.venues.kraken import Kraken
from lob_alpha.record.venues.registry import make_adapter

TS = 1_793_566_970_000_000_000


def _first(adapter, name: str, kind: type):
    for raw in frames(name):
        events = adapter.parse(raw, TS)
        if events and type(events[0]) is kind:
            return events[0]
    raise AssertionError(f"no {kind} in {name}")


def test_rfc3339_ns_keeps_full_precision() -> None:
    assert rfc3339_ns("2026-11-01T21:02:50.906419Z") == 1_793_566_970_906_419_000
    assert rfc3339_ns("2026-11-01T21:02:50Z") == 1_793_566_970_000_000_000
    assert rfc3339_ns("2026-11-01T21:02:50.123456789Z") == 1_793_566_970_123_456_789


def test_rfc3339_ns_rejects_other_shapes() -> None:
    with pytest.raises(ParseError):
        rfc3339_ns("2026-11-01")


def test_binance_depth() -> None:
    ev = _first(BinanceUsdm(binance_cfg()), "binance_usdm.jsonl", DepthEvent)
    assert ev.symbol == "BTCUSDT"
    assert ev.ts_recv == TS
    assert ev.seq_first is not None
    assert ev.seq_prev is not None
    assert ev.seq is not None
    assert ev.seq_first <= ev.seq
    assert ev.ts_exchange % 1_000_000 == 0
    assert all(px_str is None for _, _, px_str, _ in ev.bids)


def test_binance_book_ticker() -> None:
    ev = _first(BinanceUsdm(binance_cfg()), "binance_usdm.jsonl", BboEvent)
    assert ev.bid_px == 77896.7
    assert ev.ask_px == 77896.8
    assert ev.seq == 11986099377374


def test_binance_agg_trade_side_is_seller_initiated_when_buyer_is_maker() -> None:
    adapter = BinanceUsdm(binance_cfg())
    template = {"e": "aggTrade", "T": 1, "a": 7, "s": "BTCUSDT", "p": "1.0", "q": "2.0"}
    for buyer_is_maker, side in ((True, -1), (False, 1)):
        raw = orjson.dumps({"stream": "x", "data": {**template, "m": buyer_is_maker}})
        ev = adapter.parse(raw, TS)[0]
        assert isinstance(ev, TradeEvent)
        assert ev.side == side
        assert ev.trade_id == "7"
        assert ev.is_aggregated


def test_binance_ack_and_unknown_events() -> None:
    adapter = BinanceUsdm(binance_cfg())
    assert adapter.parse(b'{"result":null,"id":1}', TS) == []
    with pytest.raises(ParseError):
        adapter.parse(b'{"stream":"x","data":{"e":"mystery"}}', TS)
    with pytest.raises(ParseError):
        adapter.parse(b'{"stream":"x","data":{"e":"depthUpdate"}}', TS)


def test_binance_connections_split_public_and_market() -> None:
    public, market = BinanceUsdm(binance_cfg()).connections(["BTCUSDT", "ETHUSDT"])
    assert public.url == (
        "wss://x/public/stream?streams=btcusdt@depth@100ms/btcusdt@bookTicker/"
        "ethusdt@depth@100ms/ethusdt@bookTicker"
    )
    assert market.url == "wss://x/market/stream?streams=btcusdt@aggTrade/ethusdt@aggTrade"
    assert public.book
    assert not public.trades
    assert market.trades
    assert not market.book


def _ev(first: int, last: int, prev: int) -> DepthEvent:
    return DepthEvent("BTCUSDT", 0, TS, [], [], seq=last, seq_first=first, seq_prev=prev)


def _snap(last_update_id: int) -> Snapshot:
    return Snapshot("BTCUSDT", 0, TS, [], [], last_update_id)


def test_binance_bridge_drops_old_events_and_starts_at_the_straddling_one() -> None:
    adapter = BinanceUsdm(binance_cfg())
    result = adapter.bridge(_snap(105), [_ev(90, 99, 89), _ev(100, 110, 99), _ev(111, 120, 110)])
    assert result.status == "bridged"
    assert [e.seq for e in result.applied] == [110, 120]


def test_binance_bridge_accepts_the_boundary_u_equal_last_update_id() -> None:
    result = BinanceUsdm(binance_cfg()).bridge(_snap(110), [_ev(100, 110, 99)])
    assert result.status == "bridged"


def test_binance_bridge_waits_when_snapshot_is_ahead_of_the_buffer() -> None:
    result = BinanceUsdm(binance_cfg()).bridge(_snap(200), [_ev(100, 110, 99)])
    assert result.status == "wait"


def test_binance_bridge_stale_when_first_event_is_past_the_snapshot() -> None:
    result = BinanceUsdm(binance_cfg()).bridge(_snap(50), [_ev(100, 110, 99)])
    assert result.status == "stale"


def test_binance_check_uses_pu() -> None:
    adapter = BinanceUsdm(binance_cfg())
    book = BookL2(0.1)
    assert adapter.check(None, _ev(1, 2, 0), book).ok
    assert adapter.check(_ev(1, 2, 0), _ev(3, 4, 2), book).ok
    assert adapter.check(_ev(1, 2, 0), _ev(5, 6, 4), book).reason == "seq_gap"


def test_binance_snapshot_parse_and_bad_body() -> None:
    adapter = BinanceUsdm(binance_cfg())
    raw = b'{"lastUpdateId":9,"T":5,"E":6,"bids":[["1.0","2.0"]],"asks":[["3.0","4.0"]]}'
    snapshot = adapter.parse_snapshot(raw, "BTCUSDT", TS)
    assert snapshot.seq == 9
    assert snapshot.bids == [(1.0, 2.0, None, None)]
    with pytest.raises(ParseError):
        adapter.parse_snapshot(b'{"code":-1003}', "BTCUSDT", TS)


def test_kraken_book_keeps_wire_text_of_numbers() -> None:
    ev = _first(Kraken(kraken_cfg()), "kraken.jsonl", SnapshotEvent)
    assert ev.symbol == "BTC/USD"
    assert ev.checksum is not None
    for px, qty, px_str, qty_str in ev.bids + ev.asks:
        assert float(px_str) == px
        assert float(qty_str) == qty
        assert len(qty_str.split(".")[1]) == 8  # trailing zeros matter to the checksum


def test_kraken_trade_side_is_the_taker_side() -> None:
    adapter = Kraken(kraken_cfg())
    template = {"symbol": "BTC/USD", "price": 1.0, "qty": 2.0, "ord_type": "limit"}
    template |= {"trade_id": 5, "timestamp": "2026-11-01T21:02:53.334453Z"}
    for side, sign in (("buy", 1), ("sell", -1)):
        raw = orjson.dumps(
            {"channel": "trade", "type": "update", "data": [{**template, "side": side}]}
        )
        ev = adapter.parse(raw, TS)[0]
        assert isinstance(ev, TradeEvent)
        assert ev.side == sign
        assert not ev.is_aggregated


def test_kraken_control_messages() -> None:
    adapter = Kraken(kraken_cfg())
    assert isinstance(adapter.parse(b'{"channel":"heartbeat"}', TS)[0], Heartbeat)
    assert adapter.parse(b'{"channel":"status","type":"update","data":[]}', TS) == []
    ok = b'{"method":"subscribe","result":{"channel":"book"},"success":true}'
    assert adapter.parse(ok, TS) == []
    with pytest.raises(ParseError, match="rejected"):
        adapter.parse(b'{"method":"subscribe","success":false,"error":"nope"}', TS)
    with pytest.raises(ParseError, match="unknown"):
        adapter.parse(b'{"channel":"mystery"}', TS)


def test_kraken_pong_is_a_clock_sample() -> None:
    ev = _first(Kraken(kraken_cfg()), "kraken.jsonl", ClockSample)
    assert ev.sent_ns > 1_700_000_000_000_000_000
    assert abs(ev.server_time_ns - ev.sent_ns) < 60_000_000_000


def test_kraken_subscribe_messages_ask_for_configured_depth() -> None:
    adapter = Kraken(kraken_cfg())
    spec = adapter.connections(["BTC/USD"])[0]
    book, trade = (orjson.loads(m) for m in adapter.subscribe_messages(spec))
    assert book["params"] == {"channel": "book", "symbol": ["BTC/USD"], "depth": 10}
    assert trade["params"] == {"channel": "trade", "symbol": ["BTC/USD"]}
    first, second = (orjson.loads(m) for m in adapter.resubscribe_messages("BTC/USD"))
    assert first["method"] == "unsubscribe"
    assert second["method"] == "subscribe"


def test_kraken_has_no_rest_snapshot_path() -> None:
    with pytest.raises(ParseError):
        Kraken(kraken_cfg()).parse_snapshot(b"{}", "BTC/USD", TS)


def test_coinbase_snapshot_and_l2update() -> None:
    adapter = Coinbase(coinbase_cfg())
    snapshot = _first(adapter, "coinbase.jsonl", SnapshotEvent)
    assert snapshot.symbol == "BTC-USD"
    assert len(snapshot.bids) == 5
    update = _first(adapter, "coinbase.jsonl", DepthEvent)
    assert update.seq is None
    assert update.bids or update.asks


def test_coinbase_match_side_is_the_maker_side() -> None:
    adapter = Coinbase(coinbase_cfg())
    template = {"type": "match", "trade_id": 3, "size": "1", "price": "2", "sequence": 9}
    template |= {"product_id": "BTC-USD", "time": "2026-11-01T21:02:51.119047Z"}
    for maker_side, sign in (("sell", 1), ("buy", -1)):
        ev = adapter.parse(orjson.dumps({**template, "side": maker_side}), TS)[0]
        assert isinstance(ev, TradeEvent)
        assert ev.side == sign
        assert ev.seq == 9


def test_coinbase_control_messages() -> None:
    adapter = Coinbase(coinbase_cfg())
    heartbeat = _first(adapter, "coinbase.jsonl", Heartbeat)
    assert heartbeat.last_trade_id is not None
    assert adapter.parse(b'{"type":"subscriptions","channels":[]}', TS) == []
    with pytest.raises(ParseError, match="coinbase error"):
        adapter.parse(b'{"type":"error","message":"bad","reason":"nope"}', TS)
    with pytest.raises(ParseError, match="unknown"):
        adapter.parse(b'{"type":"mystery"}', TS)
    with pytest.raises(ParseError):
        adapter.parse(b'{"type":"match"}', TS)


def test_coinbase_rest_book_with_order_counts() -> None:
    adapter = Coinbase(coinbase_cfg())
    raw = b'{"bids":[["1.0","2.0",3]],"asks":[["4.0","5.0",6]],"sequence":7.0}'
    snapshot = adapter.parse_snapshot(raw, "BTC-USD", TS)
    assert snapshot.seq == 7
    assert snapshot.asks == [(4.0, 5.0, None, None)]


def test_registry_only_knows_the_decided_venues() -> None:
    assert isinstance(make_adapter("kraken", kraken_cfg()), Kraken)
    assert isinstance(make_adapter("coinbase", coinbase_cfg()), Coinbase)
    assert isinstance(make_adapter("binance-usdm", binance_cfg()), BinanceUsdm)
    with pytest.raises(ValueError, match="okx"):
        make_adapter("okx", kraken_cfg())
