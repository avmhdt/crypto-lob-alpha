"""Kraken spot WS v2: `book` with CRC32 checksums and `trade`."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx
import orjson

from lob_alpha.book.checksum import kraken_checksum
from lob_alpha.book.l2 import ASK, BID, BookL2
from lob_alpha.record.venues.base import (
    BridgeResult,
    CheckResult,
    ClockSample,
    ConnectionSpec,
    DepthEvent,
    Heartbeat,
    Level,
    ParseError,
    ServerTime,
    Snapshot,
    SnapshotEvent,
    TradeEvent,
    VenueEvent,
    rfc3339_ns,
)
from lob_alpha.settings import VenueConfig

VENUE = "kraken"


def _levels(rows: list[dict[str, str]]) -> list[Level]:
    out: list[Level] = []
    for row in rows:
        px, qty = row["price"], row["qty"]
        out.append((float(px), float(qty), px, qty))
    return out


class Kraken:
    venue = VENUE
    rest_snapshot = False
    max_conn_age_h: float | None = None
    keep_raw_strings = True
    contiguous_trade_ids = False
    checkpoint_kind: str | None = None

    def __init__(self, cfg: VenueConfig) -> None:
        self.cfg = cfg
        self.book_depth: int | None = cfg.book_depth

    def connections(self, symbols: Sequence[str]) -> list[ConnectionSpec]:
        return [ConnectionSpec("ws", self.cfg.ws, tuple(symbols), book=True, trades=True)]

    def _sub(self, method: str, channel: str, symbols: Sequence[str]) -> bytes:
        params: dict[str, Any] = {"channel": channel, "symbol": list(symbols)}
        if channel == "book":
            params["depth"] = self.cfg.book_depth
        return orjson.dumps({"method": method, "params": params})

    def subscribe_messages(self, conn: ConnectionSpec) -> list[bytes]:
        return [
            self._sub("subscribe", "book", conn.symbols),
            self._sub("subscribe", "trade", conn.symbols),
        ]

    def resubscribe_messages(self, symbol: str) -> list[bytes]:
        return [
            self._sub("unsubscribe", "book", [symbol]),
            self._sub("subscribe", "book", [symbol]),
        ]

    def clock_ping(self, sent_ns: int) -> bytes | None:
        """Application ping. `req_id` carries the send time so a pong alone gives the RTT."""
        return orjson.dumps({"method": "ping", "req_id": sent_ns})

    async def fetch_snapshot(self, http: httpx.AsyncClient, symbol: str) -> Snapshot | None:
        return None  # the snapshot arrives on the websocket after subscribe

    async def fetch_checkpoint(self, http: httpx.AsyncClient, symbol: str) -> Snapshot | None:
        return None  # every message carries a checksum

    async def fetch_server_time(self, http: httpx.AsyncClient) -> ServerTime | None:
        return None  # REST time has 1 s resolution; the websocket pong has microseconds

    def parse_snapshot(self, raw: bytes, symbol: str, ts_recv: int) -> Snapshot:
        raise ParseError("kraken has no REST book snapshot path")

    def parse(self, raw: bytes, ts_recv: int) -> list[VenueEvent]:
        # Books need the exact number text for the checksum, so floats stay strings here.
        if b'"channel":"book"' in raw[:32]:
            return self._parse_book(raw, ts_recv)
        message = orjson.loads(raw)
        channel = message.get("channel")
        if channel == "heartbeat":
            return [Heartbeat(ts_recv=ts_recv)]
        if channel == "trade":
            return self._parse_trades(message, ts_recv)
        if channel == "status":
            return []
        method = message.get("method")
        if method == "pong":
            return self._parse_pong(message, ts_recv)
        if method in ("subscribe", "unsubscribe"):
            if message.get("success") is not True:
                raise ParseError(f"kraken {method} rejected: {message.get('error')}")
            return []
        raise ParseError(f"unknown kraken message: {raw[:200]!r}")

    def _parse_book(self, raw: bytes, ts_recv: int) -> list[VenueEvent]:
        message = json.loads(raw, parse_float=str, parse_int=str)
        cls = SnapshotEvent if message.get("type") == "snapshot" else DepthEvent
        try:
            return [
                cls(
                    symbol=item["symbol"],
                    ts_exchange=rfc3339_ns(item["timestamp"]) if "timestamp" in item else ts_recv,
                    ts_recv=ts_recv,
                    bids=_levels(item["bids"]),
                    asks=_levels(item["asks"]),
                    checksum=int(item["checksum"]),
                )
                for item in message["data"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ParseError(f"kraken book: {exc!r}") from exc

    def _parse_trades(self, message: dict[str, Any], ts_recv: int) -> list[VenueEvent]:
        try:
            return [
                TradeEvent(
                    symbol=item["symbol"],
                    ts_exchange=rfc3339_ns(item["timestamp"]),
                    ts_recv=ts_recv,
                    trade_id=str(item["trade_id"]),
                    price=float(item["price"]),
                    qty=float(item["qty"]),
                    side=1 if item["side"] == "buy" else -1,  # `side` is the taker side
                    is_aggregated=False,
                )
                for item in message["data"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ParseError(f"kraken trade: {exc!r}") from exc

    def _parse_pong(self, message: dict[str, Any], ts_recv: int) -> list[VenueEvent]:
        try:
            return [
                ClockSample(
                    ts_recv=ts_recv,
                    sent_ns=int(message["req_id"]),
                    server_time_ns=rfc3339_ns(message["time_in"]),
                )
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ParseError(f"kraken pong: {exc!r}") from exc

    def bridge(self, snapshot: Snapshot, buffered: Sequence[DepthEvent]) -> BridgeResult:
        return BridgeResult("bridged")

    def check(self, prev: DepthEvent | None, ev: DepthEvent, book: BookL2) -> CheckResult:
        levels = self.cfg.checksum_levels
        computed = kraken_checksum(
            book.raw_levels(BID, levels), book.raw_levels(ASK, levels), levels
        )
        if computed == ev.checksum:
            return CheckResult(ok=True, expected=ev.checksum, computed=computed)
        return CheckResult(
            ok=False, reason="checksum_fail", expected=ev.checksum, computed=computed
        )
