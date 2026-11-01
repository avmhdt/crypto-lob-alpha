"""Coinbase Exchange: `level2_batch`, `matches` and `heartbeat`."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx
import orjson

from lob_alpha.book.l2 import BookL2
from lob_alpha.record.venues.base import (
    BridgeResult,
    CheckResult,
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
    timed_get,
)
from lob_alpha.settings import VenueConfig

VENUE = "coinbase"


def _levels(rows: Sequence[Sequence[Any]]) -> list[Level]:
    return [(float(row[0]), float(row[1]), None, None) for row in rows]


class Coinbase:
    venue = VENUE
    rest_snapshot = False
    max_conn_age_h: float | None = None
    book_depth: int | None = None
    keep_raw_strings = False
    contiguous_trade_ids = True
    checkpoint_kind: str | None = "rest_l2"

    def __init__(self, cfg: VenueConfig) -> None:
        self.cfg = cfg

    def connections(self, symbols: Sequence[str]) -> list[ConnectionSpec]:
        return [ConnectionSpec("ws", self.cfg.ws, tuple(symbols), book=True, trades=True)]

    def subscribe_messages(self, conn: ConnectionSpec) -> list[bytes]:
        return [
            orjson.dumps(
                {
                    "type": "subscribe",
                    "product_ids": list(conn.symbols),
                    "channels": self.cfg.channels,
                }
            )
        ]

    def resubscribe_messages(self, symbol: str) -> list[bytes]:
        request = {"product_ids": [symbol], "channels": ["level2_batch"]}
        return [
            orjson.dumps({"type": "unsubscribe", **request}),
            orjson.dumps({"type": "subscribe", **request}),
        ]

    async def fetch_snapshot(self, http: httpx.AsyncClient, symbol: str) -> Snapshot | None:
        return None  # the L2 snapshot arrives on the websocket after subscribe

    async def fetch_checkpoint(self, http: httpx.AsyncClient, symbol: str) -> Snapshot | None:
        url = f"{self.cfg.rest}/products/{symbol}/book"
        response = await http.get(url, params={"level": 2})
        ts_recv = time.time_ns()
        response.raise_for_status()
        return self.parse_snapshot(response.content, symbol, ts_recv)

    async def fetch_server_time(self, http: httpx.AsyncClient) -> ServerTime | None:
        sent_ns, response, ts_recv = await timed_get(http, f"{self.cfg.rest}/time")
        return ServerTime(
            sent_ns, round(float(response.json()["epoch"]) * 1e3) * 1_000_000, ts_recv
        )

    def clock_ping(self, sent_ns: int) -> bytes | None:
        return None

    def parse_snapshot(self, raw: bytes, symbol: str, ts_recv: int) -> Snapshot:
        """Parse the REST `/book?level=2` body used for checkpoints."""
        try:
            body = orjson.loads(raw)
            return Snapshot(
                symbol=symbol,
                ts_exchange=rfc3339_ns(body["time"]) if "time" in body else ts_recv,
                ts_recv=ts_recv,
                bids=_levels(body["bids"]),
                asks=_levels(body["asks"]),
                seq=int(body["sequence"]),
                raw=raw,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ParseError(f"coinbase book: {exc!r}") from exc

    def parse(self, raw: bytes, ts_recv: int) -> list[VenueEvent]:
        message = orjson.loads(raw)
        kind = message.get("type")
        try:
            if kind == "l2update":
                bids: list[Level] = []
                asks: list[Level] = []
                for side, price, size in message["changes"]:
                    (bids if side == "buy" else asks).append(
                        (float(price), float(size), None, None)
                    )
                return [
                    DepthEvent(
                        symbol=message["product_id"],
                        ts_exchange=rfc3339_ns(message["time"]),
                        ts_recv=ts_recv,
                        bids=bids,
                        asks=asks,
                    )
                ]
            if kind == "match":
                return [
                    TradeEvent(
                        symbol=message["product_id"],
                        ts_exchange=rfc3339_ns(message["time"]),
                        ts_recv=ts_recv,
                        trade_id=str(message["trade_id"]),
                        price=float(message["price"]),
                        qty=float(message["size"]),
                        side=-1 if message["side"] == "buy" else 1,  # `side` is the maker side
                        is_aggregated=False,
                        seq=message["sequence"],
                    )
                ]
            if kind == "heartbeat":
                return [
                    Heartbeat(
                        ts_recv=ts_recv,
                        symbol=message["product_id"],
                        last_trade_id=message["last_trade_id"],
                    )
                ]
            if kind == "snapshot":
                return [
                    SnapshotEvent(
                        symbol=message["product_id"],
                        ts_exchange=rfc3339_ns(message["time"]) if "time" in message else ts_recv,
                        ts_recv=ts_recv,
                        bids=_levels(message["bids"]),
                        asks=_levels(message["asks"]),
                    )
                ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ParseError(f"coinbase {kind}: {exc!r}") from exc
        if kind in ("subscriptions", "last_match"):
            return []
        if kind == "error":
            raise ParseError(f"coinbase error: {message.get('message')}: {message.get('reason')}")
        raise ParseError(f"unknown coinbase message: {raw[:200]!r}")

    def bridge(self, snapshot: Snapshot, buffered: Sequence[DepthEvent]) -> BridgeResult:
        return BridgeResult("bridged")

    def check(self, prev: DepthEvent | None, ev: DepthEvent, book: BookL2) -> CheckResult:
        return CheckResult(ok=True)  # L2 carries no sequence; watchdogs and REST cover it
