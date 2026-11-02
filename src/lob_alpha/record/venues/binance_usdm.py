"""Binance USD-M perpetuals: `/public` depth and bookTicker, `/market` aggTrade."""

from __future__ import annotations

import time
from collections.abc import Sequence

import httpx
import orjson

from lob_alpha.book.l2 import BookL2
from lob_alpha.record.venues.base import (
    BboEvent,
    BridgeResult,
    CheckResult,
    ConnectionSpec,
    DepthEvent,
    Level,
    ParseError,
    ServerTime,
    Snapshot,
    TradeEvent,
    VenueEvent,
    timed_get,
)
from lob_alpha.settings import VenueConfig

VENUE = "binance-usdm"
MS = 1_000_000


def _levels(rows: Sequence[Sequence[str]]) -> list[Level]:
    return [(float(p), float(q), None, None) for p, q in rows]


class BinanceUsdm:
    venue = VENUE
    rest_snapshot = True
    max_conn_age_h: float | None = 24.0
    book_depth: int | None = None
    keep_raw_strings = False
    contiguous_trade_ids = True
    checkpoint_kind: str | None = "rest_l2"

    def __init__(self, cfg: VenueConfig) -> None:
        self.cfg = cfg

    def connections(self, symbols: Sequence[str]) -> list[ConnectionSpec]:
        lower = [s.lower() for s in symbols]
        depth, bbo, trades = (self.cfg.streams[k] for k in ("depth", "bbo", "trades"))
        public = "/".join(depth.format(s=s) for s in lower)
        ticker = "/".join(bbo.format(s=s) for s in lower)
        market = "/".join(trades.format(s=s) for s in lower)
        return [
            ConnectionSpec(
                "public",
                f"{self.cfg.ws_public}?streams={public}",
                tuple(symbols),
                book=True,
                trades=False,
            ),
            # bookTicker is most of the message rate, so a drop here leaves the book LIVE.
            ConnectionSpec(
                "ticker",
                f"{self.cfg.ws_public}?streams={ticker}",
                tuple(symbols),
                book=False,
                trades=False,
                bbo=True,
            ),
            ConnectionSpec(
                "market",
                f"{self.cfg.ws_market}?streams={market}",
                tuple(symbols),
                book=False,
                trades=True,
            ),
        ]

    def subscribe_messages(self, conn: ConnectionSpec) -> list[bytes]:
        return []  # streams are named in the connection URL

    def resubscribe_messages(self, symbol: str) -> list[bytes]:
        return []  # a resync is a new REST snapshot; the diff stream never stops

    async def fetch_snapshot(self, http: httpx.AsyncClient, symbol: str) -> Snapshot | None:
        url = f"{self.cfg.rest}/fapi/v1/depth"
        params: dict[str, str | int] = {"symbol": symbol, "limit": self.cfg.snapshot_limit}
        response = await http.get(url, params=params)
        ts_recv = time.time_ns()
        response.raise_for_status()
        return self.parse_snapshot(response.content, symbol, ts_recv)

    async def fetch_checkpoint(self, http: httpx.AsyncClient, symbol: str) -> Snapshot | None:
        return await self.fetch_snapshot(http, symbol)

    async def fetch_server_time(self, http: httpx.AsyncClient) -> ServerTime | None:
        sent_ns, response, ts_recv = await timed_get(http, f"{self.cfg.rest}/fapi/v1/time")
        return ServerTime(sent_ns, int(response.json()["serverTime"]) * MS, ts_recv)

    def clock_ping(self, sent_ns: int) -> bytes | None:
        return None

    def parse_snapshot(self, raw: bytes, symbol: str, ts_recv: int) -> Snapshot:
        try:
            body = orjson.loads(raw)
            return Snapshot(
                symbol=symbol,
                ts_exchange=int(body["T"]) * MS,
                ts_recv=ts_recv,
                bids=_levels(body["bids"]),
                asks=_levels(body["asks"]),
                seq=int(body["lastUpdateId"]),
                raw=raw,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ParseError(f"binance depth snapshot: {exc!r}") from exc

    def parse(self, raw: bytes, ts_recv: int) -> list[VenueEvent]:
        message = orjson.loads(raw)
        data = message.get("data") if isinstance(message, dict) else None
        if not isinstance(data, dict):
            return []  # subscription acks carry no data payload
        try:
            kind = data["e"]
            if kind == "depthUpdate":
                return [
                    DepthEvent(
                        symbol=data["s"],
                        ts_exchange=data["T"] * MS,
                        ts_recv=ts_recv,
                        bids=_levels(data["b"]),
                        asks=_levels(data["a"]),
                        seq=data["u"],
                        seq_first=data["U"],
                        seq_prev=data["pu"],
                    )
                ]
            if kind == "bookTicker":
                return [
                    BboEvent(
                        symbol=data["s"],
                        ts=data["T"] * MS,
                        ts_recv=ts_recv,
                        bid_px=float(data["b"]),
                        bid_qty=float(data["B"]),
                        ask_px=float(data["a"]),
                        ask_qty=float(data["A"]),
                        seq=data["u"],
                    )
                ]
            if kind == "aggTrade":
                return [
                    TradeEvent(
                        symbol=data["s"],
                        ts_exchange=data["T"] * MS,
                        ts_recv=ts_recv,
                        trade_id=str(data["a"]),
                        price=float(data["p"]),
                        qty=float(data["q"]),
                        side=-1 if data["m"] else 1,  # m=true: buyer is maker, seller aggressed
                        is_aggregated=True,
                    )
                ]
        except (KeyError, TypeError, ValueError) as exc:
            raise ParseError(f"binance {data.get('e')}: {exc!r}") from exc
        raise ParseError(f"unknown binance event type: {kind!r}")

    def bridge(self, snapshot: Snapshot, buffered: Sequence[DepthEvent]) -> BridgeResult:
        """Documented local-book procedure: first applied event has U <= lastUpdateId <= u."""
        last_update_id = snapshot.seq
        if last_update_id is None:
            raise ParseError("binance snapshot has no lastUpdateId")
        kept = [ev for ev in buffered if ev.seq is not None and ev.seq >= last_update_id]
        if not kept:
            return BridgeResult("wait")
        first = kept[0]
        if first.seq_first is None or first.seq_first > last_update_id:
            return BridgeResult("stale")
        return BridgeResult("bridged", applied=kept)

    def check(self, prev: DepthEvent | None, ev: DepthEvent, book: BookL2) -> CheckResult:
        if prev is None or ev.seq_prev == prev.seq:
            return CheckResult(ok=True)
        return CheckResult(ok=False, reason="seq_gap")
