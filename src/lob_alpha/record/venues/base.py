"""Normalised events and the adapter interface every venue implements."""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import httpx

if TYPE_CHECKING:
    from lob_alpha.book.l2 import BookL2

# price, qty, raw price string, raw qty string (strings only for checksum venues)
Level = tuple[float, float, str | None, str | None]


class ParseError(ValueError):
    """A data message did not have the documented shape. Never guessed around (R-03)."""


class SyncState(enum.StrEnum):
    CONNECTING = "CONNECTING"
    SNAPSHOT = "SNAPSHOT"
    SYNCING = "SYNCING"
    LIVE = "LIVE"
    GAP = "GAP"
    RESYNC = "RESYNC"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class ConnectionSpec:
    """One logical websocket. `name` is stable across reconnects."""

    name: str
    url: str
    symbols: tuple[str, ...]
    book: bool = True
    trades: bool = True
    bbo: bool = False


@dataclass(slots=True)
class DepthEvent:
    symbol: str
    ts_exchange: int
    ts_recv: int
    bids: list[Level]
    asks: list[Level]
    seq: int | None = None
    seq_first: int | None = None
    seq_prev: int | None = None
    checksum: int | None = None


@dataclass(slots=True)
class SnapshotEvent(DepthEvent):
    """A full book pushed on the websocket (Kraken, Coinbase L2)."""


@dataclass(slots=True)
class Snapshot:
    """A full book fetched over REST (Binance)."""

    symbol: str
    ts_exchange: int
    ts_recv: int
    bids: list[Level]
    asks: list[Level]
    seq: int | None
    raw: bytes = b""


@dataclass(slots=True)
class TradeEvent:
    symbol: str
    ts_exchange: int
    ts_recv: int
    trade_id: str
    price: float
    qty: float
    side: int
    is_aggregated: bool
    seq: int | None = None


@dataclass(slots=True)
class BboEvent:
    symbol: str
    ts: int
    ts_recv: int
    bid_px: float
    bid_qty: float
    ask_px: float
    ask_qty: float
    seq: int | None = None


@dataclass(slots=True)
class Heartbeat:
    ts_recv: int
    symbol: str | None = None
    last_trade_id: int | None = None


@dataclass(slots=True)
class ClockSample:
    """A server-time reading delivered on the websocket (Kraken ping/pong)."""

    ts_recv: int
    sent_ns: int
    server_time_ns: int


@dataclass(frozen=True, slots=True)
class ServerTime:
    """One REST server-time probe."""

    sent_ns: int
    server_time_ns: int
    ts_recv: int


@dataclass(slots=True)
class BridgeResult:
    """`bridged` carries the events to apply, `wait` needs more diffs, `stale` a new snapshot."""

    status: Literal["bridged", "wait", "stale"]
    applied: list[DepthEvent] = field(default_factory=list)
    remaining: list[DepthEvent] = field(default_factory=list)


@dataclass(slots=True)
class CheckResult:
    ok: bool
    reason: str | None = None
    expected: int | None = None
    computed: int | None = None


VenueEvent = DepthEvent | TradeEvent | BboEvent | SnapshotEvent | Heartbeat | ClockSample


@runtime_checkable
class VenueAdapter(Protocol):
    venue: str
    rest_snapshot: bool  # True when SNAPSHOT is a REST call, False when pushed on the websocket
    max_conn_age_h: float | None  # documented connection lifetime, None when there is none
    book_depth: int | None  # truncate the local book to this after each message
    keep_raw_strings: bool
    contiguous_trade_ids: bool
    checkpoint_kind: str | None  # REST book checkpoints, None when checksums cover the venue

    def connections(self, symbols: Sequence[str]) -> list[ConnectionSpec]: ...
    def subscribe_messages(self, conn: ConnectionSpec) -> list[bytes]: ...
    def resubscribe_messages(self, symbol: str) -> list[bytes]: ...
    async def fetch_snapshot(self, http: httpx.AsyncClient, symbol: str) -> Snapshot | None: ...
    async def fetch_checkpoint(self, http: httpx.AsyncClient, symbol: str) -> Snapshot | None: ...
    async def fetch_server_time(self, http: httpx.AsyncClient) -> ServerTime | None: ...
    def clock_ping(self, sent_ns: int) -> bytes | None: ...
    def parse_snapshot(self, raw: bytes, symbol: str, ts_recv: int) -> Snapshot: ...
    def parse(self, raw: bytes, ts_recv: int) -> list[VenueEvent]: ...
    def bridge(self, snapshot: Snapshot, buffered: Sequence[DepthEvent]) -> BridgeResult: ...
    def check(self, prev: DepthEvent | None, ev: DepthEvent, book: BookL2) -> CheckResult: ...


@lru_cache(maxsize=4096)
def _whole_seconds_ns(prefix: str) -> int:
    parsed = datetime.fromisoformat(prefix).replace(tzinfo=UTC)
    return int(parsed.timestamp()) * 1_000_000_000


def rfc3339_ns(text: str) -> int:
    """Parse `YYYY-MM-DDTHH:MM:SS[.fff...]Z` to integer ns without going through a float."""
    if len(text) < 20 or text[-1] != "Z":
        raise ParseError(f"not an RFC3339 UTC timestamp: {text!r}")
    base = _whole_seconds_ns(text[:19])
    if text[19] == "Z":
        return base
    digits = text[20:-1]
    return base + int(digits.ljust(9, "0")[:9])
