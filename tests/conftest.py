"""Shared fixtures and the hypothesis `ci` profile."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from hypothesis import settings

from lob_alpha.record.sync import GapRow
from lob_alpha.record.venues.base import BboEvent, Level, TradeEvent
from lob_alpha.settings import VenueConfig

settings.register_profile("ci", max_examples=200, deadline=None)
settings.register_profile("dev", max_examples=50, deadline=None)
settings.load_profile("ci" if os.environ.get("CI") else "dev")

FIXTURES = Path(__file__).parent / "fixtures"


def frames(name: str) -> list[bytes]:
    """Raw lines of a committed sample-message file."""
    return (FIXTURES / name).read_bytes().splitlines()


@dataclass(slots=True)
class Batch:
    ts_recv: int
    seq: int
    batch_id: int
    is_snapshot: bool
    epoch: int
    bids: list[Level]
    asks: list[Level]


@dataclass(slots=True)
class MemorySink:
    """Collects every row a pipeline emits."""

    batches: list[Batch] = field(default_factory=list)
    trades: list[tuple[TradeEvent, int]] = field(default_factory=list)
    bbos: list[BboEvent] = field(default_factory=list)
    checkpoints: list[tuple[str, bool, int | None, int | None]] = field(default_factory=list)
    levels: list[str] = field(default_factory=list)
    clocks: list[tuple[int, int]] = field(default_factory=list)
    gaps: list[GapRow] = field(default_factory=list)

    def book_batch(
        self,
        venue: str,
        symbol: str,
        ts_exchange: int,
        ts_recv: int,
        seq: int,
        batch_id: int,
        is_snapshot: bool,
        seq_first: int | None,
        seq_prev: int | None,
        sync_epoch: int,
        bids: list[Level],
        asks: list[Level],
    ) -> None:
        self.batches.append(
            Batch(ts_recv, seq, batch_id, is_snapshot, sync_epoch, list(bids), list(asks))
        )

    def trade(self, venue: str, ev: TradeEvent, batch_id: int) -> None:
        self.trades.append((ev, batch_id))

    def bbo(self, venue: str, ev: BboEvent) -> None:
        self.bbos.append(ev)

    def checkpoint(
        self,
        venue: str,
        symbol: str,
        checkpoint_id: str,
        ts_recv: int,
        kind: str,
        seq: int,
        batch_id: int,
        expected: int | None,
        computed: int | None,
        ok: bool,
        n_levels: int,
    ) -> None:
        self.checkpoints.append((kind, ok, expected, computed))

    def checkpoint_levels(
        self,
        venue: str,
        symbol: str,
        ts_recv: int,
        checkpoint_id: str,
        bids: list[Level],
        asks: list[Level],
    ) -> None:
        self.levels.append(checkpoint_id)

    def clock(
        self, venue: str, ts_recv: int, server_time_ns: int, rtt_ns: int, offset_ns: int
    ) -> None:
        self.clocks.append((rtt_ns, offset_ns))

    def gap(self, row: GapRow) -> None:
        self.gaps.append(row)


@pytest.fixture
def sink() -> MemorySink:
    return MemorySink()


def binance_cfg() -> VenueConfig:
    return VenueConfig(
        ws_public="wss://x/public/stream",
        ws_market="wss://x/market/stream",
        rest="https://x",
        symbols=["BTCUSDT", "ETHUSDT"],
        streams={"depth": "{s}@depth@100ms", "bbo": "{s}@bookTicker", "trades": "{s}@aggTrade"},
    )


def kraken_cfg() -> VenueConfig:
    return VenueConfig(ws="wss://x", rest="https://x", symbols=["BTC/USD"], book_depth=10)


def coinbase_cfg() -> VenueConfig:
    return VenueConfig(
        ws="wss://x",
        rest="https://x",
        symbols=["BTC-USD"],
        channels=["level2_batch", "matches", "heartbeat"],
    )
