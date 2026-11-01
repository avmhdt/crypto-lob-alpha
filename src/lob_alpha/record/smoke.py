"""`lob record smoke`: connect, sync one symbol, write temp Parquet and report (M1)."""

from __future__ import annotations

import asyncio
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import websockets
from websockets.asyncio.client import ClientConnection

from lob_alpha.instruments import Instruments
from lob_alpha.record.sync import Action, NeedSnapshot, Resubscribe, VenuePipeline
from lob_alpha.record.venues.base import ConnectionSpec, SyncState, VenueAdapter
from lob_alpha.record.venues.registry import make_adapter
from lob_alpha.record.writer import ParquetSink
from lob_alpha.settings import RecorderConfig


@dataclass(slots=True)
class SmokeResult:
    venue: str
    symbol: str
    seconds: float
    frames: int
    reached_live: bool
    live_after_s: float | None
    rows_written: int
    feed_latency_ms_p50: float | None
    checksum_failures: int
    resyncs: int

    @property
    def ok(self) -> bool:
        return self.reached_live and self.rows_written > 0


async def _handle(
    actions: list[Action],
    adapter: VenueAdapter,
    pipeline: VenuePipeline,
    http: httpx.AsyncClient,
    sockets: dict[str, ClientConnection],
) -> None:
    for action in actions:
        if isinstance(action, NeedSnapshot):
            snapshot = await adapter.fetch_snapshot(http, action.symbol)
            if snapshot is not None:
                more = pipeline.ingest_control(
                    snapshot.ts_recv, {"k": "snapshot", "symbol": action.symbol}, snapshot.raw
                )
                await _handle(more, adapter, pipeline, http, sockets)
        elif isinstance(action, Resubscribe):
            for message in adapter.resubscribe_messages(action.symbol):
                await sockets[action.conn].send(message)


async def _reader(
    spec: ConnectionSpec,
    adapter: VenueAdapter,
    pipeline: VenuePipeline,
    http: httpx.AsyncClient,
    sockets: dict[str, ClientConnection],
    counter: list[int],
) -> None:
    conn_id = f"{spec.name}.g0"
    async with websockets.connect(
        spec.url, max_size=None, compression=None, proxy=None, open_timeout=15
    ) as socket:
        sockets[conn_id] = socket
        pipeline.ingest_control(
            time.time_ns(), {"k": "conn_open", "conn": conn_id, "logical": spec.name}
        )
        for message in adapter.subscribe_messages(spec):
            await socket.send(message)
        while True:
            raw = await socket.recv(decode=False)
            ts_recv = time.time_ns()
            counter[0] += 1
            actions = pipeline.ingest_frame(conn_id, ts_recv, raw)
            if actions:
                await _handle(actions, adapter, pipeline, http, sockets)


async def smoke(
    config: RecorderConfig, instruments: Instruments, venue: str, symbol: str, seconds: float
) -> SmokeResult:
    """Run one symbol for `seconds`. Rows land in a temp directory that is removed afterwards."""
    cfg = config.venues[venue].model_copy(update={"symbols": [symbol]})
    adapter = make_adapter(venue, cfg)
    tick = instruments.get(venue, symbol).tick_size
    counter = [0]
    with tempfile.TemporaryDirectory(prefix="lob-smoke-") as tmp:
        sink = ParquetSink(Path(tmp), config.writer, run="smoke")
        pipeline = VenuePipeline(adapter, sink, {symbol: tick})
        book = pipeline.books[symbol]
        sockets: dict[str, ClientConnection] = {}
        started = time.monotonic()
        live_after: float | None = None
        async with httpx.AsyncClient(timeout=15, trust_env=False) as http:
            tasks = [
                asyncio.create_task(_reader(spec, adapter, pipeline, http, sockets, counter))
                for spec in adapter.connections([symbol])
            ]
            try:
                while time.monotonic() - started < seconds:
                    await asyncio.sleep(0.1)
                    for task in tasks:
                        if task.done():
                            task.result()
                    if live_after is None and book.state is SyncState.LIVE:
                        live_after = time.monotonic() - started
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        sink.close()
        status = pipeline.status(time.time_ns())
        return SmokeResult(
            venue=venue,
            symbol=symbol,
            seconds=seconds,
            frames=counter[0],
            reached_live=live_after is not None,
            live_after_s=live_after,
            rows_written=sink.rows_written,
            feed_latency_ms_p50=status["feed_latency_ms_p50"],
            checksum_failures=book.resyncs if adapter.keep_raw_strings else 0,
            resyncs=book.resyncs,
        )
