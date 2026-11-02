"""Local stand-ins for the exchanges: a websocket server and a tiny HTTP server.

Both venues generate a deterministic stream and keep the true book, so tests can inject
faults and then compare what the recorder wrote with what really happened.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable
from pathlib import Path

import websockets
from websockets.asyncio.server import ServerConnection

from lob_alpha.book.checksum import kraken_checksum
from lob_alpha.instruments import Instrument, Instruments
from lob_alpha.settings import HealthConfig, RecorderConfig, VenueConfig, WriterConfig

SYMBOL = "BTCUSDT"
TICK = 0.1


def _px(ticks: int) -> str:
    return f"{ticks * TICK:.1f}"


class FakeHttp:
    """Minimal HTTP/1.1 server answering `routes[path](query)` with JSON bytes."""

    def __init__(self) -> None:
        self.routes: dict[str, Callable[[], bytes]] = {}
        self.status = 200
        self.server: asyncio.Server | None = None
        self.port = 0
        self.hits: dict[str, int] = {}

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                request = await reader.readuntil(b"\r\n\r\n")
                path = request.split(b" ")[1].decode().split("?")[0]
                self.hits[path] = self.hits.get(path, 0) + 1
                body = self.routes[path]()
                head = f"HTTP/1.1 {self.status} X\r\nContent-Length: {len(body)}\r\n\r\n"
                writer.write(head.encode() + body)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()


class FakeBinance:
    """Depth, bookTicker and aggTrade for one symbol, with fault switches."""

    def __init__(self, seed: int = 1, interval_s: float = 0.01) -> None:
        self.rng = random.Random(seed)
        self.interval_s = interval_s
        self.http = FakeHttp()
        self.ws: websockets.Server | None = None
        self.port = 0
        self.public: set[ServerConnection] = set()
        self.ticker: set[ServerConnection] = set()
        self.market: set[ServerConnection] = set()
        self.bids: dict[int, str] = {1000 - i: "1.000" for i in range(1, 21)}
        self.asks: dict[int, str] = {1000 + i: "1.000" for i in range(1, 21)}
        self.update_id = 100
        self.trade_id = 500
        self.silent = False
        self.skip_next = False
        self.stale_snapshots = 0
        self.task: asyncio.Task[None] | None = None
        self.history: list[tuple[int, dict[int, str], dict[int, str]]] = []
        self.trades_sent: list[int] = []
        self.http.routes["/fapi/v1/depth"] = self._depth
        self.http.routes["/fapi/v1/time"] = lambda: json.dumps(
            {"serverTime": time.time_ns() // 1_000_000}
        ).encode()

    def _depth(self) -> bytes:
        update_id = self.update_id
        if self.stale_snapshots > 0:
            self.stale_snapshots -= 1
            update_id = self.update_id - 10_000
        return json.dumps(
            {
                "lastUpdateId": update_id,
                "T": time.time_ns() // 1_000_000,
                "E": time.time_ns() // 1_000_000,
                "bids": [[_px(t), q] for t, q in sorted(self.bids.items(), reverse=True)],
                "asks": [[_px(t), q] for t, q in sorted(self.asks.items())],
            }
        ).encode()

    async def _handler(self, connection: ServerConnection) -> None:
        path = connection.request.path
        if "bookTicker" in path:
            group = self.ticker
        elif path.startswith("/public"):
            group = self.public
        else:
            group = self.market
        group.add(connection)
        try:
            await connection.wait_closed()
        finally:
            group.discard(connection)

    def _mutate(self) -> tuple[list[list[str]], list[list[str]]]:
        book, sign = (self.bids, -1) if self.rng.random() < 0.5 else (self.asks, 1)
        ticks = 1000 + sign * self.rng.randint(1, 25)
        qty = "0.000" if self.rng.random() < 0.3 else f"{self.rng.randint(1, 9)}.000"
        if qty == "0.000":
            book.pop(ticks, None)
        else:
            book[ticks] = qty
        change = [[_px(ticks), qty]]
        return (change, []) if sign == -1 else ([], change)

    async def _send(self, group: set[ServerConnection], payload: str) -> None:
        for connection in list(group):
            try:
                await connection.send(payload)
            except websockets.ConnectionClosed:
                group.discard(connection)

    def _frame(self, stream: str, data: dict[str, object]) -> str:
        return json.dumps({"stream": stream, "data": data})

    async def _pump(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            now_ms = time.time_ns() // 1_000_000
            bids, asks = self._mutate()
            first = self.update_id + 1
            previous = self.update_id
            self.update_id += 5
            self.history.append((self.update_id, dict(self.bids), dict(self.asks)))
            if self.silent:
                continue
            if self.skip_next:
                self.skip_next = False
                continue
            depth = {
                "e": "depthUpdate", "E": now_ms, "T": now_ms, "s": SYMBOL, "U": first,
                "u": self.update_id, "pu": previous, "b": bids, "a": asks,
            }  # fmt: skip
            await self._send(self.public, self._frame("btcusdt@depth@100ms", depth))
            ticker = {
                "e": "bookTicker", "T": now_ms, "u": self.update_id, "s": SYMBOL,
                "b": _px(max(self.bids)), "B": "1", "a": _px(min(self.asks)), "A": "1",
            }  # fmt: skip
            await self._send(self.ticker, self._frame("btcusdt@bookTicker", ticker))
            self.trade_id += 1
            self.trades_sent.append(self.trade_id)
            trade = {
                "e": "aggTrade", "T": now_ms, "a": self.trade_id, "s": SYMBOL,
                "p": _px(1000), "q": "0.001", "m": self.trade_id % 2 == 0,
            }  # fmt: skip
            await self._send(self.market, self._frame("btcusdt@aggTrade", trade))

    async def drop_public(self) -> None:
        for connection in list(self.public):
            await connection.close()

    async def drop_ticker(self) -> None:
        for connection in list(self.ticker):
            await connection.close()

    async def start(self) -> None:
        await self.http.start()
        self.ws = await websockets.serve(self._handler, "127.0.0.1", 0)
        self.port = self.ws.sockets[0].getsockname()[1]
        self.task = asyncio.create_task(self._pump())

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
        if self.ws is not None:
            self.ws.close()
        await self.http.stop()

    def config(self) -> VenueConfig:
        return VenueConfig(
            ws_public=f"ws://127.0.0.1:{self.port}/public/stream",
            ws_market=f"ws://127.0.0.1:{self.port}/market/stream",
            rest=f"http://127.0.0.1:{self.http.port}",
            symbols=[SYMBOL],
            streams={"depth": "{s}@depth", "bbo": "{s}@bookTicker", "trades": "{s}@aggTrade"},
            checkpoint_interval_s=0.3,
        )


class FakeKraken:
    """Book with real CRC32 checksums, ping/pong and trades."""

    def __init__(self, seed: int = 2, interval_s: float = 0.01) -> None:
        self.rng = random.Random(seed)
        self.interval_s = interval_s
        self.ws: websockets.Server | None = None
        self.port = 0
        self.bids = {1000 - i: f"{i}.00000000" for i in range(1, 16)}
        self.asks = {1000 + i: f"{i}.00000000" for i in range(1, 16)}
        self.subscribed: set[ServerConnection] = set()
        self.corrupt_next = False
        self.task: asyncio.Task[None] | None = None
        self.trade_id = 0

    def _checksum(self) -> int:
        bids = [(_px(t), self.bids[t]) for t in sorted(self.bids, reverse=True)]
        asks = [(_px(t), self.asks[t]) for t in sorted(self.asks)]
        return kraken_checksum(bids, asks)

    def _levels(self, book: dict[int, str], reverse: bool) -> str:
        rows = [f'{{"price":{_px(t)},"qty":{book[t]}}}' for t in sorted(book, reverse=reverse)]
        return "[" + ",".join(rows) + "]"

    def _book_message(self, kind: str, bids: str, asks: str, checksum: int) -> str:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())
        return (
            f'{{"channel":"book","type":"{kind}","data":[{{"symbol":"BTC/USD","bids":{bids},'
            f'"asks":{asks},"checksum":{checksum},"timestamp":"{stamp}"}}]}}'
        )

    async def _handler(self, connection: ServerConnection) -> None:
        try:
            async for message in connection:
                request = json.loads(message)
                method = request.get("method")
                if method == "ping":
                    now = time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())
                    await connection.send(
                        json.dumps({"method": "pong", "req_id": request["req_id"], "time_in": now})
                    )
                    continue
                params = request["params"]
                ack = {"method": method, "result": params, "success": True}
                await connection.send(json.dumps(ack))
                if params["channel"] != "book":
                    continue
                if method == "unsubscribe":
                    self.subscribed.discard(connection)
                else:
                    snapshot = self._book_message(
                        "snapshot",
                        self._levels(self.bids, True),
                        self._levels(self.asks, False),
                        self._checksum(),
                    )
                    await connection.send(snapshot)
                    self.subscribed.add(connection)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.subscribed.discard(connection)

    async def _pump(self) -> None:
        while True:
            await asyncio.sleep(self.interval_s)
            ticks = 1000 - self.rng.randint(1, 15)
            qty = f"{self.rng.randint(1, 9)}.{self.rng.randint(0, 99):02d}000000"
            self.bids[ticks] = qty
            row = f'[{{"price":{_px(ticks)},"qty":{qty}}}]'
            checksum = self._checksum()
            if self.corrupt_next:
                self.corrupt_next = False
                checksum += 1
            message = self._book_message("update", row, "[]", checksum)
            self.trade_id += 1
            trade = {
                "channel": "trade",
                "type": "update",
                "data": [
                    {
                        "symbol": "BTC/USD", "side": "buy", "price": 100.0, "qty": 1.0,
                        "ord_type": "limit", "trade_id": self.trade_id,
                        "timestamp": "2026-11-01T21:02:53.334453Z",
                    }
                ],
            }  # fmt: skip
            for connection in list(self.subscribed):
                try:
                    await connection.send(message)
                    await connection.send(json.dumps(trade))
                except websockets.ConnectionClosed:
                    self.subscribed.discard(connection)

    async def start(self) -> None:
        self.ws = await websockets.serve(self._handler, "127.0.0.1", 0)
        self.port = self.ws.sockets[0].getsockname()[1]
        self.task = asyncio.create_task(self._pump())

    async def stop(self) -> None:
        if self.task is not None:
            self.task.cancel()
        if self.ws is not None:
            self.ws.close()

    def config(self) -> VenueConfig:
        return VenueConfig(
            ws=f"ws://127.0.0.1:{self.port}",
            rest="http://127.0.0.1:1",
            symbols=["BTC/USD"],
            book_depth=10,
        )


def instruments() -> Instruments:
    common = {
        "lot_size": 0.001, "contract_type": "perp", "contract_value": 1.0, "maker_fee_bps": 1.0,
        "taker_fee_bps": 1.0, "effective_from": "2026-01-01", "source_url": "u",
        "fee_source_url": "f", "fee_read_on": "2026-11-01",
    }  # fmt: skip
    return Instruments(
        [
            Instrument.model_validate(
                {"venue": "binance-usdm", "symbol": SYMBOL, "tick_size": TICK, **common}
            ),
            Instrument.model_validate(
                {"venue": "kraken", "symbol": "BTC/USD", "tick_size": TICK, **common}
            ),
        ]
    )


def recorder_config(venues: dict[str, VenueConfig], **health: float) -> RecorderConfig:
    settings = {
        "stale_after_s": 0.5, "backoff_initial_s": 0.05, "backoff_max_s": 0.2,
        "proactive_reconnect_h": 23.0, "clock_probe_interval_s": 0.2, "status_interval_s": 0.1,
        "min_free_gb": 0.0, "hard_stop_free_gb": -1.0,
    }  # fmt: skip
    return RecorderConfig(
        version=1,
        venue_set=list(venues),
        writer=WriterConfig(flush_interval_s=0.2),
        health=HealthConfig(**{**settings, **health}),
        venues=venues,
    )


def parquet_rows(root: Path, dataset: str) -> list[dict[str, object]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, object]] = []
    for path in sorted((root / dataset).rglob("*.parquet")):
        rows += pq.read_table(path, partitioning=None).to_pylist()
    return rows
