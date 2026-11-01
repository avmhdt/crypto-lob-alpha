"""Sans-IO sync core: the per-stream state machine of Fig. 2 for one venue.

`VenuePipeline` does no I/O and reads no clock. Its only inputs are websocket frames and
control records, and every control record is written to the raw archive before it is
applied. Live recording and `lob record recover` therefore run the same code over the same
inputs, which is what makes recovered Parquet row-identical to the live files (FR-09).

Timers are the one thing that needs the wall clock: `timers(now)` turns elapsed time into
control records for the caller to archive and feed back in.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any, Protocol

from lob_alpha.book.l2 import ASK, BID, BookL2
from lob_alpha.record.venues.base import (
    BboEvent,
    ClockSample,
    ConnectionSpec,
    DepthEvent,
    Heartbeat,
    Level,
    ParseError,
    Snapshot,
    SnapshotEvent,
    SyncState,
    TradeEvent,
    VenueAdapter,
)

CLOCK_LIMIT_NS = 50_000_000
CLOCK_STRIKES = 5
OVERLAP_NS = 5_000_000_000
PROMOTION_TIMEOUT_NS = 120_000_000_000
BUFFER_LIMIT = 20_000
SEEN_TRADES = 20_000
CHECKPOINT_LEVELS_PER_SIDE = 1000

Log = Callable[..., None]


def _no_log(*_args: Any, **_kwargs: Any) -> None:
    return None


@dataclass(frozen=True, slots=True)
class NeedSnapshot:
    symbol: str


@dataclass(frozen=True, slots=True)
class Resubscribe:
    symbol: str
    conn: str


@dataclass(frozen=True, slots=True)
class CloseConn:
    conn: str
    reason: str


Action = NeedSnapshot | Resubscribe | CloseConn


@dataclass(frozen=True, slots=True)
class GapRow:
    gap_id: str
    venue: str
    symbol: str
    stream: str
    start_ns: int
    end_ns: int
    duration_ms: float
    reason: str
    seq_before: int | None
    seq_after: int | None
    host: str
    recorder_version: str


class Sink(Protocol):
    """Where normalised rows go. Implemented by the Parquet writer and by tests."""

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
        bids: Sequence[Level],
        asks: Sequence[Level],
    ) -> None: ...

    def trade(self, venue: str, ev: TradeEvent, batch_id: int) -> None: ...
    def bbo(self, venue: str, ev: BboEvent) -> None: ...

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
    ) -> None: ...

    def checkpoint_levels(
        self,
        venue: str,
        symbol: str,
        ts_recv: int,
        checkpoint_id: str,
        bids: Sequence[Level],
        asks: Sequence[Level],
    ) -> None: ...

    def clock(
        self, venue: str, ts_recv: int, server_time_ns: int, rtt_ns: int, offset_ns: int
    ) -> None: ...

    def gap(self, row: GapRow) -> None: ...


@dataclass(slots=True)
class OpenGap:
    start_ns: int
    reason: str
    seq_before: int | None


class _Context:
    """What every stream shares: identity, the sink and logging."""

    def __init__(
        self, adapter: VenueAdapter, sink: Sink, host: str, version: str, log: Log
    ) -> None:
        self.adapter = adapter
        self.venue = adapter.venue
        self.sink = sink
        self.host = host
        self.version = version
        self.log = log

    def emit_gap(
        self,
        symbol: str,
        stream: str,
        start_ns: int,
        end_ns: int,
        reason: str,
        seq_before: int | None,
        seq_after: int | None,
    ) -> None:
        key = f"{self.venue}|{symbol}|{stream}|{start_ns}|{reason}"
        row = GapRow(
            gap_id=hashlib.sha1(key.encode()).hexdigest()[:16],
            venue=self.venue,
            symbol=symbol,
            stream=stream,
            start_ns=start_ns,
            end_ns=end_ns,
            duration_ms=(end_ns - start_ns) / 1e6,
            reason=reason,
            seq_before=seq_before,
            seq_after=seq_after,
            host=self.host,
            recorder_version=self.version,
        )
        self.log(
            "gap", venue=self.venue, symbol=symbol, stream=stream, reason=reason,
            duration_ms=row.duration_ms,
        )  # fmt: skip
        self.sink.gap(row)


class _Stream:
    name = ""

    def __init__(self, ctx: _Context, symbol: str) -> None:
        self.ctx = ctx
        self.symbol = symbol
        self.state = SyncState.CONNECTING
        self.gap: OpenGap | None = None
        self.last_good_ns = 0
        self.last_seq: int | None = None
        self.msgs = 0

    def _to(self, state: SyncState, reason: str) -> None:
        if state is self.state:
            return
        self.ctx.log(
            "transition", venue=self.ctx.venue, symbol=self.symbol, stream=self.name,
            old=str(self.state), new=str(state), reason=reason,
        )  # fmt: skip
        self.state = state

    def open_gap(self, reason: str) -> None:
        if self.gap is None and self.last_good_ns:
            self.gap = OpenGap(self.last_good_ns, reason, self.last_seq)

    def close_gap(self, ts: int, seq_after: int | None) -> None:
        gap = self.gap
        if gap is None:
            return
        self.gap = None
        self.ctx.emit_gap(
            self.symbol, self.name, gap.start_ns, ts, gap.reason, gap.seq_before, seq_after
        )

    def flush_gap(self, ts: int) -> None:
        """Close an open gap at shutdown so the interval is on disk."""
        self.close_gap(ts, None)
        self._to(SyncState.STOPPED, "sigterm")


class BookStream(_Stream):
    """One L2 book stream: buffering, snapshot bridge, LIVE checks, gaps and resync."""

    name = "book"

    def __init__(self, ctx: _Context, symbol: str, tick_size: float) -> None:
        super().__init__(ctx, symbol)
        adapter = ctx.adapter
        self.book = BookL2(tick_size, keep_raw_strings=adapter.keep_raw_strings, strict=False)
        self.depth_limit = adapter.book_depth
        self.epoch = 0
        self.last_batch = 0
        self.prev: DepthEvent | None = None
        self.buffer: deque[DepthEvent] = deque(maxlen=BUFFER_LIMIT)
        self.pending: Snapshot | None = None
        self.snapshot_requested = False
        self.last_msg_ns = 0
        self.sync_since = 0
        self.overlap = False
        self.linked = False
        self.crossed = 0
        self.resyncs = 0

    def _batch(self, ts: int) -> int:
        self.last_batch = max(ts, self.last_batch + 1)
        return self.last_batch

    def _apply(self, ev: DepthEvent | Snapshot) -> None:
        book = self.book
        for px, qty, px_str, qty_str in ev.bids:
            book.apply(BID, px, qty, px_str, qty_str)
        for px, qty, px_str, qty_str in ev.asks:
            book.apply(ASK, px, qty, px_str, qty_str)
        if self.depth_limit is not None:
            book.truncate(self.depth_limit)

    def on_connected(self, ts: int) -> None:
        self.book.clear()
        self.prev = None
        self.buffer.clear()
        self.pending = None
        self.snapshot_requested = False
        self.sync_since = ts
        self.last_msg_ns = ts
        self._to(SyncState.SNAPSHOT, "subscribed")

    def on_conn_down(self, ts: int, reason: str) -> None:
        self.open_gap(reason)
        self.book.clear()
        self.prev = None
        self.buffer.clear()
        self.pending = None
        self.snapshot_requested = False
        self.overlap = False
        self.linked = False
        self._to(SyncState.CONNECTING, reason)

    def fail(self, ts: int, reason: str, ev: DepthEvent | None) -> list[Action]:
        """LIVE -> GAP -> RESYNC. Other streams and the process are untouched (FR-07)."""
        self._to(SyncState.GAP, reason)
        self.open_gap(reason)
        self.book.clear()
        self.prev = None
        self.pending = None
        self.buffer.clear()
        self.sync_since = ts
        self.last_msg_ns = ts
        self.resyncs += 1
        self._to(SyncState.RESYNC, reason)
        if self.ctx.adapter.rest_snapshot:
            if ev is not None:
                self.buffer.append(ev)
            self.snapshot_requested = True
            return [NeedSnapshot(self.symbol)]
        return [Resubscribe(self.symbol, "")]

    def _live(self, ev: DepthEvent, ts: int) -> list[Action]:
        self._apply(ev)
        result = self.ctx.adapter.check(self.prev, ev, self.book)
        batch = self._batch(ts)
        seq = ev.seq if ev.seq is not None else batch
        if result.expected is not None:
            self.ctx.sink.checkpoint(
                self.ctx.venue, self.symbol, str(batch), ts, "kraken_crc32", seq, batch,
                result.expected, result.computed, result.ok,
                self.book.depth(BID) + self.book.depth(ASK),
            )  # fmt: skip
        if not result.ok:
            return self.fail(ts, result.reason or "seq_gap", ev)
        self.ctx.sink.book_batch(
            self.ctx.venue, self.symbol, ev.ts_exchange, ts, seq, batch, False,
            ev.seq_first, ev.seq_prev, self.epoch, ev.bids, ev.asks,
        )  # fmt: skip
        self.prev = ev
        self.last_seq = seq
        self.last_good_ns = ts
        if self.book.is_crossed():
            self.crossed += 1
            self.ctx.log(
                "crossed_book", venue=self.ctx.venue, symbol=self.symbol, seq=seq, ts_recv=ts
            )
        return []

    def _standby_filter(self, ev: DepthEvent, primary: bool) -> bool:
        """Decide whether an event is applied during make-before-break. True means apply."""
        prev = self.prev
        if prev is None or prev.seq is None or ev.seq is None:
            return primary
        if ev.seq <= prev.seq:  # already applied from the other connection
            if not primary:
                self.linked = True
            return False
        if ev.seq_prev == prev.seq:
            if not primary:
                self.linked = True
            return True
        return primary  # a standby ahead of the chain waits; a primary here is a real gap

    def on_depth(self, ev: DepthEvent, primary: bool) -> list[Action]:
        self.msgs += 1
        self.last_msg_ns = ev.ts_recv
        if self.state is SyncState.LIVE:
            if self.overlap and not self._standby_filter(ev, primary):
                return []
            if not primary and not self.overlap:
                return []
            return self._live(ev, ev.ts_recv)
        if not primary or not self.ctx.adapter.rest_snapshot:
            return []
        self.buffer.append(ev)
        if self.pending is not None:
            return self._try_bridge()
        if not self.snapshot_requested:
            self.snapshot_requested = True
            return [NeedSnapshot(self.symbol)]
        return []

    def on_rest_snapshot(self, snapshot: Snapshot) -> list[Action]:
        self.pending = snapshot
        self.snapshot_requested = False
        self._to(SyncState.SYNCING, "snap_ok")
        return self._try_bridge()

    def on_snapshot_error(self, ts: int, error: str) -> None:
        self.snapshot_requested = False
        self.sync_since = ts
        self.open_gap("snapshot_error")
        self.ctx.log("snapshot_error", venue=self.ctx.venue, symbol=self.symbol, error=error)

    def _try_bridge(self) -> list[Action]:
        snapshot = self.pending
        if snapshot is None:
            return []
        result = self.ctx.adapter.bridge(snapshot, list(self.buffer))
        if result.status == "wait":
            self.buffer.clear()
            return []
        if result.status == "stale":
            self.pending = None
            self.open_gap("stale_snapshot")
            self._to(SyncState.RESYNC, "stale_snapshot")
            self.snapshot_requested = True
            return [NeedSnapshot(self.symbol)]
        self.buffer.clear()
        self.pending = None
        self.epoch += 1
        self.book.clear()
        self._apply(snapshot)
        ts = snapshot.ts_recv
        batch = self._batch(ts)
        seq = snapshot.seq if snapshot.seq is not None else batch
        self.ctx.sink.book_batch(
            self.ctx.venue, self.symbol, snapshot.ts_exchange, ts, seq, batch, True,
            None, None, self.epoch, snapshot.bids, snapshot.asks,
        )  # fmt: skip
        self.prev = None
        self._to(SyncState.LIVE, "bridged")
        actions: list[Action] = []
        for ev in result.applied:
            # A diff buffered before the snapshot arrived was not usable until now. Rows keep
            # the ts_recv order the replay rule needs, and never claim an earlier known-at time.
            ts = max(ev.ts_recv, snapshot.ts_recv)
            actions.extend(self._live(ev, ts))
            if self.state is not SyncState.LIVE:
                return actions
        self.close_gap(ts, self.last_seq)
        return actions

    def on_ws_snapshot(self, ev: SnapshotEvent, primary: bool) -> list[Action]:
        self.msgs += 1
        self.last_msg_ns = ev.ts_recv
        if not primary:
            return []
        self.book.clear()
        self._apply(ev)
        self._to(SyncState.SYNCING, "snap_ok")
        result = self.ctx.adapter.check(None, ev, self.book)
        batch = self._batch(ev.ts_recv)
        seq = batch
        if result.expected is not None:
            self.ctx.sink.checkpoint(
                self.ctx.venue, self.symbol, str(batch), ev.ts_recv, "kraken_crc32", seq, batch,
                result.expected, result.computed, result.ok,
                self.book.depth(BID) + self.book.depth(ASK),
            )  # fmt: skip
        if not result.ok:
            return self.fail(ev.ts_recv, result.reason or "checksum_fail", None)
        self.epoch += 1
        self.ctx.sink.book_batch(
            self.ctx.venue, self.symbol, ev.ts_exchange, ev.ts_recv, seq, batch, True,
            None, None, self.epoch, ev.bids, ev.asks,
        )  # fmt: skip
        self.prev = ev
        self.last_seq = seq
        self.last_good_ns = ev.ts_recv
        self._to(SyncState.LIVE, "bridged")
        self.close_gap(ev.ts_recv, seq)
        return []

    def on_checkpoint(self, kind: str, snapshot: Snapshot) -> None:
        checkpoint_id = f"{kind}-{snapshot.ts_recv}"
        n_levels = len(snapshot.bids) + len(snapshot.asks)
        self.ctx.sink.checkpoint(
            self.ctx.venue, self.symbol, checkpoint_id, snapshot.ts_recv, kind,
            snapshot.seq or 0, self.last_batch, None, None, self.state is SyncState.LIVE, n_levels,
        )  # fmt: skip
        self.ctx.sink.checkpoint_levels(
            self.ctx.venue, self.symbol, snapshot.ts_recv, checkpoint_id,
            snapshot.bids[:CHECKPOINT_LEVELS_PER_SIDE], snapshot.asks[:CHECKPOINT_LEVELS_PER_SIDE],
        )  # fmt: skip

    def dump(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "state": str(self.state),
            "epoch": self.epoch,
            "last_batch": self.last_batch,
            "last_seq": self.last_seq,
            "last_good_ns": self.last_good_ns,
            "prev_seq": self.prev.seq if self.prev else None,
        }
        if self.ctx.adapter.keep_raw_strings:  # the checksum needs the book itself
            state["bids"] = self.book.raw_levels(BID, self.book.depth(BID))
            state["asks"] = self.book.raw_levels(ASK, self.book.depth(ASK))
        return state

    def load(self, state: dict[str, Any]) -> None:
        self.state = SyncState(state["state"])
        self.epoch = state["epoch"]
        self.last_batch = state["last_batch"]
        self.last_seq = state["last_seq"]
        self.last_good_ns = state["last_good_ns"]
        self.prev = None
        if state["prev_seq"] is not None or self.state is SyncState.LIVE:
            self.prev = DepthEvent(self.symbol, 0, 0, [], [], seq=state["prev_seq"])
        self.book.clear()
        for side, key in ((BID, "bids"), (ASK, "asks")):
            for px, qty in state.get(key, ()):
                self.book.apply(side, float(px), float(qty), px, qty)


class FlowStream(_Stream):
    """Trades or bbo. No snapshot: LIVE as soon as its connection delivers frames."""

    def __init__(self, ctx: _Context, symbol: str, name: str) -> None:
        super().__init__(ctx, symbol)
        self.name = name
        self.last_batch = 0
        self.last_id: int | None = None
        self.seen: dict[str, None] = {}
        self.duplicates = 0

    def on_frame(self, ts: int) -> None:
        if self.state is not SyncState.LIVE:
            self._to(SyncState.LIVE, "first_frame")
            self.close_gap(ts, None)

    def on_conn_down(self, ts: int, reason: str, last_frame_ns: int) -> None:
        self.last_good_ns = max(self.last_good_ns, last_frame_ns)
        self.open_gap(reason)
        self.last_id = None
        self._to(SyncState.CONNECTING, reason)

    def on_trade(self, ev: TradeEvent, overlap: bool) -> None:
        self.msgs += 1
        trade_id = ev.trade_id
        if trade_id in self.seen:
            self.duplicates += 1
            return
        self.seen[trade_id] = None
        if len(self.seen) > SEEN_TRADES:
            del self.seen[next(iter(self.seen))]
        adapter = self.ctx.adapter
        if adapter.contiguous_trade_ids:
            number = int(trade_id)
            if not overlap and self.last_id is not None and number > self.last_id + 1:
                self.ctx.emit_gap(
                    self.symbol, "trades", self.last_good_ns or ev.ts_recv, ev.ts_recv, "seq_gap",
                    self.last_id, number,
                )  # fmt: skip
            self.last_id = max(number, self.last_id or 0)
            self.last_seq = self.last_id
        self.last_batch = max(ev.ts_recv, self.last_batch + 1)
        self.last_good_ns = ev.ts_recv
        self.ctx.sink.trade(self.ctx.venue, ev, self.last_batch)

    def on_bbo(self, ev: BboEvent, overlap: bool) -> None:
        self.msgs += 1
        if overlap and ev.seq is not None and self.last_seq and ev.seq <= self.last_seq:
            return
        self.last_seq = ev.seq
        self.last_good_ns = ev.ts_recv
        self.ctx.sink.bbo(self.ctx.venue, ev)

    def dump(self) -> dict[str, Any]:
        return {
            "state": str(self.state),
            "last_batch": self.last_batch,
            "last_id": self.last_id,
            "last_seq": self.last_seq,
            "last_good_ns": self.last_good_ns,
            "seen": list(self.seen)[-1000:],
        }

    def load(self, state: dict[str, Any]) -> None:
        self.state = SyncState(state["state"])
        self.last_batch = state["last_batch"]
        self.last_id = state["last_id"]
        self.last_seq = state["last_seq"]
        self.last_good_ns = state["last_good_ns"]
        self.seen = dict.fromkeys(state["seen"])


@dataclass(slots=True)
class _Conn:
    conn_id: str
    logical: str
    opened_ns: int
    last_frame_ns: int
    last_ts: int = 0


class VenuePipeline:
    """Every stream of one venue, driven by frames and control records."""

    def __init__(
        self,
        adapter: VenueAdapter,
        sink: Sink,
        ticks: Mapping[str, float],
        *,
        host: str = "",
        version: str = "",
        stale_after_s: float = 30.0,
        log: Log = _no_log,
    ) -> None:
        self.adapter = adapter
        self.venue = adapter.venue
        self.symbols = list(ticks)
        self.specs = {c.name: c for c in adapter.connections(self.symbols)}
        self.ctx = _Context(adapter, sink, host, version, log)
        self.log = log
        self.stale_after_ns = int(stale_after_s * 1e9)
        self.books: dict[str, BookStream] = {}
        self.trades: dict[str, FlowStream] = {}
        self.bbos: dict[str, FlowStream] = {}
        for spec in self.specs.values():
            for symbol in spec.symbols:
                if spec.book:
                    self.books[symbol] = BookStream(self.ctx, symbol, ticks[symbol])
                if spec.trades:
                    self.trades[symbol] = FlowStream(self.ctx, symbol, "trades")
                if spec.bbo:
                    self.bbos[symbol] = FlowStream(self.ctx, symbol, "bbo")
        self.conns: dict[str, _Conn] = {}
        self.active: dict[str, str] = {}
        self.standby: dict[str, str] = {}
        self.parse_errors = 0
        self.ts_regressions = 0
        self.latency_ns: deque[int] = deque(maxlen=4096)
        self.clock_offset_ns: int | None = None
        self.clock_strikes = 0
        self.clock_bad_since = 0

    def streams(self) -> list[_Stream]:
        return [*self.books.values(), *self.trades.values(), *self.bbos.values()]

    def _logical_streams(self, logical: str) -> tuple[list[BookStream], list[FlowStream]]:
        spec = self.specs[logical]
        books = [self.books[s] for s in spec.symbols] if spec.book else []
        flows: list[FlowStream] = []
        if spec.trades:
            flows += [self.trades[s] for s in spec.symbols]
        if spec.bbo:
            flows += [self.bbos[s] for s in spec.symbols]
        return books, flows

    # ---- frames ---------------------------------------------------------------------------

    def ingest_frame(self, conn_id: str, ts_recv: int, raw: bytes) -> list[Action]:
        """Process one websocket frame. Unknown connections are ignored."""
        conn = self.conns.get(conn_id)
        if conn is None:
            return []
        if ts_recv < conn.last_ts:
            self.ts_regressions += 1
        conn.last_ts = ts_recv
        conn.last_frame_ns = ts_recv
        primary = self.active.get(conn.logical) == conn_id
        overlap = conn.logical in self.standby
        if primary:
            _, flows = self._logical_streams(conn.logical)
            for flow in flows:
                if flow.state is not SyncState.LIVE:
                    flow.on_frame(ts_recv)
        try:
            events = self.adapter.parse(raw, ts_recv)
        except ValueError as exc:
            return self._on_parse_error(conn, exc, raw)
        actions: list[Action] = []
        for ev in events:
            if isinstance(ev, SnapshotEvent):
                stream = self.books.get(ev.symbol)
                if stream is not None:
                    actions.extend(stream.on_ws_snapshot(ev, primary))
            elif isinstance(ev, DepthEvent):
                stream = self.books.get(ev.symbol)
                if stream is not None:
                    self.latency_ns.append(ts_recv - ev.ts_exchange)
                    actions.extend(self._rewrite(conn, stream.on_depth(ev, primary)))
            elif isinstance(ev, TradeEvent):
                flow_stream = self.trades.get(ev.symbol)
                if flow_stream is not None:
                    flow_stream.on_trade(ev, overlap)
            elif isinstance(ev, BboEvent):
                flow_stream = self.bbos.get(ev.symbol)
                if flow_stream is not None:
                    flow_stream.on_bbo(ev, overlap)
            elif isinstance(ev, ClockSample):
                self._on_clock(ev.ts_recv, ev.server_time_ns, ev.sent_ns)
            elif isinstance(ev, Heartbeat):
                pass
        return actions

    def _rewrite(self, conn: _Conn, actions: list[Action]) -> list[Action]:
        return [
            Resubscribe(a.symbol, conn.conn_id) if isinstance(a, Resubscribe) else a
            for a in actions
        ]

    def _on_parse_error(self, conn: _Conn, exc: ValueError, raw: bytes) -> list[Action]:
        """A shape we do not understand on a data message: resync, never guess (R-03)."""
        self.parse_errors += 1
        self.log("parse_error", venue=self.venue, conn=conn.conn_id, error=str(exc)[:300])
        if not isinstance(exc, ParseError):
            return []
        return [CloseConn(conn.conn_id, "snapshot_error")]

    # ---- control records ------------------------------------------------------------------

    def ingest_control(self, ts: int, ctrl: dict[str, Any], body: bytes = b"") -> list[Action]:
        """Apply one control record. Every record here is also in the raw archive."""
        kind = ctrl["k"]
        if kind == "conn_open":
            return self._conn_open(ts, ctrl["conn"], ctrl["logical"], bool(ctrl.get("standby")))
        if kind == "conn_close":
            return self._conn_close(ts, ctrl["conn"], ctrl["reason"])
        if kind == "snapshot":
            return self.books[ctrl["symbol"]].on_rest_snapshot(
                self.adapter.parse_snapshot(body, ctrl["symbol"], ts)
            )
        if kind == "checkpoint":
            self.books[ctrl["symbol"]].on_checkpoint(
                ctrl["checkpoint"], self.adapter.parse_snapshot(body, ctrl["symbol"], ts)
            )
            return []
        if kind == "snapshot_error":
            self.books[ctrl["symbol"]].on_snapshot_error(ts, ctrl["error"])
            return []
        if kind == "stale":
            stream = self.books[ctrl["symbol"]]
            return self._rewrite_book(stream, stream.fail(ts, "stale", None))
        if kind == "promote":
            self._promote(ts, ctrl["logical"], ctrl["conn"])
            return []
        if kind == "clock":
            self._on_clock(ts, ctrl["server_time_ns"], ctrl["sent_ns"])
            return []
        if kind == "run_start":
            self.seed(ctrl.get("streams", {}), ctrl.get("reason", "process_restart"))
            return []
        if kind == "state":
            self.load(ctrl["state"])
            return []
        raise ValueError(f"unknown control record {kind!r}")

    def _rewrite_book(self, stream: BookStream, actions: list[Action]) -> list[Action]:
        logical = next((n for n, s in self.specs.items() if stream.symbol in s.symbols and s.book))
        conn_id = self.active.get(logical, "")
        return [
            Resubscribe(a.symbol, conn_id) if isinstance(a, Resubscribe) else a for a in actions
        ]

    def _conn_open(self, ts: int, conn_id: str, logical: str, standby: bool) -> list[Action]:
        self.conns[conn_id] = _Conn(conn_id, logical, ts, ts)
        books, _ = self._logical_streams(logical)
        if standby:
            self.standby[logical] = conn_id
            for book in books:
                book.overlap = True
                book.linked = False
            return []
        self.active[logical] = conn_id
        for book in books:
            book.on_connected(ts)
        return []

    def _conn_close(self, ts: int, conn_id: str, reason: str) -> list[Action]:
        conn = self.conns.pop(conn_id, None)
        if conn is None:
            return []
        logical = conn.logical
        books, flows = self._logical_streams(logical)
        if self.standby.get(logical) == conn_id:
            del self.standby[logical]
            for book in books:
                book.overlap = False
                book.linked = False
            return []
        if self.active.get(logical) != conn_id:
            return []
        replacement = self.standby.get(logical)
        if replacement is not None and books and all(b.linked for b in books):
            self._promote(ts, logical, replacement)
            return []
        del self.active[logical]
        for book in books:
            book.on_conn_down(ts, reason)
        for flow in flows:
            flow.on_conn_down(ts, reason, conn.last_frame_ns)
        if replacement is not None:
            del self.standby[logical]
            self.active[logical] = replacement
            for book in books:
                book.on_connected(ts)
        return []

    def _promote(self, ts: int, logical: str, conn_id: str) -> None:
        old = self.active.get(logical)
        self.active[logical] = conn_id
        self.standby.pop(logical, None)
        if old is not None:
            self.conns.pop(old, None)
        books, _ = self._logical_streams(logical)
        for book in books:
            book.overlap = False
            book.linked = False
        self.log("promoted", venue=self.venue, logical=logical, conn=conn_id, old=old, ts=ts)

    def _on_clock(self, ts: int, server_time_ns: int, sent_ns: int) -> None:
        rtt = ts - sent_ns
        offset = server_time_ns - (sent_ns + rtt // 2)
        self.clock_offset_ns = offset
        self.ctx.sink.clock(self.venue, ts, server_time_ns, rtt, offset)
        if abs(offset) > CLOCK_LIMIT_NS:
            self.clock_strikes += 1
            if self.clock_strikes == 1:
                self.clock_bad_since = ts
            return
        if self.clock_strikes >= CLOCK_STRIKES:
            self.ctx.emit_gap("*", "book", self.clock_bad_since, ts, "clock", None, None)
        self.clock_strikes = 0

    # ---- timers ---------------------------------------------------------------------------

    def timers(self, now: int) -> tuple[list[dict[str, Any]], list[Action]]:
        """Turn elapsed time into control records to archive and actions to run."""
        controls: list[dict[str, Any]] = []
        actions: list[Action] = []
        stale = self.stale_after_ns
        for conn in self.conns.values():
            if now - conn.last_frame_ns > stale:
                actions.append(CloseConn(conn.conn_id, "stale"))
        if actions:
            return controls, actions
        for logical, conn_id in self.active.items():
            for book in self._logical_streams(logical)[0]:
                if book.state is SyncState.LIVE:
                    if now - book.last_msg_ns > stale:
                        controls.append({"k": "stale", "symbol": book.symbol})
                elif now - book.sync_since > stale:
                    book.sync_since = now
                    if self.adapter.rest_snapshot:
                        book.snapshot_requested = True
                        actions.append(NeedSnapshot(book.symbol))
                    else:
                        actions.append(CloseConn(conn_id, "stale"))
        for logical, conn_id in self.standby.items():
            books, _ = self._logical_streams(logical)
            age = now - self.conns[conn_id].opened_ns
            if age >= OVERLAP_NS and all(b.linked for b in books):
                controls.append({"k": "promote", "logical": logical, "conn": conn_id})
            elif age > PROMOTION_TIMEOUT_NS:
                actions.append(CloseConn(conn_id, "standby_abandoned"))
        return controls, actions

    # ---- state ----------------------------------------------------------------------------

    def seed(self, previous: dict[str, Any], reason: str) -> None:
        """Start from the previous run: epochs continue and the downtime becomes a gap."""
        for stream in self.streams():
            saved = previous.get(f"{stream.symbol}|{stream.name}")
            if saved is None:
                continue
            if isinstance(stream, BookStream):
                stream.epoch = saved.get("epoch", 0)
            stream.last_good_ns = saved.get("last_good_ns", 0)
            stream.last_seq = saved.get("last_seq")
            stream.open_gap(reason)

    def dump(self) -> dict[str, Any]:
        """Stream state needed to replay an hour of raw frames on its own."""
        return {
            "books": {s: b.dump() for s, b in self.books.items()},
            "trades": {s: t.dump() for s, t in self.trades.items()},
            "bbos": {s: b.dump() for s, b in self.bbos.items()},
            "conns": {
                c.conn_id: {"logical": c.logical, "opened": c.opened_ns, "last": c.last_frame_ns}
                for c in self.conns.values()
            },
            "active": self.active,
            "standby": self.standby,
        }

    def load(self, state: dict[str, Any]) -> None:
        """Restore `dump` output at the top of a replayed hour."""
        for symbol, saved in state["books"].items():
            self.books[symbol].load(saved)
        for symbol, saved in state["trades"].items():
            self.trades[symbol].load(saved)
        for symbol, saved in state["bbos"].items():
            self.bbos[symbol].load(saved)
        self.conns = {
            conn_id: _Conn(conn_id, c["logical"], c["opened"], c["last"])
            for conn_id, c in state["conns"].items()
        }
        self.active = dict(state["active"])
        self.standby = dict(state["standby"])
        for logical in self.standby:
            for book in self._logical_streams(logical)[0]:
                book.overlap = True

    def status(self, now: int) -> dict[str, Any]:
        """Snapshot for `status.json`."""
        rows: list[dict[str, Any]] = []
        for stream in self.streams():
            row: dict[str, Any] = {
                "symbol": stream.symbol,
                "stream": stream.name,
                "state": str(stream.state),
                "last_good_ns": stream.last_good_ns,
                "last_seq": stream.last_seq,
                "age_s": round((now - stream.last_good_ns) / 1e9, 1)
                if stream.last_good_ns
                else None,
                "msgs": stream.msgs,
            }
            if isinstance(stream, BookStream):
                row.update(epoch=stream.epoch, crossed=stream.crossed, resyncs=stream.resyncs)
            if isinstance(stream, FlowStream):
                row.update(duplicates=stream.duplicates)
            rows.append(row)
        return {
            "streams": rows,
            "parse_errors": self.parse_errors,
            "ts_regressions": self.ts_regressions,
            "feed_latency_ms_p50": round(median(self.latency_ns) / 1e6, 2)
            if self.latency_ns
            else None,
            "clock_offset_ms": round(self.clock_offset_ns / 1e6, 2)
            if self.clock_offset_ns is not None
            else None,
        }


__all__ = [
    "Action",
    "BookStream",
    "CloseConn",
    "ConnectionSpec",
    "FlowStream",
    "GapRow",
    "NeedSnapshot",
    "Resubscribe",
    "Sink",
    "VenuePipeline",
]
