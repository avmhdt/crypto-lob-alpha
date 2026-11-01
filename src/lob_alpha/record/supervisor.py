"""The always-on recorder: websocket connections, timers and everything around the sync core."""

from __future__ import annotations

import asyncio
import contextlib
import os
import random
import signal
import socket
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import WebSocketException

from lob_alpha import __version__
from lob_alpha.instruments import Instruments
from lob_alpha.record import health, raw_archive
from lob_alpha.record.raw_archive import RawArchive
from lob_alpha.record.sync import Action, CloseConn, NeedSnapshot, Resubscribe, VenuePipeline
from lob_alpha.record.venues.base import ConnectionSpec, ParseError, SyncState
from lob_alpha.record.venues.registry import make_adapter
from lob_alpha.record.writer import ParquetSink
from lob_alpha.settings import RecorderConfig, Settings

log = structlog.get_logger("recorder")

SEC = 1_000_000_000
STATS_INTERVAL_S = 60.0
DISK_INTERVAL_S = 60.0
PRUNE_INTERVAL_S = 3600.0
STANDBY_RETRY_S = 300.0
RECOVER_DELAY_S = 120.0
HARD_STOP_MARGIN_GB = 1.0
RESET_BACKOFF_AFTER_S = 60.0
INGEST_SAMPLES = 20_000

ClientFactory = Callable[[], httpx.AsyncClient]


class RateLimiter:
    """Token bucket keeping REST calls under 10% of a venue's budget (NFR-11)."""

    def __init__(self, capacity: float, per_second: float) -> None:
        self.capacity = capacity
        self.per_second = per_second
        self.tokens = capacity
        self.updated = time.monotonic()
        self.blocked_until = 0.0

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.per_second)
            self.updated = now
            if now >= self.blocked_until and self.tokens >= 1:
                self.tokens -= 1
                return
            await asyncio.sleep(max(self.blocked_until - now, 1 / self.per_second, 0.05))

    def penalise(self, seconds: float) -> None:
        self.blocked_until = max(self.blocked_until, time.monotonic() + seconds)


def _run_id(ts_ns: int) -> str:
    return datetime.fromtimestamp(ts_ns / 1e9, tz=UTC).strftime("%Y%m%dT%H%M%S")


def _deployed_sha() -> str:
    try:
        return Path("DEPLOYED_SHA").read_text(encoding="ascii").strip()[:12]
    except OSError:
        return ""


class VenueRunner:
    """Connections, REST calls and timers for one venue, around its `VenuePipeline`."""

    def __init__(self, recorder: Recorder, venue: str) -> None:
        self.recorder = recorder
        self.venue = venue
        self.cfg = recorder.config.venues[venue]
        self.adapter = make_adapter(venue, self.cfg)
        ticks = {s: recorder.instruments.get(venue, s).tick_size for s in self.cfg.symbols}
        self.pipeline = VenuePipeline(
            self.adapter,
            recorder.sink,
            ticks,
            host=recorder.host,
            version=recorder.version,
            stale_after_s=recorder.config.health.stale_after_s,
            log=self._log,
        )
        self.raw = recorder.raw
        self.sockets: dict[str, ClientConnection] = {}
        self.lives: dict[str, asyncio.Task[None]] = {}
        self.close_reasons: dict[str, str] = {}
        self.background: set[asyncio.Task[Any]] = set()
        self.tasks: list[asyncio.Task[Any]] = []
        self.fetching: set[str] = set()
        self.generation = 0
        self.hour = 0
        self.ingest_ns: deque[int] = deque(maxlen=INGEST_SAMPLES)
        self.frames = 0
        self.next_standby: dict[str, float] = {}
        self.http: httpx.AsyncClient = recorder.http_factory()
        # Binance: 20 weight per snapshot, 10 per minute is 200 of 2400. Coinbase: 1 per 5 s.
        self.limiter = RateLimiter(6, 1 / 6) if self.adapter.rest_snapshot else RateLimiter(3, 0.2)

    def _log(self, event: str, **fields: Any) -> None:
        if event == "crossed_book":
            log.warning(event, **fields)
        else:
            log.info(event, **fields)

    # ---- inputs ---------------------------------------------------------------------------

    def _roll(self, ts: int) -> None:
        """At the first input of a new hour, archive the state an hourly replay starts from."""
        hour = ts // raw_archive.NS_PER_HOUR
        if hour != self.hour:
            first = self.hour == 0
            self.hour = hour
            if not first:
                state = {"k": "state", "state": self.pipeline.dump()}
                self.raw.control(self.venue, hour * raw_archive.NS_PER_HOUR, state)

    def frame(self, conn_id: str, ts: int, raw: bytes) -> None:
        started = time.perf_counter_ns()
        self._roll(ts)
        self.raw.frame(self.venue, conn_id, ts, raw)
        actions = self.pipeline.ingest_frame(conn_id, ts, raw)
        self.frames += 1
        if actions:
            self.run_actions(actions)
        self.ingest_ns.append(time.perf_counter_ns() - started)

    def control(self, ctrl: dict[str, Any], body: bytes = b"", ts: int | None = None) -> None:
        stamp = time.time_ns() if ts is None else ts
        self._roll(stamp)
        self.raw.control(self.venue, stamp, ctrl, body)
        if ctrl["k"] == "promote":
            self._promote(ctrl, stamp)
            return
        try:
            actions = self.pipeline.ingest_control(stamp, ctrl, body)
        except ParseError as exc:
            log.error("bad_rest_body", venue=self.venue, error=str(exc)[:300])
            symbol = ctrl.get("symbol")
            if symbol is not None and ctrl["k"] == "snapshot":
                self.control({"k": "snapshot_error", "symbol": symbol, "error": str(exc)[:200]})
            return
        if actions:
            self.run_actions(actions)

    def _promote(self, ctrl: dict[str, Any], ts: int) -> None:
        old = self.pipeline.active.get(ctrl["logical"])
        self.pipeline.ingest_control(ts, ctrl)
        if old is not None:
            self.close(old, "promoted")

    def spawn(self, work: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(work)
        self.background.add(task)
        task.add_done_callback(self.background.discard)

    def run_actions(self, actions: list[Action]) -> None:
        for action in actions:
            if isinstance(action, NeedSnapshot):
                if action.symbol not in self.fetching:
                    self.fetching.add(action.symbol)
                    self.spawn(self._fetch_snapshot(action.symbol))
            elif isinstance(action, Resubscribe):
                self.spawn(self._resubscribe(action))
            elif isinstance(action, CloseConn):
                self.close(action.conn, action.reason)

    def close(self, conn_id: str, reason: str) -> None:
        ws = self.sockets.get(conn_id)
        if ws is not None and conn_id not in self.close_reasons:
            self.close_reasons[conn_id] = reason
            self.spawn(ws.close())

    # ---- REST -----------------------------------------------------------------------------

    async def _fetch_snapshot(self, symbol: str) -> None:
        try:
            await self.limiter.acquire()
            snapshot = await self.adapter.fetch_snapshot(self.http, symbol)
        except httpx.HTTPStatusError as exc:
            self._respect_limits(exc)
            self.control({"k": "snapshot_error", "symbol": symbol, "error": str(exc)[:200]})
        except (httpx.HTTPError, ValueError) as exc:
            self.control({"k": "snapshot_error", "symbol": symbol, "error": repr(exc)[:200]})
        else:
            if snapshot is not None:
                ctrl = {"k": "snapshot", "symbol": symbol}
                self.control(ctrl, snapshot.raw, snapshot.ts_recv)
        finally:
            self.fetching.discard(symbol)

    def _respect_limits(self, exc: httpx.HTTPStatusError) -> None:
        if exc.response.status_code in (418, 429):
            retry_after = float(exc.response.headers.get("Retry-After", 60))
            self.limiter.penalise(retry_after)
            log.warning("rest_rate_limited", venue=self.venue, retry_after=retry_after)

    async def _resubscribe(self, action: Resubscribe) -> None:
        ws = self.sockets.get(action.conn)
        if ws is None:
            return
        with contextlib.suppress(WebSocketException, OSError):
            for message in self.adapter.resubscribe_messages(action.symbol):
                await ws.send(message)

    async def _checkpoint_loop(self) -> None:
        kind = self.adapter.checkpoint_kind
        if kind is None:
            return
        symbols = self.cfg.symbols
        pause = self.cfg.checkpoint_interval_s / len(symbols)
        while True:
            for symbol in symbols:
                await asyncio.sleep(pause)
                if self.pipeline.books[symbol].state is not SyncState.LIVE:
                    continue
                try:
                    await self.limiter.acquire()
                    snapshot = await self.adapter.fetch_checkpoint(self.http, symbol)
                except httpx.HTTPStatusError as exc:
                    self._respect_limits(exc)
                    log.warning("checkpoint_failed", venue=self.venue, symbol=symbol)
                except (httpx.HTTPError, ValueError) as exc:
                    log.warning("checkpoint_failed", venue=self.venue, error=repr(exc)[:200])
                else:
                    if snapshot is not None:
                        ctrl = {"k": "checkpoint", "symbol": symbol, "checkpoint": kind}
                        self.control(ctrl, snapshot.raw, snapshot.ts_recv)

    async def _clock_loop(self) -> None:
        interval = self.recorder.config.health.clock_probe_interval_s
        while True:
            await asyncio.sleep(interval)
            ping = self.adapter.clock_ping(time.time_ns())
            if ping is not None:
                ws = next(iter(self.sockets.values()), None)
                if ws is not None:
                    with contextlib.suppress(WebSocketException, OSError):
                        await ws.send(ping)
                continue
            try:
                sample = await self.adapter.fetch_server_time(self.http)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                log.warning("clock_probe_failed", venue=self.venue, error=repr(exc)[:200])
                continue
            if sample is not None:
                ctrl = {
                    "k": "clock",
                    "server_time_ns": sample.server_time_ns,
                    "sent_ns": sample.sent_ns,
                }
                self.control(ctrl, ts=sample.ts_recv)

    # ---- connections ----------------------------------------------------------------------

    def _new_conn_id(self, spec: ConnectionSpec) -> str:
        conn_id = f"{spec.name}.g{self.generation}"
        self.generation += 1
        return conn_id

    async def _life(self, spec: ConnectionSpec, conn_id: str, standby: bool) -> None:
        """One websocket connection from open to close."""
        opened = False
        try:
            async with websockets.connect(
                spec.url,
                max_size=None,
                max_queue=None,
                compression=None,
                proxy=None,  # never through a proxy (spec section 2)
                open_timeout=15,
                close_timeout=2,
                ping_interval=20,
                ping_timeout=20,
            ) as ws:
                self.sockets[conn_id] = ws
                ctrl: dict[str, Any] = {"k": "conn_open", "conn": conn_id, "logical": spec.name}
                if standby:
                    ctrl["standby"] = True
                self.control(ctrl)
                opened = True
                for message in self.adapter.subscribe_messages(spec):
                    await ws.send(message)
                while True:
                    raw = await ws.recv(decode=False)
                    self.frame(conn_id, time.time_ns(), raw)
        except (OSError, WebSocketException, TimeoutError) as exc:
            log.info("connection_ended", venue=self.venue, conn=conn_id, error=repr(exc)[:200])
        finally:
            self.sockets.pop(conn_id, None)
            self.lives.pop(conn_id, None)
            reason = self.close_reasons.pop(conn_id, "disconnect")
            if opened and not self.recorder.stopping:
                self.control({"k": "conn_close", "conn": conn_id, "reason": reason})

    async def _manager(self, spec: ConnectionSpec) -> None:
        """Keeps one logical connection alive, with exponential backoff and jitter."""
        health_cfg = self.recorder.config.health
        backoff = health_cfg.backoff_initial_s
        while True:
            await self.recorder.writable.wait()
            current = self.pipeline.active.get(spec.name)
            task = self.lives.get(current) if current else None
            started = time.monotonic()
            if task is None:
                conn_id = self._new_conn_id(spec)
                task = asyncio.create_task(self._life(spec, conn_id, False))
                self.lives[conn_id] = task
            await asyncio.wait([task])
            if not task.cancelled() and task.exception() is not None:
                log.error("connection_task_crashed", venue=self.venue, error=repr(task.exception()))
            promoted = self.pipeline.active.get(spec.name) in self.lives
            if promoted:
                backoff = health_cfg.backoff_initial_s
                continue
            if time.monotonic() - started > RESET_BACKOFF_AFTER_S:
                backoff = health_cfg.backoff_initial_s
            await asyncio.sleep(backoff * random.uniform(0.5, 1.0))
            backoff = min(backoff * 2, health_cfg.backoff_max_s)

    def _open_standbys(self, now_ns: int) -> None:
        limit_h = self.adapter.max_conn_age_h
        if limit_h is None:
            return
        age_limit = int(
            min(limit_h, self.recorder.config.health.proactive_reconnect_h) * 3600 * SEC
        )
        specs = self.pipeline.specs
        for logical, conn_id in list(self.pipeline.active.items()):
            conn = self.pipeline.conns.get(conn_id)
            if conn is None or logical in self.pipeline.standby:
                continue
            due = self.next_standby.get(logical, 0.0)
            if now_ns - conn.opened_ns >= age_limit and time.monotonic() >= due:
                self.next_standby[logical] = time.monotonic() + STANDBY_RETRY_S
                standby_id = self._new_conn_id(specs[logical])
                log.info("make_before_break", venue=self.venue, logical=logical, conn=standby_id)
                task = asyncio.create_task(self._life(specs[logical], standby_id, True))
                self.lives[standby_id] = task

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(min(1.0, self.pipeline.stale_after_ns / SEC / 4))
            now = time.time_ns()
            controls, actions = self.pipeline.timers(now)
            for ctrl in controls:
                self.control(ctrl)
            if actions:
                self.run_actions(actions)
            self._open_standbys(now)

    # ---- lifecycle ------------------------------------------------------------------------

    def start(self, previous: dict[str, Any], reason: str) -> None:
        ctrl = {
            "k": "run_start",
            "run": self.recorder.run_id,
            "streams": previous,
            "reason": reason,
        }
        self.control(ctrl)
        for spec in self.pipeline.specs.values():
            self.tasks.append(asyncio.create_task(self._manager(spec)))
        self.tasks.append(asyncio.create_task(self._tick_loop()))
        self.tasks.append(asyncio.create_task(self._checkpoint_loop()))
        self.tasks.append(asyncio.create_task(self._clock_loop()))

    def hold_for_disk(self) -> None:
        for conn_id in list(self.sockets):
            self.close(conn_id, "disk_guard")

    async def stop(self) -> None:
        for task in self.tasks + list(self.lives.values()) + list(self.background):
            task.cancel()
        pending = self.tasks + list(self.lives.values()) + list(self.background)
        await asyncio.gather(*pending, return_exceptions=True)
        now = time.time_ns()
        for stream in self.pipeline.streams():
            stream.flush_gap(now)
        await self.http.aclose()

    def status(self, now: int) -> dict[str, Any]:
        block = self.pipeline.status(now)
        samples = sorted(self.ingest_ns)
        block["ingest_us_p50"] = samples[len(samples) // 2] / 1000 if samples else None
        block["ingest_us_p99"] = samples[int(len(samples) * 0.99)] / 1000 if samples else None
        block["frames"] = self.frames
        block["conns"] = list(self.sockets)
        return block


class Recorder:
    """The `lob record run` process."""

    def __init__(
        self,
        config: RecorderConfig,
        settings: Settings,
        instruments: Instruments,
        *,
        data_dir: Path | None = None,
        http_factory: ClientFactory | None = None,
        auto_recover: bool = True,
    ) -> None:
        self.config = config
        self.settings = settings
        self.instruments = instruments
        self.data_dir = data_dir or settings.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.started_ns = time.time_ns()
        self.run_id = _run_id(self.started_ns)
        self.host = socket.gethostname()
        sha = _deployed_sha()
        self.version = f"{__version__}+{sha}" if sha else __version__
        self.http_factory = http_factory or (
            lambda: httpx.AsyncClient(timeout=15, trust_env=False, http2=False)
        )
        self.auto_recover = auto_recover
        self.sink = ParquetSink(self.data_dir, config.writer, run=self.run_id)
        self.raw = RawArchive(
            self.data_dir,
            self.run_id,
            flush_interval_s=config.raw_archive.flush_interval_s,
        )
        self.raw.paused = not config.raw_archive.enabled
        self.writable = asyncio.Event()
        self.writable.set()
        self.stop_event = asyncio.Event()
        self.stopping = False
        self.disk_guarded = False
        self.disk: health.DiskState = health.disk_state(self.data_dir)
        self.runners = [VenueRunner(self, venue) for venue in config.venue_set]
        self.last_frames: dict[str, int] = {}

    def request_stop(self) -> None:
        self.stop_event.set()

    def status(self) -> dict[str, Any]:
        now = time.time_ns()
        return {
            "ts_ns": now,
            "ts_utc": datetime.fromtimestamp(now / 1e9, tz=UTC).isoformat(),
            "run": self.run_id,
            "pid": os.getpid(),
            "host": self.host,
            "boot_id": health.boot_id(),
            "version": self.version,
            "stopped": self.stopping,
            "disk": {
                "free_gb": round(self.disk.free_gb, 1),
                "used_pct": round(self.disk.used_pct, 1),
                "guarded": self.disk_guarded,
                "raw_paused": self.raw.paused,
            },
            "writer": {
                "rows_written": self.sink.rows_written,
                "files_closed": self.sink.files_closed,
                "flush_errors": self.sink.flush_errors,
                "last_error": self.sink.last_error,
            },
            "raw": {
                "dropped": self.raw.dropped,
                "errors": self.raw.errors,
                "last_error": self.raw.last_error,
            },
            "venues": {r.venue: r.status(now) for r in self.runners},
        }

    async def _status_loop(self) -> None:
        path = self.data_dir / health.STATUS_FILE
        while True:
            await asyncio.sleep(self.config.health.status_interval_s)
            await asyncio.to_thread(health.write_json_atomic, path, self.status())

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            self.sink.tick()

    async def _stats_loop(self) -> None:
        while True:
            await asyncio.sleep(STATS_INTERVAL_S)
            now = time.time_ns()
            for runner in self.runners:
                previous = self.last_frames.get(runner.venue, 0)
                self.last_frames[runner.venue] = runner.frames
                block = runner.status(now)
                log.info(
                    "venue_stats",
                    venue=runner.venue,
                    msgs_per_s=round((runner.frames - previous) / STATS_INTERVAL_S, 1),
                    rows_written=self.sink.rows_written,
                    ingest_us_p99=block["ingest_us_p99"],
                    feed_latency_ms_p50=block["feed_latency_ms_p50"],
                    clock_offset_ms=block["clock_offset_ms"],
                )
                for stream in block["streams"]:
                    log.info("stream_stats", venue=runner.venue, **stream)

    def _apply_disk_state(self, state: health.DiskState) -> None:
        self.disk = state
        cfg = self.config.health
        if self.disk.used_pct >= health.WARN_USED_PCT:
            log.warning("disk_over_80_pct", used_pct=round(self.disk.used_pct, 1))
        if not self.config.raw_archive.enabled:
            pass
        elif self.disk.free_gb < cfg.min_free_gb and not self.raw.paused:
            log.warning("raw_archive_paused", free_gb=round(self.disk.free_gb, 1))
            self.raw.paused = True
        elif self.disk.free_gb >= cfg.min_free_gb and self.raw.paused:
            log.info("raw_archive_resumed", free_gb=round(self.disk.free_gb, 1))
            self.raw.paused = False
        if self.disk.free_gb < cfg.hard_stop_free_gb and not self.disk_guarded:
            log.error("disk_guard_stop", free_gb=round(self.disk.free_gb, 1))
            self.disk_guarded = True
            self.writable.clear()
            for runner in self.runners:
                runner.hold_for_disk()
        elif self.disk.free_gb >= cfg.hard_stop_free_gb + HARD_STOP_MARGIN_GB and self.disk_guarded:
            log.info("disk_guard_clear", free_gb=round(self.disk.free_gb, 1))
            self.disk_guarded = False
            self.writable.set()

    async def _disk_loop(self) -> None:
        while True:
            await asyncio.sleep(min(DISK_INTERVAL_S, self.config.health.status_interval_s * 6))
            self._apply_disk_state(await asyncio.to_thread(health.disk_state, self.data_dir))

    async def _prune_loop(self) -> None:
        while True:
            removed = await asyncio.to_thread(
                raw_archive.prune,
                self.data_dir,
                self.config.raw_archive.retention_days,
                date.today(),
            )
            for path in removed:
                log.info("raw_archive_pruned", path=str(path))
            await asyncio.sleep(PRUNE_INTERVAL_S)

    async def _recover_interrupted(self, config_path: Path) -> None:
        """A crash leaves hours only in the raw archive. Rebuild them at low priority."""
        await asyncio.sleep(RECOVER_DELAY_S)
        process = await asyncio.create_subprocess_exec(
            "nice", "-n", "19", sys.executable, "-m", "lob_alpha.cli",
            "record", "recover", "--from-raw", "--interrupted", "--config", str(config_path),
            stdout=asyncio.subprocess.DEVNULL,
        )  # fmt: skip
        code = await process.wait()
        log.info("interrupted_hours_recovered", exit_code=code)

    async def _wait_for_stop(
        self,
        stop_task: asyncio.Task[Any],
        watched: set[asyncio.Task[Any]],
        duration_s: float | None,
    ) -> asyncio.Task[Any] | None:
        """Wait for a stop request or the end of `duration_s`. Returns a crashed task, if any."""
        deadline = None if duration_s is None else time.monotonic() + duration_s
        pending = watched
        while True:
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            done, pending = await asyncio.wait(
                pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if not done or stop_task in done:
                return None
            for task in done:
                if not task.cancelled() and task.exception() is not None:
                    log.error("recorder_task_crashed", error=repr(task.exception()))
                    return task

    async def run(
        self, duration_s: float | None = None, config_path: Path | None = None
    ) -> dict[str, Any]:
        """Record until SIGTERM, or for `duration_s`. Returns the final status."""
        loop = asyncio.get_running_loop()
        for name in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, ValueError, RuntimeError):
                loop.add_signal_handler(name, self.request_stop)
        status = health.read_status(self.data_dir)
        saved, reason = health.previous_streams(status)
        capture = health.note_capture_start(self.data_dir, self.started_ns, self.host)
        log.info(
            "recorder_start", run=self.run_id, version=self.version, reason=reason,
            capture_started=capture["started_utc"], venues=self.config.venue_set,
        )  # fmt: skip
        for runner in self.runners:
            runner.start(saved.get(runner.venue, {}), reason)
        stop_task = asyncio.create_task(self.stop_event.wait())
        tasks = [
            asyncio.create_task(self._status_loop()),
            asyncio.create_task(self._flush_loop()),
            asyncio.create_task(self._stats_loop()),
            asyncio.create_task(self._disk_loop()),
            asyncio.create_task(self._prune_loop()),
        ]
        if self.auto_recover and config_path is not None and sys.platform != "win32":
            tasks.append(asyncio.create_task(self._recover_interrupted(config_path)))
        watched = {stop_task, *tasks, *(t for r in self.runners for t in r.tasks)}
        crashed = await self._wait_for_stop(stop_task, watched, duration_s)
        self.stopping = True
        stop_task.cancel()
        for task in tasks:
            task.cancel()
        await asyncio.gather(stop_task, *tasks, return_exceptions=True)
        for runner in self.runners:
            await runner.stop()
        await asyncio.to_thread(self.sink.close)
        await asyncio.to_thread(self.raw.close)
        final = self.status()
        health.write_json_atomic(self.data_dir / health.STATUS_FILE, final)
        log.info("recorder_stopped", run=self.run_id, rows_written=self.sink.rows_written)
        if crashed is not None:  # stop cleanly, then let systemd restart a whole process
            raise RuntimeError("recorder task crashed") from crashed.exception()
        return final
