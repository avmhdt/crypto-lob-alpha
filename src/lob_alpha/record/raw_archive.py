"""Rolling raw archive: every input the sync core saw, in `ts_recv\\tpayload` lines (FR-09).

Websocket frames go to `HH-<conn>-<run>.jsonl.zst`. Everything else the pipeline consumes
(REST bodies, connection events, timers, the hourly state record) goes to
`HH-control-<run>.jsonl.zst` as `ts_recv\\t!<json>\\t<body>`. Together they are a complete
input log, so `lob record recover` can replay an hour through the same code.

Lines are compressed on a writer thread and pushed to the OS at least once a second, so
`kill -9` loses at most about a second of frames (NFR-02). A killed file ends in a truncated
zstd frame, which the reader stops at cleanly.
"""

from __future__ import annotations

import heapq
import os
import queue
import shutil
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import IO, Any

import orjson
import zstandard

NS_PER_HOUR = 3_600_000_000_000
CONTROL = "control"
MAX_QUEUE = 2_000_000


def archive_root(data_dir: Path) -> Path:
    return data_dir / "raw" / "ws"


def _hour_parts(ts: int) -> tuple[str, str]:
    moment = datetime.fromtimestamp(ts // 1_000_000_000, tz=UTC)
    return moment.strftime("%Y-%m-%d"), moment.strftime("%H")


def _flat(payload: bytes) -> bytes:
    return payload.replace(b"\n", b" ") if b"\n" in payload else payload


@dataclass(slots=True)
class _File:
    hour: int
    handle: IO[bytes]
    stream: Any


class RawArchive:
    """Appends lines from the event loop; a thread compresses and writes them."""

    def __init__(
        self, data_dir: Path, run: str, *, level: int = 3, flush_interval_s: float = 1.0
    ) -> None:
        self.root = archive_root(data_dir)
        self.run = run
        self.flush_interval_s = flush_interval_s
        self.level = level
        self.paused = False
        self.dropped = 0
        self.errors = 0
        self.last_error = ""
        self.last_flush_ns = 0
        self._queue: queue.Queue[tuple[str, str, int, bytes] | None] = queue.Queue()
        self._files: dict[tuple[str, str], _File] = {}
        self._thread = threading.Thread(target=self._drain, name="raw-archive", daemon=True)
        self._thread.start()

    def frame(self, venue: str, conn: str, ts: int, payload: bytes) -> None:
        """Queue one websocket frame."""
        self._put(venue, conn, ts, b"%d\t%s\n" % (ts, _flat(payload)))

    def control(self, venue: str, ts: int, ctrl: dict[str, Any], body: bytes = b"") -> None:
        """Queue one control record, with an optional raw body such as a REST response."""
        self._put(venue, CONTROL, ts, b"%d\t!%s\t%s\n" % (ts, orjson.dumps(ctrl), _flat(body)))

    def _put(self, venue: str, name: str, ts: int, line: bytes) -> None:
        if self.paused or self._queue.qsize() > MAX_QUEUE:
            self.dropped += 1
            return
        self._queue.put((venue, name, ts, line))

    def close(self) -> None:
        """Write everything queued and end every zstd frame."""
        self._queue.put(None)
        self._thread.join()

    def _open(self, venue: str, name: str, ts: int) -> _File:
        day, hour = _hour_parts(ts)
        directory = self.root / f"venue={venue}" / f"date={day}"
        directory.mkdir(parents=True, exist_ok=True)
        handle = open(directory / f"{hour}-{name}-{self.run}.jsonl.zst", "ab")  # noqa: SIM115
        # A compressor context serves one stream, so every open file needs its own.
        compressor = zstandard.ZstdCompressor(level=self.level)
        stream = compressor.stream_writer(handle, closefd=False)
        return _File(ts // NS_PER_HOUR, handle, stream)

    def _close_file(self, file: _File) -> None:
        file.stream.flush(zstandard.FLUSH_FRAME)
        file.handle.flush()
        os.fsync(file.handle.fileno())
        file.handle.close()

    def _flush_all(self) -> None:
        for file in self._files.values():
            file.stream.flush(zstandard.FLUSH_BLOCK)
            file.handle.flush()
        self.last_flush_ns = time.time_ns()

    def _write(self, venue: str, name: str, ts: int, line: bytes) -> None:
        key = (venue, name)
        file = self._files.get(key)
        if file is None or file.hour != ts // NS_PER_HOUR:
            if file is not None:
                self._close_file(file)
            file = self._open(venue, name, ts)
            self._files[key] = file
        file.stream.write(line)

    def _drain(self) -> None:
        next_flush = time.monotonic() + self.flush_interval_s
        running = True
        while running:
            wait = max(0.0, next_flush - time.monotonic())
            try:
                item = self._queue.get(timeout=wait)
                if item is None:
                    running = False
                else:
                    self._write(*item)
            except queue.Empty:
                pass
            except OSError as exc:  # a full or failing disk must not stop the recorder
                self.errors += 1
                self.last_error = repr(exc)
            if time.monotonic() >= next_flush:
                try:
                    self._flush_all()
                except OSError as exc:
                    self.errors += 1
                    self.last_error = repr(exc)
                next_flush = time.monotonic() + self.flush_interval_s
        for file in self._files.values():
            self._close_file(file)
        self._files.clear()


@dataclass(slots=True)
class Record:
    """One archived input. `ctrl` is set for control records, otherwise this is a frame."""

    ts: int
    order: int
    conn: str
    run: str
    payload: bytes
    ctrl: dict[str, Any] | None


def _split_name(path: Path) -> tuple[str, str]:
    stem = path.name.removesuffix(".jsonl.zst")
    name, run = stem[3:].rsplit("-", 1)
    return name, run


def read_lines(path: Path) -> Iterator[bytes]:
    """Lines of one archive file, stopping cleanly at a truncated tail.

    Files of dead connections and killed runs end without a zstd frame footer. `stream_reader`
    can silently lose the end of such a file, so this feeds a `decompressobj` instead, which
    returns everything decodable.
    """
    decoder = zstandard.ZstdDecompressor().decompressobj(read_across_frames=True)
    tail = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            try:
                data = decoder.decompress(chunk)
            except zstandard.ZstdError:
                break
            lines = (tail + data).split(b"\n")
            tail = lines.pop()
            yield from lines


def read_file(path: Path) -> Iterator[Record]:
    """Parsed records of one archive file."""
    name, run = _split_name(path)
    for line in read_lines(path):
        stamp, _, rest = line.partition(b"\t")
        if not rest:
            continue
        if name == CONTROL:
            meta, _, body = rest.removeprefix(b"!").partition(b"\t")
            yield Record(int(stamp), 0, name, run, body, orjson.loads(meta))
        else:
            yield Record(int(stamp), 1, name, run, rest, None)


def hour_files(data_dir: Path, venue: str, day: date, hour: int) -> list[Path]:
    directory = archive_root(data_dir) / f"venue={venue}" / f"date={day.isoformat()}"
    return sorted(directory.glob(f"{hour:02d}-*.jsonl.zst"))


def read_hour(data_dir: Path, venue: str, day: date, hour: int) -> Iterator[Record]:
    """Every record of one venue-hour, merged by `ts_recv`, control records first on a tie."""
    readers = [read_file(path) for path in hour_files(data_dir, venue, day, hour)]
    return heapq.merge(*readers, key=lambda record: (record.ts, record.order))


def prune(data_dir: Path, retention_days: int, today: date) -> list[Path]:
    """Delete `date=` folders older than the retention window (D-06)."""
    cutoff = today - timedelta(days=retention_days)
    removed: list[Path] = []
    for venue_dir in archive_root(data_dir).glob("venue=*"):
        for day_dir in venue_dir.glob("date=*"):
            if date.fromisoformat(day_dir.name.removeprefix("date=")) < cutoff:
                shutil.rmtree(day_dir)
                removed.append(day_dir)
    return removed
