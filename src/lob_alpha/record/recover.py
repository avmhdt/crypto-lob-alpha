"""`lob record recover`: rebuild Parquet by replaying the raw archive (FR-09, NFR-02)."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import unquote

from lob_alpha.instruments import Instruments
from lob_alpha.record import health
from lob_alpha.record.durable import has_parquet_footer
from lob_alpha.record.raw_archive import hour_files, read_hour
from lob_alpha.record.sync import VenuePipeline
from lob_alpha.record.venues.registry import make_adapter
from lob_alpha.record.writer import ParquetSink
from lob_alpha.settings import RecorderConfig

ACTIVE_STATUS_AGE_NS = 60_000_000_000
TMP_PATTERN = re.compile(r"/venue=([^/]+)/(?:symbol=[^/]+/)?date=(\d{4}-\d{2}-\d{2})/part-(\d{2})")


@dataclass(slots=True)
class RecoverNamer:
    """Names a run's part exactly as the live writer did, and protects closed files."""

    replace: bool
    runs: dict[tuple[Path, str], dict[str, int]] = field(default_factory=dict)
    skipped: list[Path] = field(default_factory=list)
    written: list[Path] = field(default_factory=list)

    def __call__(self, directory: Path, hour_stem: str, run: str) -> Path | None:
        order = self.runs.setdefault((directory, hour_stem), {})
        index = order.setdefault(run, len(order))
        name = f"{hour_stem}.parquet" if index == 0 else f"{hour_stem}-r{index}.parquet"
        path = directory / name
        if path.exists() and not self.replace and has_parquet_footer(path):
            if path not in self.skipped:
                self.skipped.append(path)
            return None
        if path not in self.written:
            self.written.append(path)
        return path


def active_run(data_dir: Path) -> str | None:
    """The run still writing to this data directory, if any."""
    status = health.read_status(data_dir)
    if status is None or status.get("stopped"):
        return None
    if time.time_ns() - int(status.get("ts_ns", 0)) > ACTIVE_STATUS_AGE_NS:
        return None
    run = status.get("run")
    return run if isinstance(run, str) else None


def _pipeline(
    config: RecorderConfig, instruments: Instruments, venue: str, sink: ParquetSink, day: date
) -> VenuePipeline:
    cfg = config.venues[venue]
    ticks = {s: instruments.get(venue, s, day).tick_size for s in cfg.symbols}
    return VenuePipeline(make_adapter(venue, cfg), sink, ticks)


def replay_hour(
    config: RecorderConfig,
    instruments: Instruments,
    data_dir: Path,
    venue: str,
    day: date,
    hour: int,
    sink: ParquetSink,
    skip_run: str | None,
) -> int:
    """Feed one venue-hour of raw records through the sync core. Returns records replayed."""
    pipeline = _pipeline(config, instruments, venue, sink, day)
    count = 0
    run: str | None = None
    for record in read_hour(data_dir, venue, day, hour):
        if record.run == skip_run:
            continue
        if record.run != run:
            run = record.run
            sink.set_run(run)
            pipeline = _pipeline(config, instruments, venue, sink, day)
        count += 1
        if record.ctrl is None:
            pipeline.ingest_frame(record.conn, record.ts, record.payload)
        else:
            try:
                pipeline.ingest_control(record.ts, record.ctrl, record.payload)
            except ValueError:
                continue  # a bad REST body; the archive holds the snapshot_error that followed
    return count


@dataclass(slots=True)
class RecoverResult:
    records: int
    written: list[Path]
    skipped: list[Path]


def recover(
    config: RecorderConfig,
    instruments: Instruments,
    data_dir: Path,
    venue: str,
    day: date,
    hours: list[int],
    *,
    replace: bool = False,
    out_dir: Path | None = None,
) -> RecoverResult:
    """Rebuild the given hours of one venue. Closed files stay unless `replace` is set."""
    namer = RecoverNamer(replace)
    midnight = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) // 3600
    records = 0
    skip_run = active_run(data_dir)
    for hour in hours:
        if not hour_files(data_dir, venue, day, hour):
            continue
        sink = ParquetSink(
            out_dir or data_dir,
            config.writer,
            run="recover",
            namer=namer,
            write_gaps=False,
            hours={midnight + hour},
        )
        records += replay_hour(config, instruments, data_dir, venue, day, hour, sink, skip_run)
        sink.close()
        if sink.flush_errors:
            raise RuntimeError(f"recover failed for {venue} {day} {hour}: {sink.last_error}")
    return RecoverResult(records, namer.written, namer.skipped)


def interrupted_hours(data_dir: Path) -> list[tuple[str, date, int]]:
    """Venue-hours with a `.parquet.tmp` on disk, or a closed part with no footer.

    The live run's own files show up here too. That is harmless: `recover` skips the active
    run, so such an hour only rebuilds what older runs left behind. A missing footer means
    the host went down after the rename, before the file's last pages reached the disk.
    """
    broken = [p for p in data_dir.rglob("part-*.parquet") if not has_parquet_footer(p)]
    found: set[tuple[str, date, int]] = set()
    for path in [*data_dir.rglob("part-*.parquet.tmp"), *broken]:
        match = TMP_PATTERN.search(path.as_posix())
        if match is not None and "/gaps/" not in path.as_posix():
            venue, day_text, hour_text = match.groups()
            found.add((unquote(venue), date.fromisoformat(day_text), int(hour_text)))
    return sorted(found)
