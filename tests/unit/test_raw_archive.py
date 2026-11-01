from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from lob_alpha.record.raw_archive import (
    RawArchive,
    hour_files,
    prune,
    read_file,
    read_hour,
    read_lines,
)

SEC = 1_000_000_000
T0 = int(datetime(2026, 11, 2, 14, 0, 1, tzinfo=UTC).timestamp()) * SEC


def test_frames_and_control_records_round_trip(tmp_path: Path) -> None:
    archive = RawArchive(tmp_path, "run1", flush_interval_s=0.01)
    archive.control("kraken", T0, {"k": "conn_open", "conn": "ws.g0", "logical": "ws"})
    archive.frame("kraken", "ws.g0", T0 + 1, b'{"a":1}')
    archive.control("kraken", T0 + 2, {"k": "snapshot", "symbol": "X"}, b'{"body":\n2}')
    archive.close()
    records = list(read_hour(tmp_path, "kraken", date(2026, 11, 2), 14))
    assert [r.ts for r in records] == [T0, T0 + 1, T0 + 2]
    assert records[0].ctrl == {"k": "conn_open", "conn": "ws.g0", "logical": "ws"}
    assert records[1].ctrl is None
    assert records[1].payload == b'{"a":1}'
    assert records[1].conn == "ws.g0"
    assert records[1].run == "run1"
    assert records[2].payload == b'{"body": 2}'


def test_file_layout_is_hourly_per_connection(tmp_path: Path) -> None:
    archive = RawArchive(tmp_path, "run1")
    archive.frame("kraken", "ws.g0", T0, b"{}")
    archive.frame("kraken", "ws.g0", T0 + 3600 * SEC, b"{}")
    archive.close()
    directory = tmp_path / "raw" / "ws" / "venue=kraken" / "date=2026-11-02"
    assert sorted(p.name for p in directory.iterdir()) == [
        "14-ws.g0-run1.jsonl.zst",
        "15-ws.g0-run1.jsonl.zst",
    ]


def test_records_merge_by_time_across_connections_with_control_first(tmp_path: Path) -> None:
    archive = RawArchive(tmp_path, "run1")
    archive.frame("x", "a.g0", T0 + 2, b"late")
    archive.frame("x", "b.g0", T0 + 1, b"early")
    archive.control("x", T0 + 2, {"k": "state"})
    archive.close()
    records = list(read_hour(tmp_path, "x", date(2026, 11, 2), 14))
    assert [r.payload for r in records] == [b"early", b"", b"late"]


def test_a_truncated_tail_stops_cleanly_at_the_last_full_line(tmp_path: Path) -> None:
    archive = RawArchive(tmp_path, "run1", flush_interval_s=0.01)
    for i in range(200):
        archive.frame("x", "a.g0", T0 + i, b'{"n":%d}' % i)
    archive.close()
    path = hour_files(tmp_path, "x", date(2026, 11, 2), 14)[0]
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 5])
    lines = list(read_lines(path))
    assert len(lines) < 200 or lines[-1].endswith(b"}")
    assert all(line.endswith(b"}") for line in lines)


def test_paused_archive_counts_what_it_drops(tmp_path: Path) -> None:
    archive = RawArchive(tmp_path, "run1")
    archive.paused = True
    archive.frame("x", "a.g0", T0, b"{}")
    archive.close()
    assert archive.dropped == 1
    assert not (tmp_path / "raw").exists()


def test_write_failures_are_counted_not_raised(tmp_path: Path) -> None:
    (tmp_path / "raw").write_text("a file where a folder must go", encoding="utf-8")
    archive = RawArchive(tmp_path, "run1")
    archive.frame("x", "a.g0", T0, b"{}")
    archive.close()
    assert archive.errors == 1
    assert archive.last_error


def test_prune_removes_only_days_past_retention(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "ws" / "venue=x"
    for day in ("2026-11-01", "2026-11-02", "2026-11-09"):
        (root / f"date={day}").mkdir(parents=True)
    removed = prune(tmp_path, 7, date(2026, 11, 9))
    assert [p.name for p in removed] == ["date=2026-11-01"]
    assert sorted(p.name for p in root.iterdir()) == ["date=2026-11-02", "date=2026-11-09"]


def test_read_file_parses_run_from_the_name(tmp_path: Path) -> None:
    archive = RawArchive(tmp_path, "20261102T140001")
    archive.frame("x", "public.g3", T0, b"{}")
    archive.close()
    path = hour_files(tmp_path, "x", date(2026, 11, 2), 14)[0]
    record = next(read_file(path))
    assert (record.conn, record.run) == ("public.g3", "20261102T140001")
