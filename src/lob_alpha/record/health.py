"""Status file, disk accounting and capture bookkeeping (NFR-10, section 7d)."""

from __future__ import annotations

import os
import platform
import shutil
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import orjson

GIB = 1024**3
STATUS_FILE = "status.json"
CAPTURE_FILE = "capture.json"
WARN_USED_PCT = 80.0


def boot_id() -> str:
    """Identifies this OS boot, so a restart can be told from a host outage."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError:
        return ""


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = orjson.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def read_status(data_dir: Path) -> dict[str, Any] | None:
    return read_json(data_dir / STATUS_FILE)


def status_age_s(status: dict[str, Any]) -> float:
    return (time.time_ns() - int(status.get("ts_ns", 0))) / 1e9


def note_capture_start(data_dir: Path, ts_ns: int, host: str) -> dict[str, Any]:
    """Record when capture first started. Later runs leave the file alone."""
    path = data_dir / CAPTURE_FILE
    existing = read_json(path)
    if existing is not None:
        return existing
    started = datetime.fromtimestamp(ts_ns / 1e9, tz=UTC)
    payload = {"started_ns": ts_ns, "started_utc": started.isoformat(), "host": host}
    write_json_atomic(path, payload)
    return payload


def previous_streams(status: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], str]:
    """Per-venue saved stream state from the last status file, and why the recorder stopped."""
    if status is None:
        return {}, "process_restart"
    reason = "host_down" if status.get("boot_id") != boot_id() else "process_restart"
    saved: dict[str, dict[str, Any]] = {}
    for venue, block in status.get("venues", {}).items():
        saved[venue] = {f"{s['symbol']}|{s['stream']}": s for s in block.get("streams", [])}
    return saved, reason


@dataclass(frozen=True, slots=True)
class DiskState:
    free_gb: float
    used_pct: float
    total_gb: float


def disk_state(path: Path) -> DiskState:
    usage = shutil.disk_usage(path)
    return DiskState(usage.free / GIB, 100.0 * usage.used / usage.total, usage.total / GIB)


def dated_sizes(data_dir: Path) -> dict[date, dict[str, int]]:
    """Bytes per UTC date per top-level dataset, from the `date=` folders on disk."""
    sizes: dict[date, dict[str, int]] = {}
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        day_part = next((p for p in path.parts if p.startswith("date=")), None)
        if day_part is None:
            continue
        day = date.fromisoformat(day_part.removeprefix("date="))
        relative = path.relative_to(data_dir).parts
        dataset = "raw" if relative[0] == "raw" else relative[0]
        bucket = sizes.setdefault(day, {})
        bucket[dataset] = bucket.get(dataset, 0) + path.stat().st_size
    return sizes


def disk_report(data_dir: Path, planned_days: int, retention_days: int) -> dict[str, Any]:
    """Measured daily rate and a projection to the planned end of capture."""
    state = disk_state(data_dir)
    sizes = dated_sizes(data_dir)
    capture = read_json(data_dir / CAPTURE_FILE) or {}
    now = datetime.now(tz=UTC)
    report: dict[str, Any] = {
        "platform": platform.node(),
        "free_gb": round(state.free_gb, 1),
        "used_pct": round(state.used_pct, 1),
        "warn_over_80_pct": state.used_pct >= WARN_USED_PCT,
        "capture_started_utc": capture.get("started_utc"),
        "days": {
            day.isoformat(): {k: round(v / GIB, 3) for k, v in sorted(bucket.items())}
            for day, bucket in sorted(sizes.items())
        },
    }
    started_ns = capture.get("started_ns")
    if not started_ns:
        return report
    elapsed_days = (now.timestamp() - started_ns / 1e9) / 86_400
    canonical = sum(v for b in sizes.values() for k, v in b.items() if k != "raw")
    raw = sum(v for b in sizes.values() for k, v in b.items() if k == "raw")
    report["elapsed_days"] = round(elapsed_days, 3)
    if elapsed_days < 1 / 24:
        return report
    canonical_rate = canonical / GIB / elapsed_days
    raw_rate = raw / GIB / elapsed_days
    end = datetime.fromtimestamp(started_ns / 1e9, tz=UTC) + timedelta(days=planned_days)
    remaining = max(0.0, planned_days - elapsed_days)
    raw_target = raw_rate * min(retention_days + 1, planned_days)
    growth = canonical_rate * remaining + max(0.0, raw_target - raw / GIB)
    report.update(
        canonical_gb_per_day=round(canonical_rate, 2),
        raw_gb_per_day=round(raw_rate, 2),
        rate_is_provisional=elapsed_days < 1,
        planned_end_utc=end.date().isoformat(),
        projected_growth_gb=round(growth, 1),
        projected_free_gb_at_end=round(state.free_gb - growth, 1),
    )
    return report
