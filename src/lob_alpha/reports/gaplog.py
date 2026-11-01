"""`lob record gaplog`: publish the gap log and the coverage table (FR-10)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from lob_alpha import schemas

NS_PER_S = 1_000_000_000
NOT_DOWNTIME = {"clock"}  # data still arrived; research treats these with and without


def _ns(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * NS_PER_S


def _iso(ns_column: str) -> pl.Expr:
    return (
        pl.from_epoch(pl.col(ns_column), time_unit="ns")
        .dt.strftime("%Y-%m-%dT%H:%M:%S%.3fZ")
        .alias(ns_column.replace("_ns", "_utc"))
    )


def load_gaps(data_dir: Path) -> pl.DataFrame:
    """Every gap row on disk, one row per `gap_id`, sorted by start."""
    files = sorted((data_dir / "gaps").rglob("*.parquet"))
    if not files:
        empty = pl.from_arrow(schemas.GAPS.empty_table())
        assert isinstance(empty, pl.DataFrame)
        return empty
    frames = [pl.read_parquet(path, hive_partitioning=False) for path in files]
    return pl.concat(frames).unique(subset="gap_id", keep="first").sort("start_ns", "gap_id")


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union of closed intervals, so overlapping gaps count once."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


@dataclass(frozen=True, slots=True)
class Stream:
    venue: str
    symbol: str
    stream: str


def coverage_table(
    gaps: pl.DataFrame, streams: list[Stream], start_ns: int, end_ns: int
) -> pl.DataFrame:
    """LIVE coverage per stream over the window, with the longest single gap."""
    rows: list[dict[str, object]] = []
    window = end_ns - start_ns
    for item in streams:
        chosen = gaps.filter(
            (pl.col("venue") == item.venue)
            & (pl.col("symbol") == item.symbol)
            & (pl.col("stream") == item.stream)
            & ~pl.col("reason").is_in(list(NOT_DOWNTIME))
        )
        clipped = [
            (max(s, start_ns), min(e, end_ns))
            for s, e in chosen.select("start_ns", "end_ns").rows()
            if e > start_ns and s < end_ns
        ]
        merged = merge_intervals(clipped)
        down = sum(e - s for s, e in merged)
        longest = max((e - s for s, e in merged), default=0)
        rows.append(
            {
                "venue": item.venue,
                "symbol": item.symbol,
                "stream": item.stream,
                "window_s": window / NS_PER_S,
                "downtime_s": down / NS_PER_S,
                "coverage_pct": round(100.0 * (window - down) / window, 4),
                "n_gaps": len(merged),
                "longest_gap_s": longest / NS_PER_S,
            }
        )
    return pl.DataFrame(rows)


def publish(
    data_dir: Path,
    streams: list[Stream],
    start: date,
    end: date,
    out: Path,
    coverage_out: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Write `gap_log.csv` and `coverage.csv` for the UTC days from `start` to `end`."""
    start_ns = _ns(start)
    end_ns = min(_ns(end + timedelta(days=1)), time.time_ns())
    capture = data_dir / "capture.json"
    if capture.exists():
        started = pl.read_json(capture).item(0, "started_ns")
        start_ns = max(start_ns, int(started))
    gaps = load_gaps(data_dir)
    in_window = gaps.filter((pl.col("end_ns") > start_ns) & (pl.col("start_ns") < end_ns))
    log = in_window.with_columns(_iso("start_ns"), _iso("end_ns"))
    coverage = coverage_table(gaps, streams, start_ns, end_ns)
    out.parent.mkdir(parents=True, exist_ok=True)
    coverage_out.parent.mkdir(parents=True, exist_ok=True)
    log.write_csv(out)
    coverage.write_csv(coverage_out)
    return log, coverage
