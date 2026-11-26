"""`lob dq report`: row counts, coverage, duplicates, clock offsets and complete days."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast
from urllib.parse import unquote

import duckdb
import polars as pl
import pyarrow.parquet as pq

from lob_alpha.record import health
from lob_alpha.reports.gaplog import NS_PER_DAY, Stream, _ns, coverage_by_day, load_gaps

COMPLETE_PCT = 99.0  # DQ-10
DATASETS = ("book_l2", "trades", "bbo", "checkpoints", "clock")


def _query(sql: str) -> pl.DataFrame:
    relation = duckdb.sql(sql)
    assert relation is not None
    return cast(pl.DataFrame, relation.pl())


def row_counts(data_dir: Path, start: date, end: date) -> pl.DataFrame:
    """Rows per (dataset, venue, symbol, date) from Parquet footers only."""
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for dataset in DATASETS:
        for path in (data_dir / dataset).glob("venue=*/**/date=*/part-*.parquet"):
            parts = {p.split("=", 1)[0]: p.split("=", 1)[1] for p in path.parent.parts if "=" in p}
            day = parts.get("date", "")
            if not (str(start) <= day <= str(end)):
                continue
            symbol = unquote(parts.get("symbol", "*"))
            counts[(dataset, parts["venue"], symbol, day)] += pq.read_metadata(path).num_rows
    rows = [
        {"dataset": k[0], "venue": k[1], "symbol": k[2], "date": k[3], "rows": v}
        for k, v in sorted(counts.items())
    ]
    return pl.DataFrame(
        rows,
        schema={
            "dataset": pl.String,
            "venue": pl.String,
            "symbol": pl.String,
            "date": pl.String,
            "rows": pl.Int64,
        },
    )


def trade_duplicates(data_dir: Path, start: date, end: date) -> pl.DataFrame:
    """Rows minus distinct trade ids per day (DQ-09: the writer drops duplicates)."""
    glob = str(data_dir / "trades" / "**" / "*.parquet")
    sql = f"""
        select venue, symbol, date, count(*) as trades,
               count(*) - count(distinct trade_id) as duplicates
        from (select venue, symbol, trade_id,
                     regexp_extract(filename, 'date=([0-9-]+)', 1) as date
              from read_parquet('{glob}', hive_partitioning = false, filename = true))
        where date between '{start}' and '{end}'
        group by all order by all
    """
    return _query(sql)


def clock_offsets(data_dir: Path, start: date, end: date) -> pl.DataFrame:
    glob = str(data_dir / "clock" / "**" / "*.parquet")
    sql = f"""
        select venue, date, count(*) as probes,
               round(median(offset_ns) / 1e6, 2) as median_ms,
               round(quantile_cont(abs(offset_ns), 0.95) / 1e6, 2) as p95_abs_ms,
               round(max(abs(offset_ns)) / 1e6, 2) as max_abs_ms
        from (select venue, offset_ns, regexp_extract(filename, 'date=([0-9-]+)', 1) as date
              from read_parquet('{glob}', hive_partitioning = false, filename = true))
        where date between '{start}' and '{end}'
        group by all order by all
    """
    return _query(sql)


def utc_today() -> date:
    return datetime.now(UTC).date()


def complete_days(data_dir: Path, streams: list[Stream], start: date, end: date) -> pl.DataFrame:
    """Coverage per stream per day and the DQ-10 verdict (book and trades both >= 99%).

    A day counts as whole only when it starts at or after the capture start and has ended
    (UTC): coverage only sees logged gaps, so the hours before the first start, or the rest
    of a day still in progress, would otherwise look covered.
    """
    gaps = load_gaps(data_dir)
    capture = health.read_json(data_dir / health.CAPTURE_FILE) or {}
    first_ns = int(capture.get("started_ns", 0))
    lo = _ns(start)
    hi = min(_ns(end) + NS_PER_DAY, _ns(utc_today()))  # today has not ended yet
    daily = coverage_by_day(gaps, streams, lo, hi)
    pivot = (
        daily.filter(pl.col("stream").is_in(["book", "trades"]))
        .group_by("venue", "symbol", "date")
        .agg(
            pl.col("coverage_pct").filter(pl.col("stream") == "book").first().alias("book_pct"),
            pl.col("coverage_pct").filter(pl.col("stream") == "trades").first().alias("trades_pct"),
            pl.col("complete_window").all().alias("whole_day"),
        )
        .with_columns(
            (
                pl.col("whole_day")
                & (
                    pl.col("date")
                    .cast(pl.String)
                    .str.to_date()
                    .cast(pl.Datetime("ns"))
                    .dt.epoch("ns")
                    >= first_ns
                )
            ).alias("whole_day")
        )
        .with_columns(
            (
                pl.col("whole_day")
                & (pl.col("book_pct") >= COMPLETE_PCT)
                & (pl.col("trades_pct") >= COMPLETE_PCT)
            ).alias("complete")
        )
        .with_columns(pl.col("date").cast(pl.String))
        .sort("venue", "symbol", "date")
    )
    return pivot


def _md(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "_none_\n"
    cols = frame.columns
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in frame.iter_rows():
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def report(data_dir: Path, streams: list[Stream], start: date, end: date) -> str:
    counts = row_counts(data_dir, start, end)
    wide = counts.pivot(on="dataset", index=["venue", "symbol", "date"], values="rows").sort(
        "venue", "symbol", "date"
    )
    complete = complete_days(data_dir, streams, start, end)
    dups = trade_duplicates(data_dir, start, end)
    clock = clock_offsets(data_dir, start, end)
    n_complete = complete.filter(pl.col("complete")).height
    parts = [
        f"# Data-quality report, {start} to {end}\n",
        "Generated by `lob dq report`. Days are UTC (DQ-10). A complete day has the whole day "
        "inside the capture and at least 99% LIVE coverage for both book and trades. `clock` "
        "rows are not downtime (DQ-15).\n",
        f"## Complete days\n\n{n_complete} of {complete.height} symbol-days are complete.\n\n"
        f"{_md(complete)}",
        f"## Row counts\n\n{_md(wide)}",
        "## Trade duplicates (DQ-09)\n\nDuplicates are rows minus distinct trade ids per day. "
        f"The writer drops them, so this must be 0.\n\n{_md(dups)}",
        "## Clock offsets (DQ-15)\n\nVenue time minus host time per probe. "
        f"Negative means the host is fast.\n\n{_md(clock)}",
    ]
    return "\n".join(parts)
