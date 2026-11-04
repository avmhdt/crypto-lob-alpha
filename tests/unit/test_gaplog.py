"""Per-day coverage, the table the AC-W-01 check reads."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from lob_alpha import schemas
from lob_alpha.reports.gaplog import NS_PER_DAY, NS_PER_S, Stream, coverage_by_day

DAY = int(datetime(2026, 11, 3, tzinfo=UTC).timestamp()) * NS_PER_S
STREAM = [Stream("kraken", "BTC/USD", "book")]


def gaps(*rows: tuple[int, int, str]) -> pl.DataFrame:
    empty = pl.from_arrow(schemas.GAPS.empty_table())
    assert isinstance(empty, pl.DataFrame)
    extra = pl.DataFrame(
        [
            {
                "gap_id": f"g{i}",
                "venue": "kraken",
                "symbol": "BTC/USD",
                "stream": "book",
                "start_ns": s,
                "end_ns": e,
                "duration_ms": (e - s) / 1e6,
                "reason": reason,
                "seq_before": None,
                "seq_after": None,
                "host": "h",
                "recorder_version": "v",
            }
            for i, (s, e, reason) in enumerate(rows)
        ],
        schema=empty.schema,
    )
    return pl.concat([empty, extra])


def test_gap_across_midnight_is_split_between_the_two_days() -> None:
    table = coverage_by_day(
        gaps((DAY + NS_PER_DAY - 600 * NS_PER_S, DAY + NS_PER_DAY + 300 * NS_PER_S, "stale")),
        STREAM,
        DAY,
        DAY + 2 * NS_PER_DAY,
    )
    assert table["date"].to_list() == ["2026-11-03", "2026-11-04"]
    assert table["downtime_s"].to_list() == [600.0, 300.0]
    assert table["complete_window"].to_list() == [True, True]
    assert table["meets_99"].to_list() == [True, True]


def test_a_day_under_99_fails_and_clock_rows_are_not_downtime() -> None:
    table = coverage_by_day(
        gaps((DAY, DAY + 900 * NS_PER_S, "disconnect"), (DAY, DAY + NS_PER_DAY, "clock")),
        STREAM,
        DAY,
        DAY + NS_PER_DAY,
    )
    assert table.item(0, "downtime_s") == 900.0
    assert table.item(0, "coverage_pct") < 99.0
    assert table.item(0, "meets_99") is False


def test_a_partial_first_day_is_marked_incomplete() -> None:
    table = coverage_by_day(gaps(), STREAM, DAY + 3600 * NS_PER_S, DAY + NS_PER_DAY)
    assert table.item(0, "complete_window") is False
    assert table.item(0, "window_s") == 23 * 3600
