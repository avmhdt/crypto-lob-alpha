from __future__ import annotations

import time
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from lob_alpha import schemas
from lob_alpha.record.sync import GapRow
from lob_alpha.record.venues.base import BboEvent, TradeEvent
from lob_alpha.record.writer import ParquetSink, free_part, hour_stem, partition_dir
from lob_alpha.settings import WriterConfig

HOUR = 3_600_000_000_000
TS = 498_213 * HOUR + 123  # 2026-11-01T21:00
CFG = WriterConfig()


def book(sink: ParquetSink, ts: int = TS, symbol: str = "BTC/USD") -> None:
    sink.book_batch(
        "kraken", symbol, ts - 5, ts, 1, 2, True, None, None, 3,
        [(100.0, 1.0, "100.0", "1.00000000")],
        [(101.0, 2.0, "101.0", "2.00000000"), (102.0, 3.0, "102.0", "3.00000000")],
    )  # fmt: skip


def test_hour_and_date_come_from_ts_recv() -> None:
    assert hour_stem(TS // HOUR) == "part-21"
    path = partition_dir(Path("d"), "book_l2", "kraken", "BTC/USD", TS // HOUR)
    assert path.as_posix() == "d/book_l2/venue=kraken/symbol=BTC%2FUSD/date=2026-11-01"


def test_book_rows_land_in_an_atomically_renamed_hourly_file(tmp_path: Path) -> None:
    sink = ParquetSink(tmp_path, CFG)
    book(sink)
    sink.close()
    files = list(tmp_path.rglob("*.parquet"))
    assert [f.name for f in files] == ["part-21.parquet"]
    assert not list(tmp_path.rglob("*.tmp"))
    table = pq.read_table(files[0], partitioning=None)
    assert table.schema.equals(schemas.BOOK_L2)
    assert table.num_rows == 3
    assert table.column("side").to_pylist() == [1, -1, -1]
    assert table.column("px_str").to_pylist()[0] == "100.0"
    assert sink.files_closed == 1
    assert sink.rows_written == 3


def test_hive_partition_decodes_back_to_the_native_symbol(tmp_path: Path) -> None:
    sink = ParquetSink(tmp_path, CFG)
    book(sink)
    sink.close()
    dataset = ds.dataset(tmp_path / "book_l2", format="parquet", partitioning="hive")
    assert "BTC/USD" in str(dataset.to_table().column("symbol").unique().to_pylist())


def test_all_datasets_match_their_schemas(tmp_path: Path) -> None:
    sink = ParquetSink(tmp_path, CFG)
    sink.trade("kraken", TradeEvent("BTC/USD", TS, TS, "9", 1.0, 2.0, 1, False), 5)
    sink.bbo("kraken", BboEvent("BTC/USD", TS, TS, 1.0, 2.0, 3.0, 4.0, 7))
    sink.checkpoint("kraken", "BTC/USD", "c1", TS, "kraken_crc32", 1, 2, 3, 3, True, 20)
    sink.checkpoint_levels(
        "kraken", "BTC/USD", TS, "c1", [(1.0, 2.0, None, None)], [(3.0, 4.0, None, None)]
    )
    sink.clock("kraken", TS, TS + 1, 10, 1)
    sink.gap(GapRow("g", "kraken", "BTC/USD", "book", TS, TS + 1, 1.0, "stale", 1, 2, "h", "v"))
    sink.close()
    for name in ("trades", "bbo", "checkpoints", "checkpoint_levels", "clock", "gaps"):
        files = list((tmp_path / name).rglob("*.parquet"))
        assert len(files) == 1, name
        assert pq.read_table(files[0], partitioning=None).schema.equals(schemas.DATASETS[name])


def test_rows_split_by_hour(tmp_path: Path) -> None:
    sink = ParquetSink(tmp_path, CFG)
    book(sink, TS)
    book(sink, TS + HOUR)
    sink.close()
    assert sorted(f.name for f in tmp_path.rglob("*.parquet")) == [
        "part-21.parquet",
        "part-22.parquet",
    ]


def test_restart_within_the_hour_gets_an_r_suffix_and_skips_leftover_tmp(tmp_path: Path) -> None:
    first = ParquetSink(tmp_path, CFG)
    book(first)
    first.close()
    directory = next(tmp_path.rglob("part-21.parquet")).parent
    (directory / "part-21-r1.parquet.tmp").write_bytes(b"killed")
    second = ParquetSink(tmp_path, CFG)
    book(second)
    second.close()
    assert sorted(f.name for f in directory.glob("*.parquet")) == [
        "part-21-r2.parquet",
        "part-21.parquet",
    ]


def test_free_part_counts_up(tmp_path: Path) -> None:
    assert free_part(tmp_path, "part-01", "x").name == "part-01.parquet"
    (tmp_path / "part-01.parquet").touch()
    assert free_part(tmp_path, "part-01", "x").name == "part-01-r1.parquet"


def test_row_group_flushes_before_close(tmp_path: Path) -> None:
    sink = ParquetSink(tmp_path, WriterConfig(flush_interval_s=0.0))
    book(sink, time.time_ns())  # the current hour is not finished yet
    sink.tick()
    sink.wait_idle()
    assert sink.rows_written == 3
    assert list(tmp_path.rglob("*.parquet.tmp"))
    sink.close()


def test_hours_filter_drops_other_hours(tmp_path: Path) -> None:
    sink = ParquetSink(tmp_path, CFG, hours={TS // HOUR})
    book(sink, TS)
    book(sink, TS + HOUR)
    sink.close()
    assert [f.name for f in tmp_path.rglob("*.parquet")] == ["part-21.parquet"]


def test_writer_errors_are_counted_not_raised(tmp_path: Path) -> None:
    sink = ParquetSink(tmp_path, CFG)
    sink.book_batch(
        "kraken", "X", "bad", TS, 1, 2, True, None, None, 3, [(1.0, 1.0, None, None)], []
    )  # type: ignore[arg-type]
    sink.close()
    assert sink.flush_errors == 1
    assert sink.last_error


def test_gaps_can_be_switched_off(tmp_path: Path) -> None:
    sink = ParquetSink(tmp_path, CFG, write_gaps=False)
    sink.gap(GapRow("g", "kraken", "X", "book", TS, TS + 1, 1.0, "stale", None, None, "h", "v"))
    sink.close()
    assert not list(tmp_path.rglob("*.parquet"))


@pytest.mark.parametrize("name", ["part-21.parquet.tmp"])
def test_readers_skip_tmp_files(tmp_path: Path, name: str) -> None:
    sink = ParquetSink(tmp_path, CFG)
    book(sink)
    sink.close()
    directory = next(tmp_path.rglob("part-21.parquet")).parent
    (directory / name).write_bytes(b"partial")
    assert len(list(directory.glob("*.parquet"))) == 1
