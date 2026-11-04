"""The recorder against local fake venues with injected faults (AC-W-06)."""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
import yaml
from fake_venue import SYMBOL, FakeBinance, FakeKraken, instruments, parquet_rows, recorder_config
from typer.testing import CliRunner

from lob_alpha import cli
from lob_alpha.book.l2 import ASK, BID, BookL2
from lob_alpha.instruments import dump_instruments
from lob_alpha.record import health
from lob_alpha.record.raw_archive import read_hour
from lob_alpha.record.recover import active_run, interrupted_hours, recover
from lob_alpha.record.supervisor import Recorder
from lob_alpha.record.venues.base import SyncState
from lob_alpha.reports.gaplog import Stream, publish
from lob_alpha.settings import RecorderConfig, Settings

Body = Callable[[Recorder], Awaitable[None]]


@pytest.fixture
async def binance() -> AsyncIterator[FakeBinance]:
    venue = FakeBinance()
    await venue.start()
    yield venue
    await venue.stop()


@pytest.fixture
async def kraken() -> AsyncIterator[FakeKraken]:
    venue = FakeKraken()
    await venue.start()
    yield venue
    await venue.stop()


def settings() -> Settings:
    return Settings(_env_file=None)


async def wait_for(check: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.01)


def book_live(recorder: Recorder, venue: str, symbol: str) -> bool:
    runner = next(r for r in recorder.runners if r.venue == venue)
    return runner.pipeline.books[symbol].state is SyncState.LIVE


async def record(config: RecorderConfig, data: Path, body: Body) -> dict[str, object]:
    recorder = Recorder(config, settings(), instruments(), data_dir=data, auto_recover=False)
    run = asyncio.create_task(recorder.run())
    try:
        await body(recorder)
    finally:
        recorder.request_stop()
    return await run


def rebuild(root: Path) -> tuple[BookL2, list[dict[str, object]]]:
    """Replay `book_l2` rows the way the offline rebuild will, batch by batch."""
    rows = parquet_rows(root, "book_l2")
    rows.sort(key=lambda r: (r["ts_recv"], r["batch_id"]))
    book = BookL2(0.1)
    current: tuple[object, object] = (None, None)
    for row in rows:
        key = (row["batch_id"], row["is_snapshot"])
        if row["is_snapshot"] and key != current:
            book.clear()
        current = key
        book.apply(row["side"], row["price"], row["qty"])
    return book, rows


def assert_matches_truth(book: BookL2, venue: FakeBinance, last_seq: int) -> None:
    update_id, bids, asks = next(h for h in venue.history if h[0] == last_seq)
    assert update_id == last_seq
    for side, truth in ((BID, bids), (ASK, asks)):
        sign = -1 if side == BID else 1
        keys = sorted(truth, key=lambda t: sign * t)
        levels = book.top_n(len(keys))[0 if side == BID else 1]
        assert [round(float(p) * 10) for p, _ in levels] == keys
        assert [float(q) for _, q in levels] == [float(truth[k]) for k in keys]


async def test_clean_run_writes_every_dataset_and_rebuilds_to_truth(
    tmp_path: Path, binance: FakeBinance
) -> None:
    async def body(recorder: Recorder) -> None:
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        await asyncio.sleep(1.0)

    final = await record(recorder_config({"binance-usdm": binance.config()}), tmp_path, body)
    assert final["writer"]["flush_errors"] == 0
    assert not list(tmp_path.rglob("*.parquet.tmp"))
    assert (
        not list((tmp_path / "gaps").rglob("*.parquet")) if (tmp_path / "gaps").exists() else True
    )
    book, rows = rebuild(tmp_path)
    assert_matches_truth(book, binance, max(r["seq"] for r in rows))
    trades = parquet_rows(tmp_path, "trades")
    ids = [int(r["trade_id"]) for r in trades]
    assert ids == sorted(set(ids))
    assert all(r["is_aggregated"] for r in trades)
    assert parquet_rows(tmp_path, "bbo")
    assert parquet_rows(tmp_path, "clock")
    checkpoints = parquet_rows(tmp_path, "checkpoints")
    assert {r["kind"] for r in checkpoints} == {"rest_l2"}
    assert all(r["ok_live"] for r in checkpoints)
    assert parquet_rows(tmp_path, "checkpoint_levels")
    capture = json.loads((tmp_path / "capture.json").read_text())
    assert capture["started_ns"] > 0


async def test_disconnect_logs_a_gap_and_resyncs_without_a_restart(
    tmp_path: Path, binance: FakeBinance
) -> None:
    async def body(recorder: Recorder) -> None:
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        await asyncio.sleep(0.2)
        await binance.drop_public()
        await asyncio.sleep(0.1)
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        await asyncio.sleep(0.3)

    await record(recorder_config({"binance-usdm": binance.config()}), tmp_path, body)
    gaps = parquet_rows(tmp_path, "gaps")
    assert {(g["stream"], g["reason"]) for g in gaps} == {("book", "disconnect")}
    book_rows = parquet_rows(tmp_path, "book_l2")
    assert max(r["sync_epoch"] for r in book_rows) == 2
    book, rows = rebuild(tmp_path)
    assert_matches_truth(book, binance, max(r["seq"] for r in rows))


async def test_dropping_the_ticker_connection_leaves_the_book_alone(
    tmp_path: Path, binance: FakeBinance
) -> None:
    async def body(recorder: Recorder) -> None:
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        await asyncio.sleep(0.2)
        await binance.drop_ticker()
        await asyncio.sleep(0.5)
        assert book_live(recorder, "binance-usdm", SYMBOL)

    await record(recorder_config({"binance-usdm": binance.config()}), tmp_path, body)
    gaps = parquet_rows(tmp_path, "gaps")
    assert {(g["stream"], g["reason"]) for g in gaps} == {("bbo", "disconnect")}
    assert {r["sync_epoch"] for r in parquet_rows(tmp_path, "book_l2")} == {1}


async def test_silence_triggers_the_watchdog(tmp_path: Path, binance: FakeBinance) -> None:
    async def body(recorder: Recorder) -> None:
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        await asyncio.sleep(0.2)
        binance.silent = True
        await wait_for(lambda: not book_live(recorder, "binance-usdm", SYMBOL), 4.0)
        await asyncio.sleep(0.1)
        binance.silent = False
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL), 8.0)
        await asyncio.sleep(0.3)

    await record(recorder_config({"binance-usdm": binance.config()}), tmp_path, body)
    assert {"stale"} == {
        g["reason"] for g in parquet_rows(tmp_path, "gaps") if g["stream"] == "book"
    }


async def test_sequence_gap_resyncs_one_stream(tmp_path: Path, binance: FakeBinance) -> None:
    async def body(recorder: Recorder) -> None:
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        await asyncio.sleep(0.2)
        binance.skip_next = True
        await asyncio.sleep(0.2)
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        await asyncio.sleep(0.2)

    await record(recorder_config({"binance-usdm": binance.config()}), tmp_path, body)
    gaps = parquet_rows(tmp_path, "gaps")
    assert [(g["stream"], g["reason"]) for g in gaps] == [("book", "seq_gap")]
    assert gaps[0]["seq_after"] > gaps[0]["seq_before"]
    book, rows = rebuild(tmp_path)
    assert_matches_truth(book, binance, max(r["seq"] for r in rows))


async def test_stale_snapshot_is_retried_until_it_bridges(
    tmp_path: Path, binance: FakeBinance
) -> None:
    binance.stale_snapshots = 2

    async def body(recorder: Recorder) -> None:
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL), 8.0)
        await asyncio.sleep(0.2)

    await record(recorder_config({"binance-usdm": binance.config()}), tmp_path, body)
    assert binance.http.hits["/fapi/v1/depth"] >= 3


async def test_rest_failure_recovers_when_the_venue_returns(
    tmp_path: Path, binance: FakeBinance
) -> None:
    binance.http.status = 500

    async def body(recorder: Recorder) -> None:
        await asyncio.sleep(0.3)
        assert not book_live(recorder, "binance-usdm", SYMBOL)
        binance.http.status = 200
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL), 10.0)

    await record(recorder_config({"binance-usdm": binance.config()}), tmp_path, body)


async def test_checksum_failure_resubscribes_kraken(tmp_path: Path, kraken: FakeKraken) -> None:
    async def body(recorder: Recorder) -> None:
        await wait_for(lambda: book_live(recorder, "kraken", "BTC/USD"))
        await asyncio.sleep(0.2)
        kraken.corrupt_next = True
        await asyncio.sleep(0.2)
        await wait_for(lambda: book_live(recorder, "kraken", "BTC/USD"))
        await asyncio.sleep(0.3)

    await record(recorder_config({"kraken": kraken.config()}), tmp_path, body)
    gaps = parquet_rows(tmp_path, "gaps")
    assert [(g["stream"], g["reason"]) for g in gaps] == [("book", "checksum_fail")]
    checks = parquet_rows(tmp_path, "checkpoints")
    assert [r["ok_live"] for r in checks].count(False) == 1
    assert parquet_rows(tmp_path, "clock")


async def test_make_before_break_has_zero_gap_and_no_duplicates(
    tmp_path: Path, binance: FakeBinance
) -> None:
    config = recorder_config({"binance-usdm": binance.config()}, proactive_reconnect_h=0.0005)

    async def body(recorder: Recorder) -> None:
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        runner = recorder.runners[0]
        await wait_for(lambda: runner.generation >= 6, 12.0)  # all three connections replaced
        await wait_for(lambda: not runner.pipeline.standby and len(runner.sockets) == 3, 12.0)
        await asyncio.sleep(0.5)

    await record(config, tmp_path, body)
    assert not (tmp_path / "gaps").exists() or not parquet_rows(tmp_path, "gaps")
    rows = parquet_rows(tmp_path, "book_l2")
    assert sum(1 for r in rows if r["is_snapshot"]) > 0
    assert {r["sync_epoch"] for r in rows} == {1}  # never resynced
    book, rows = rebuild(tmp_path)
    assert_matches_truth(book, binance, max(r["seq"] for r in rows))
    diffs: dict[int, tuple[object, object]] = {}
    for row in rows:
        if not row["is_snapshot"]:
            diffs[row["seq"]] = (row["seq_first"], row["seq_prev"])
    ordered = sorted(diffs)
    for before, after in itertools.pairwise(ordered):
        assert diffs[after][1] == before  # the pu chain is unbroken across the switch
    trades = [int(r["trade_id"]) for r in parquet_rows(tmp_path, "trades")]
    assert len(trades) == len(set(trades))
    assert sorted(trades) == list(range(min(trades), max(trades) + 1))


async def test_recover_rebuilds_row_identical_files(tmp_path: Path, binance: FakeBinance) -> None:
    async def body(recorder: Recorder) -> None:
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        await asyncio.sleep(0.3)
        binance.skip_next = True
        await asyncio.sleep(0.3)
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        await asyncio.sleep(0.5)

    data = tmp_path / "live"
    config = recorder_config({"binance-usdm": binance.config()})
    await record(config, data, body)
    now = datetime.now(tz=UTC)
    out = tmp_path / "rebuilt"
    result = recover(
        config, instruments(), data, "binance-usdm", now.date(), [now.hour], out_dir=out
    )
    assert result.records > 0
    assert result.written
    for dataset in ("book_l2", "trades", "bbo", "checkpoints", "checkpoint_levels", "clock"):
        assert parquet_rows(out, dataset) == parquet_rows(data, dataset), dataset


async def test_recover_refuses_closed_files_unless_replace(
    tmp_path: Path, binance: FakeBinance
) -> None:
    async def body(recorder: Recorder) -> None:
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        await asyncio.sleep(0.3)

    config = recorder_config({"binance-usdm": binance.config()})
    await record(config, tmp_path, body)
    now = datetime.now(tz=UTC)
    before = parquet_rows(tmp_path, "book_l2")
    first = recover(config, instruments(), tmp_path, "binance-usdm", now.date(), [now.hour])
    assert first.skipped
    assert not first.written
    second = recover(
        config, instruments(), tmp_path, "binance-usdm", now.date(), [now.hour], replace=True
    )
    assert second.written
    assert parquet_rows(tmp_path, "book_l2") == before


async def test_restart_logs_process_restart_and_uses_an_r_part(
    tmp_path: Path, binance: FakeBinance
) -> None:
    async def body(recorder: Recorder) -> None:
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        await asyncio.sleep(0.3)

    config = recorder_config({"binance-usdm": binance.config()})
    await record(config, tmp_path, body)
    await asyncio.sleep(1.1)  # run ids have one-second resolution
    await record(config, tmp_path, body)
    gaps = parquet_rows(tmp_path, "gaps")
    assert {g["reason"] for g in gaps} == {"process_restart"}
    assert {g["stream"] for g in gaps} == {"book", "trades", "bbo"}
    book_files = sorted(p.name for p in (tmp_path / "book_l2").rglob("*.parquet"))
    assert len(book_files) == 2
    assert any("-r1" in name for name in book_files) or len({n[:7] for n in book_files}) == 2
    rows = parquet_rows(tmp_path, "book_l2")
    assert max(r["sync_epoch"] for r in rows) == 2  # epochs continue across runs


async def test_disk_guard_stops_capture_and_resumes(
    tmp_path: Path, binance: FakeBinance, monkeypatch: pytest.MonkeyPatch
) -> None:
    free = [100.0]

    def fake_disk(_path: Path) -> health.DiskState:
        return health.DiskState(free[0], 85.0, 200.0)

    monkeypatch.setattr(health, "disk_state", fake_disk)
    config = recorder_config(
        {"binance-usdm": binance.config()}, hard_stop_free_gb=5.0, min_free_gb=20.0
    )

    async def body(recorder: Recorder) -> None:
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        await asyncio.sleep(0.2)
        free[0] = 10.0
        await wait_for(lambda: recorder.raw.paused, 3.0)
        free[0] = 2.0
        await wait_for(lambda: recorder.disk_guarded, 3.0)
        await wait_for(lambda: not recorder.runners[0].sockets, 3.0)
        free[0] = 100.0
        await wait_for(lambda: not recorder.disk_guarded, 3.0)
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL), 5.0)
        assert not recorder.raw.paused
        await asyncio.sleep(0.2)

    await record(config, tmp_path, body)
    gaps = parquet_rows(tmp_path, "gaps")
    assert {g["reason"] for g in gaps} == {"disk_guard"}


async def test_status_cli_and_gaplog(
    tmp_path: Path, binance: FakeBinance, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def body(recorder: Recorder) -> None:
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        await asyncio.sleep(0.2)
        binance.skip_next = True
        await asyncio.sleep(0.2)
        await wait_for(lambda: book_live(recorder, "binance-usdm", SYMBOL))
        await wait_for(lambda: health.read_status(tmp_path) is not None)
        status = health.read_status(tmp_path)
        assert status is not None
        assert status["run"] == recorder.run_id
        assert active_run(tmp_path) == recorder.run_id
        await asyncio.sleep(0.2)

    await record(recorder_config({"binance-usdm": binance.config()}), tmp_path, body)
    assert active_run(tmp_path) is None
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    runner = CliRunner()
    outcome = runner.invoke(cli.app, ["record", "status", "--disk"])
    assert outcome.exit_code == 0, outcome.output
    assert "BTCUSDT" in outcome.output
    assert "capture_started_utc" in outcome.output
    outcome = runner.invoke(cli.app, ["record", "status", "--json"])
    assert json.loads(outcome.output)["status"]["run"]
    today = datetime.now(tz=UTC).date()
    log, coverage = publish(
        tmp_path,
        [Stream("binance-usdm", SYMBOL, "book")],
        today,
        today,
        tmp_path / "gap_log.csv",
        tmp_path / "coverage.csv",
    )
    assert log.height == 1
    assert log.item(0, "start_utc").endswith("Z")
    assert coverage.item(0, "coverage_pct") < 100
    assert coverage.item(0, "n_gaps") == 1
    by_day = tmp_path / "by_day.csv"
    publish(
        tmp_path,
        [Stream("binance-usdm", SYMBOL, "book")],
        today,
        today,
        tmp_path / "gap_log.csv",
        tmp_path / "coverage.csv",
        by_day,
    )
    daily = pl.read_csv(by_day)
    assert daily.height == 1
    assert daily.item(0, "date") == today.isoformat()
    assert daily.item(0, "complete_window") is False
    assert daily.item(0, "n_gaps") == 1


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL is POSIX")
async def test_kill_nine_loses_at_most_a_second_and_recover_restores(
    tmp_path: Path, binance: FakeBinance
) -> None:
    """NFR-02. A real process, a real SIGKILL, and a real second run afterwards."""
    data = tmp_path / "data"
    data.mkdir()
    cwd = tmp_path / "cwd"
    (cwd / "config").mkdir(parents=True)
    dump_instruments(instruments(), cwd / "config" / "instruments.yaml")
    config = recorder_config({"binance-usdm": binance.config()})
    config_path = cwd / "config" / "recorder.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    env = {**os.environ, "DATA_ROOT": str(data), "LOB_VENUE_SET": "binance-usdm"}
    script = textwrap.dedent(
        """
        import asyncio, sys
        from pathlib import Path
        from lob_alpha.instruments import load_instruments
        from lob_alpha.record.supervisor import Recorder
        from lob_alpha.settings import Settings, load_recorder_config
        settings = Settings(_env_file=None)
        config = load_recorder_config(Path("config/recorder.yaml"), settings)
        recorder = Recorder(config, settings, load_instruments(), auto_recover=False)
        asyncio.run(recorder.run())
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script], cwd=cwd, env=env, stdout=subprocess.DEVNULL
    )
    try:
        await wait_for(
            lambda: (
                (health.read_status(data) or {})
                .get("venues", {})
                .get("binance-usdm", {})
                .get("frames", 0)
                > 150
            ),
            15.0,
        )
        await asyncio.sleep(1.5)
        process.send_signal(signal.SIGKILL)
        killed_ns = time.time_ns()
        process.wait(timeout=5)
    finally:
        process.kill()
    now = datetime.now(tz=UTC)
    assert list(data.rglob("*.parquet.tmp"))
    records = list(read_hour(data, "binance-usdm", now.date(), now.hour))
    frames = [r for r in records if r.ctrl is None]
    assert frames
    assert killed_ns - max(r.ts for r in frames) < 1_500_000_000
    assert interrupted_hours(data) == [("binance-usdm", now.date(), now.hour)]
    status = health.read_status(data)
    assert status is not None
    status["ts_ns"] = 0
    health.write_json_atomic(data / health.STATUS_FILE, status)
    result = recover(config, instruments(), data, "binance-usdm", now.date(), [now.hour])
    assert result.written
    assert not list((data / "book_l2").rglob("*.parquet.tmp"))
    rows = parquet_rows(data, "book_l2")
    assert rows
    stamps = [r["ts_recv"] for r in rows]
    assert killed_ns - max(stamps) < 1_500_000_000
    book = BookL2(0.1)
    for row in sorted(rows, key=lambda r: (r["ts_recv"], r["batch_id"])):
        book.apply(row["side"], row["price"], row["qty"])
    assert_matches_truth_partial(book, binance, max(r["seq"] for r in rows))


def assert_matches_truth_partial(book: BookL2, venue: FakeBinance, last_seq: int) -> None:
    _, bids, asks = next(h for h in venue.history if h[0] == last_seq)
    assert round(book.best_bid()[0] * 10) == max(bids)
    assert round(book.best_ask()[0] * 10) == min(asks)
