"""`VenueRunner` driven directly with synthetic times, with no network at all.

This is how the top-of-hour state record gets tested: real time cannot be made to cross an
hour inside a test, but `frame` and `control` take the timestamp as an argument.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import orjson
import pytest
from fake_venue import SYMBOL, FakeBinance, FakeKraken, instruments, parquet_rows, recorder_config
from typer.testing import CliRunner

from conftest import frames, kraken_cfg
from lob_alpha import cli
from lob_alpha.instruments import dump_instruments
from lob_alpha.record import health
from lob_alpha.record.raw_archive import prune, read_hour
from lob_alpha.record.recover import interrupted_hours, recover
from lob_alpha.record.supervisor import RateLimiter, Recorder, VenueRunner
from lob_alpha.record.venues.base import SyncState
from lob_alpha.record.venues.coinbase import Coinbase
from lob_alpha.settings import Settings, VenueConfig

SEC = 1_000_000_000
HOUR_END = int(datetime(2026, 11, 2, 15, 0, 0, tzinfo=UTC).timestamp()) * SEC
DAY = date(2026, 11, 2)


def settings() -> Settings:
    return Settings(_env_file=None)


def offline_recorder(tmp_path: Path, venue: str, cfg: VenueConfig) -> tuple[Recorder, VenueRunner]:
    config = recorder_config({venue: cfg})
    recorder = Recorder(config, settings(), instruments(), data_dir=tmp_path, auto_recover=False)
    return recorder, recorder.runners[0]


async def close(recorder: Recorder, runner: VenueRunner) -> None:
    await runner.http.aclose()
    recorder.sink.close()
    recorder.raw.close()


def depth(index: int) -> bytes:
    data = {
        "e": "depthUpdate", "E": 1, "T": 1, "s": SYMBOL, "U": index * 10 + 1,
        "u": index * 10 + 10, "pu": index * 10, "b": [["100.0", str(index + 1)]], "a": [],
    }  # fmt: skip
    return orjson.dumps({"stream": "x", "data": data})


async def test_hour_rollover_state_lets_one_hour_replay_alone(tmp_path: Path) -> None:
    """Binance across a top of hour: recover hour 15 alone and match the live file."""
    cfg = FakeBinance().config()
    recorder, runner = offline_recorder(tmp_path / "live", "binance-usdm", cfg)
    runner.hour = HOUR_END // (3600 * SEC) - 1
    runner.raw.control(runner.venue, HOUR_END - 60 * SEC, {"k": "run_start", "streams": {}})
    runner.pipeline.seed({}, "process_restart")
    start = {"k": "conn_open", "conn": "public.g0", "logical": "public"}
    market = {"k": "conn_open", "conn": "market.g0", "logical": "market"}
    runner.control(start, ts=HOUR_END - 50 * SEC)
    runner.control(market, ts=HOUR_END - 50 * SEC)
    runner.frame("public.g0", HOUR_END - 40 * SEC, depth(1))
    body = orjson.dumps({"lastUpdateId": 15, "T": 1, "bids": [], "asks": [["101.0", "1"]]})
    runner.control({"k": "snapshot", "symbol": SYMBOL}, body, HOUR_END - 30 * SEC)
    assert runner.pipeline.books[SYMBOL].state is SyncState.LIVE
    trade = {"e": "aggTrade", "T": 1, "a": 7, "s": SYMBOL, "p": "1", "q": "1", "m": False}
    runner.frame("market.g0", HOUR_END - 20 * SEC, orjson.dumps({"data": trade}))
    runner.frame("public.g0", HOUR_END - 10 * SEC, depth(2))
    for index in range(3, 8):
        runner.frame("public.g0", HOUR_END + index * SEC, depth(index))
    runner.frame("market.g0", HOUR_END + 9 * SEC, orjson.dumps({"data": {**trade, "a": 8}}))
    await close(recorder, runner)
    records = list(read_hour(tmp_path / "live", "binance-usdm", DAY, 15))
    assert records[0].ctrl is not None
    assert records[0].ctrl["k"] == "state"
    out = tmp_path / "rebuilt"
    config = recorder.config
    result = recover(
        config, instruments(), tmp_path / "live", "binance-usdm", DAY, [15], out_dir=out
    )
    assert result.written
    live = [r for r in parquet_rows(tmp_path / "live", "book_l2") if r["ts_recv"] >= HOUR_END]
    assert parquet_rows(out, "book_l2") == live
    trades = [r for r in parquet_rows(tmp_path / "live", "trades") if r["ts_recv"] >= HOUR_END]
    assert parquet_rows(out, "trades") == trades
    assert not list((out / "gaps").rglob("*")) if (out / "gaps").exists() else True


async def test_kraken_book_survives_the_state_record(tmp_path: Path) -> None:
    """The checksum needs the book itself, so the state record must carry it exactly."""
    recorder, runner = offline_recorder(tmp_path / "live", "kraken", kraken_cfg())
    runner.hour = HOUR_END // (3600 * SEC) - 1
    lines = frames("kraken.jsonl")
    middle = len(lines) // 2
    runner.control({"k": "conn_open", "conn": "ws.g0", "logical": "ws"}, ts=HOUR_END - 50 * SEC)
    for i, raw in enumerate(lines[:middle]):
        runner.frame("ws.g0", HOUR_END - 40 * SEC + i, raw)
    assert runner.pipeline.books["BTC/USD"].state is SyncState.LIVE
    for i, raw in enumerate(lines[middle:]):
        runner.frame("ws.g0", HOUR_END + i, raw)
    await close(recorder, runner)
    assert runner.pipeline.books["BTC/USD"].resyncs == 0
    out = tmp_path / "rebuilt"
    recover(recorder.config, instruments(), tmp_path / "live", "kraken", DAY, [15], out_dir=out)
    live = [r for r in parquet_rows(tmp_path / "live", "checkpoints") if r["ts_recv"] >= HOUR_END]
    rebuilt = parquet_rows(out, "checkpoints")
    assert rebuilt == live
    assert rebuilt
    assert all(r["ok_live"] for r in rebuilt)


async def test_parse_error_on_rest_body_becomes_snapshot_error(tmp_path: Path) -> None:
    recorder, runner = offline_recorder(tmp_path, "binance-usdm", FakeBinance().config())
    runner.control({"k": "conn_open", "conn": "public.g0", "logical": "public"})
    runner.control({"k": "snapshot", "symbol": SYMBOL}, b'{"code":-1003}')
    book = runner.pipeline.books[SYMBOL]
    assert book.state is SyncState.SNAPSHOT
    await close(recorder, runner)


async def test_a_late_periodic_status_cannot_undo_the_final_one(tmp_path: Path) -> None:
    recorder, runner = offline_recorder(tmp_path, "binance-usdm", FakeBinance().config())
    recorder.write_status({"stopped": False})
    recorder.write_status({"stopped": True}, final=True)
    recorder.write_status({"stopped": False})
    assert health.read_status(tmp_path) == {"stopped": True}
    await close(recorder, runner)


async def test_rate_limiter_holds_after_a_penalty() -> None:
    limiter = RateLimiter(1, 1000)
    await limiter.acquire()
    limiter.penalise(0.05)
    started = asyncio.get_running_loop().time()
    await limiter.acquire()
    assert asyncio.get_running_loop().time() - started >= 0.04


async def test_http_429_blocks_further_rest_calls(tmp_path: Path) -> None:
    recorder, runner = offline_recorder(tmp_path, "binance-usdm", FakeBinance().config())
    response = httpx.Response(
        429, headers={"Retry-After": "120"}, request=httpx.Request("GET", "http://x")
    )
    runner._respect_limits(
        httpx.HTTPStatusError("slow down", request=response.request, response=response)
    )
    assert runner.limiter.blocked_until > asyncio.get_running_loop().time()
    await close(recorder, runner)


async def test_coinbase_rest_checkpoint_and_clock() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/time":
            return httpx.Response(200, json={"iso": "x", "epoch": 1793567025.462})
        assert request.url.params["level"] == "2"
        return httpx.Response(
            200,
            json={
                "bids": [["1.0", "2.0", 3]],
                "asks": [["4.0", "5.0", 6]],
                "sequence": 7,
                "time": "2026-11-01T21:03:45.462000Z",
            },
        )

    adapter = Coinbase(VenueConfig(rest="https://x", symbols=["BTC-USD"]))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        snapshot = await adapter.fetch_checkpoint(http, "BTC-USD")
        sample = await adapter.fetch_server_time(http)
    assert snapshot is not None
    assert snapshot.seq == 7
    assert snapshot.raw
    assert sample is not None
    assert sample.server_time_ns == 1793567025462 * 1_000_000
    assert sample.ts_recv >= sample.sent_ns


async def test_clock_probe_keeps_connection_setup_out_of_the_rtt() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"epoch": 1793567025.462})

    adapter = Coinbase(VenueConfig(rest="https://x", symbols=["BTC-USD"]))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        await adapter.fetch_server_time(http)
    assert calls == 2


async def test_clock_probe_failure_on_the_warm_up_call_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    adapter = Coinbase(VenueConfig(rest="https://x", symbols=["BTC-USD"]))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.fetch_server_time(http)


def write_config(cwd: Path, venue: str, cfg: VenueConfig) -> Path:
    import yaml

    (cwd / "config").mkdir(parents=True, exist_ok=True)
    dump_instruments(instruments(), cwd / "config" / "instruments.yaml")
    config = recorder_config({venue: cfg})
    path = cwd / "config" / "recorder.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    return path


def test_cli_recover_interrupted_and_gaplog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()

    async def body() -> None:
        recorder, runner = offline_recorder(data, "kraken", kraken_cfg())
        runner.hour = HOUR_END // (3600 * SEC)
        runner.control({"k": "conn_open", "conn": "ws.g0", "logical": "ws"}, ts=HOUR_END + 1)
        for i, raw in enumerate(frames("kraken.jsonl")):
            runner.frame("ws.g0", HOUR_END + 2 + i, raw)
        await close(recorder, runner)

    asyncio.run(body())
    closed = next((data / "book_l2").rglob("*.parquet"))
    closed.rename(closed.with_name(closed.name + ".tmp"))  # as a kill would leave it
    assert interrupted_hours(data) == [("kraken", DAY, 15)]
    config = write_config(tmp_path / "cwd", "kraken", kraken_cfg())
    monkeypatch.setenv("DATA_ROOT", str(data))
    monkeypatch.setenv("LOB_VENUE_SET", "kraken")
    monkeypatch.chdir(tmp_path / "cwd")
    runner = CliRunner()
    args = ["record", "recover", "--from-raw", "--config", str(config)]
    outcome = runner.invoke(cli.app, [*args, "--interrupted"])
    assert outcome.exit_code == 0, outcome.output
    assert "recovered" in outcome.output
    assert closed.exists()
    outcome = runner.invoke(
        cli.app, [*args, "--venue", "kraken", "--date", "2026-11-02", "--hour", "15"]
    )
    assert "kept closed file" in outcome.output
    assert runner.invoke(cli.app, args).exit_code != 0
    assert runner.invoke(cli.app, ["record", "recover", "--config", str(config)]).exit_code != 0
    outcome = runner.invoke(
        cli.app,
        [
            "record", "gaplog", "--start", "2026-11-02", "--end", "2026-11-02",
            "--out", str(tmp_path / "g.csv"), "--coverage-out", str(tmp_path / "c.csv"),
            "--config", str(config),
        ],
    )  # fmt: skip
    assert outcome.exit_code == 0, outcome.output
    assert (tmp_path / "g.csv").exists()
    assert (tmp_path / "c.csv").exists()


def test_a_closed_file_with_no_footer_is_found_and_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a VM shutdown left on 2026-11-02: a final name on a file missing its last pages."""
    data = tmp_path / "data"
    data.mkdir()

    async def body() -> None:
        recorder, runner = offline_recorder(data, "kraken", kraken_cfg())
        runner.hour = HOUR_END // (3600 * SEC)
        runner.control({"k": "conn_open", "conn": "ws.g0", "logical": "ws"}, ts=HOUR_END + 1)
        for i, raw in enumerate(frames("kraken.jsonl")):
            runner.frame("ws.g0", HOUR_END + 2 + i, raw)
        await close(recorder, runner)

    asyncio.run(body())
    closed = next((data / "book_l2").rglob("*.parquet"))
    before = parquet_rows(data, "book_l2")
    closed.write_bytes(closed.read_bytes()[:4096])
    assert interrupted_hours(data) == [("kraken", DAY, 15)]
    config = write_config(tmp_path / "cwd", "kraken", kraken_cfg())
    monkeypatch.setenv("DATA_ROOT", str(data))
    monkeypatch.setenv("LOB_VENUE_SET", "kraken")
    monkeypatch.chdir(tmp_path / "cwd")
    args = ["record", "recover", "--from-raw", "--interrupted", "--config", str(config)]
    outcome = CliRunner().invoke(cli.app, args)
    assert outcome.exit_code == 0, outcome.output
    assert parquet_rows(data, "book_l2") == before
    assert interrupted_hours(data) == []


def test_status_without_a_status_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert CliRunner().invoke(cli.app, ["record", "status"]).exit_code == 1


def test_disk_report_projects_to_the_planned_end(tmp_path: Path) -> None:
    now = datetime.now(tz=UTC)
    started = now.timestamp() - 2 * 86_400
    health.write_json_atomic(
        tmp_path / health.CAPTURE_FILE,
        {"started_ns": int(started * 1e9), "started_utc": "x"},
    )
    day = tmp_path / "book_l2" / "venue=v" / "symbol=s" / f"date={now.date()}"
    day.mkdir(parents=True)
    (day / "part-00.parquet").write_bytes(b"x" * 2048)
    raw = tmp_path / "raw" / "ws" / "venue=v" / f"date={now.date()}"
    raw.mkdir(parents=True)
    (raw / "00-a-b.jsonl.zst").write_bytes(b"x" * 1024)
    report = health.disk_report(tmp_path, 28, 7)
    assert report["rate_is_provisional"] is False
    assert report["canonical_gb_per_day"] >= 0
    assert report["projected_free_gb_at_end"] <= report["free_gb"]
    assert report["days"][now.date().isoformat()]["raw"] >= 0


def test_prune_handles_an_empty_archive(tmp_path: Path) -> None:
    assert prune(tmp_path, 7, DAY) == []


async def test_standby_abandoned_when_it_never_links(tmp_path: Path) -> None:
    recorder, runner = offline_recorder(tmp_path, "binance-usdm", FakeBinance().config())
    pipeline = runner.pipeline
    t0 = HOUR_END
    runner.hour = t0 // (3600 * SEC)
    runner.control({"k": "conn_open", "conn": "public.g0", "logical": "public"}, ts=t0)
    runner.frame("public.g0", t0 + 1, depth(1))
    runner.control(
        {"k": "snapshot", "symbol": SYMBOL},
        orjson.dumps({"lastUpdateId": 15, "T": 1, "bids": [], "asks": []}),
        t0 + 2,
    )
    ctrl = {"k": "conn_open", "conn": "public.g1", "logical": "public", "standby": True}
    runner.control(ctrl, ts=t0 + 3)
    assert pipeline.standby == {"public": "public.g1"}
    later = t0 + 121 * SEC
    pipeline.conns["public.g0"].last_frame_ns = later
    pipeline.conns["public.g1"].last_frame_ns = later
    pipeline.books[SYMBOL].last_msg_ns = later
    _, actions = pipeline.timers(later)
    assert [a.conn for a in actions] == ["public.g1"]
    runner.control(
        {"k": "conn_close", "conn": "public.g1", "reason": "standby_abandoned"}, ts=later
    )
    assert pipeline.standby == {}
    assert pipeline.books[SYMBOL].overlap is False
    assert pipeline.books[SYMBOL].state is SyncState.LIVE
    await close(recorder, runner)


async def test_primary_gap_during_overlap_still_resyncs(tmp_path: Path) -> None:
    recorder, runner = offline_recorder(tmp_path, "binance-usdm", FakeBinance().config())
    pipeline = runner.pipeline
    t0 = HOUR_END
    runner.hour = t0 // (3600 * SEC)
    runner.control({"k": "conn_open", "conn": "public.g0", "logical": "public"}, ts=t0)
    runner.frame("public.g0", t0 + 1, depth(1))
    runner.control(
        {"k": "snapshot", "symbol": SYMBOL},
        orjson.dumps({"lastUpdateId": 15, "T": 1, "bids": [], "asks": []}),
        t0 + 2,
    )
    ctrl = {"k": "conn_open", "conn": "public.g1", "logical": "public", "standby": True}
    runner.control(ctrl, ts=t0 + 3)
    runner.frame("public.g1", t0 + 4, depth(2))  # links the chain
    runner.frame("public.g0", t0 + 5, depth(2))  # duplicate from the old connection
    runner.frame("public.g0", t0 + 6, depth(4))  # the primary skipped an event: a real gap
    assert pipeline.books[SYMBOL].state is SyncState.RESYNC
    await close(recorder, runner)


async def test_active_dying_during_overlap_promotes_a_linked_standby(tmp_path: Path) -> None:
    recorder, runner = offline_recorder(tmp_path, "binance-usdm", FakeBinance().config())
    pipeline = runner.pipeline
    t0 = HOUR_END
    runner.hour = t0 // (3600 * SEC)
    runner.control({"k": "conn_open", "conn": "public.g0", "logical": "public"}, ts=t0)
    runner.frame("public.g0", t0 + 1, depth(1))
    runner.control(
        {"k": "snapshot", "symbol": SYMBOL},
        orjson.dumps({"lastUpdateId": 15, "T": 1, "bids": [], "asks": []}),
        t0 + 2,
    )
    ctrl = {"k": "conn_open", "conn": "public.g1", "logical": "public", "standby": True}
    runner.control(ctrl, ts=t0 + 3)
    runner.frame("public.g1", t0 + 4, depth(2))
    assert pipeline.books[SYMBOL].linked
    runner.control({"k": "conn_close", "conn": "public.g0", "reason": "disconnect"}, ts=t0 + 5)
    assert pipeline.active == {"public": "public.g1"}
    assert pipeline.books[SYMBOL].state is SyncState.LIVE
    runner.frame("public.g1", t0 + 6, depth(3))
    assert pipeline.books[SYMBOL].last_seq == 40
    await close(recorder, runner)
    assert not (tmp_path / "gaps").exists()


async def test_active_dying_before_the_standby_links_resyncs_on_it(tmp_path: Path) -> None:
    recorder, runner = offline_recorder(tmp_path, "binance-usdm", FakeBinance().config())
    pipeline = runner.pipeline
    t0 = HOUR_END
    runner.hour = t0 // (3600 * SEC)
    runner.control({"k": "conn_open", "conn": "public.g0", "logical": "public"}, ts=t0)
    runner.frame("public.g0", t0 + 1, depth(1))
    runner.control(
        {"k": "snapshot", "symbol": SYMBOL},
        orjson.dumps({"lastUpdateId": 15, "T": 1, "bids": [], "asks": []}),
        t0 + 2,
    )
    ctrl = {"k": "conn_open", "conn": "public.g1", "logical": "public", "standby": True}
    runner.control(ctrl, ts=t0 + 3)
    runner.control({"k": "conn_close", "conn": "public.g0", "reason": "disconnect"}, ts=t0 + 5)
    assert pipeline.active == {"public": "public.g1"}
    book = pipeline.books[SYMBOL]
    assert book.state is SyncState.SNAPSHOT
    assert book.gap is not None
    assert book.gap.reason == "disconnect"
    await close(recorder, runner)


async def test_kraken_ping_reaches_the_socket(tmp_path: Path) -> None:
    venue = FakeKraken()
    await venue.start()
    try:
        recorder, _ = offline_recorder(tmp_path, "kraken", venue.config())
        task = asyncio.create_task(recorder.run(duration_s=1.0))
        final = await task
        assert final["venues"]["kraken"]["clock_offset_ms"] is not None
        assert parquet_rows(tmp_path, "clock")
    finally:
        await venue.stop()
