"""`lob record smoke` against a local websocket server replaying recorded frames."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path

import pytest
import websockets
from typer.testing import CliRunner

from conftest import frames
from lob_alpha import cli
from lob_alpha.fetch.metadata import Spec
from lob_alpha.instruments import Instrument, Instruments, dump_instruments, load_instruments
from lob_alpha.record.smoke import SmokeResult, smoke
from lob_alpha.settings import RecorderConfig, VenueConfig

ENTRY = {
    "tick_size": 0.01, "lot_size": 1e-8, "contract_type": "spot", "contract_value": 1.0,
    "maker_fee_bps": 40.0, "taker_fee_bps": 60.0, "effective_from": date(2026, 1, 1),
    "source_url": "u", "fee_source_url": "f", "fee_read_on": date(2026, 11, 1),
}  # fmt: skip


async def _replay(connection, name: str) -> None:
    await connection.recv()
    for raw in frames(name):
        await connection.send(raw.decode())
    await asyncio.sleep(30)


@pytest.fixture
async def coinbase_server() -> AsyncIterator[str]:
    async def handler(connection) -> None:
        await _replay(connection, "coinbase.jsonl")

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}"


def config(url: str) -> RecorderConfig:
    block = VenueConfig(
        ws=url, rest="http://127.0.0.1:1", symbols=["BTC-USD"], channels=["level2_batch"]
    )
    return RecorderConfig(version=1, venue_set=["coinbase"], venues={"coinbase": block})


async def test_smoke_reaches_live_and_writes_rows(coinbase_server: str) -> None:
    table = Instruments([Instrument(venue="coinbase", symbol="BTC-USD", **ENTRY)])
    result = await smoke(config(coinbase_server), table, "coinbase", "BTC-USD", 3.0)
    assert result.ok
    assert result.frames >= 7
    assert result.rows_written > 0
    assert result.feed_latency_ms_p50 is not None


async def test_smoke_surfaces_a_refused_connection() -> None:
    table = Instruments([Instrument(venue="coinbase", symbol="BTC-USD", **ENTRY)])
    with pytest.raises(OSError):
        await smoke(config("ws://127.0.0.1:1"), table, "coinbase", "BTC-USD", 0.5)


def test_cli_smoke_exit_code_follows_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    from lob_alpha.record import smoke as smoke_module

    good = SmokeResult("coinbase", "BTC-USD", 1.0, 5, True, 0.1, 9, 1.0, 0, 0)
    bad = SmokeResult("coinbase", "BTC-USD", 1.0, 0, False, None, 0, None, 0, 0)
    runner = CliRunner()
    for result, code in ((good, 0), (bad, 1)):

        async def fake(*_args: object, result: SmokeResult = result) -> SmokeResult:
            return result

        monkeypatch.setattr(smoke_module, "smoke", fake)
        outcome = runner.invoke(
            cli.app, ["record", "smoke", "--venue", "coinbase", "--seconds", "1"]
        )
        assert outcome.exit_code == code, outcome.output


def test_cli_meta_fetch_writes_dated_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "instruments.yaml"
    dump_instruments(Instruments([Instrument(venue="coinbase", symbol="BTC-USD", **ENTRY)]), path)
    spec = Spec("BTC-USD", 0.05, 1e-8, "spot", 1.0, "u2")
    monkeypatch.setitem(cli.FETCHERS, "coinbase", lambda _http, _cfg: [spec])
    runner = CliRunner()
    args = ["meta", "fetch", "--venue", "coinbase", "--instruments", str(path)]
    outcome = runner.invoke(cli.app, args)
    assert outcome.exit_code == 0, outcome.output
    assert "tick 0.01 -> 0.05" in outcome.output
    assert len(load_instruments(path).entries) == 2
    assert "no changes" in runner.invoke(cli.app, args).output
