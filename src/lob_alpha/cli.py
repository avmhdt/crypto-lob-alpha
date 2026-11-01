"""The `lob` command line."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from lob_alpha.fetch.metadata import FETCHERS, merge_specs
from lob_alpha.instruments import DEFAULT_PATH, dump_instruments, load_instruments
from lob_alpha.settings import Settings, load_recorder_config

app = typer.Typer(no_args_is_help=True, add_completion=False)
record_app = typer.Typer(no_args_is_help=True, help="Recorder commands.")
meta_app = typer.Typer(no_args_is_help=True, help="Instrument metadata.")
app.add_typer(record_app, name="record")
app.add_typer(meta_app, name="meta")

ConfigOption = Annotated[Path, typer.Option("--config", help="Recorder config file.")]
DEFAULT_CONFIG = Path("config/recorder.yaml")


def run_async[T](main: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine on uvloop where it exists."""
    if sys.platform != "win32":
        import uvloop  # noqa: PLC0415  # not installed on Windows

        return uvloop.run(main)
    return asyncio.run(main)


@record_app.command("smoke")
def record_smoke(
    venue: Annotated[str, typer.Option(help="Venue id.")],
    symbol: Annotated[str | None, typer.Option(help="Venue-native symbol.")] = None,
    seconds: Annotated[float, typer.Option(help="How long to run.")] = 60.0,
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Connect, sync one symbol, write temp Parquet, print rates and latency."""
    from lob_alpha.record.smoke import smoke  # noqa: PLC0415  # keeps `lob --help` fast

    recorder = load_recorder_config(config)
    chosen = symbol or recorder.venues[venue].symbols[0]
    result = run_async(smoke(recorder, load_instruments(), venue, chosen, seconds))
    typer.echo(f"venue={result.venue} symbol={result.symbol} seconds={result.seconds:g}")
    typer.echo(f"frames={result.frames} rate={result.frames / result.seconds:.1f}/s")
    typer.echo(f"reached_live={result.reached_live} live_after_s={result.live_after_s}")
    typer.echo(f"rows_written={result.rows_written} resyncs={result.resyncs}")
    typer.echo(f"feed_latency_ms_p50={result.feed_latency_ms_p50}")
    if not result.ok:
        raise typer.Exit(code=1)


@meta_app.command("fetch")
def meta_fetch(
    venue: Annotated[str | None, typer.Option(help="Only this venue.")] = None,
    config: ConfigOption = DEFAULT_CONFIG,
    instruments_path: Annotated[Path, typer.Option("--instruments")] = DEFAULT_PATH,
) -> None:
    """Write dated tick, lot and contract changes into the instruments file."""
    settings = Settings()
    recorder = load_recorder_config(config, settings)
    instruments = load_instruments(instruments_path)
    today = date.today()
    changes: list[str] = []
    with httpx.Client(timeout=30, trust_env=False) as http:
        for name in [venue] if venue else recorder.venue_set:
            specs = FETCHERS[name](http, recorder.venues[name])
            changes += merge_specs(instruments, name, specs, today)
    if any(": tick " in line for line in changes):
        dump_instruments(instruments, instruments_path)
    for line in changes or ["no changes"]:
        typer.echo(line)
