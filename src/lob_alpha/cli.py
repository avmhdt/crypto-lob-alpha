"""The `lob` command line."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Coroutine
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import httpx
import structlog
import typer

from lob_alpha.fetch.metadata import FETCHERS, merge_specs
from lob_alpha.instruments import DEFAULT_PATH, dump_instruments, load_instruments
from lob_alpha.record import health
from lob_alpha.settings import RecorderConfig, Settings, load_recorder_config

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


def configure_logging(settings: Settings) -> None:
    """JSON logs to stdout (journald), or to a file when `LOB_LOG_DIR` is set."""
    factory: Any = structlog.PrintLoggerFactory()
    if settings.lob_log_dir:
        directory = Path(settings.lob_log_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        stream = (directory / "recorder.log").open("a", encoding="utf-8", buffering=1)
        factory = structlog.PrintLoggerFactory(file=stream)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.lob_log_level.upper(), logging.INFO)
        ),
        logger_factory=factory,
    )


def _pairs(recorder: RecorderConfig) -> list[tuple[str, str]]:
    return [(v, s) for v in recorder.venue_set for s in recorder.venues[v].symbols]


@record_app.command("run")
def record_run(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """The service process. SIGTERM flushes and closes every file."""
    from lob_alpha.record.supervisor import Recorder  # noqa: PLC0415

    settings = Settings()
    configure_logging(settings)
    recorder = load_recorder_config(config, settings)
    instruments = load_instruments(config.parent / "instruments.yaml")
    instruments.require(_pairs(recorder))

    async def main() -> None:
        await Recorder(recorder, settings, instruments).run(config_path=config.resolve())

    run_async(main())


def _print_status(status: dict[str, Any]) -> None:
    age = health.status_age_s(status)
    typer.echo(
        f"run={status.get('run')} version={status.get('version')} updated {age:.0f}s ago "
        f"stopped={status.get('stopped')}"
    )
    disk = status.get("disk", {})
    writer = status.get("writer", {})
    typer.echo(
        f"disk free={disk.get('free_gb')} GB used={disk.get('used_pct')}% "
        f"guarded={disk.get('guarded')} raw_paused={disk.get('raw_paused')}"
    )
    typer.echo(
        f"writer rows={writer.get('rows_written')} files_closed={writer.get('files_closed')} "
        f"errors={writer.get('flush_errors')}"
    )
    for venue, block in status.get("venues", {}).items():
        typer.echo(
            f"\n{venue}: frames={block.get('frames')} feed_latency_ms_p50="
            f"{block.get('feed_latency_ms_p50')} clock_offset_ms={block.get('clock_offset_ms')} "
            f"ingest_us_p99={block.get('ingest_us_p99')} parse_errors={block.get('parse_errors')}"
        )
        for s in block.get("streams", []):
            extra = f" epoch={s['epoch']} resyncs={s['resyncs']}" if "epoch" in s else ""
            extra += f" crossed={s['crossed']}" if "crossed" in s else ""
            typer.echo(
                f"  {s['symbol']:<9} {s['stream']:<6} {s['state']:<10} age={s['age_s']}s "
                f"msgs={s['msgs']}{extra}"
            )


@record_app.command("status")
def record_status(
    disk: Annotated[bool, typer.Option("--disk", help="Disk rate and projection.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Raw status JSON.")] = False,
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Per-stream state, last message age, disk use and projection."""
    settings = Settings()
    status = health.read_status(settings.data_dir)
    if status is None:
        typer.echo(f"no status file under {settings.data_dir}")
        raise typer.Exit(code=1)
    report: dict[str, Any] | None = None
    if disk:
        recorder = load_recorder_config(config, settings)
        report = health.disk_report(
            settings.data_dir,
            recorder.health.planned_capture_days,
            recorder.raw_archive.retention_days,
        )
    if as_json:
        typer.echo(json.dumps({"status": status, "disk": report}, indent=2))
        return
    _print_status(status)
    if report is not None:
        typer.echo("\n" + json.dumps(report, indent=2))


@record_app.command("recover")
def record_recover(
    from_raw: Annotated[bool, typer.Option("--from-raw", help="Replay the raw archive.")] = False,
    venue: Annotated[str | None, typer.Option(help="Venue id.")] = None,
    day: Annotated[str | None, typer.Option("--date", help="UTC date YYYY-MM-DD.")] = None,
    hour: Annotated[int | None, typer.Option(help="UTC hour, default every hour.")] = None,
    replace: Annotated[bool, typer.Option("--replace", help="Overwrite closed files.")] = False,
    interrupted: Annotated[
        bool, typer.Option("--interrupted", help="Every hour with a leftover .tmp file.")
    ] = False,
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Rebuild Parquet from the raw archive. Closed files stay unless --replace."""
    from lob_alpha.record.recover import interrupted_hours, recover  # noqa: PLC0415

    if not from_raw:
        raise typer.BadParameter("--from-raw is the only supported source")
    settings = Settings()
    recorder = load_recorder_config(config, settings)
    instruments = load_instruments(config.parent / "instruments.yaml")
    data_dir = settings.data_dir
    if interrupted:
        targets = interrupted_hours(data_dir)
    elif venue is None or day is None:
        raise typer.BadParameter("give --venue and --date, or --interrupted")
    else:
        hours = [hour] if hour is not None else list(range(24))
        targets = [(venue, date.fromisoformat(day), h) for h in hours]
    written = skipped = 0
    for target_venue, target_day, target_hour in targets:
        if target_venue not in recorder.venues:
            continue
        result = recover(
            recorder,
            instruments,
            data_dir,
            target_venue,
            target_day,
            [target_hour],
            replace=replace,
        )
        written += len(result.written)
        skipped += len(result.skipped)
        for path in result.skipped:
            typer.echo(f"kept closed file {path}")
    typer.echo(f"recovered {written} files, kept {skipped}")


@record_app.command("gaplog")
def record_gaplog(
    start: Annotated[str, typer.Option(help="First UTC date.")],
    end: Annotated[str, typer.Option(help="Last UTC date.")],
    out: Annotated[Path, typer.Option(help="Gap log CSV.")] = Path("reports/gap_log.csv"),
    coverage_out: Annotated[Path, typer.Option("--coverage-out")] = Path(
        "reports/tables/coverage.csv"
    ),
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Publish the gap log and the per-stream coverage table."""
    from lob_alpha.reports.gaplog import Stream, publish  # noqa: PLC0415

    settings = Settings()
    recorder = load_recorder_config(config, settings)
    streams: list[Stream] = []
    for venue in recorder.venue_set:
        cfg = recorder.venues[venue]
        kinds = ["book", "trades", "bbo"] if venue == "binance-usdm" else ["book", "trades"]
        streams += [Stream(venue, s, k) for s in cfg.symbols for k in kinds]
    log, coverage = publish(
        settings.data_dir,
        streams,
        date.fromisoformat(start),
        date.fromisoformat(end),
        out,
        coverage_out,
    )
    typer.echo(f"{log.height} gaps to {out}; coverage for {coverage.height} streams")
    typer.echo(str(coverage.select("venue", "symbol", "stream", "coverage_pct", "longest_gap_s")))


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


if __name__ == "__main__":
    app()
