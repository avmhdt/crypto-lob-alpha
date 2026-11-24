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
book_app = typer.Typer(no_args_is_help=True, help="Offline book rebuild and validation.")
dq_app = typer.Typer(no_args_is_help=True, help="Data-quality reports.")
validate_app = typer.Typer(no_args_is_help=True, help="External cross-checks.")
app.add_typer(record_app, name="record")
app.add_typer(meta_app, name="meta")
app.add_typer(book_app, name="book")
app.add_typer(dq_app, name="dq")
app.add_typer(validate_app, name="validate")
VALIDATION_DIR = Path("reports/validation")

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


def _streams(recorder: RecorderConfig) -> list[Any]:
    from lob_alpha.reports.gaplog import Stream  # noqa: PLC0415

    streams: list[Stream] = []
    for venue in recorder.venue_set:
        cfg = recorder.venues[venue]
        kinds = ["book", "trades", "bbo"] if venue == "binance-usdm" else ["book", "trades"]
        streams += [Stream(venue, s, k) for s in cfg.symbols for k in kinds]
    return streams


def _select(
    recorder: RecorderConfig, venue: str | None, symbol: str | None
) -> list[tuple[str, str]]:
    pairs = [(v, s) for v, s in _pairs(recorder) if venue in (None, v) and symbol in (None, s)]
    if not pairs:
        raise typer.BadParameter(f"no configured stream for venue={venue} symbol={symbol}")
    return pairs


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
    by_day_out: Annotated[Path, typer.Option("--by-day-out")] = Path(
        "reports/tables/coverage_by_day.csv"
    ),
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Publish the gap log and the per-stream coverage table."""
    from lob_alpha.reports.gaplog import Stream, publish  # noqa: PLC0415

    settings = Settings()
    recorder = load_recorder_config(config, settings)
    streams: list[Stream] = _streams(recorder)
    log, coverage = publish(
        settings.data_dir,
        streams,
        date.fromisoformat(start),
        date.fromisoformat(end),
        out,
        coverage_out,
        by_day_out,
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


Day = Annotated[str, typer.Option(help="UTC date YYYY-MM-DD.")]
VenueOpt = Annotated[str | None, typer.Option("--venue", help="Venue id; default every venue.")]
SymbolOpt = Annotated[str | None, typer.Option("--symbol", help="Venue-native symbol.")]


@book_app.command("rebuild")
def book_rebuild(
    start: Day,
    end: Day,
    venue: VenueOpt = None,
    symbol: SymbolOpt = None,
    write: Annotated[bool, typer.Option("--write/--no-write", help="Write interim tables.")] = True,
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Replay book_l2 into interim/{topn,bbo_rebuilt} and record per-day validation stats."""
    import time  # noqa: PLC0415

    from lob_alpha.validate.rebuild import Rebuilder, save_stats  # noqa: PLC0415

    settings = Settings()
    recorder = load_recorder_config(config, settings)
    instruments = load_instruments(config.parent / "instruments.yaml")
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    for v, s in _select(recorder, venue, symbol):
        cfg = recorder.venues[v]
        tick = instruments.get(v, s, first).tick_size
        rebuilder = Rebuilder(
            settings.data_dir, v, s, tick, depth=getattr(cfg, "book_depth", None),
            raw=(v == "kraken"), write=write,
        )  # fmt: skip
        t0 = time.perf_counter()
        stats = rebuilder.run(first, last)
        save_stats(settings.data_dir, stats)
        rows = sum(st.rows for st in stats)
        crossed = sum(st.crossed + st.locked for st in stats)
        typer.echo(
            f"{v} {s}: {len(stats)} days, {rows} rows, {crossed} crossed or locked, "
            f"{time.perf_counter() - t0:.0f} s"
        )


@book_app.command("validate")
def book_validate(
    start: Day,
    end: Day,
    venue: VenueOpt = None,
    symbol: SymbolOpt = None,
    out_dir: Annotated[Path, typer.Option("--out-dir")] = VALIDATION_DIR,
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Write checkpoints_summary.csv and crossed_books.csv from rebuild stats.

    Exits 1 on any failure.
    """
    from lob_alpha.validate import report  # noqa: PLC0415
    from lob_alpha.validate.dq import complete_days  # noqa: PLC0415
    from lob_alpha.validate.rebuild import load_stats  # noqa: PLC0415

    settings = Settings()
    recorder = load_recorder_config(config, settings)
    wanted = set(_select(recorder, venue, symbol))
    stats = [
        st for st in load_stats(settings.data_dir)
        if (st["venue"], st["symbol"]) in wanted and start <= st["date"] <= end
    ]  # fmt: skip
    if not stats:
        typer.echo("no rebuild stats in range: run `lob book rebuild` first")
        raise typer.Exit(code=1)
    full = complete_days(
        settings.data_dir, _streams(recorder), date.fromisoformat(start), date.fromisoformat(end)
    )
    complete = {
        (r["venue"], r["symbol"], r["date"])
        for r in full.filter(full["complete"]).iter_rows(named=True)
    }
    summary = report.summary_rows(stats, complete)
    crossed = report.crossed_rows(stats)
    report.write_csv(out_dir / "checkpoints_summary.csv", summary, report.SUMMARY_FIELDS)
    report.write_csv(out_dir / "crossed_books.csv", crossed, report.CROSSED_FIELDS)
    report.write_csv(out_dir / "checkpoint_failures.csv", report.failure_rows(stats))
    ok2, text2 = report.verdict_ac_w_02(crossed)
    ok3, text3 = report.verdict_ac_w_03(summary)
    typer.echo(f"AC-W-02 {'PASS' if ok2 else 'FAIL'}: {text2}")
    typer.echo(f"AC-W-03 {'PASS' if ok3 else 'FAIL'}: {text3}")
    if not (ok2 and ok3):
        raise typer.Exit(code=1)


@dq_app.command("report")
def dq_report(
    start: Day,
    end: Day,
    out: Annotated[Path, typer.Option("--out")] = VALIDATION_DIR / "dq_report.md",
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Row counts, coverage, duplicates, clock offsets and complete days, as Markdown."""
    from lob_alpha.validate.dq import report  # noqa: PLC0415

    settings = Settings()
    recorder = load_recorder_config(config, settings)
    text = report(
        settings.data_dir, _streams(recorder), date.fromisoformat(start), date.fromisoformat(end)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    typer.echo(f"wrote {out}")


@validate_app.command("external")
def validate_external(
    kind: Annotated[str, typer.Option(help="binance-vision or tardis.")],
    venue: Annotated[str, typer.Option(help="Venue id.")],
    symbol: Annotated[str, typer.Option(help="Venue-native symbol.")],
    day: Day,
    tardis_symbol: Annotated[str | None, typer.Option(help="Tardis symbol if it differs.")] = None,
    out_dir: Annotated[Path, typer.Option("--out-dir")] = VALIDATION_DIR,
) -> None:
    """Cross-check one symbol-day against Binance Vision or a Tardis free sample (FR-16)."""
    from lob_alpha.validate import external  # noqa: PLC0415

    settings = Settings()
    when = date.fromisoformat(day)
    result: Any
    if kind == "binance-vision":
        result = external.binance_vision(settings.data_dir, venue, symbol, when)
        external.upsert_csv(out_dir / "external_binance_vision.csv", result)
    elif kind == "tardis":
        tick = load_instruments().get(venue, symbol, when).tick_size
        result = external.tardis(settings.data_dir, venue, symbol, when, tick, tardis_symbol)
        external.upsert_csv(out_dir / "external_tardis.csv", result)
    else:
        raise typer.BadParameter("kind must be binance-vision or tardis")
    typer.echo(json.dumps(result.__dict__, indent=1))


features_app = typer.Typer(no_args_is_help=True, help="Features and labels (M4).")
labels_app = typer.Typer(no_args_is_help=True, help="Labels (built with the features).")
ofi_app = typer.Typer(no_args_is_help=True, help="OFI replication.")
app.add_typer(features_app, name="features")
app.add_typer(labels_app, name="labels")
app.add_typer(ofi_app, name="ofi")
ResearchConfig = Annotated[Path, typer.Option("--config", help="Research config file.")]
DEFAULT_RESEARCH = Path("config/research.yaml")


def _research(path: Path) -> dict[str, Any]:
    import yaml  # noqa: PLC0415

    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return dict(loaded or {})


def _universe(cfg: dict[str, Any], venue: str | None, symbol: str | None) -> list[tuple[str, str]]:
    pairs = [(u["venue"], u["symbol"]) for u in cfg.get("universe", [])]
    pairs = [(v, s) for v, s in pairs if venue in (None, v) and symbol in (None, s)]
    if not pairs:
        raise typer.BadParameter(f"no universe entry for venue={venue} symbol={symbol}")
    return pairs


@features_app.command("build")
def features_build(
    start: Day,
    end: Day,
    venue: VenueOpt = None,
    symbol: SymbolOpt = None,
    grid: Annotated[str, typer.Option(help="time, event or both.")] = "both",
    config: ResearchConfig = DEFAULT_RESEARCH,
) -> None:
    """Build processed/features and processed/labels from interim/topn and trades."""
    from datetime import timedelta  # noqa: PLC0415

    from lob_alpha.features.build import Params, build_day  # noqa: PLC0415

    cfg = _research(config)
    params = Params.from_config(cfg)
    grids = ("time", "event") if grid == "both" else (grid,)
    settings = Settings()
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    for v, s in _universe(cfg, venue, symbol):
        day = first
        while day <= last:
            res, _ = build_day(settings.data_dir, v, s, day, params, grids=grids)
            typer.echo(
                f"{v} {s} {day}: time {res.time_rows} rows, event {res.event_rows} rows, "
                f"{res.bytes_written / 1e6:.0f} MB, {res.seconds:.0f} s"
            )
            day += timedelta(days=1)


@labels_app.command("build")
def labels_build(
    start: Day,
    end: Day,
    venue: VenueOpt = None,
    symbol: SymbolOpt = None,
    grid: Annotated[str, typer.Option(help="time, event or both.")] = "both",
    config: ResearchConfig = DEFAULT_RESEARCH,
) -> None:
    """Same pass as `lob features build`: labels are written beside the features."""
    features_build(start, end, venue, symbol, grid, config)


@ofi_app.command("replicate")
def ofi_replicate(
    start: Day,
    end: Day,
    config: ResearchConfig = DEFAULT_RESEARCH,
    recorder_config: Annotated[Path, typer.Option("--recorder-config")] = DEFAULT_CONFIG,
    out: Annotated[Path, typer.Option("--out")] = Path("reports/tables/ofi_replication.csv"),
    figures: Annotated[Path, typer.Option("--figures")] = Path("reports/figures"),
) -> None:
    """CKS OFI regressions per complete symbol-day (DQ-10); exit 1 if AC-W-04 fails."""
    from lob_alpha.research import ofi  # noqa: PLC0415
    from lob_alpha.validate.dq import complete_days  # noqa: PLC0415

    cfg = _research(config)
    settings = Settings()
    recorder = load_recorder_config(recorder_config, settings)
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    full = complete_days(settings.data_dir, _streams(recorder), first, last)
    done = {
        (r["venue"], r["symbol"], r["date"])
        for r in full.filter(full["complete"]).iter_rows(named=True)
    }
    rows: list[Any] = []
    for v, s in _universe(cfg, None, None):
        days = sorted(d for vv, ss, d in done if (vv, ss) == (v, s))
        for d in days:
            rows += ofi.replicate_day(settings.data_dir, v, s, date.fromisoformat(d))
        if days:
            ofi.scatter(
                figures / f"ofi_scatter_{v}_{s.replace('/', '-')}.png",
                settings.data_dir,
                v,
                s,
                date.fromisoformat(days[-1]),
            )
    ofi.write_table(out, rows)
    ok, text = ofi.verdict(ofi.read_table(out))
    typer.echo(f"{len(rows)} rows to {out}")
    typer.echo(f"AC-W-04 {'PASS' if ok else 'FAIL'}: {text}")
    if not ok:
        raise typer.Exit(code=1)


def _complete_days(
    venue: str, symbol: str, start: str | None, end: str | None, recorder_config: Path
) -> list[date]:
    from lob_alpha.validate.dq import complete_days  # noqa: PLC0415

    settings = Settings()
    recorder = load_recorder_config(recorder_config, settings)
    first = date.fromisoformat(start) if start else date(2026, 11, 1)
    last = date.fromisoformat(end) if end else date.today()
    full = complete_days(settings.data_dir, _streams(recorder), first, last)
    rows = full.filter(full["complete"]).iter_rows(named=True)
    return sorted(
        date.fromisoformat(str(r["date"]))
        for r in rows
        if (r["venue"], r["symbol"]) == (venue, symbol)
    )


def _model_grid(base: dict[str, Any], sets: list[str]) -> list[dict[str, Any]]:
    """Expand list-valued options into one config per combination; --set fixes a key."""
    import itertools  # noqa: PLC0415

    import yaml  # noqa: PLC0415

    fixed = dict(base)
    for kv in sets:
        key, _, value = kv.partition("=")
        fixed[key] = yaml.safe_load(value)
    keys = [k for k, v in fixed.items() if isinstance(v, list) and k != "alphas"]
    combos = itertools.product(*[fixed[k] for k in keys]) if keys else [()]
    return [fixed | dict(zip(keys, c, strict=True)) for c in combos]


@app.command("train")
def train(
    model: Annotated[str, typer.Option("--model", help="ridge or lgbm.")],
    venue: Annotated[str, typer.Option("--venue")],
    symbol: Annotated[str, typer.Option("--symbol")],
    target: Annotated[str, typer.Option("--target")] = "dmid_ticks_time_1000",
    set_: Annotated[list[str] | None, typer.Option("--set", help="key=value.")] = None,
    folds: Annotated[str | None, typer.Option(help="Fold indices, e.g. 9,10.")] = None,
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option()] = None,
    permute: Annotated[bool, typer.Option(help="AC-W-05 permutation run.")] = False,
    notes: Annotated[str, typer.Option()] = "",
    config: ResearchConfig = DEFAULT_RESEARCH,
    recorder_config: Annotated[Path, typer.Option("--recorder-config")] = DEFAULT_CONFIG,
    results: Annotated[Path, typer.Option("--results")] = Path("results"),
) -> None:
    """Walk-forward fit and OOS predictions; one run per grid point; prints each run_id."""
    from lob_alpha.models.train import Runner, RunSpec, Target  # noqa: PLC0415

    cfg = _research(config)
    tgt = Target.parse(target)
    mcfg = cfg.get("models", {})
    features = list(mcfg.get("feature_sets", {})[tgt.clock])
    from lob_alpha.validate.rebuild import partition  # noqa: PLC0415

    settings = Settings()
    complete = _complete_days(venue, symbol, start, end, recorder_config)

    def built(d: date) -> bool:
        base = settings.data_dir / "processed"
        return all(
            any((partition(base, ds, venue, symbol, d) / f"clock={tgt.clock}").glob("*.parquet"))
            for ds in ("features", "labels")
        )

    days = [d for d in complete if built(d)]
    missing = sorted(set(complete) - set(days))
    if missing:
        typer.echo(f"complete days without features yet, left out: {[str(d) for d in missing]}")
    inst = load_instruments().get(venue, symbol, days[-1] if days else None)
    wf = cfg.get("walk_forward", {})
    grid = _model_grid(dict(mcfg.get(model, {})), set_ or [])
    for model_cfg in grid:
        spec = RunSpec(
            venue=venue,
            symbol=symbol,
            target=target,
            model=model,
            model_cfg=model_cfg,
            features=features,
            days=days,
            tick_size=inst.tick_size,
            taker_fee_bps=inst.taker_fee_bps,
            train_days=int(wf.get("train_days", 5)),
            val_days=int(wf.get("val_days", 1)),
            test_days=int(wf.get("test_days", 1)),
            step_days=int(wf.get("step_days", 1)),
            purge_s=float(wf.get("purge_s", 31)),
            folds=[int(x) for x in folds.split(",")] if folds else None,
            permute=permute,
            notes=notes,
        )
        res = Runner(spec, settings.data_dir, results).run()
        p = res.pooled
        typer.echo(
            f"{res.run_id} {model} {venue} {symbol} {target} {len(res.folds)} folds: "
            f"OOS R2 {p['oos_r2']:.4f}, IC {p['ic_spearman']:.4f}, "
            f"hit {p['hit_rate_nonzero']:.3f}, trades {p['n_trades']}"
        )


@app.command("decay")
def decay_cmd(
    run_id: Annotated[str, typer.Option("--run-id")],
    horizons: Annotated[bool, typer.Option(help="Horizon curve (one run per horizon).")] = True,
    latencies: Annotated[bool, typer.Option(help="Latency curve (clock runs).")] = True,
    config: ResearchConfig = DEFAULT_RESEARCH,
    results: Annotated[Path, typer.Option("--results")] = Path("results"),
    tables: Annotated[Path, typer.Option("--tables")] = Path("reports/tables"),
) -> None:
    """Alpha decay vs horizon and added latency; half-life with CI (FR-21)."""
    from lob_alpha import explog  # noqa: PLC0415
    from lob_alpha.models.train import Runner, RunSpec, Target  # noqa: PLC0415
    from lob_alpha.research import decay  # noqa: PLC0415

    cfg = _research(config)
    settings = Settings()
    base = json.loads((results / "runs" / run_id / "config.json").read_text(encoding="utf-8"))
    tgt = Target.parse(base["target"])
    inst = load_instruments().get(base["venue"], base["symbol"])
    meta: dict[str, str | int] = {
        "venue": base["venue"],
        "symbol": base["symbol"],
        "clock": tgt.clock,
        "model": base["model"],
        "run_id": run_id,
    }
    rows: list[decay.CurveRow] = []
    if horizons:
        lab = cfg.get("labels", {})
        hs = lab.get("horizons_ms" if tgt.clock == "time" else "horizons_events", [])
        done = {
            explog.canonical_json(json.loads(r["params_json"])): r["run_id"]
            for r in explog.read_rows(results / explog.TRIAL_LOG.name)
            if r["x_status"] == "ok"
        }
        for h in hs:
            name = f"dmid_ticks_{tgt.clock}_{h}"
            params = base | {"target": name}
            rid = done.get(explog.canonical_json(params))
            if rid is None:
                spec = RunSpec.from_params(params, inst.tick_size, notes=f"decay of {run_id}")
                rid = Runner(spec, settings.data_dir, results).run().run_id
            stats = decay.run_day_stats(settings.data_dir, rid)
            rows.append(decay.curve_row(meta | {"run_id": rid}, int(h), 0, stats))
            typer.echo(f"horizon {h}: run {rid} OOS R2 {rows[-1].oos_r2:.4f}")
    if latencies and tgt.clock == "time":
        lat_rows, lat_stats = [], {}
        for lat in cfg.get("labels", {}).get("latencies_ms", [0]):
            stats = decay.latency_day_stats(
                settings.data_dir, run_id, base["venue"], base["symbol"], tgt.horizon, int(lat)
            )
            lat_stats[int(lat)] = stats
            lat_rows.append(decay.curve_row(meta, tgt.horizon, int(lat), stats))
            typer.echo(f"latency {lat} ms: OOS R2 {lat_rows[-1].oos_r2:.4f}")
        rows += [r for r in lat_rows if r.latency_ms > 0 or not horizons]
        hl = decay.half_life_ci(lat_stats)
        decay.write_half_life(tables / "half_life.csv", meta, tgt.horizon, hl)
        typer.echo(f"half-life at {tgt.horizon} ms: {hl}")
    if rows:
        decay.write_rows(
            tables / "alpha_decay.csv",
            rows,
            ("venue", "symbol", "clock", "model", "horizon", "latency_ms"),
        )


@app.command("importance")
def importance_cmd(
    run_id: Annotated[str, typer.Option("--run-id")],
    results: Annotated[Path, typer.Option("--results")] = Path("results"),
    tables: Annotated[Path, typer.Option("--tables")] = Path("reports/tables"),
    figures: Annotated[Path, typer.Option("--figures")] = Path("reports/figures"),
) -> None:
    """Gain and mean |SHAP| per feature on OOS days, with a mechanism sentence (FR-26)."""
    from lob_alpha.research import importance  # noqa: PLC0415

    cfg = json.loads((results / "runs" / run_id / "config.json").read_text(encoding="utf-8"))
    rows = importance.run_importance(Settings().data_dir, results, run_id)
    importance.write_table(tables / "importance.csv", cfg, run_id, rows)
    sym = cfg["symbol"].replace("/", "")
    importance.figure(
        figures / f"shap_summary_{cfg['venue']}_{sym}_{cfg['target']}.png",
        f"{cfg['venue']} {cfg['symbol']} {cfg['target']}: mean |SHAP|, {rows[0].folds} folds",
        rows,
    )
    for r in rows[:10]:
        typer.echo(f"{r.shap_rank:2d} {r.feature:24s} {r.mean_abs_shap:.4f} gain {r.gain_rank}")


@app.command("stability")
def stability_cmd(
    run_id: Annotated[list[str], typer.Option("--run-id", help="One or more runs.")],
    transfer: Annotated[
        list[str] | None, typer.Option(help="Cross-venue test stream, venue:symbol.")
    ] = None,
    recorder_config: Annotated[Path, typer.Option("--recorder-config")] = DEFAULT_CONFIG,
    results: Annotated[Path, typer.Option("--results")] = Path("results"),
    tables: Annotated[Path, typer.Option("--tables")] = Path("reports/tables"),
    figures: Annotated[Path, typer.Option("--figures")] = Path("reports/figures"),
) -> None:
    """Per-test-day OOS R2 with 95% CIs, forest plot, cross-venue transfer (FR-25)."""
    import csv  # noqa: PLC0415
    from dataclasses import asdict  # noqa: PLC0415

    from lob_alpha.research import decay, stability  # noqa: PLC0415

    for rid in run_id:
        for tv in transfer or []:
            venue_b, _, symbol_b = tv.partition(":")
            cfg = json.loads((results / "runs" / rid / "config.json").read_text(encoding="utf-8"))
            insts = load_instruments()
            days_b = {
                str(d) for d in _complete_days(venue_b, symbol_b, None, None, recorder_config)
            }
            row = stability.transfer(
                Settings().data_dir,
                results,
                rid,
                venue_b,
                symbol_b,
                insts.get(venue_b, symbol_b).tick_size,
                insts.get(cfg["venue"], cfg["symbol"]).tick_size,
                days_b,
            )
            path = tables / "cross_venue.csv"
            old: list[dict[str, str]] = []
            if path.exists():
                with path.open(encoding="utf-8", newline="") as fh:
                    old = [
                        r
                        for r in csv.DictReader(fh)
                        if (r["run_id"], r["test_venue"]) != (rid, venue_b)
                    ]
            new = {k: decay._fmt(v) for k, v in asdict(row).items()}
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(new), lineterminator="\n")
                w.writeheader()
                w.writerows([*old, new])
            typer.echo(f"{rid} -> {tv}: {row.n_days} days, OOS R2 {row.oos_r2:.4f}")
    for rid in run_id:
        cfg = json.loads((results / "runs" / rid / "config.json").read_text(encoding="utf-8"))
        rows = stability.run_days(Settings().data_dir, cfg, rid)
        stability.write_table(tables / "stability.csv", rows)
        pos = sum(r.oos_r2 > 0 for r in rows)
        typer.echo(f"{rid} {cfg['venue']} {cfg['symbol']}: {pos}/{len(rows)} days OOS R2 > 0")
    with (tables / "stability.csv").open(encoding="utf-8", newline="") as fh:
        stability.forest(figures / "stability_forest.png", list(csv.DictReader(fh)))


@app.command("event-vs-clock")
def event_vs_clock_cmd(
    run_id: Annotated[str, typer.Option("--run-id", help="A clock-time run.")],
    clock_horizons: Annotated[str, typer.Option(help="Comma-separated ms.")] = "1000,5000",
    config: ResearchConfig = DEFAULT_RESEARCH,
    results: Annotated[Path, typer.Option("--results")] = Path("results"),
    tables: Annotated[Path, typer.Option("--tables")] = Path("reports/tables"),
) -> None:
    """Clock vs event models at matched horizons, with day-block CIs (FR-27)."""
    from lob_alpha import explog  # noqa: PLC0415
    from lob_alpha.models.train import Runner, RunSpec, Target  # noqa: PLC0415
    from lob_alpha.research import decay, event_clock  # noqa: PLC0415

    cfg = _research(config)
    settings = Settings()
    base = json.loads((results / "runs" / run_id / "config.json").read_text(encoding="utf-8"))
    if Target.parse(base["target"]).clock != "time":
        raise typer.BadParameter("--run-id must be a clock-time run")
    inst = load_instruments().get(base["venue"], base["symbol"])
    done = {
        explog.canonical_json(json.loads(r["params_json"])): r["run_id"]
        for r in explog.read_rows(results / explog.TRIAL_LOG.name)
        if r["x_status"] == "ok"
    }

    def ensure(params: dict[str, Any]) -> str:
        rid = done.get(explog.canonical_json(params))
        if rid is None:
            spec = RunSpec.from_params(params, inst.tick_size, notes=f"event-vs-clock of {run_id}")
            rid = Runner(spec, settings.data_dir, results).run().run_id
            done[explog.canonical_json(params)] = rid
        return rid

    first_train = [date.fromisoformat(d) for d in base["days"][:5]]
    hev = [int(h) for h in cfg.get("labels", {}).get("horizons_events", [])]
    durations = event_clock.median_event_duration_ms(
        settings.data_dir, base["venue"], base["symbol"], first_train, hev
    )
    event_feats = list(cfg["models"]["feature_sets"]["event"])
    rows = []
    for hc in (int(x) for x in clock_horizons.split(",")):
        he = event_clock.match(hc, durations)
        crid = ensure(base | {"target": f"dmid_ticks_time_{hc}"})
        erid = ensure(base | {"target": f"dmid_ticks_event_{he}", "features": event_feats})
        cs, es = (
            decay.run_day_stats(settings.data_dir, crid),
            decay.run_day_stats(settings.data_dir, erid),
        )
        clo, chi = decay.bootstrap(cs)
        elo, ehi = decay.bootstrap(es)
        rows.append(
            event_clock.Comparison(
                base["venue"],
                base["symbol"],
                hc,
                he,
                durations[he],
                crid,
                erid,
                decay.pooled_r2(cs),
                clo,
                chi,
                decay.pooled_r2(es),
                elo,
                ehi,
                len(cs),
                len(es),
            )
        )
        typer.echo(
            f"{hc} ms vs {he} batches (median {durations[he]:.0f} ms): clock R2 "
            f"{rows[-1].clock_r2:.4f}, event R2 {rows[-1].event_r2:.4f}"
        )
    event_clock.write_table(tables / "event_vs_clock.csv", rows)


@app.command("export-model")
def export_model_cmd(
    run_id: Annotated[str, typer.Option("--run-id")],
    version: Annotated[str, typer.Option("--version", help="semver, e.g. 1.0.0.")],
    fold: Annotated[int | None, typer.Option(help="Fold to export; default the last.")] = None,
    provisional: Annotated[bool, typer.Option(help="Mark the artifact provisional.")] = False,
    out: Annotated[Path, typer.Option("--out")] = Path("artifacts/models"),
    results: Annotated[Path, typer.Option("--results")] = Path("results"),
) -> None:
    """Write the signal model artifact (contract v1.0.0) from one fold of a LightGBM run."""
    from lob_alpha import export  # noqa: PLC0415

    cfg = json.loads((results / "runs" / run_id / "config.json").read_text(encoding="utf-8"))
    inst = load_instruments().get(cfg["venue"], cfg["symbol"])
    extra: dict[str, Any] = {}
    if provisional:
        extra = {
            "provisional": True,
            "status": (
                "provisional pre-release for integration only; not for published results. "
                "The final artifact follows the end of the capture window."
            ),
        }
    res = export.export_model(
        run_id=run_id,
        results=results,
        data_root=Settings().data_dir,
        out_root=out,
        version=version,
        tick=inst.tick_size,
        lot=inst.lot_size,
        fold=fold,
        extra_metadata=extra,
    )
    typer.echo(
        f"{res.folder}: {res.n_parity} parity rows, reload max |diff| {res.max_abs_diff:.2e}"
    )


@app.command("verify")
def verify(
    ac_id: Annotated[str, typer.Argument(help="Acceptance criterion, e.g. AC-W-03.")],
    out_dir: Annotated[Path, typer.Option("--out-dir")] = VALIDATION_DIR,
) -> None:
    """Run the check listed for an acceptance criterion; exit 0 = pass."""
    from lob_alpha.validate import report  # noqa: PLC0415

    if ac_id == "AC-W-02":
        ok, text = report.verdict_ac_w_02(report.read_csv(out_dir / "crossed_books.csv"))
    elif ac_id == "AC-W-03":
        ok, text = report.verdict_ac_w_03(report.read_csv(out_dir / "checkpoints_summary.csv"))
    elif ac_id == "AC-W-04":
        from lob_alpha.research import ofi  # noqa: PLC0415

        ok, text = ofi.verdict(ofi.read_table(Path("reports/tables/ofi_replication.csv")))
    elif ac_id == "AC-W-10":
        from lob_alpha import explog  # noqa: PLC0415

        problems = explog.check_logs(Path("results"))
        ok, text = not problems, "; ".join(problems) or "trial and experiment logs consistent"
    elif ac_id == "AC-W-11":
        from lob_alpha import export  # noqa: PLC0415

        ok, text = export.verify_artifacts(Path("artifacts/models"))
    elif ac_id == "AC-W-05":
        from lob_alpha import explog  # noqa: PLC0415

        ok, text = explog.verdict_permutation(Path("results"))
    else:
        typer.echo(f"{ac_id}: no automated check yet")
        raise typer.Exit(code=2)
    typer.echo(f"{ac_id} {'PASS' if ok else 'FAIL'}: {text}")
    if not ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
