"""External cross-checks (FR-16, AC-H-08). Reference data stays under raw/external (DQ-14).

* Binance Vision: the daily USD-M `aggTrades` file against the recorded `aggTrade` ids.
  Trades are matched on the exchange trade time, so a day is the venue's UTC day. Ids
  missing inside a logged trades gap (widened by `GAP_MARGIN_S` for receive latency) are
  expected and counted apart; completeness is reported over all ids and over the ids that
  fall in LIVE time.
* Tardis: the free first-of-month `incremental_book_L2` file is replayed through the same
  kernel and compared with the rebuilt top 10 once per second, on exchange time.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt
import polars as pl

from lob_alpha.book.replay import (
    CRC_TABLE,
    NO_LEVEL,
    TOP_N,
    BookState,
    empty_probes,
    run_batches,
)
from lob_alpha.reports.gaplog import _ns, load_gaps
from lob_alpha.validate.rebuild import day_files, partition

VISION = "https://data.binance.vision/data/futures/um/daily/aggTrades"
TARDIS = "https://datasets.tardis.dev/v1"
TARDIS_EXCHANGE = {"binance-usdm": "binance-futures", "coinbase": "coinbase", "kraken": "kraken"}
GAP_MARGIN_S = 2.0
RATE_CAP = 2_000_000.0  # bytes/s
NS = 1_000_000_000
MS = 1_000_000


def external_dir(data_dir: Path, kind: str) -> Path:
    return data_dir / "raw" / "external" / kind


def download(
    url: str, dest: Path, *, client: httpx.Client | None = None, max_bytes_per_s: float = RATE_CAP
) -> Path:
    """Fetch once into `dest`; a finished file is never fetched again.

    The rate is capped so a check never competes with the recorder's own connections on the
    home link (bursts of PC traffic hurt the Binance feed first, capture log 2026-11-02).
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    own = client is None
    http = client or httpx.Client(timeout=120, follow_redirects=True)
    try:
        with http.stream("GET", url) as response:
            response.raise_for_status()
            start = time.monotonic()
            done = 0
            with tmp.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 18):
                    handle.write(chunk)
                    done += len(chunk)
                    ahead = done / max_bytes_per_s - (time.monotonic() - start)
                    if ahead > 0:
                        time.sleep(ahead)
    finally:
        if own:
            http.close()
    tmp.replace(dest)
    return dest


# ---- Binance Vision ------------------------------------------------------------------------


@dataclass
class VisionResult:
    venue: str
    symbol: str
    date: str
    vision_ids: int
    recorded_ids: int
    missing: int
    missing_in_gaps: int
    missing_live: int
    extra: int
    completeness_pct: float
    completeness_live_pct: float
    sha256_ok: bool


def vision_ids(zip_path: Path) -> pl.DataFrame:
    """agg_trade_id and transact_time (ms) from a Binance Vision aggTrades zip."""
    with zipfile.ZipFile(zip_path) as archive:
        name = archive.namelist()[0]
        raw = archive.read(name)
    first = raw.split(b"\n", 1)[0]
    has_header = not first[:1].isdigit()
    frame = pl.read_csv(
        io.BytesIO(raw),
        has_header=has_header,
        new_columns=[
            "agg_trade_id",
            "price",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "transact_time",
            "is_buyer_maker",
        ],
        schema_overrides={"agg_trade_id": pl.Int64, "transact_time": pl.Int64},
    )
    return frame.select("agg_trade_id", "transact_time")


def check_sha256(zip_path: Path, checksum_path: Path) -> bool:
    expected = checksum_path.read_text(encoding="ascii").split()[0].lower()
    return hashlib.sha256(zip_path.read_bytes()).hexdigest() == expected


def recorded_ids(data_dir: Path, venue: str, symbol: str, day: date) -> pl.DataFrame:
    """Recorded trades whose exchange time falls on `day`, from that day and its neighbours."""
    lo, hi = _ns(day), _ns(day + timedelta(days=1))
    parts = [
        pl.read_parquet(p, columns=["trade_id", "ts_exchange"])
        for d in (day - timedelta(days=1), day, day + timedelta(days=1))
        for p in day_files(data_dir, "trades", venue, symbol, d)
    ]
    if not parts:
        return pl.DataFrame(schema={"agg_trade_id": pl.Int64, "ts_exchange": pl.Int64})
    return (
        pl.concat(parts)
        .filter((pl.col("ts_exchange") >= lo) & (pl.col("ts_exchange") < hi))
        .select(pl.col("trade_id").cast(pl.Int64).alias("agg_trade_id"), "ts_exchange")
        .unique("agg_trade_id")
    )


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 5) if whole else 0.0


def compare_vision(
    vision: pl.DataFrame, recorded: pl.DataFrame, gaps: list[tuple[int, int]],
    venue: str, symbol: str, day: date, sha_ok: bool,
) -> VisionResult:  # fmt: skip
    v_ids = vision["agg_trade_id"].unique()
    r_ids = recorded["agg_trade_id"].unique()
    missing = vision.filter(~pl.col("agg_trade_id").is_in(r_ids.implode()))
    extra = int((~r_ids.is_in(v_ids.implode())).sum())
    t = missing["transact_time"].to_numpy().astype(np.int64) * MS
    in_gap = np.zeros(len(t), dtype=bool)
    margin = int(GAP_MARGIN_S * NS)
    for start, end in gaps:
        in_gap |= (t >= start - margin) & (t <= end + margin)
    n_v = len(v_ids)
    n_missing = missing.height
    n_in_gap = int(in_gap.sum())
    live_total = n_v - n_in_gap
    return VisionResult(
        venue=venue, symbol=symbol, date=str(day), vision_ids=n_v, recorded_ids=len(r_ids),
        missing=n_missing, missing_in_gaps=n_in_gap, missing_live=n_missing - n_in_gap, extra=extra,
        completeness_pct=round(100.0 * (n_v - n_missing) / n_v, 5) if n_v else 0.0,
        completeness_live_pct=_pct(live_total - (n_missing - n_in_gap), live_total),
        sha256_ok=sha_ok,
    )  # fmt: skip


def trade_gaps(data_dir: Path, venue: str, symbol: str) -> list[tuple[int, int]]:
    gaps = load_gaps(data_dir)
    rows = gaps.filter(
        (pl.col("venue") == venue) & (pl.col("symbol").is_in([symbol, "*"]))
        & (pl.col("stream") == "trades") & (pl.col("reason") != "clock")
    )  # fmt: skip
    return [(int(a), int(b)) for a, b in rows.select("start_ns", "end_ns").iter_rows()]


def binance_vision(data_dir: Path, venue: str, symbol: str, day: date) -> VisionResult:
    if venue != "binance-usdm":
        raise ValueError("Binance Vision covers binance-usdm only")
    name = f"{symbol}-aggTrades-{day}.zip"
    folder = external_dir(data_dir, "binance_vision") / symbol
    with httpx.Client(timeout=300, follow_redirects=True) as http:
        zip_path = download(f"{VISION}/{symbol}/{name}", folder / name, client=http)
        sum_path = download(
            f"{VISION}/{symbol}/{name}.CHECKSUM", folder / f"{name}.CHECKSUM", client=http
        )
    ok = check_sha256(zip_path, sum_path)
    return compare_vision(
        vision_ids(zip_path), recorded_ids(data_dir, venue, symbol, day),
        trade_gaps(data_dir, venue, symbol), venue, symbol, day, ok,
    )  # fmt: skip


# ---- Tardis ---------------------------------------------------------------------------------


@dataclass
class TardisResult:
    venue: str
    symbol: str
    date: str
    seconds_compared: int
    seconds_equal: int
    agreement_pct: float
    seconds_skipped: int
    first_diff_utc: str


def tardis_top(path: Path, tick: float, lo: int, size: int) -> pl.DataFrame:
    """Replay a Tardis incremental_book_L2 csv.gz and return the top 10 at each batch.

    Rows sharing a `local_timestamp` form one batch. A run of snapshot rows that follows
    incremental rows starts a new book.
    """
    with gzip.open(path, "rb") as handle:
        frame = pl.read_csv(
            handle.read(),
            columns=["timestamp", "local_timestamp", "is_snapshot", "side", "price", "amount"],
        )
    frame = frame.with_columns(
        pl.when(pl.col("side") == "bid").then(1).otherwise(-1).cast(pl.Int8).alias("s"),
        (pl.col("price") / tick).round(0).cast(pl.Int64).alias("tick"),
    )
    local = frame["local_timestamp"].to_numpy()
    snap = frame["is_snapshot"].to_numpy().astype(bool)
    new_batch = np.ones(len(local), dtype=bool)
    new_batch[1:] = local[1:] != local[:-1]
    starts = np.flatnonzero(new_batch)
    bstart = np.concatenate([starts, [len(local)]]).astype(np.int64)
    first_snap = snap[starts]
    prev_snap = np.concatenate([[False], first_snap[:-1]])
    bsnap = first_snap & ~prev_snap
    n = len(starts)
    state = BookState(lo, size, raw=False)
    out: dict[str, npt.NDArray[Any]] = {k: np.empty((n, TOP_N), np.int64) for k in ("bpx", "apx")}
    out |= {k: np.empty((n, TOP_N), np.float64) for k in ("bq", "aq")}
    flags = np.zeros(n, np.int8)
    pr = empty_probes()
    empty = np.zeros((0, 1), np.uint8)
    run_batches(
        state.qty, state.meta, state.px_raw, state.px_len, state.qty_raw, state.qty_len, lo, 0,
        frame["s"].to_numpy(), frame["tick"].to_numpy(),
        frame["amount"].to_numpy().astype(np.float64),
        empty, np.zeros(0, np.uint8), empty, np.zeros(0, np.uint8), bstart, bsnap, 0, n,
        out["bpx"], out["bq"], out["apx"], out["aq"], flags,
        np.zeros(n, np.bool_), np.full(n, -1, np.int64), CRC_TABLE,
        pr["pr_batch"], pr["pr_snap_off"], pr["snap_tick"], pr["snap_qty"],
        pr["pr_ex_off"], pr["ex_tick"], pr["pr_out"],
    )  # fmt: skip
    ts = frame["timestamp"].to_numpy()[starts] * 1000  # microseconds -> ns
    return _top_frame(ts, out)


def _top_frame(ts: np.ndarray, out: dict[str, np.ndarray]) -> pl.DataFrame:
    cols: dict[str, Any] = {"ts": ts.astype(np.int64)}
    for k in range(TOP_N):
        cols[f"bid_px_{k + 1}"] = out["bpx"][:, k]
        cols[f"bid_qty_{k + 1}"] = out["bq"][:, k]
        cols[f"ask_px_{k + 1}"] = out["apx"][:, k]
        cols[f"ask_qty_{k + 1}"] = out["aq"][:, k]
    return pl.DataFrame(cols).sort("ts")


def compare_top(
    ours: pl.DataFrame,
    theirs: pl.DataFrame,
    live: list[tuple[int, int]],
    venue: str,
    symbol: str,
    day: date,
) -> TardisResult:
    """Compare as-of states at every whole second inside our LIVE intervals."""
    lo, hi = _ns(day), _ns(day + timedelta(days=1))
    grid = np.arange(lo, hi, NS, dtype=np.int64)
    keep = np.zeros(len(grid), dtype=bool)
    for lo_ns, hi_ns in live:
        keep |= (grid >= lo_ns) & (grid < hi_ns)
    skipped = int((~keep).sum())
    samples = pl.DataFrame({"ts": grid[keep]})
    cols = [c for c in theirs.columns if c != "ts"]
    a = samples.join_asof(ours.select(["ts", *cols]), on="ts").drop_nulls()
    b = samples.join_asof(theirs, on="ts").drop_nulls()
    both = a.join(b, on="ts", suffix="_t")
    equal = np.ones(both.height, dtype=bool)
    for c in cols:
        x = both[c].fill_null(NO_LEVEL if "px" in c else 0).to_numpy()
        y = both[f"{c}_t"].fill_null(NO_LEVEL if "px" in c else 0).to_numpy()
        if "qty" in c:
            equal &= np.isclose(x, y, rtol=1e-9, atol=1e-12)
        else:
            equal &= x == y
    first = both["ts"].to_numpy()[~equal]
    from datetime import UTC, datetime  # noqa: PLC0415

    return TardisResult(
        venue=venue, symbol=symbol, date=str(day), seconds_compared=both.height,
        seconds_equal=int(equal.sum()),
        agreement_pct=round(100.0 * equal.sum() / both.height, 4) if both.height else 0.0,
        seconds_skipped=skipped,
        first_diff_utc=(
            datetime.fromtimestamp(first[0] / 1e9, tz=UTC).isoformat() if len(first) else ""
        ),
    )  # fmt: skip


def our_top(data_dir: Path, venue: str, symbol: str, day: date) -> pl.DataFrame:
    """Rebuilt top 10 keyed on exchange time, from interim/topn."""
    parts = [
        pl.read_parquet(p)
        for p in sorted(
            partition(data_dir / "interim", "topn", venue, symbol, day).glob("part-*.parquet")
        )
    ]
    if not parts:
        raise FileNotFoundError(
            f"no interim/topn for {venue} {symbol} {day}: run `lob book rebuild` first"
        )
    frame = pl.concat(parts).sort("ts_exchange")
    return frame.rename({"ts_exchange": "ts"}).drop(
        "ts_recv", "batch_id", "event_idx", "sync_epoch"
    )


def book_live(data_dir: Path, venue: str, symbol: str, day: date) -> list[tuple[int, int]]:
    """LIVE intervals of the book on `day`: the day minus logged book gaps, `clock` aside."""
    gaps = load_gaps(data_dir).filter(
        (pl.col("venue") == venue) & (pl.col("symbol").is_in([symbol, "*"]))
        & (pl.col("stream") == "book") & (pl.col("reason") != "clock")
    ).sort("start_ns")  # fmt: skip
    lo, hi = _ns(day), _ns(day + timedelta(days=1))
    first = min(
        (
            int(x)
            for x in pl.read_parquet(
                day_files(data_dir, "book_l2", venue, symbol, day)[0], columns=["ts_recv"]
            )["ts_recv"][:1]
        ),
        default=hi,
    )
    live: list[tuple[int, int]] = []
    cursor = max(lo, first)
    for a, b in gaps.select("start_ns", "end_ns").iter_rows():
        if b <= cursor or a >= hi:
            continue
        if a > cursor:
            live.append((cursor, min(a, hi)))
        cursor = max(cursor, b)
    if cursor < hi:
        live.append((cursor, hi))
    return live


def tardis(
    data_dir: Path,
    venue: str,
    symbol: str,
    day: date,
    tick: float,
    tardis_symbol: str | None = None,
) -> TardisResult:
    if day.day != 1:
        raise ValueError("Tardis free samples exist only for the first day of each month")
    exchange = TARDIS_EXCHANGE[venue]
    tsym = tardis_symbol or symbol.replace("/", "-")
    url = f"{TARDIS}/{exchange}/incremental_book_L2/{day:%Y/%m/%d}/{tsym}.csv.gz"
    path = download(url, external_dir(data_dir, "tardis") / exchange / f"{day}_{tsym}.csv.gz")
    ours = our_top(data_dir, venue, symbol, day)
    mid = int(np.nanmedian(ours["bid_px_1"].to_numpy().astype(float)))
    lo = int(mid * 0.7)
    theirs = tardis_top(path, tick, lo, int(mid * 1.4) - lo + 1)
    return compare_top(ours, theirs, book_live(data_dir, venue, symbol, day), venue, symbol, day)


def upsert_csv(path: Path, result: VisionResult | TardisResult) -> None:
    """Keep one row per (venue, symbol, date) in the external_*.csv report."""
    row = {k: v for k, v in asdict(result).items()}
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [
                r
                for r in csv.DictReader(handle)
                if (r["venue"], r["symbol"], r["date"])
                != (row["venue"], row["symbol"], row["date"])
            ]
    rows.append(row)
    rows.sort(key=lambda r: (r["venue"], r["symbol"], r["date"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
