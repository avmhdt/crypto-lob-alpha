"""`lob book rebuild|validate`, `lob verify`, `lob dq report` and the external cross-checks."""

from __future__ import annotations

import gzip
import hashlib
import zipfile
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_rebuild import D1, D2, Capture, t, trades, write
from typer.testing import CliRunner

from lob_alpha import schemas
from lob_alpha.book.checksum import kraken_checksum
from lob_alpha.book.l2 import ASK, BID
from lob_alpha.cli import app
from lob_alpha.validate import external, report

V, S = "kraken", "BTC/USD"


def kraken_capture(root: Path, *, corrupt: bool) -> None:
    cap = Capture(V, S, 0.1, raw=True)
    cps: list[dict[str, Any]] = []

    def cp(batch: int, ts: int, bad: bool = False) -> None:
        bids, asks = cap.book.raw_levels(BID, 10), cap.book.raw_levels(ASK, 10)
        value = kraken_checksum(bids, asks) + (1 if bad else 0)
        cps.append(
            {"checkpoint_id": str(batch), "ts_recv": ts, "venue": V, "symbol": S,
             "kind": "kraken_crc32", "seq": batch, "batch_id": batch, "expected": value,
             "computed_live": value, "ok_live": True, "n_levels": 10}
        )  # fmt: skip

    snap = [(BID, f"{100 - i * 0.1:.1f}", "1.00000000") for i in range(5)]
    snap += [(ASK, f"{100.1 + i * 0.1:.1f}", "1.00000000") for i in range(5)]
    ts = t(D1, 1, 0)
    cp(cap.add(ts, snap, snap=True), ts)
    ts = t(D1, 1, 1)
    cp(cap.add(ts, [(BID, "99.9", "2.00000000")]), ts, bad=corrupt)
    cap.flush(root, D1, "part-01.parquet")
    write(root, "checkpoints", V, S, D1, "part-01.parquet", cps)
    trades(root, V, S, D1, [100.0])
    clock = [
        {"ts_recv": t(D1, 1, i), "venue": V, "server_time_ns": 0, "rtt_ns": 1,
         "offset_ns": -2_000_000}
        for i in range(3)
    ]  # fmt: skip
    folder = root / "clock" / f"venue={V}" / f"date={D1}"
    folder.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(clock, schema=schemas.CLOCK), folder / "part-01.parquet")


@pytest.mark.parametrize("corrupt", [False, True])
def test_rebuild_validate_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: bool
) -> None:
    data = tmp_path / "data"
    kraken_capture(data, corrupt=corrupt)
    monkeypatch.setenv("DATA_ROOT", str(data))
    out = tmp_path / "validation"
    runner = CliRunner()
    args = ["--start", str(D1), "--end", str(D1), "--venue", V, "--symbol", S]
    res = runner.invoke(app, ["book", "rebuild", *args])
    assert res.exit_code == 0, res.output
    assert "0 crossed or locked" in res.output
    res = runner.invoke(app, ["book", "validate", *args, "--out-dir", str(out)])
    assert res.exit_code == (1 if corrupt else 0), res.output
    assert "AC-W-02 PASS" in res.output
    summary = report.read_csv(out / "checkpoints_summary.csv")
    assert summary[0]["kind"] == "kraken_crc32" and summary[0]["compared"] == "2"
    assert summary[0]["mismatched"] == ("1" if corrupt else "0")
    res = runner.invoke(app, ["verify", "AC-W-03", "--out-dir", str(out)])
    assert res.exit_code == (1 if corrupt else 0), res.output
    res = runner.invoke(app, ["verify", "AC-W-02", "--out-dir", str(out)])
    assert res.exit_code == 0
    assert runner.invoke(app, ["verify", "AC-W-99"]).exit_code == 2
    failures = (out / "checkpoint_failures.csv").read_text(encoding="utf-8")
    assert ("kraken_crc32" in failures) == corrupt

    md = tmp_path / "dq.md"
    res = runner.invoke(
        app, ["dq", "report", "--start", str(D1), "--end", str(D2), "--out", str(md)]
    )
    assert res.exit_code == 0, res.output
    text = md.read_text(encoding="utf-8")
    assert "Complete days" in text and "-2.0" in text and "| kraken | BTC/USD |" in text


def test_validate_without_stats_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    res = CliRunner().invoke(app, ["book", "validate", "--start", str(D1), "--end", str(D1)])
    assert res.exit_code == 1


def test_verdicts_flag_exclusions_and_short_days() -> None:
    rows = [
        {"venue": "coinbase", "symbol": "X", "kind": "rest_l2", "date": "d1", "total": "100",
         "excluded_nonlive": "2", "excluded_gap": "0", "compared": "98", "mismatched": "0",
         "meets_min_per_day": "False"},
    ]  # fmt: skip
    ok, text = report.verdict_ac_w_03(rows)
    assert not ok and "2.0% excluded" in text and "under 144" in text
    assert report.verdict_ac_w_03([]) == (False, "no checkpoints")
    assert report.verdict_ac_w_02([])[0] is False
    ok, text = report.verdict_ac_w_02(
        [{"venue": "a", "symbol": "b", "date": "d", "crossed": "1", "locked": "0"}]
    )
    assert not ok and "1 crossed" in text


# ---- Binance Vision ------------------------------------------------------------------------


def _vision_zip(path: Path, ids: list[int], times: list[int], header: bool) -> None:
    lines = (
        ["agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker"]
        if header
        else []
    )
    lines += [f"{i},100.0,1.0,{i},{i},{ms},true" for i, ms in zip(ids, times, strict=True)]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(path.stem + ".csv", "\n".join(lines) + "\n")


@pytest.mark.parametrize("header", [True, False])
def test_vision_completeness_separates_gaps(tmp_path: Path, header: bool) -> None:
    day = date(2026, 11, 9)
    base = t(day, 0, 0) // 1_000_000
    zip_path = tmp_path / "BTCUSDT-aggTrades-2026-11-09.zip"
    _vision_zip(zip_path, [1, 2, 3, 4, 5], [base + 1000 * i for i in range(5)], header)
    checksum = tmp_path / "x.CHECKSUM"
    checksum.write_text(
        hashlib.sha256(zip_path.read_bytes()).hexdigest() + "  x.zip\n", encoding="ascii"
    )
    assert external.check_sha256(zip_path, checksum)
    vision = external.vision_ids(zip_path)
    recorded = pl.DataFrame({"agg_trade_id": [1, 2, 5, 9], "ts_exchange": [0, 0, 0, 0]})
    # ids 3 and 4 trade at 2 s and 3 s; the gap at 2 s plus the 2 s margin covers both
    gap = (t(day, 0, 2) - 100, t(day, 0, 2) + 100, None)
    res = external.compare_vision(vision, recorded, [gap], "binance-usdm", "BTCUSDT", day, True)
    assert res.vision_ids == 5 and res.missing == 2 and res.extra == 1
    assert res.missing_in_gaps == 2 and res.missing_live == 0
    assert res.completeness_pct == 60.0 and res.completeness_live_pct == 100.0
    out = tmp_path / "ext.csv"
    external.upsert_csv(out, res)
    external.upsert_csv(out, res)
    assert len(report.read_csv(out)) == 1


def test_recorded_ids_filter_on_exchange_day(tmp_path: Path) -> None:
    rows = [
        {"ts_exchange": t(D1, 23, 59), "ts_recv": t(D2, 0, 0.5), "venue": "binance-usdm",
         "symbol": "BTCUSDT", "trade_id": "7", "price": 1.0, "qty": 1.0, "side": 1, "batch_id": 1,
         "is_aggregated": True},
        {"ts_exchange": t(D2, 0, 1), "ts_recv": t(D2, 0, 1), "venue": "binance-usdm",
         "symbol": "BTCUSDT", "trade_id": "8", "price": 1.0, "qty": 1.0, "side": 1, "batch_id": 2,
         "is_aggregated": True},
    ]  # fmt: skip
    write(tmp_path, "trades", "binance-usdm", "BTCUSDT", D2, "part-00.parquet", rows)
    got = external.recorded_ids(tmp_path, "binance-usdm", "BTCUSDT", D1)
    assert got["agg_trade_id"].to_list() == [7]
    with pytest.raises(ValueError):
        external.binance_vision(tmp_path, "kraken", "BTC/USD", D1)


# ---- Tardis ---------------------------------------------------------------------------------


def test_tardis_replay_and_agreement(tmp_path: Path) -> None:
    day = date(2026, 12, 1)
    t0 = t(day, 0, 0) // 1000  # microseconds
    lines = ["exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount"]
    lines += [f"x,S,{t0},{t0},true,bid,100.0,1", f"x,S,{t0},{t0},true,ask,100.1,1"]
    lines += [f"x,S,{t0 + 1_500_000},{t0 + 1_500_000},false,bid,100.0,3"]
    lines += [
        f"x,S,{t0 + 5_000_000},{t0 + 5_000_000},true,bid,99.0,1",
        f"x,S,{t0 + 5_000_000},{t0 + 5_000_000},true,ask,99.5,1",
    ]
    path = tmp_path / "t.csv.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("\n".join(lines) + "\n")
    theirs = external.tardis_top(path, 0.1, 900, 200)
    assert theirs.height == 3
    assert theirs["bid_px_1"].to_list() == [1000, 1000, 990]  # snapshot resets the book
    assert theirs["bid_qty_1"].to_list() == [1.0, 3.0, 1.0]
    ours = theirs.clone()
    live = [(t(day, 0, 0), t(day, 0, 10))]
    res = external.compare_top(ours, theirs, live, "kraken", "BTC/USD", day)
    assert res.seconds_compared == 10 and res.agreement_pct == 100.0
    worse = ours.with_columns(
        pl.when(pl.col("ts") > t(day, 0, 4))
        .then(5.0)
        .otherwise(pl.col("bid_qty_1"))
        .alias("bid_qty_1")
    )
    res = external.compare_top(worse, theirs, live, "kraken", "BTC/USD", day)
    assert res.seconds_equal == 5 and res.first_diff_utc.startswith("2026-12-01T00:00:05")
    with pytest.raises(ValueError):
        external.tardis(tmp_path, "kraken", "BTC/USD", date(2026, 12, 2), 0.1)


def test_book_live_subtracts_gaps(tmp_path: Path) -> None:
    cap = Capture(V, S, 0.1)
    cap.add(t(D1, 0, 0), [(BID, "1.0", "1")], snap=True)
    cap.flush(tmp_path, D1, "part-00.parquet")
    gaps = [{"gap_id": "g", "venue": V, "symbol": S, "stream": "book", "start_ns": t(D1, 5, 0),
             "end_ns": t(D1, 6, 0), "duration_ms": 3.6e6, "reason": "disconnect", "host": "h",
             "recorder_version": "v"}]  # fmt: skip
    write(tmp_path, "gaps", V, S, D1, "part-05.parquet", gaps)
    live = external.book_live(tmp_path, V, S, D1)
    assert live == [(t(D1, 0, 0), t(D1, 5, 0)), (t(D1, 6, 0), t(D2, 0, 0))]
    assert np.isclose(sum(b - a for a, b in live) / 3.6e12, 23.0)


def test_download_once_with_rate_cap(tmp_path: Path) -> None:
    import httpx

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=b"x" * 4000)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        dest = tmp_path / "a" / "f.zip"
        external.download("https://example.test/f.zip", dest, client=client, max_bytes_per_s=1e9)
        external.download("https://example.test/f.zip", dest, client=client)
    assert dest.read_bytes() == b"x" * 4000
    assert len(calls) == 1


def test_complete_days_start_after_capture_start(tmp_path: Path) -> None:
    import json

    from lob_alpha.reports.gaplog import Stream
    from lob_alpha.validate.dq import complete_days

    (tmp_path / "capture.json").write_text(
        json.dumps({"started_ns": t(D1, 22, 0)}), encoding="utf-8"
    )
    streams = [Stream(V, S, "book"), Stream(V, S, "trades")]
    days = complete_days(tmp_path, streams, D1, D2)
    assert days.filter(pl.col("date") == str(D1))["complete"].item() is False
    assert days.filter(pl.col("date") == str(D2))["complete"].item() is True


def test_vision_gap_by_last_id_covers_a_lagging_feed() -> None:
    day = date(2026, 11, 3)
    base = t(day, 13, 0) // 1_000_000
    vision = pl.DataFrame(
        {"agg_trade_id": list(range(1, 11)), "transact_time": [base + 1000 * i for i in range(10)]}
    )
    recorded = pl.DataFrame({"agg_trade_id": [1, 2, 9, 10], "ts_exchange": [0, 0, 0, 0]})
    # received-time bounds far from the lost trades, but the last id before the gap was 2
    gap = (t(day, 14, 0), t(day, 14, 0) + 4 * 10**9, 2)
    res = external.compare_vision(vision, recorded, [gap], "binance-usdm", "BTCUSDT", day, True)
    assert res.missing == 6 and res.missing_in_gaps == 6 and res.missing_live == 0


def test_complete_days_leave_out_today(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from lob_alpha.reports.gaplog import Stream
    from lob_alpha.validate import dq

    (tmp_path / "capture.json").write_text(
        json.dumps({"started_ns": t(D1, 0, 0)}), encoding="utf-8"
    )
    streams = [Stream(V, S, "book"), Stream(V, S, "trades")]
    monkeypatch.setattr(dq, "utc_today", lambda: D2)  # D2 is still in progress
    days = dq.complete_days(tmp_path, streams, D1, D2)
    assert days.filter(pl.col("date") == str(D1))["complete"].item() is True
    assert days.filter(pl.col("date") == str(D2)).is_empty()
