"""Trial log in the trial-log contract v1.0 format, experiment log beside it (AC-W-10)."""

from __future__ import annotations

import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path

import pyarrow.parquet as pq

from lob_alpha import explog


def _row(**kw: object) -> explog.TrialRow:
    base = dict(
        family="lob_ridge",
        params={"b": 1, "a": [1.5, "x"]},
        venue="binance-usdm",
        symbol="BTCUSDT",
        clock="time",
        target="dmid_ticks_time_1000",
        horizon=1000,
        latency_ms=0,
    )
    return explog.TrialRow(**(base | kw))  # type: ignore[arg-type]


def test_header_rows_series_and_check(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    days = [date(2026, 11, 3) + timedelta(days=k) for k in range(4)]
    rets = [0.001, -0.002, 0.0, 0.003]
    rid = explog.append_trial(_row(dates=days, returns=rets, metrics={"oos_r2": 0.07}), log)
    explog.append_trial(_row(status="failed", notes="boom"), log)
    raw = log.read_bytes()
    assert b"\r\n" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    with log.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    assert tuple(rows[0]) == explog.TRIAL_COLUMNS
    assert tuple(rows[0][:10]) == explog.CONTRACT_COLUMNS
    assert all(c.startswith("x_") for c in rows[0][10:])
    ok = dict(zip(rows[0], rows[1], strict=True))
    assert ok["run_id"] == rid
    assert ok["params_json"] == '{"a":[1.5,"x"],"b":1}'
    assert ok["start"] == "2026-11-03" and ok["end"] == "2026-11-06" and ok["n_obs"] == "4"
    assert ok["returns_path"] == f"runs/{rid}/returns.parquet"
    assert ok["x_periods_per_year"] == "365"
    import statistics

    want = statistics.fmean(rets) / statistics.stdev(rets) * math.sqrt(365)
    assert abs(float(ok["sharpe_annual"]) - want) < 1e-9
    series = pq.read_table(tmp_path / ok["returns_path"])
    assert series.column("ret").to_pylist() == rets
    assert str(series.schema.field("date").type) == "date32[day]"
    bad = dict(zip(rows[0], rows[2], strict=True))
    assert bad["n_obs"] == "0" and bad["sharpe_annual"] == "" and bad["returns_path"] == ""
    assert bad["x_status"] == "failed"
    exp = explog.read_rows(tmp_path / "experiment_log.csv")
    assert [r["run_id"] for r in exp] == [rows[1][0], rows[2][0]]
    assert tuple(exp[0]) == explog.EXPERIMENT_COLUMNS
    assert exp[0]["oos_r2"] == "0.07"
    assert explog.check_logs(tmp_path) == []
    (tmp_path / "runs" / "orphan").mkdir()
    assert any("without a trial row" in p for p in explog.check_logs(tmp_path))


def test_permutation_verdict(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    assert not explog.verdict_permutation(tmp_path)[0]
    days = [date(2026, 11, 3), date(2026, 11, 4)]
    explog.append_trial(
        _row(params={"permute": True}, dates=days, returns=[0.0, 0.1], metrics={"oos_r2": -0.002}),
        log,
    )
    ok, text = explog.verdict_permutation(tmp_path)
    assert ok, text


def test_canonical_json_matches_contract() -> None:
    params = {"z": 1, "a": {"y": 2.5, "b": None}, "u": "é"}
    assert explog.canonical_json(params) == json.dumps(
        params, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
