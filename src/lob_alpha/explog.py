"""Trial log and experiment log (FR-29, AC-W-10).

`results/trial_log.csv` follows backtest-overfitting-lab's Trial-log contract v1.0 (frozen
2026-11-01): the ten contract columns, then only `x_` columns. The writer is the contract's
reference writer, extended with this project's `x_` scope columns. Each row's return series
is `results/runs/<run_id>/returns.parquet` (git-ignored). `results/experiment_log.csv` holds
the prediction and execution metrics of the same run id.

Both logs are append-only. Logging happens inside the shared evaluation (`models.train`),
so no run can skip it; a failed run is logged with n_obs = 0.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

TRIAL_LOG = Path("results/trial_log.csv")
EXPERIMENT_LOG = Path("results/experiment_log.csv")
PERIODS_PER_YEAR = 365

CONTRACT_COLUMNS = (
    "run_id",
    "timestamp",
    "git_sha",
    "strategy_family",
    "params_json",
    "start",
    "end",
    "n_obs",
    "sharpe_annual",
    "returns_path",
)
TRIAL_COLUMNS = (
    *CONTRACT_COLUMNS,
    "x_periods_per_year",
    "x_venue",
    "x_symbol",
    "x_clock",
    "x_target",
    "x_horizon",
    "x_latency_ms",
    "x_status",
    "x_notes",
)
EXPERIMENT_COLUMNS = (
    "run_id",
    "venue",
    "symbol",
    "clock",
    "target",
    "horizon",
    "latency_ms",
    "n_samples",
    "oos_r2",
    "ic_spearman",
    "hit_rate_nonzero",
    "mae_ticks",
    "edge_bps",
    "net_bps_taker",
    "net_bps_maker",
    "n_trades",
    "sharpe_252",
    "status",
    "notes",
)
FAMILIES = ("lob_ofi_ols", "lob_ridge", "lob_lgbm", "lob_exec_taker", "lob_exec_maker")


def canonical_json(params: dict[str, Any]) -> str:
    return json.dumps(
        params, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def git_sha(log_path: Path) -> str:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    try:
        sha = git("rev-parse", "HEAD").strip()
        root = Path(git("rev-parse", "--show-toplevel").strip())
        log = log_path.resolve().relative_to(root.resolve()).as_posix()
        exp = (log_path.parent / EXPERIMENT_LOG.name).resolve().relative_to(root.resolve())
        status = git(
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            ".",
            f":(exclude){log}",
            f":(exclude){exp.as_posix()}",
        )
    except (OSError, ValueError, subprocess.CalledProcessError):
        return "0" * 40 + "-dirty"
    return f"{sha}-dirty" if status.strip() else sha


@dataclass
class TrialRow:
    family: str
    params: dict[str, Any]
    venue: str
    symbol: str
    clock: str
    target: str
    horizon: int
    latency_ms: int
    dates: list[date] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    status: str = "ok"  # ok | failed | discarded
    notes: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)  # experiment-log columns
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))


def sharpe(returns: list[float], periods: int) -> float:
    if len(returns) < 2:
        return 0.0
    std = statistics.stdev(returns)
    return statistics.fmean(returns) / std * math.sqrt(periods) if std > 0 else 0.0


def _append(path: Path, columns: tuple[str, ...], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, restval="", lineterminator="\n")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.10g}"
    return str(value)


def append_trial(row: TrialRow, path: Path = TRIAL_LOG) -> str:
    """Write the return series (if any), append the trial row and the experiment row."""
    if row.family not in FAMILIES:
        raise ValueError(f"unknown strategy family {row.family}")
    failed = row.status == "failed" or not row.returns
    out: dict[str, str] = {
        "run_id": row.run_id,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(path),
        "strategy_family": row.family,
        "params_json": canonical_json(row.params),
        "n_obs": "0",
        "x_periods_per_year": str(PERIODS_PER_YEAR),
        "x_venue": row.venue,
        "x_symbol": row.symbol,
        "x_clock": row.clock,
        "x_target": row.target,
        "x_horizon": str(row.horizon),
        "x_latency_ms": str(row.latency_ms),
        "x_status": row.status,
        "x_notes": row.notes,
    }
    if not failed:
        if any(b <= a for a, b in zip(row.dates, row.dates[1:], strict=False)):
            raise ValueError("return dates must be strictly increasing")
        if not all(math.isfinite(r) for r in row.returns):
            raise ValueError("returns must be finite")
        relative = f"runs/{row.run_id}/returns.parquet"
        target = path.parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "date": pa.array(row.dates, pa.date32()),
                "ret": pa.array(row.returns, pa.float64()),
            }
        )
        pq.write_table(table, target)
        out |= {
            "start": row.dates[0].isoformat(),
            "end": row.dates[-1].isoformat(),
            "n_obs": str(len(row.returns)),
            "sharpe_annual": f"{sharpe(row.returns, PERIODS_PER_YEAR):.10g}",
            "returns_path": relative,
        }
    _append(path, TRIAL_COLUMNS, out)
    exp = {c: _fmt(row.metrics.get(c)) for c in EXPERIMENT_COLUMNS}
    exp |= {
        "run_id": row.run_id,
        "venue": row.venue,
        "symbol": row.symbol,
        "clock": row.clock,
        "target": row.target,
        "horizon": str(row.horizon),
        "latency_ms": str(row.latency_ms),
        "sharpe_252": "" if failed else f"{sharpe(row.returns, 252):.10g}",
        "status": row.status,
        "notes": row.notes,
    }
    _append(path.parent / EXPERIMENT_LOG.name, EXPERIMENT_COLUMNS, exp)
    return row.run_id


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def check_logs(results: Path) -> list[str]:
    """AC-W-10 problems: header, one row per run id in both logs, no orphan run folders."""
    problems: list[str] = []
    trial = results / TRIAL_LOG.name
    exp = results / EXPERIMENT_LOG.name
    if trial.exists():
        with trial.open(encoding="utf-8", newline="") as fh:
            header = next(csv.reader(fh), [])
        if tuple(header[:10]) != CONTRACT_COLUMNS:
            problems.append(f"trial log header starts {header[:10]}")
        if any(not h.startswith("x_") for h in header[10:]):
            problems.append("trial log has a non-x_ extra column")
        raw = trial.read_bytes()
        if b"\r\n" in raw or raw.startswith(b"\xef\xbb\xbf"):
            problems.append("trial log must be LF, without BOM")
    t_ids = [r["run_id"] for r in read_rows(trial)]
    e_ids = [r["run_id"] for r in read_rows(exp)]
    if len(set(t_ids)) != len(t_ids):
        problems.append("duplicate run_id in the trial log")
    if set(t_ids) != set(e_ids) or len(e_ids) != len(set(e_ids)):
        problems.append("trial log and experiment log run ids differ")
    runs = results / "runs"
    if runs.exists():
        orphans = sorted(p.name for p in runs.iterdir() if p.is_dir() and p.name not in set(t_ids))
        if orphans:
            problems.append(f"run folders without a trial row: {orphans[:5]}")
    for r in read_rows(trial):
        try:
            uuid.UUID(r["run_id"], version=4)
        except ValueError:
            problems.append(f"bad run_id {r['run_id']}")
    return problems


PERMUTATION_MAX_R2 = 0.001


def verdict_permutation(results: Path) -> tuple[bool, str]:
    """AC-W-05 data check: the latest permutation run has pooled OOS R^2 <= 0.001."""
    trial = {r["run_id"]: r for r in read_rows(results / TRIAL_LOG.name)}
    exp = {r["run_id"]: r for r in read_rows(results / EXPERIMENT_LOG.name)}
    runs = [
        rid
        for rid, r in trial.items()
        if r["x_status"] == "ok" and json.loads(r["params_json"]).get("permute") is True
    ]
    if not runs:
        return False, "no permutation run in the trial log (lob train --permute)"
    rid = max(runs, key=lambda k: trial[k]["timestamp"])
    r2 = float(exp[rid]["oos_r2"])
    ok = r2 <= PERMUTATION_MAX_R2
    return ok, f"permutation run {rid}: pooled OOS R2 {r2:.5f} (max {PERMUTATION_MAX_R2})"
