"""Signal model artifact contract v1.0.0: writer and loader (FR-28, AC-W-11, AC-H-10).

`export_model` writes `artifacts/models/<model_name>/<model_version>/` from one fold of a
LightGBM run: `model.txt`, `feature_spec.json`, `metadata.json`, `parity_sample.csv`.
`load_model` reads a version folder back, checks every SHA-256 and the LightGBM
major.minor, and predicts from normalised features. docs/model_artifact.md restates the
contract and every feature definition with a worked example.

A version folder is immutable: export refuses to write into an existing one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from lob_alpha.models import featureset
from lob_alpha.models.data import DayPanel, load_day
from lob_alpha.models.train import Target
from lob_alpha.validate.rebuild import partition

F64 = npt.NDArray[np.float64]
SPEC_VERSION = "1.0.0"
PARITY_ROWS = 1200
PARITY_START_HOUR = 12
NS_PER_HOUR = 3_600_000_000_000
COMPATIBILITY = (
    "consumers MUST reject a different spec_version major; MUST apply normalization before predict"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_name(venue: str, symbol: str, target: str) -> str:
    return f"lob-{venue}-{symbol.replace('/', '')}-{target}"


def _definition(d: featureset.FeatureDef) -> str:
    w = f"{d.window} {d.window_unit}" if d.window is not None else ""
    win = {
        "ms": f"over the time window (t - {d.window} ms, t]",
        "events": f"over the last {d.window} book batches up to t",
        "trades": f"over the last {d.window} trades known at t",
        None: "",
    }[d.window_unit]
    if d.kind == "ofi":
        return (
            "sum of Cont-Kukanov-Stoikov e_n at level 1 between consecutive book batches, "
            f"{win}, divided by params.depth_norm (base units / base units)"
        )
    if d.kind == "mlofi":
        return (
            f"sum of e_n computed with level-{d.levels[0]} prices and sizes {win}, "
            "divided by params.depth_norm"
        )
    if d.kind == "integrated_ofi":
        return (
            "sum over l = 1..10 of params.weights[l-1] x (sum of level-l e_n "
            f"{win}), divided by params.depth_norm"
        )
    if d.kind == "queue_imbalance":
        n = max(d.levels)
        return (
            f"(sum bid_qty_1..{n} - sum ask_qty_1..{n}) / (sum bid_qty_1..{n} + sum ask_qty_1..{n})"
        )
    if d.kind == "microprice":
        return (
            "Stoikov (2018) adjustment g(I, S) in ticks from params.table rows "
            "[spread_ticks, imbalance_lower_edge, g], I = bid_qty_1 / (bid_qty_1 + ask_qty_1), "
            "S = ask_1 - bid_1 in ticks clamped to 1..params.max_spread_ticks; the last row of "
            "spread S whose edge <= I"
        )
    if d.kind == "weighted_mid":
        return (
            "(bid_1 ask_qty_1 + ask_1 bid_qty_1) / (bid_qty_1 + ask_qty_1) - (bid_1 + ask_1) / 2, "
            "ticks"
        )
    if d.kind == "spread":
        return "ask_1 - bid_1, ticks"
    if d.kind == "trade_sign_imbalance":
        wt = "trade size" if d.weight == "volume" else "1"
        return (
            f"sum(side x w) / sum(w) {win}, side +1 buyer-initiated, -1 seller-initiated, w = {wt}"
        )
    if d.kind == "mid_changes":
        return f"number of book batches {win} whose mid differs from the previous batch's mid"
    if d.kind == "realized_vol":
        return (
            "sqrt of the sum of squared changes of the mid (ticks) between consecutive "
            f"100 ms grid points {win}"
        )
    raise ValueError(f"no definition for kind {d.kind} ({w})")


def _params(d: featureset.FeatureDef, prep: featureset.Prep) -> dict[str, Any]:
    if d.kind in ("ofi", "mlofi"):
        return {"depth_norm": prep.depth_norm}
    if d.kind == "integrated_ofi":
        return {"depth_norm": prep.depth_norm, "weights": prep.weights[d.name]}
    if d.kind == "microprice":
        assert prep.micro is not None
        return {
            "imbalance_buckets": prep.micro.n_imb,
            "max_spread_ticks": prep.micro.max_spread,
            "table": prep.micro.to_rows(),
        }
    if d.kind == "trade_sign_imbalance":
        return {"weight": d.weight}
    if d.kind == "realized_vol":
        return {"sample_ms": 100}
    return {}


def feature_spec(
    name: str,
    version: str,
    venue: str,
    symbol: str,
    tick: float,
    lot: float,
    target: Target,
    prep: featureset.Prep,
) -> dict[str, Any]:
    feats = []
    for j, d in enumerate(prep.defs):
        feats.append(
            {
                "name": d.name,
                "kind": d.kind,
                "dtype": "float64",
                "levels": d.levels,
                "window": d.window,
                "window_unit": d.window_unit,
                "params": _params(d, prep),
                "definition": _definition(d),
                "normalization": {
                    "method": "zscore_clip",
                    "mean": float(prep.mean[j]),
                    "std": float(prep.std[j]),
                    "clip": featureset.CLIP,
                },
            }
        )
    return {
        "spec_version": SPEC_VERSION,
        "model_name": name,
        "model_version": version,
        "venue": venue,
        "symbols": [symbol],
        "tick_size": tick,
        "lot_size": lot,
        "book_depth": 10,
        "timestamp": "ts_recv",
        "clock": target.clock,
        "sample_grid_ms": 100 if target.clock == "time" else None,
        "target": {
            "name": target.name,
            "unit": "ticks",
            "horizon": {
                "kind": "clock" if target.clock == "time" else "event",
                "value": target.horizon,
                "unit": "ms" if target.clock == "time" else "events",
            },
            "latency_ms": target.latency_ms,
        },
        "features": feats,
        "compatibility": COMPATIBILITY,
    }


def _g17(x: float) -> str:
    return "" if not math.isfinite(x) else f"{x:.17g}"


def _ts_exchange(
    root: Path, venue: str, symbol: str, day: date, t: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Exchange time of the last book batch known (running-max ts_recv) at each grid time."""
    folder = partition(root / "interim", "topn", venue, symbol, day)
    hours = sorted({int((x % (24 * NS_PER_HOUR)) // NS_PER_HOUR) for x in (t[0], t[-1])})
    files = [
        f
        for f in sorted(folder.glob("part-*.parquet"))
        if int(f.name[5:7]) in range(hours[0] - 1, hours[-1] + 1)
    ]
    tab = pa.concat_tables([pq.read_table(f, columns=["ts_recv", "ts_exchange"]) for f in files])
    known = np.maximum.accumulate(np.asarray(tab.column("ts_recv").to_numpy(), dtype=np.int64))
    tex = np.asarray(tab.column("ts_exchange").to_numpy(), dtype=np.int64)
    i = np.searchsorted(known, t, side="right") - 1
    if np.any(i < 0):
        raise ValueError("parity rows start before the first batch read")
    return tex[i]


def parity_rows(panel: DayPanel, raw: F64, n_rows: int = PARITY_ROWS) -> slice:
    """First run of `n_rows` consecutive 100 ms grid rows from 12:00 UTC with every feature
    present: no grid point missing (no gap) and no NaN."""
    ts = panel.ts
    day0 = ts[0] - ts[0] % (24 * NS_PER_HOUR)
    start = int(np.searchsorted(ts, day0 + PARITY_START_HOUR * NS_PER_HOUR))
    good = ~np.isnan(raw).any(axis=1)
    step = np.r_[False, np.diff(ts) == 100_000_000]
    i = start
    while i + n_rows <= len(ts):
        win_good = good[i : i + n_rows]
        win_step = step[i + 1 : i + n_rows]
        if win_good.all() and win_step.all():
            return slice(i, i + n_rows)
        bad = np.flatnonzero(~win_good)
        bad_s = np.flatnonzero(~win_step) + 1
        jump = max(bad.max() if len(bad) else 0, bad_s.max() if len(bad_s) else 0)
        i += int(jump) + 1
    raise ValueError("no clean parity window on the test day")


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() + ("-dirty" if dirty else "")


@dataclass
class ExportResult:
    folder: Path
    name: str
    version: str
    n_parity: int
    max_abs_diff: float


def export_model(
    *,
    run_id: str,
    results: Path,
    data_root: Path,
    out_root: Path,
    version: str,
    tick: float,
    lot: float,
    fold: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> ExportResult:
    import lightgbm as lgb  # noqa: PLC0415

    run_dir = results / "runs" / run_id
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    met = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if cfg["model"] != "lgbm":
        raise ValueError("the artifact contract carries a LightGBM model; this run is ridge")
    folds = {f["fold"]: f for f in met["folds"]}
    k = max(folds) if fold is None else fold
    finfo = folds[k]
    fdir = data_root / "runs" / run_id / f"fold_{k:02d}"
    prep = featureset.Prep.from_json(json.loads((fdir / "prep.json").read_text(encoding="utf-8")))
    target = Target.parse(cfg["target"])
    venue, symbol = cfg["venue"], cfg["symbol"]
    name = model_name(venue, symbol, target.name)
    folder = out_root / name / version
    if folder.exists():
        raise FileExistsError(f"{folder} exists; a version folder is immutable")
    test_day = date.fromisoformat(finfo["test"][0])
    if test_day <= date.fromisoformat(finfo["val"][-1]):
        raise ValueError("evaluation day must be after the validation days")
    panel = load_day(
        data_root, venue, symbol, test_day, target.clock, featureset.stored_columns(prep.names), []
    )
    raw, norm = prep.transform(panel.cols)
    rows = parity_rows(panel, raw)
    booster = lgb.Booster(model_file=str(fdir / "model.txt"))
    pred = np.asarray(booster.predict(norm[rows]), dtype=np.float64)
    tex = _ts_exchange(data_root, venue, symbol, test_day, panel.ts[rows])

    folder.mkdir(parents=True)
    try:
        shutil.copyfile(fdir / "model.txt", folder / "model.txt")
        spec = feature_spec(name, version, venue, symbol, tick, lot, target, prep)
        (folder / "feature_spec.json").write_text(
            json.dumps(spec, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        header = ["ts_exchange", "ts_recv", "symbol"]
        header += prep.names + [f"{n}__norm" for n in prep.names] + ["prediction"]
        with (folder / "parity_sample.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(header)
            for i, r in enumerate(range(rows.start, rows.stop)):
                w.writerow(
                    [str(int(tex[i])), str(int(panel.ts[r])), symbol]
                    + [_g17(float(v)) for v in raw[r]]
                    + [_g17(float(v)) for v in norm[r]]
                    + [_g17(float(pred[i]))]
                )
        meta: dict[str, Any] = {
            "git_sha": git_head(),
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lightgbm_version": lgb.__version__,
            "model_sha256": sha256(folder / "model.txt"),
            "files_sha256": {
                "feature_spec.json": sha256(folder / "feature_spec.json"),
                "parity_sample.csv": sha256(folder / "parity_sample.csv"),
            },
            "fold": k,
            "train_start": finfo["train"][0],
            "train_end": finfo["train"][-1],
            "valid_start": finfo["val"][0],
            "valid_end": finfo["val"][-1],
            "train_days": finfo["train"],
            "validation_days": finfo["val"],
            "target_unit": "ticks",
            "metrics": {
                "oos_r2": finfo["oos_r2"],
                "ic_spearman": finfo["ic_spearman"],
                "n_test_days": len(finfo["test"]),
            },
            "test_days": finfo["test"],
            "run_id": run_id,
        } | (extra_metadata or {})
        (folder / "metadata.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    except Exception:
        shutil.rmtree(folder)
        raise
    loaded = load_model(folder)
    check = loaded.predict(norm[rows])
    return ExportResult(folder, name, version, len(pred), float(np.max(np.abs(check - pred))))


@dataclass
class LoadedModel:
    folder: Path
    spec: dict[str, Any]
    metadata: dict[str, Any]
    booster: Any

    @property
    def feature_names(self) -> list[str]:
        return [f["name"] for f in self.spec["features"]]

    def normalize(self, raw: F64) -> F64:
        out = np.empty_like(raw)
        for j, f in enumerate(self.spec["features"]):
            nm = f["normalization"]
            if nm["method"] == "none":
                out[:, j] = raw[:, j]
            elif nm["method"] == "zscore_clip":
                z = (raw[:, j] - nm["mean"]) / nm["std"]
                out[:, j] = np.clip(z, -nm["clip"], nm["clip"])
            else:
                raise ValueError(f"unknown normalization {nm['method']}")
        return out

    def predict(self, norm: F64) -> F64:
        return np.asarray(self.booster.predict(norm), dtype=np.float64)


def load_model(folder: Path) -> LoadedModel:
    """Read a version folder; verify SHA-256 values, spec_version major, LightGBM major.minor."""
    import lightgbm as lgb  # noqa: PLC0415

    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    spec = json.loads((folder / "feature_spec.json").read_text(encoding="utf-8"))
    if spec["spec_version"].split(".")[0] != SPEC_VERSION.split(".", maxsplit=1)[0]:
        raise ValueError(f"spec_version {spec['spec_version']} is not major {SPEC_VERSION}")
    if sha256(folder / "model.txt") != meta["model_sha256"]:
        raise ValueError("model.txt SHA-256 differs from metadata.json")
    for fname, digest in meta["files_sha256"].items():
        if sha256(folder / fname) != digest:
            raise ValueError(f"{fname} SHA-256 differs from metadata.json")
    if lgb.__version__.split(".")[:2] != meta["lightgbm_version"].split(".")[:2]:
        raise ValueError(
            f"LightGBM {lgb.__version__} does not match the artifact's {meta['lightgbm_version']}"
        )
    booster = lgb.Booster(model_file=str(folder / "model.txt"))
    if booster.feature_name() != [f["name"] for f in spec["features"]]:
        raise ValueError("model.txt feature order differs from feature_spec.json")
    return LoadedModel(folder, spec, meta, booster)


def read_parity(folder: Path) -> tuple[list[str], F64, F64, F64]:
    """(feature names, raw features, normalised features, predictions) of parity_sample.csv."""
    with (folder / "parity_sample.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    names = [h for h in header[3:-1] if not h.endswith("__norm")]
    data = np.array([[float(v) if v else np.nan for v in r[3:]] for r in body])
    k = len(names)
    return names, data[:, :k], data[:, k : 2 * k], data[:, -1]


SCHEMA_PATH = Path("docs/feature_spec.schema.json")
PARITY_TOL = 1e-9


def verify_artifact(folder: Path, schema_path: Path = SCHEMA_PATH) -> list[str]:
    """AC-W-11 problems for one version folder (empty when it round-trips)."""
    problems: list[str] = []
    try:
        model = load_model(folder)
    except (OSError, ValueError, KeyError) as exc:
        return [f"{folder}: {exc}"]
    try:
        import jsonschema  # noqa: PLC0415
    except ImportError:
        jsonschema = None
    if jsonschema is not None and schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        errors = list(jsonschema.Draft202012Validator(schema).iter_errors(model.spec))
        problems += [f"{folder}: schema: {e.message}" for e in errors[:5]]
    names, raw, norm, pred = read_parity(folder)
    if names != model.feature_names:
        problems.append(f"{folder}: parity columns differ from feature_spec.json")
        return problems
    if len(pred) < 1000:
        problems.append(f"{folder}: {len(pred)} parity rows (< 1000)")
    got = model.predict(norm)
    diff = float(np.max(np.abs(got - pred)))
    if not diff <= PARITY_TOL:
        problems.append(f"{folder}: prediction max |diff| {diff:.3e} > {PARITY_TOL}")
    renorm = model.normalize(raw)
    ndiff = float(np.nanmax(np.abs(renorm - norm)))
    if not ndiff <= PARITY_TOL:
        problems.append(f"{folder}: normalisation max |diff| {ndiff:.3e} > {PARITY_TOL}")
    return problems


def verify_artifacts(root: Path) -> tuple[bool, str]:
    folders = sorted(p.parent for p in root.glob("*/*/metadata.json"))
    if not folders:
        return False, f"no artifact under {root}"
    problems = [p for f in folders for p in verify_artifact(f)]
    if problems:
        return False, "; ".join(problems)
    return True, f"{len(folders)} artifact version(s) round-trip within {PARITY_TOL}"
