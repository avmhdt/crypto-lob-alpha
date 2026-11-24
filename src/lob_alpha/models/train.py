"""`lob train`: walk-forward fit, OOS predictions, run folder and log rows (FR-20, FR-29).

One run = one (venue, symbol, clock, target, model configuration) over every fold of the
stream's complete days. Per fold: purge the boundary days, fit `Prep` on the training days
only, fit the model (ridge alpha / LightGBM early stopping on the validation day), predict
the test day. Outputs:

* `results/runs/<run_id>/config.json`, `metrics.json`, `returns.parquet` (git-ignored);
* `<DATA_ROOT>/runs/<run_id>/predictions.parquet` and `fold_<k>/` (prep.json, model.txt
  for LightGBM), kept out of the repository because they are large;
* one row in `results/trial_log.csv` and one in `results/experiment_log.csv`, also when
  the run fails.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from lob_alpha import explog
from lob_alpha.execsim import touch
from lob_alpha.models import featureset, fit, metrics, splits
from lob_alpha.models.data import DayPanel, load_day

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
MS = 1_000_000
TARGET_RE = re.compile(r"dmid_ticks_(time|event)_(\d+)(?:_lat(\d+))?")


@dataclass(frozen=True)
class Target:
    name: str
    clock: str
    horizon: int
    latency_ms: int

    @classmethod
    def parse(cls, name: str) -> Target:
        m = TARGET_RE.fullmatch(name)
        if m is None:
            raise ValueError(f"unknown target {name!r}")
        return cls(name, m.group(1), int(m.group(2)), int(m.group(3) or 0))


@dataclass
class RunSpec:
    venue: str
    symbol: str
    target: str
    model: str  # ridge | lgbm
    model_cfg: dict[str, Any]
    features: list[str]
    days: list[date]
    tick_size: float
    taker_fee_bps: float
    train_days: int = 5
    val_days: int = 1
    test_days: int = 1
    step_days: int = 1
    purge_s: float = 31.0
    folds: list[int] | None = None  # subset of fold indices; None = all
    permute: bool = False  # AC-W-05 permutation run: shuffle train and val labels
    seed: int = fit.SEED
    exec_quantile: float = touch.DEFAULT_QUANTILE
    notes: str = ""

    @classmethod
    def from_params(cls, params: dict[str, Any], tick_size: float, notes: str = "") -> RunSpec:
        """Rebuild a spec from a run's config.json (the trial row's params_json)."""
        wf = params["walk_forward"]
        return cls(
            venue=params["venue"],
            symbol=params["symbol"],
            target=params["target"],
            model=params["model"],
            model_cfg=dict(params["model_cfg"]),
            features=list(params["features"]),
            days=[date.fromisoformat(d) for d in params["days"]],
            tick_size=tick_size,
            taker_fee_bps=float(params["exec"]["taker_fee_bps"]),
            train_days=int(wf["train_days"]),
            val_days=int(wf["val_days"]),
            test_days=int(wf["test_days"]),
            step_days=int(wf["step_days"]),
            purge_s=float(wf["purge_s"]),
            folds=wf["folds"],
            permute=bool(params["permute"]),
            seed=int(params["seed"]),
            exec_quantile=float(params["exec"]["quantile"]),
            notes=notes,
        )

    @property
    def tgt(self) -> Target:
        return Target.parse(self.target)

    def params(self) -> dict[str, Any]:
        return {
            "venue": self.venue,
            "symbol": self.symbol,
            "clock": self.tgt.clock,
            "target": self.target,
            "model": self.model,
            "model_cfg": dict(sorted(self.model_cfg.items())),
            "features": list(self.features),
            "days": [d.isoformat() for d in self.days],
            "walk_forward": {
                "train_days": self.train_days,
                "val_days": self.val_days,
                "test_days": self.test_days,
                "step_days": self.step_days,
                "purge_s": self.purge_s,
                "folds": self.folds,
            },
            "permute": self.permute,
            "seed": self.seed,
            "exec": {
                "check": "touch_taker_v0",
                "quantile": self.exec_quantile,
                "taker_fee_bps": self.taker_fee_bps,
            },
        }


@dataclass
class FoldResult:
    fold: splits.Fold
    prep: featureset.Prep
    model: fit.Fitted
    train_mean: float
    threshold: float
    n_train: int
    n_val: int
    scores: metrics.Scores
    seconds: float


@dataclass
class RunResult:
    run_id: str
    folds: list[FoldResult] = field(default_factory=list)
    pooled: dict[str, Any] = field(default_factory=dict)
    per_day: list[dict[str, Any]] = field(default_factory=list)


def _concat(parts: list[DayPanel]) -> tuple[dict[str, F64], dict[str, F64], list[slice], I64]:
    cols = {k: np.concatenate([p.cols[k] for p in parts]) for k in parts[0].cols}
    labs = {k: np.concatenate([p.labels[k] for p in parts]) for k in parts[0].labels}
    segs, a = [], 0
    for p in parts:
        segs.append(slice(a, a + p.n))
        a += p.n
    ts = np.concatenate([p.ts for p in parts])
    return cols, labs, segs, ts


def exit_values(panel: DayPanel, tgt: Target) -> tuple[F64, F64, F64, I64]:
    """Entry mid, entry spread, exit spread and entry-to-exit span for every row."""
    mid, spread, ts = panel.cols["mid_ticks"], panel.cols["spread_ticks"], panel.ts
    n = panel.n
    if tgt.clock == "time":
        lat, hor = tgt.latency_ms * MS, tgt.horizon * MS
        ie = np.searchsorted(ts, ts + lat, side="right") - 1
        ix = np.searchsorted(ts, ts + lat + hor, side="right") - 1
        span = np.full(n, lat + hor, dtype=np.int64)
    else:
        ie = np.arange(n)
        ix = np.minimum(np.arange(n) + tgt.horizon, n - 1)
        span = (ts[ix] - ts).astype(np.int64)
    return mid[ie], spread[ie], spread[ix], span


class Runner:
    def __init__(self, spec: RunSpec, data_root: Path, results: Path) -> None:
        self.spec = spec
        self.data_root = data_root
        self.results = results
        self.cache: dict[date, DayPanel] = {}
        self.columns = featureset.stored_columns(spec.features)

    def _folds(self) -> list[splits.Fold]:
        s = self.spec
        folds = splits.walk_forward(s.days, s.train_days, s.val_days, s.test_days, s.step_days)
        if s.folds is not None:
            folds = [f for f in folds if f.index in set(s.folds)]
        return folds

    def panel(self, day: date) -> DayPanel:
        if day not in self.cache:
            s = self.spec
            self.cache[day] = load_day(
                self.data_root, s.venue, s.symbol, day, s.tgt.clock, self.columns, [s.target]
            )
        return self.cache[day]

    def _purged(self, fold: splits.Fold, day: date) -> DayPanel:
        p = self.panel(day)
        if splits.needs_purge(fold, day):
            return p.take(splits.purge_keep(p.ts, day, self.spec.purge_s))
        return p

    def run_fold(
        self, fold: splits.Fold, rng: np.random.Generator
    ) -> tuple[FoldResult, list[tuple[DayPanel, F64]]]:
        s = self.spec
        t0 = time.perf_counter()
        for d in list(self.cache):
            if d not in fold.all_days():
                del self.cache[d]
        tr_cols, tr_labs, segs, _ = _concat([self._purged(fold, d) for d in fold.train])
        va_cols, va_labs, _, _ = _concat([self._purged(fold, d) for d in fold.val])
        prep = featureset.Prep(list(s.features)).fit(tr_cols, segs)
        y, yv = tr_labs[s.target], va_labs[s.target]
        keep, keep_v = ~np.isnan(y), ~np.isnan(yv)
        _, x = prep.transform(tr_cols)
        _, xv = prep.transform(va_cols)
        x, y, xv, yv = x[keep], y[keep], xv[keep_v], yv[keep_v]
        if s.permute:
            y, yv = rng.permutation(y), rng.permutation(yv)
        train_mean = float(np.mean(y))
        if s.model == "ridge":
            alphas = [float(a) for a in s.model_cfg.get("alphas", [0.1, 1, 10, 100, 1000])]
            model = fit.fit_ridge(x, y, xv, yv, alphas, train_mean)
        elif s.model == "lgbm":
            cfg = dict(s.model_cfg) | {"seed": s.seed}
            model = fit.fit_lgbm(x, y, xv, yv, cfg, list(s.features), train_mean)
        else:
            raise ValueError(f"unknown model {s.model}")
        threshold = touch.threshold_from_train(model.predict(x), s.exec_quantile)
        tests = []
        ys, yhs = [], []
        for d in fold.test:
            p = self.panel(d)
            _, xt = prep.transform(p.cols)
            yhat = model.predict(xt)
            tests.append((p, yhat))
            ok = ~np.isnan(p.labels[s.target])
            ys.append(p.labels[s.target][ok])
            yhs.append(yhat[ok])
        sc = metrics.score(np.concatenate(ys), np.concatenate(yhs), train_mean)
        res = FoldResult(
            fold, prep, model, train_mean, threshold, len(y), len(yv), sc, time.perf_counter() - t0
        )
        return res, tests

    def run(self, run_id: str | None = None) -> RunResult:
        s = self.spec
        row = explog.TrialRow(
            family=f"lob_{s.model}",
            params=s.params(),
            venue=s.venue,
            symbol=s.symbol,
            clock=s.tgt.clock,
            target=s.target,
            horizon=s.tgt.horizon,
            latency_ms=s.tgt.latency_ms,
            notes=s.notes,
        )
        if run_id is not None:
            row.run_id = run_id
        out = self.results / "runs" / row.run_id
        heavy = self.data_root / "runs" / row.run_id
        out.mkdir(parents=True, exist_ok=True)
        heavy.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(
            json.dumps(s.params(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result = RunResult(row.run_id)
        try:
            self._run(result, out, heavy, row)
        except Exception as exc:
            row.status = "failed"
            row.notes = (s.notes + " | " if s.notes else "") + f"failed: {exc!r}"[:400]
            (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            explog.append_trial(row, self.results / explog.TRIAL_LOG.name)
            raise
        explog.append_trial(row, self.results / explog.TRIAL_LOG.name)
        return result

    def _run(self, result: RunResult, out: Path, heavy: Path, row: explog.TrialRow) -> None:
        s = self.spec
        rng = np.random.default_rng(s.seed)
        folds = self._folds()
        if not folds:
            raise ValueError(f"no folds from {len(s.days)} complete days")
        pred_parts: list[pa.Table] = []
        sse = sst = 0.0
        all_y, all_yh = [], []
        dates, rets = [], []
        trades = net_sum = gross_sum = 0.0
        for fold in folds:
            fr, tests = self.run_fold(fold, rng)
            result.folds.append(fr)
            fdir = heavy / f"fold_{fold.index:02d}"
            fdir.mkdir(exist_ok=True)
            (fdir / "prep.json").write_text(json.dumps(fr.prep.to_json()), encoding="utf-8")
            if fr.model.kind == "lgbm":
                fr.model.booster.save_model(
                    str(fdir / "model.txt"), num_iteration=fr.model.best_iteration
                )
            else:
                (fdir / "ridge.json").write_text(
                    json.dumps({"coef": fr.model.coef, "intercept": fr.model.intercept}),
                    encoding="utf-8",
                )
            for p, yhat in tests:
                y = p.labels[s.target]
                ok = ~np.isnan(y)
                sse += float(np.sum((y[ok] - yhat[ok]) ** 2))
                sst += float(np.sum((y[ok] - fr.train_mean) ** 2))
                all_y.append(y[ok])
                all_yh.append(yhat[ok])
                day_sc = metrics.score(y[ok], yhat[ok], fr.train_mean)
                mid_e, spr_e, spr_x, span = exit_values(p, s.tgt)
                tt = touch.touch_taker(
                    p.ts, yhat, y, mid_e, spr_e, spr_x, span, fr.threshold, s.taker_fee_bps
                )
                ret = float(np.sum(tt.net_bps) * 1e-4)
                dates.append(p.day)
                rets.append(ret)
                trades += len(tt.idx)
                net_sum += float(np.sum(tt.net_bps))
                gross_sum += float(np.sum(tt.gross_bps))
                result.per_day.append(
                    {"day": p.day.isoformat(), "fold": fold.index}
                    | day_sc.as_dict()
                    | {
                        "n_trades": len(tt.idx),
                        "net_bps_mean": float(np.mean(tt.net_bps)) if len(tt.idx) else None,
                        "ret": ret,
                    }
                )
                pred_parts.append(
                    pa.table(
                        {
                            "day": pa.array([p.day] * p.n, pa.date32()),
                            "ts": pa.array(p.ts, pa.int64()),
                            "fold": pa.array(np.full(p.n, fold.index), pa.int16()),
                            "y": pa.array(y, pa.float64()),
                            "yhat": pa.array(yhat, pa.float64()),
                            "train_mean": pa.array(np.full(p.n, fr.train_mean), pa.float64()),
                        }
                    )
                )
        pq.write_table(pa.concat_tables(pred_parts), heavy / "predictions.parquet")
        yy, yh = np.concatenate(all_y), np.concatenate(all_yh)
        pooled = metrics.score(yy, yh, 0.0).as_dict()
        pooled["oos_r2"] = 1.0 - sse / sst if sst > 0 else float("nan")
        result.pooled = pooled | {
            "n_test_days": len(dates),
            "n_trades": int(trades),
            "edge_bps": gross_sum / trades if trades else None,
            "net_bps_taker": net_sum / trades if trades else None,
        }
        fold_rows = [
            {
                "fold": fr.fold.index,
                "train": [d.isoformat() for d in fr.fold.train],
                "val": [d.isoformat() for d in fr.fold.val],
                "test": [d.isoformat() for d in fr.fold.test],
                "n_train": fr.n_train,
                "n_val": fr.n_val,
                "model": fr.model.params,
                "best_iteration": fr.model.best_iteration,
                "val_r2": fr.model.val_r2,
                "grid": fr.model.grid,
                "train_mean": fr.train_mean,
                "threshold": fr.threshold,
                "seconds": round(fr.seconds, 1),
            }
            | fr.scores.as_dict()
            for fr in result.folds
        ]
        (out / "metrics.json").write_text(
            json.dumps(
                {"pooled": result.pooled, "folds": fold_rows, "per_day": result.per_day},
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        row.dates, row.returns = dates, rets
        row.metrics = {
            "n_samples": pooled["n"],
            "oos_r2": pooled["oos_r2"],
            "ic_spearman": pooled["ic_spearman"],
            "hit_rate_nonzero": pooled["hit_rate_nonzero"],
            "mae_ticks": pooled["mae_ticks"],
            "edge_bps": result.pooled["edge_bps"],
            "net_bps_taker": result.pooled["net_bps_taker"],
            "n_trades": int(trades),
        }
