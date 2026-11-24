"""Ridge baseline and LightGBM regression on one walk-forward fold (FR-20).

Inputs are the normalised features from `featureset.Prep` (train-only z-scores, clipped).
Ridge gets missing values as 0, the training mean; LightGBM sees them as missing.

* Ridge: scikit-learn `Ridge` for each alpha in the grid; the alpha with the best
  validation-day OOS R^2 is kept (ties: the smaller alpha).
* LightGBM: one configuration per run (objective, num_leaves, ...), early stopping on the
  validation day's L2 loss, `deterministic=true`, `force_row_wise=true`, a fixed thread
  count and seed (NFR-05).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from lob_alpha.models.metrics import oos_r2

F64 = npt.NDArray[np.float64]
SEED = 20261102


@dataclass
class Fitted:
    kind: str  # ridge | lgbm
    params: dict[str, Any]
    coef: list[float] = field(default_factory=list)
    intercept: float = 0.0
    booster: Any = None
    best_iteration: int = 0
    val_r2: float = float("nan")
    grid: list[dict[str, float]] = field(default_factory=list)

    def predict(self, x: F64) -> F64:
        if self.kind == "ridge":
            out: F64 = np.nan_to_num(x, nan=0.0) @ np.asarray(self.coef) + self.intercept
            return out
        pred = self.booster.predict(x, num_iteration=self.best_iteration)
        return np.asarray(pred, dtype=np.float64)


def fit_ridge(x: F64, y: F64, xv: F64, yv: F64, alphas: list[float], train_mean: float) -> Fitted:
    from sklearn.linear_model import Ridge  # noqa: PLC0415

    xz, xvz = np.nan_to_num(x, nan=0.0), np.nan_to_num(xv, nan=0.0)
    best: Fitted | None = None
    grid = []
    for a in sorted(alphas):
        model = Ridge(alpha=a, fit_intercept=True, solver="cholesky").fit(xz, y)
        r2 = oos_r2(yv, np.asarray(model.predict(xvz), dtype=np.float64), train_mean)
        grid.append({"alpha": float(a), "val_r2": r2})
        if best is None or r2 > best.val_r2:
            best = Fitted(
                "ridge",
                {"alpha": float(a)},
                coef=[float(c) for c in model.coef_],
                intercept=float(model.intercept_),
                val_r2=r2,
            )
    assert best is not None
    best.grid = grid
    return best


def lgbm_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """LightGBM parameters for one run from the resolved model config."""
    return {
        "objective": cfg.get("objective", "regression"),
        "num_leaves": int(cfg.get("num_leaves", 63)),
        "learning_rate": float(cfg.get("learning_rate", 0.05)),
        "min_data_in_leaf": int(cfg.get("min_data_in_leaf", 500)),
        "feature_fraction": float(cfg.get("feature_fraction", 0.8)),
        "deterministic": bool(cfg.get("deterministic", True)),
        "force_row_wise": True,
        "num_threads": int(cfg.get("num_threads", 8)),
        "seed": int(cfg.get("seed", SEED)),
        "metric": "l2",
        "verbosity": -1,
    }


def fit_lgbm(
    x: F64, y: F64, xv: F64, yv: F64, cfg: dict[str, Any], names: list[str], train_mean: float
) -> Fitted:
    import lightgbm as lgb  # noqa: PLC0415

    params = lgbm_params(cfg)
    rounds = int(cfg.get("n_estimators", 2000))
    stop = int(cfg.get("early_stopping_rounds", 100))
    dtrain = lgb.Dataset(x, label=y, feature_name=names, free_raw_data=True)
    dval = lgb.Dataset(xv, label=yv, reference=dtrain, free_raw_data=True)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=rounds,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[lgb.early_stopping(stop, first_metric_only=True, verbose=False)],
    )
    best = int(booster.best_iteration) or rounds
    fitted = Fitted("lgbm", params | {"n_estimators": rounds, "early_stopping_rounds": stop})
    fitted.booster = booster
    fitted.best_iteration = best
    fitted.val_r2 = oos_r2(yv, fitted.predict(xv), train_mean)
    return fitted
