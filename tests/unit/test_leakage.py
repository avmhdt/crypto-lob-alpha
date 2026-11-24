"""Train-only fitting of normalisers, integrated-OFI PCA and micro-price; no test leakage
(AC-W-05)."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from lob_alpha.models import featureset
from lob_alpha.models.train import Runner, RunSpec

sys.path.insert(0, str(Path(__file__).parents[1] / "fixtures"))
from make_panels import FEATURES, SYMBOL, TARGET, VENUE, make_days

FIRST = date(2026, 3, 2)


def _spec(days: list[date], model: str = "ridge", **kw: object) -> RunSpec:
    cfg: dict[str, object] = {"alphas": [0.1, 10.0]}
    if model == "lgbm":
        cfg = {
            "n_estimators": 40,
            "early_stopping_rounds": 10,
            "num_threads": 1,
            "min_data_in_leaf": 50,
            "num_leaves": 7,
        }
    return RunSpec(VENUE, SYMBOL, TARGET, model, cfg, list(FEATURES), days, 0.1, 5.0, **kw)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[date]]:
    root = tmp_path_factory.mktemp("panels")
    return root, make_days(root, FIRST, 8, skip=(3,))


def test_prep_ignores_non_training_rows(data: tuple[Path, list[date]]) -> None:
    root, days = data
    r = Runner(_spec(days), root, root / "results")
    train = [r.panel(d) for d in days[:5]]
    cols = {k: np.concatenate([p.cols[k] for p in train]) for k in train[0].cols}
    a = featureset.Prep(list(FEATURES)).fit(cols)
    b = featureset.Prep(list(FEATURES)).fit({k: v.copy() for k, v in cols.items()})
    assert a.to_json() == b.to_json()
    # a test day with extreme values cannot move a fitted training statistic
    test = r.panel(days[6])
    shifted = {k: v * 100 + 7 for k, v in test.cols.items()}
    c = featureset.Prep(list(FEATURES)).fit(cols)
    c.transform(shifted)
    assert c.to_json() == a.to_json()


@pytest.mark.parametrize("model", ["ridge", "lgbm"])
def test_test_labels_never_reach_the_model(data: tuple[Path, list[date]], model: str) -> None:
    root, days = data
    r1 = Runner(_spec(days, model, folds=[0]), root, root / "r1")
    f1, t1 = r1.run_fold(r1._folds()[0], np.random.default_rng(0))
    r2 = Runner(_spec(days, model, folds=[0]), root, root / "r2")
    fold = r2._folds()[0]
    for d in fold.test:
        p = r2.panel(d)
        p.labels[TARGET] = -p.labels[TARGET] * 3 + 11
    f2, t2 = r2.run_fold(fold, np.random.default_rng(0))
    assert f1.prep.to_json() == f2.prep.to_json()
    assert f1.threshold == f2.threshold
    for (_, a), (_, b) in zip(t1, t2, strict=True):
        assert np.array_equal(a, b)


def test_planted_signal_found_and_permutation_kills_it(data: tuple[Path, list[date]]) -> None:
    root, days = data
    real = Runner(_spec(days), root, root / "res").run()
    assert real.pooled["oos_r2"] > 0.2
    perm = Runner(_spec(days, permute=True), root, root / "res").run()
    assert perm.pooled["oos_r2"] <= 0.001
