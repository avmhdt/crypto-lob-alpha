"""Metrics, ridge and LightGBM fits, the touch-taker check, feature parsing."""

from __future__ import annotations

import numpy as np
import pytest

from lob_alpha.execsim.touch import threshold_from_train, touch_taker
from lob_alpha.models import featureset, fit, metrics
from lob_alpha.models.train import Target


def test_oos_r2_uses_training_mean() -> None:
    y = np.array([1.0, -1.0, 2.0, 0.0])
    assert metrics.oos_r2(y, np.full(4, 0.5), 0.5) == 0.0
    assert metrics.oos_r2(y, y, 0.3) == 1.0
    s = metrics.score(y, np.array([0.5, -0.1, 0.0, 3.0]), 0.0)
    assert s.hit_rate_nonzero == pytest.approx(2 / 3)  # 0.0 prediction is a miss
    assert s.mae_ticks == pytest.approx((0.5 + 0.9 + 2.0 + 3.0) / 4)


def test_spearman_with_ties() -> None:
    a = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 20.0, 25.0, 40.0])
    assert metrics.spearman(a, b) == pytest.approx(1.0)
    assert metrics.spearman(a, -b) == pytest.approx(-1.0)


def test_ridge_recovers_planted_coefficients() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(20_000, 3))
    y = x @ np.array([0.5, -0.2, 0.0]) + 0.1 + rng.normal(0, 0.1, 20_000)
    xv = rng.normal(size=(5_000, 3))
    yv = xv @ np.array([0.5, -0.2, 0.0]) + 0.1
    m = fit.fit_ridge(x, y, xv, yv, [0.1, 1e6], float(y.mean()))
    assert m.params["alpha"] == 0.1
    assert np.allclose(m.coef, [0.5, -0.2, 0.0], atol=0.01)
    assert m.val_r2 > 0.99
    x[0, 0] = np.nan
    assert np.isfinite(m.predict(x)).all()


def test_lgbm_is_deterministic() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5_000, 4))
    y = np.sin(x[:, 0]) + rng.normal(0, 0.1, 5_000)
    cfg = {
        "n_estimators": 50,
        "early_stopping_rounds": 10,
        "num_threads": 2,
        "min_data_in_leaf": 20,
        "num_leaves": 7,
    }
    names = ["a", "b", "c", "d"]
    a = fit.fit_lgbm(x[:4000], y[:4000], x[4000:], y[4000:], cfg, names, 0.0)
    b = fit.fit_lgbm(x[:4000], y[:4000], x[4000:], y[4000:], cfg, names, 0.0)
    assert np.array_equal(a.predict(x), b.predict(x))
    assert a.val_r2 > 0.8


def test_touch_taker_hand_computed() -> None:
    # rows every 100 ms; horizon 1 s -> a trade blocks the next 10 rows
    ts = np.arange(30, dtype=np.int64) * 100_000_000
    yhat = np.zeros(30)
    yhat[[0, 5, 12, 25]] = [2.0, 3.0, -2.0, 0.1]
    y = np.zeros(30)
    y[[0, 12]] = [4.0, -1.0]
    mid = np.full(30, 1000.0)
    spr_e = np.full(30, 2.0)
    spr_x = np.full(30, 4.0)
    span = np.full(30, 1_000_000_000, dtype=np.int64)
    tt = touch_taker(ts, yhat, y, mid, spr_e, spr_x, span, threshold=1.0, fee_bps=5.0)
    assert tt.idx.tolist() == [0, 12]  # row 5 is inside the first trade, row 25 below threshold
    assert tt.side.tolist() == [1.0, -1.0]
    # long: (4 - 1 - 2) / 1000 * 1e4 - 10 = 0 bps; short: (1 - 3) / 1000 * 1e4 - 10 = -30 bps
    assert tt.net_bps.tolist() == pytest.approx([0.0, -30.0])
    assert tt.gross_bps.tolist() == pytest.approx([40.0, 10.0])
    assert threshold_from_train(np.array([-3.0, 1.0, 2.0, np.nan]), 0.5) == 2.0


def test_feature_names_parse() -> None:
    d = featureset.parse("mlofi_l7_ms1000")
    assert (d.kind, d.levels, d.window, d.window_unit) == ("mlofi", [7], 1000, "ms")
    d = featureset.parse("tsi_cnt_tr200")
    assert (d.kind, d.weight, d.window_unit) == ("trade_sign_imbalance", "count", "trades")
    d = featureset.parse("integrated_ofi_ev50")
    assert d.stored_columns[0] == "mlofi_l1_ev50" and len(d.stored_columns) == 10
    assert featureset.parse("qi_l5").levels == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        featureset.parse("imb_l1")
    t = Target.parse("dmid_ticks_time_5000_lat25")
    assert (t.clock, t.horizon, t.latency_ms) == ("time", 5000, 25)
    assert Target.parse("dmid_ticks_event_50").clock == "event"
