"""Feature importance on OOS days: LightGBM gain and mean |SHAP| (FR-26, AC-H-04).

SHAP values come from LightGBM's own TreeSHAP (`predict(pred_contrib=True)`), the same
exact path-dependent algorithm as `shap.TreeExplainer` on a LightGBM model, without the
extra dependency. Per fold: up to `sample` test-day rows (fixed seed), mean |SHAP| per
feature in ticks of predicted mid change, and the booster's total gain share. Across
folds: the mean of both, then ranks. Each feature gets one plain-language sentence on the
mechanism, keyed by its kind.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt

from lob_alpha.models import featureset
from lob_alpha.models.data import load_day

F64 = npt.NDArray[np.float64]
SEED = 20261102

MECHANISM = {
    "ofi": "Net limit-order flow at the touch (bids added or asks removed minus the reverse) "
    "pushes the mid up before trades do.",
    "mlofi": "The same order-flow pressure deeper in the book: queue building behind the touch "
    "signals where the touch will move.",
    "integrated_ofi": "Order flow summed across ten levels with PCA weights: one number for "
    "book-wide buying or selling pressure.",
    "queue_imbalance": "A thin ask queue against a thick bid queue is likely to be eaten first, "
    "so the next mid move is up.",
    "microprice": "Stoikov's micro-price: the expected mid at the next move given imbalance "
    "and spread, a direct forecast of the drift.",
    "weighted_mid": "The size-weighted mid sits nearer the thinner side, where the next price "
    "change is likely to happen.",
    "spread": "A wide spread means a mid move is half a spread away from either side being "
    "filled, so moves are larger and more uncertain.",
    "trade_sign_imbalance": "Aggressive buying or selling tends to continue for a while "
    "(order splitting), so recent trade signs predict the next ones.",
    "mid_changes": "How busy the mid has been: activity scales the size of the next move.",
    "realized_vol": "Recent volatility scales the size of the next move.",
}


@dataclass
class Importance:
    feature: str
    kind: str
    mean_abs_shap: float
    gain_share: float
    shap_rank: int
    gain_rank: int
    folds: int


def fold_importance(
    data_root: Path,
    results: Path,
    run_id: str,
    fold: int,
    cfg: dict[str, object],
    sample: int = 200_000,
) -> tuple[F64, F64, list[str]]:
    import lightgbm as lgb  # noqa: PLC0415

    fdir = data_root / "runs" / run_id / f"fold_{fold:02d}"
    prep = featureset.Prep.from_json(json.loads((fdir / "prep.json").read_text(encoding="utf-8")))
    booster = lgb.Booster(model_file=str(fdir / "model.txt"))
    metrics = json.loads((results / "runs" / run_id / "metrics.json").read_text(encoding="utf-8"))
    finfo = next(f for f in metrics["folds"] if f["fold"] == fold)
    xs = []
    for d in finfo["test"]:
        panel = load_day(
            data_root,
            str(cfg["venue"]),
            str(cfg["symbol"]),
            date.fromisoformat(d),
            "time" if "time" in str(cfg["target"]) else "event",
            featureset.stored_columns(prep.names),
            [],
        )
        xs.append(prep.transform(panel.cols)[1])
    x = np.concatenate(xs)
    rng = np.random.default_rng(SEED + fold)
    if len(x) > sample:
        x = x[np.sort(rng.choice(len(x), sample, replace=False))]
    contrib = np.asarray(booster.predict(x, pred_contrib=True), dtype=np.float64)[:, :-1]
    gain = np.asarray(booster.feature_importance("gain"), dtype=np.float64)
    return np.abs(contrib).mean(axis=0), gain / gain.sum(), prep.names


def run_importance(data_root: Path, results: Path, run_id: str) -> list[Importance]:
    cfg = json.loads((results / "runs" / run_id / "config.json").read_text(encoding="utf-8"))
    if cfg["model"] != "lgbm":
        raise ValueError("importance needs a LightGBM run")
    met = json.loads((results / "runs" / run_id / "metrics.json").read_text(encoding="utf-8"))
    shap_sum: F64 | None = None
    gain_sum: F64 | None = None
    names: list[str] = []
    folds = [f["fold"] for f in met["folds"]]
    for k in folds:
        s, g, names = fold_importance(data_root, results, run_id, k, cfg)
        shap_sum = s if shap_sum is None else shap_sum + s
        gain_sum = g if gain_sum is None else gain_sum + g
    assert shap_sum is not None and gain_sum is not None
    shap_mean, gain_mean = shap_sum / len(folds), gain_sum / len(folds)
    shap_rank = np.argsort(np.argsort(-shap_mean)) + 1
    gain_rank = np.argsort(np.argsort(-gain_mean)) + 1
    out = [
        Importance(
            n,
            featureset.parse(n).kind,
            float(shap_mean[j]),
            float(gain_mean[j]),
            int(shap_rank[j]),
            int(gain_rank[j]),
            len(folds),
        )
        for j, n in enumerate(names)
    ]
    return sorted(out, key=lambda r: r.shap_rank)


def write_table(path: Path, cfg: dict[str, str], run_id: str, rows: list[Importance]) -> None:
    cols = [
        "venue",
        "symbol",
        "target",
        "run_id",
        "feature",
        "kind",
        "mean_abs_shap_ticks",
        "gain_share",
        "shap_rank",
        "gain_rank",
        "folds",
        "mechanism",
    ]
    old: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            old = [
                r
                for r in csv.DictReader(fh)
                if (r["venue"], r["symbol"], r["target"])
                != (cfg["venue"], cfg["symbol"], cfg["target"])
            ]
    new = [
        {
            "venue": cfg["venue"],
            "symbol": cfg["symbol"],
            "target": cfg["target"],
            "run_id": run_id,
            "feature": r.feature,
            "kind": r.kind,
            "mean_abs_shap_ticks": f"{r.mean_abs_shap:.6g}",
            "gain_share": f"{r.gain_share:.6g}",
            "shap_rank": str(r.shap_rank),
            "gain_rank": str(r.gain_rank),
            "folds": str(r.folds),
            "mechanism": MECHANISM[r.kind],
        }
        for r in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(old + new)


def figure(path: Path, title: str, rows: list[Importance], top: int = 15) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sel = rows[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(sel) + 1.2), dpi=150)
    ax.barh([r.feature for r in sel], [r.mean_abs_shap for r in sel], color="#0072B2")
    ax.set_xlabel("mean |SHAP| on OOS days (ticks of predicted mid change)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
