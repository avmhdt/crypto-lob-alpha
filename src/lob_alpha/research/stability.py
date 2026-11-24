"""Stability across days, assets and venues (FR-25, AC-H-03).

Per run and test day: OOS R2 against the fold's training mean, with a 95% CI from a block
bootstrap of that day's rows in 5-minute blocks (2,000 resamples, fixed seed). One day is
one block for the day-block bootstrap, so a per-day interval has to resample within the
day. 5 minutes is 5 times the longest feature window (60 s) and 10 times the longest
label horizon (30 s), so blocks are close to independent. Per run: the pooled OOS R2 with the
day-block CI (decay.bootstrap) and the share of test days with positive OOS R2.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from lob_alpha.research import decay

F64 = npt.NDArray[np.float64]
BLOCK_NS = 5 * 60 * 1_000_000_000
SEED = 20261102


@dataclass
class DayRow:
    venue: str
    symbol: str
    clock: str
    target: str
    model: str
    run_id: str
    day: str
    n: int
    oos_r2: float
    lo: float
    hi: float


def block_ci(
    ts: npt.NDArray[np.int64], y: F64, yhat: F64, mu: F64, resamples: int = 2000, seed: int = SEED
) -> tuple[float, float, float]:
    ok = ~(np.isnan(y) | np.isnan(yhat))
    ts, y, yhat, mu = ts[ok], y[ok], yhat[ok], mu[ok]
    blocks = (ts // BLOCK_NS).astype(np.int64)
    _, inv = np.unique(blocks, return_inverse=True)
    nb = int(inv.max()) + 1 if len(inv) else 0
    sse = np.bincount(inv, weights=(y - yhat) ** 2, minlength=nb)
    sst = np.bincount(inv, weights=(y - mu) ** 2, minlength=nb)
    point = 1.0 - sse.sum() / sst.sum() if sst.sum() > 0 else float("nan")
    if nb < 2:
        return float(point), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    k = rng.integers(0, nb, size=(resamples, nb))
    r2 = 1.0 - sse[k].sum(axis=1) / sst[k].sum(axis=1)
    return float(point), float(np.quantile(r2, 0.025)), float(np.quantile(r2, 0.975))


def run_days(data_root: Path, cfg: dict[str, str], run_id: str) -> list[DayRow]:
    t = pq.read_table(data_root / "runs" / run_id / "predictions.parquet")
    days = np.array([d.isoformat() for d in t.column("day").to_pylist()])
    ts = t.column("ts").to_numpy()
    y, yhat, mu = (t.column(c).to_numpy() for c in ("y", "yhat", "train_mean"))
    clock = "time" if "_time_" in cfg["target"] else "event"
    out = []
    for d in sorted(set(days)):
        m = days == d
        r2, lo, hi = block_ci(ts[m], y[m], yhat[m], mu[m])
        n = int(np.sum(~np.isnan(y[m])))
        out.append(
            DayRow(
                cfg["venue"],
                cfg["symbol"],
                clock,
                cfg["target"],
                cfg["model"],
                run_id,
                d,
                n,
                r2,
                lo,
                hi,
            )
        )
    return out


def write_table(path: Path, rows: list[DayRow]) -> None:
    key = ("venue", "symbol", "target", "model")
    new = [{k: decay._fmt(v) for k, v in asdict(r).items()} for r in rows]
    keys = {tuple(r[k] for k in key) for r in new}
    old: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            old = [r for r in csv.DictReader(fh) if tuple(r[k] for k in key) not in keys]
    merged = sorted(old + new, key=lambda r: (*[r[k] for k in key], r["day"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])), lineterminator="\n")
        w.writeheader()
        w.writerows(merged)


def forest(path: Path, rows: list[dict[str, str]]) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = [f"{r['venue']} {r['symbol']} {r['model']} {r['day']}" for r in rows]
    r2 = np.array([float(r["oos_r2"]) for r in rows])
    lo = np.array([float(r["lo"] or "nan") for r in rows])
    hi = np.array([float(r["hi"] or "nan") for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 0.22 * len(rows) + 1.2), dpi=150)
    yy = np.arange(len(rows))[::-1]
    ax.errorbar(
        r2, yy, xerr=[r2 - lo, hi - r2], fmt="o", ms=3, color="#0072B2", ecolor="#56B4E9", capsize=2
    )
    ax.axvline(0, color="#999999", lw=0.8)
    ax.set_yticks(yy, labels, fontsize=6)
    ax.set_xlabel("OOS R2 per test day (95% CI, 5-minute block bootstrap)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


ASSET = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
    "SOL-USD": "SOL",
    "BTC/USD": "BTC",
    "ETH/USD": "ETH",
    "SOL/USD": "SOL",
}


@dataclass
class TransferRow:
    asset: str
    train_venue: str
    train_symbol: str
    test_venue: str
    test_symbol: str
    run_id: str
    target: str
    n_days: int
    oos_r2: float
    lo: float
    hi: float
    days: str


def transfer(
    data_root: Path,
    results: Path,
    run_id: str,
    test_venue: str,
    test_symbol: str,
    test_tick: float,
    train_tick: float,
    test_days: set[str],
) -> TransferRow:
    """Apply each fold's model from run `run_id` (venue A) to venue B on the fold's test
    day, where that day is complete for B too.

    Features on B are normalised with statistics fitted on B's rows of the fold's own
    training days (unsupervised, train-only), so the model sees each venue on its own
    scale. Predictions are converted from A ticks to B ticks (x tick_A / tick_B). The
    benchmark is the training mean of B's label over the same training days.
    """
    import json  # noqa: PLC0415
    from datetime import date  # noqa: PLC0415

    import lightgbm as lgb  # noqa: PLC0415

    from lob_alpha.models import featureset  # noqa: PLC0415
    from lob_alpha.models.data import load_day  # noqa: PLC0415

    cfg = json.loads((results / "runs" / run_id / "config.json").read_text(encoding="utf-8"))
    met = json.loads((results / "runs" / run_id / "metrics.json").read_text(encoding="utf-8"))
    target, clock = cfg["target"], "time" if "_time_" in cfg["target"] else "event"
    stats: list[decay.DayStat] = []
    for f in met["folds"]:
        day = f["test"][0]
        if day not in test_days:
            continue
        fdir = data_root / "runs" / run_id / f"fold_{f['fold']:02d}"
        prep_a = featureset.Prep.from_json(json.loads((fdir / "prep.json").read_text()))
        names = prep_a.names
        cols = featureset.stored_columns(names)
        try:
            train = [
                load_day(
                    data_root, test_venue, test_symbol, date.fromisoformat(d), clock, cols, [target]
                )
                for d in f["train"]
            ]
        except FileNotFoundError:
            continue
        tr = {k: np.concatenate([p.cols[k] for p in train]) for k in train[0].cols}
        segs, a = [], 0
        for p in train:
            segs.append(slice(a, a + p.n))
            a += p.n
        prep_b = featureset.Prep(names).fit(tr, segs)
        mu = float(np.nanmean(np.concatenate([p.labels[target] for p in train])))
        test = load_day(
            data_root, test_venue, test_symbol, date.fromisoformat(day), clock, cols, [target]
        )
        booster = lgb.Booster(model_file=str(fdir / "model.txt"))
        yhat = np.asarray(booster.predict(prep_b.transform(test.cols)[1]), dtype=np.float64)
        yhat *= train_tick / test_tick
        y = test.labels[target]
        stats.append(decay.day_stat(day, y, yhat, np.full(len(y), mu)))
    lo, hi = decay.bootstrap(stats)
    return TransferRow(
        ASSET.get(cfg["symbol"], cfg["symbol"]),
        cfg["venue"],
        cfg["symbol"],
        test_venue,
        test_symbol,
        run_id,
        target,
        len(stats),
        decay.pooled_r2(stats) if stats else float("nan"),
        lo,
        hi,
        " ".join(s.day for s in stats),
    )
