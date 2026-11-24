"""OFI replication after Cont, Kukanov and Stoikov 2014 (FR-19, AC-W-04).

Per symbol-day and interval length dt (1 s and 10 s), on non-overlapping intervals
[t, t + dt) aligned to the clock:

    dmid_k = mid(t + dt) - mid(t)            ticks  (label dmid_ticks_time_{dt} at t)
    OFI_k  = sum of CKS e_n over (t, t + dt]  base units (feature ofi_l1_ms{dt} at t + dt)
    dmid_k = alpha + beta OFI_k + eps_k

estimated by OLS with Newey-West (Bartlett) standard errors, lag floor(4 (n / 100)^(2/9)).
Only intervals that are LIVE end to end enter (the label is null across gaps). Reported:
beta, its HAC t-stat, R^2, the mean level-1 depth (qb_1 + qa_1) / 2 over the day, and
beta x mean depth, which CKS find roughly constant across stocks.

AC-W-04: beta > 0 with t > 5 on at least 95% of symbol-days; median R^2 at 10 s at least
0.10 per venue; beta inversely related to mean depth across symbols (Spearman < 0).
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from lob_alpha.validate.rebuild import partition

F64 = npt.NDArray[np.float64]
NS = 1_000_000_000
MS = 1_000_000
INTERVALS_MS = (1000, 10000)
MIN_T = 5.0
MIN_SHARE = 0.95
MIN_R2_10S = 0.10


@dataclass
class OlsResult:
    n: int
    alpha: float
    beta: float
    se_beta: float
    t_beta: float
    r2: float


def newey_west_lag(n: int) -> int:
    value: float = 4.0 * (n / 100.0) ** (2.0 / 9.0)
    return math.floor(value)


def ols_hac(y: F64, x: F64, lag: int | None = None) -> OlsResult:
    """y = a + b x by OLS; HAC (Bartlett kernel) standard error of b."""
    n = len(y)
    if n < 3:
        return OlsResult(n, math.nan, math.nan, math.nan, math.nan, math.nan)
    xm = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.pinv(xm.T @ xm)
    coef = xtx_inv @ (xm.T @ y)
    u = y - xm @ coef
    big_l = newey_west_lag(n) if lag is None else lag
    xu = xm * u[:, None]
    s = xu.T @ xu
    for lv in range(1, big_l + 1):
        w = 1.0 - lv / (big_l + 1.0)
        g = xu[lv:].T @ xu[:-lv]
        s += w * (g + g.T)
    cov = xtx_inv @ s @ xtx_inv
    se = float(math.sqrt(max(cov[1, 1], 0.0)))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((u * u).sum()) / ss_tot if ss_tot > 0 else math.nan
    beta = float(coef[1])
    return OlsResult(n, float(coef[0]), beta, se, beta / se if se > 0 else math.nan, r2)


def spearman(a: F64, b: F64) -> float:
    if len(a) < 3:
        return math.nan
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def interval_pairs(
    feat_ts: npt.NDArray[np.int64],
    ofi: F64,
    lab_ts: npt.NDArray[np.int64],
    dmid: F64,
    depth: F64,
    dt_ms: int,
) -> tuple[F64, F64, F64]:
    """(dmid, OFI, depth) for aligned non-overlapping intervals of dt_ms."""
    step = dt_ms * MS
    on = lab_ts % step == 0
    t0 = lab_ts[on]
    y = dmid[on]
    pos = np.searchsorted(feat_ts, t0 + step)
    have = (pos < len(feat_ts)) & (feat_ts[np.minimum(pos, len(feat_ts) - 1)] == t0 + step)
    ok = have & ~np.isnan(y)
    x = np.where(have, ofi[np.minimum(pos, len(feat_ts) - 1)], np.nan)
    d = np.where(have, depth[np.minimum(pos, len(feat_ts) - 1)], np.nan)
    ok &= ~np.isnan(x)
    return y[ok], x[ok], d[ok]


@dataclass
class DayRow:
    venue: str
    symbol: str
    date: str
    interval_ms: int
    n: int
    beta: float
    t_hac: float
    r2: float
    mean_depth: float
    beta_x_depth: float


def _read(
    root: Path, dataset: str, venue: str, symbol: str, day: date, cols: list[str]
) -> dict[str, npt.NDArray[Any]]:
    folder = partition(root / "processed", dataset, venue, symbol, day) / "clock=time"
    files = sorted(folder.glob("part-*.parquet"))
    if not files:
        return {}
    tables = [pq.read_table(f, columns=cols) for f in files]
    return {
        c: np.concatenate(
            [
                np.asarray(
                    t.column(c).to_numpy(zero_copy_only=False),
                    dtype=float if c != "ts" else np.int64,
                )
                for t in tables
            ]
        )
        for c in cols
    }


def replicate_day(root: Path, venue: str, symbol: str, day: date) -> list[DayRow]:
    rows: list[DayRow] = []
    for dt in INTERVALS_MS:
        f = _read(root, "features", venue, symbol, day, ["ts", f"ofi_l1_ms{dt}", "depth_l1"])
        lab = _read(root, "labels", venue, symbol, day, ["ts", f"dmid_ticks_time_{dt}"])
        if not f or not lab:
            continue
        y, x, _ = interval_pairs(
            f["ts"], f[f"ofi_l1_ms{dt}"], lab["ts"], lab[f"dmid_ticks_time_{dt}"], f["depth_l1"], dt
        )
        res = ols_hac(y, x)
        depth = float(np.nanmean(f["depth_l1"])) if len(f["depth_l1"]) else math.nan
        rows.append(
            DayRow(
                venue,
                symbol,
                str(day),
                dt,
                res.n,
                res.beta,
                res.t_beta,
                res.r2,
                depth,
                res.beta * depth,
            )
        )
    return rows


def write_table(path: Path, rows: list[DayRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(DayRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in asdict(r).items()}
            )


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verdict(rows: list[dict[str, str]]) -> tuple[bool, str]:
    """AC-W-04 from the replication table."""
    if not rows:
        return False, "no replication rows"
    problems: list[str] = []
    notes: list[str] = []
    for dt in INTERVALS_MS:
        sub = [r for r in rows if int(r["interval_ms"]) == dt and r["beta"] not in ("", "nan")]
        good = [r for r in sub if float(r["beta"]) > 0 and float(r["t_hac"]) > MIN_T]
        share = len(good) / len(sub) if sub else 0.0
        notes.append(
            f"{dt} ms: beta>0 & t>{MIN_T:g} on {len(good)}/{len(sub)} symbol-days "
            f"({100 * share:.1f}%)"
        )
        if share < MIN_SHARE:
            problems.append(f"{dt} ms share {100 * share:.1f}% < {100 * MIN_SHARE:.0f}%")
    ten = [r for r in rows if int(r["interval_ms"]) == 10000 and r["r2"] not in ("", "nan")]
    for venue in sorted({r["venue"] for r in ten}):
        r2 = np.array([float(r["r2"]) for r in ten if r["venue"] == venue])
        med = float(np.median(r2))
        notes.append(f"{venue} median R2 at 10 s {med:.3f}")
        if med < MIN_R2_10S:
            problems.append(f"{venue} median R2 {med:.3f} < {MIN_R2_10S}")
    per_symbol: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for r in ten:
        per_symbol.setdefault((r["venue"], r["symbol"]), []).append(
            (float(r["beta"]), float(r["mean_depth"]))
        )
    betas = np.array([np.median([b for b, _ in v]) for v in per_symbol.values()])
    depths = np.array([np.median([d for _, d in v]) for v in per_symbol.values()])
    rho = spearman(betas, depths)
    notes.append(f"Spearman(beta, depth) across {len(betas)} symbols {rho:.3f}")
    if not (rho < 0):
        problems.append(f"Spearman(beta, depth) {rho:.3f} is not < 0")
    return (not problems), "; ".join(notes + (["FAIL: " + ", ".join(problems)] if problems else []))


def scatter(path: Path, root: Path, venue: str, symbol: str, day: date, dt: int = 10000) -> None:
    """Dmid against OFI for one symbol-day, with the fitted line."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    f = _read(root, "features", venue, symbol, day, ["ts", f"ofi_l1_ms{dt}", "depth_l1"])
    lab = _read(root, "labels", venue, symbol, day, ["ts", f"dmid_ticks_time_{dt}"])
    if not f or not lab:
        return
    y, x, _ = interval_pairs(
        f["ts"], f[f"ofi_l1_ms{dt}"], lab["ts"], lab[f"dmid_ticks_time_{dt}"], f["depth_l1"], dt
    )
    res = ols_hac(y, x)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=110)
    ax.scatter(x, y, s=3, alpha=0.3, color="#4a6fa5")
    if len(x):
        xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 50)
        ax.plot(xs, res.alpha + res.beta * xs, color="#c44e52", lw=1.5)
    ax.set_xlabel(f"OFI over {dt // 1000} s (base units)")
    ax.set_ylabel("mid change (ticks)")
    ax.set_title(f"{venue} {symbol} {day}: beta {res.beta:.3g}, R2 {res.r2:.2f}", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
