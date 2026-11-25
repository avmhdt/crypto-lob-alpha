"""Model features: the stored M4 columns plus three fitted on training days only.

M4 stores features raw (docs/decisions.md, "M4 stored feature values"). Before a model sees
them, `Prep.fit` learns on the training days only (NFR-07, AC-W-05):

* `depth_norm`: mean level-1 depth (qb_1 + qa_1) / 2 over the training rows. Every OFI-type
  feature (ofi, mlofi, integrated_ofi) is divided by it (FR-17 `train_mean_l1_depth`).
* integrated OFI: weights = the first principal component of the depth-normalised level
  1..10 OFI over the training rows (covariance PCA), scaled so the absolute weights sum to 1
  and the sum of weights is positive (Cont, Cucuringu and Zhang 2023).
* micro-price: the Stoikov (2018) table g(I, S) fitted on consecutive training grid rows;
  spreads outside 1..max_spread use the nearest fitted spread.
* normalisation: z-score with the training mean and standard deviation, clipped to +-5.

`Prep.transform` turns the stored columns of any day into the model features (`raw`, as
defined, after depth normalisation) and their normalised values (`norm`, the model input).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from lob_alpha.features import microprice as mp

F64 = npt.NDArray[np.float64]
CLIP = 5.0
MLOFI_LEVELS = 10

_PATTERNS: list[tuple[str, str]] = [
    (r"ofi_l1_(ms|ev)(\d+)", "ofi"),
    (r"mlofi_l(\d+)_(ms|ev)(\d+)", "mlofi"),
    (r"integrated_ofi_(ms|ev)(\d+)", "integrated_ofi"),
    (r"qi_l(\d+)", "queue_imbalance"),
    (r"microprice_adj_ticks", "microprice"),
    (r"wmid_adj_ticks", "weighted_mid"),
    (r"spread_ticks", "spread"),
    (r"tsi_(vol|cnt)_(ms|tr)(\d+)", "trade_sign_imbalance"),
    (r"midchg_ms(\d+)", "mid_changes"),
    (r"rv_ms(\d+)", "realized_vol"),
]
UNIT = {"ms": "ms", "ev": "events", "tr": "trades"}
OFI_KINDS = ("ofi", "mlofi", "integrated_ofi")


@dataclass(frozen=True)
class FeatureDef:
    name: str
    kind: str
    levels: list[int]
    window: int | None
    window_unit: str | None
    weight: str | None = None  # trade_sign_imbalance: volume or count

    @property
    def stored_columns(self) -> list[str]:
        if self.kind == "integrated_ofi":
            tag = "ms" if self.window_unit == "ms" else "ev"
            return [f"mlofi_l{m}_{tag}{self.window}" for m in range(1, MLOFI_LEVELS + 1)]
        if self.kind == "microprice":
            return ["imb_l1", "spread_ticks"]
        return [self.name]


def parse(name: str) -> FeatureDef:
    for pattern, kind in _PATTERNS:
        m = re.fullmatch(pattern, name)
        if m is None:
            continue
        g = m.groups()
        if kind == "ofi":
            return FeatureDef(name, kind, [1], int(g[1]), UNIT[g[0]])
        if kind == "mlofi":
            return FeatureDef(name, kind, [int(g[0])], int(g[2]), UNIT[g[1]])
        if kind == "integrated_ofi":
            return FeatureDef(name, kind, list(range(1, MLOFI_LEVELS + 1)), int(g[1]), UNIT[g[0]])
        if kind == "queue_imbalance":
            return FeatureDef(name, kind, list(range(1, int(g[0]) + 1)), None, None)
        if kind == "trade_sign_imbalance":
            weight = "volume" if g[0] == "vol" else "count"
            return FeatureDef(name, kind, [], int(g[2]), UNIT[g[1]], weight)
        if kind in ("mid_changes", "realized_vol"):
            return FeatureDef(name, kind, [1], int(g[0]), "ms")
        return FeatureDef(name, kind, [1], None, None)
    raise ValueError(f"unknown model feature {name!r}")


def stored_columns(names: list[str]) -> list[str]:
    cols: list[str] = ["depth_l1", "imb_l1", "spread_ticks", "mid_ticks"]
    for n in names:
        for c in parse(n).stored_columns:
            if c not in cols:
                cols.append(c)
    return cols


@dataclass
class Prep:
    names: list[str]
    depth_norm: float = float("nan")
    weights: dict[str, list[float]] = field(default_factory=dict)  # integrated_ofi
    micro: mp.MicropriceTable | None = None
    mean: F64 = field(default_factory=lambda: np.zeros(0))
    std: F64 = field(default_factory=lambda: np.zeros(0))
    micro_buckets: int = 10
    micro_max_spread: int = 4

    @property
    def defs(self) -> list[FeatureDef]:
        return [parse(n) for n in self.names]

    def fit(self, train: dict[str, F64], segments: list[slice] | None = None) -> Prep:
        """Fit on training rows; `segments` are the day slices (the micro-price fit never
        pairs the last row of one day with the first row of the next)."""
        depth = train["depth_l1"]
        self.depth_norm = float(np.nanmean(depth))
        if not np.isfinite(self.depth_norm) or self.depth_norm <= 0:
            raise ValueError("training depth_l1 has no positive mean")
        for d in self.defs:
            if d.kind == "integrated_ofi":
                x = np.column_stack([train[c] for c in d.stored_columns]) / self.depth_norm
                x = x[~np.isnan(x).any(axis=1)]
                cov = np.cov(x, rowvar=False)
                vals, vecs = np.linalg.eigh(cov)
                w = vecs[:, int(np.argmax(vals))]
                if w.sum() < 0:
                    w = -w
                self.weights[d.name] = [float(v) for v in w / np.abs(w).sum()]
            if d.kind == "microprice":
                parts = segments or [slice(0, len(depth))]
                imb = np.concatenate([np.append(train["imb_l1"][s], np.nan) for s in parts])
                spr = np.concatenate([np.append(train["spread_ticks"][s], np.nan) for s in parts])
                mid = np.concatenate([np.append(train["mid_ticks"][s], np.nan) for s in parts])
                self.micro = mp.fit_microprice_arrays(
                    imb, spr, mid, self.micro_buckets, self.micro_max_spread
                )
        raw = self.raw(train)
        self.mean = np.nanmean(raw, axis=0)
        std = np.nanstd(raw, axis=0)
        self.std = np.where(std > 0, std, 1.0)
        return self

    def raw(self, cols: dict[str, F64]) -> F64:
        out = np.empty((len(cols["depth_l1"]), len(self.names)))
        for j, d in enumerate(self.defs):
            if d.kind in ("ofi", "mlofi"):
                out[:, j] = cols[d.name] / self.depth_norm
            elif d.kind == "integrated_ofi":
                # acc = 0; acc = acc + w_l * x_l for l = 1..10 in order (no fused multiply-add,
                # no reordering, unlike a BLAS dot product); then / depth_norm
                acc = np.zeros(len(out))
                for w_l, c in zip(self.weights[d.name], d.stored_columns, strict=True):
                    acc = acc + w_l * cols[c]
                out[:, j] = acc / self.depth_norm
            elif d.kind == "microprice":
                assert self.micro is not None
                out[:, j] = mp.lookup(self.micro, cols["imb_l1"], cols["spread_ticks"], clamp=True)
            else:
                out[:, j] = cols[d.name]
        return out

    def normalize(self, raw: F64) -> F64:
        z: F64 = np.clip((raw - self.mean) / self.std, -CLIP, CLIP)
        return z

    def transform(self, cols: dict[str, F64]) -> tuple[F64, F64]:
        raw = self.raw(cols)
        return raw, self.normalize(raw)

    def to_json(self) -> dict[str, Any]:
        return {
            "names": self.names,
            "depth_norm": self.depth_norm,
            "weights": self.weights,
            "micro": None
            if self.micro is None
            else {
                "n_imb": self.micro.n_imb,
                "max_spread": self.micro.max_spread,
                "g": self.micro.g.tolist(),
            },
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> Prep:
        p = cls(list(obj["names"]))
        p.depth_norm = float(obj["depth_norm"])
        p.weights = {k: list(v) for k, v in obj["weights"].items()}
        m = obj.get("micro")
        if m is not None:
            p.micro = mp.MicropriceTable(int(m["n_imb"]), int(m["max_spread"]), np.array(m["g"]))
            p.micro_buckets, p.micro_max_spread = p.micro.n_imb, p.micro.max_spread
        p.mean = np.asarray(obj["mean"], dtype=np.float64)
        p.std = np.asarray(obj["std"], dtype=np.float64)
        return p
