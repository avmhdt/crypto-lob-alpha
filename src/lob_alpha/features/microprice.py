"""Stoikov (2018) micro-price adjustment g(I, S) in ticks.

The state is (I, S): I = qb_1 / (qb_1 + qa_1) in `n_imb` equal buckets, S = spread in ticks
for S = 1..`max_spread`. Consecutive observations (n, n+1) give a state transition and a mid
change dM in ticks. The data is symmetrised: each observation (I, S, dM, I', S') is added
again as (1 - I, S, -dM, 1 - I', S'). That makes g antisymmetric, g(I) = -g(1 - I), so g is
0 at I = 0.5.

With Q the transitions among states with dM = 0, T those with dM != 0 (to the new state), R
the probability of each dM value != 0 and K the dM values:

    G1 = (1 - Q)^-1 R K       expected mid change at the next mid change
    B  = (1 - Q)^-1 T
    g  = G1 + B G1 + B^2 G1 + ... (iterated to convergence)

Fit on training days only (NFR-07); `microprice_adjustment` then looks up g by state.
Mid changes larger than one tick are dropped, as in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass
class MicropriceTable:
    n_imb: int
    max_spread: int
    g: F64  # (max_spread, n_imb): row s-1, column bucket
    counts: npt.NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, np.int64))

    def to_rows(self) -> list[list[float]]:
        """[[spread_ticks, imbalance_lower_edge, g], ...] for feature_spec.json.

        Bucket b covers b / n_imb <= I < (b + 1) / n_imb (the last bucket includes I = 1),
        so a consumer takes the last row of the spread whose edge is <= I.
        """
        rows = []
        for s in range(self.max_spread):
            for b in range(self.n_imb):
                rows.append([float(s + 1), b / self.n_imb, float(self.g[s, b])])
        return rows


def _bucket(imb: F64, n_imb: int) -> npt.NDArray[np.int64]:
    return np.clip(np.floor(imb * n_imb).astype(np.int64), 0, n_imb - 1)


def fit_microprice_arrays(
    imb: F64, spread: F64, mid: F64, n_imb: int, max_spread: int, *, iterations: int = 30
) -> MicropriceTable:
    """Fit g from consecutive observations of (imbalance, spread ticks, mid ticks)."""
    ok = ~(np.isnan(imb) | np.isnan(spread) | np.isnan(mid))
    i0, i1 = imb[:-1], imb[1:]
    s0, s1 = spread[:-1], spread[1:]
    d = mid[1:] - mid[:-1]
    keep = ok[:-1] & ok[1:] & (s0 >= 1) & (s0 <= max_spread) & (s1 >= 1) & (s1 <= max_spread)
    keep &= np.abs(d) <= 1.0 + 1e-12
    i0, i1, s0, s1, d = i0[keep], i1[keep], s0[keep], s1[keep], d[keep]
    # symmetrise
    i0 = np.concatenate([i0, 1.0 - i0])
    i1 = np.concatenate([i1, 1.0 - i1])
    s0 = np.concatenate([s0, s0])
    s1 = np.concatenate([s1, s1])
    d = np.concatenate([d, -d])
    n_states = n_imb * max_spread
    x0 = (s0.astype(np.int64) - 1) * n_imb + _bucket(i0, n_imb)
    x1 = (s1.astype(np.int64) - 1) * n_imb + _bucket(i1, n_imb)
    dvals = np.array([-1.0, -0.5, 0.5, 1.0])
    dk = np.searchsorted(dvals, np.round(d * 2) / 2)
    moved = np.round(d * 2) != 0
    counts = np.bincount(x0, minlength=n_states).astype(np.float64)
    safe = np.where(counts > 0, counts, 1.0)
    q = np.zeros((n_states, n_states))
    np.add.at(q, (x0[~moved], x1[~moved]), 1.0)
    t = np.zeros((n_states, n_states))
    np.add.at(t, (x0[moved], x1[moved]), 1.0)
    r = np.zeros((n_states, len(dvals)))
    np.add.at(r, (x0[moved], dk[moved]), 1.0)
    q /= safe[:, None]
    t /= safe[:, None]
    r /= safe[:, None]
    inv = np.linalg.pinv(np.eye(n_states) - q)
    g1 = inv @ r @ dvals
    bmat = inv @ t
    g = g1.copy()
    term = g1.copy()
    for _ in range(iterations):
        term = bmat @ term
        g = g + term
        if np.max(np.abs(term)) < 1e-12:
            break
    return MicropriceTable(n_imb, max_spread, g.reshape(max_spread, n_imb), counts.astype(np.int64))


def lookup(table: MicropriceTable, imb: F64, spread: F64, *, clamp: bool = False) -> F64:
    """g(I, S) per row; NaN when inputs are missing.

    Without `clamp`, NaN outside spreads 1..max_spread. With `clamp`, a spread below 1 uses
    the 1-tick row and a spread above max_spread the max_spread row (the model feature).
    """
    out = np.full(len(imb), np.nan)
    if clamp:
        spread = np.clip(np.floor(spread + 1e-9), 1, table.max_spread)
    ok = ~(np.isnan(imb) | np.isnan(spread)) & (spread >= 1) & (spread <= table.max_spread)
    s = spread[ok].astype(np.int64) - 1
    b = _bucket(imb[ok], table.n_imb)
    out[ok] = table.g[s, b]
    return out
