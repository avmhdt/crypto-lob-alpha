"""Golden features and labels on a seeded synthetic capture (AC-W-13, tolerance 1e-12).

Two checks:
* the hour-by-hour build equals a single pass over the whole stretch, so the lookback and
  look-ahead around the hour boundary lose nothing;
* selected columns equal the committed golden CSV. Regenerate with LOB_UPDATE_GOLDEN=1.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "fixtures"))
from make_synthetic import DAY, VENUES, write_capture

from lob_alpha.features import core
from lob_alpha.features.build import (
    DAY as DAY_NS,
)
from lob_alpha.features.build import (
    MS,
    NS,
    Params,
    Rows,
    book_gaps,
    build_day,
    evaluate,
    live_mask,
    read_topn,
    read_trades,
    rv_on_grid,
)
from lob_alpha.labels import clock_labels, event_labels
from lob_alpha.validate.rebuild import day_files

GOLDEN = Path(__file__).with_name("features_golden.csv")
COLUMNS = [
    "ofi_l1_ev10", "ofi_l1_ms1000", "mlofi_l3_ev50", "qi_l5", "wmid_adj_ticks", "imb_l1",
    "tsi_vol_tr50", "tsi_cnt_ms10000", "rv_ms10000", "midchg_ms1000", "dmid_ticks_time_1000",
    "dmid_ticks_time_500_lat100",
]  # fmt: skip
RTOL = 1e-12


def _concat(tables: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([t[k] for t in tables]) for k in tables[0]}


def _single_pass(
    root: Path, venue: str, symbol: str, p: Params
) -> dict[str, dict[str, np.ndarray]]:
    day_start = int(np.datetime64(str(DAY), "ns").astype(np.int64))
    rows = Rows.concat(
        [read_topn(f) for f in day_files(root / "interim", "topn", venue, symbol, DAY)]
    )
    book = core.make_book(
        rows.ts_recv, rows.ts_exchange, rows.bpx, rows.bq, rows.apx, rows.aq, rows.epoch
    )
    trades = read_trades(root, venue, symbol, DAY)
    gaps = book_gaps(root, venue, symbol, day_start, day_start + DAY_NS)
    lo = day_start + 10 * 3600 * NS
    g = np.arange(lo, lo + 2 * 3600 * NS, p.grid_ms * MS, dtype=np.int64)
    data_end = int(book.ts.max())
    t = g[live_mask(g, book, gaps, data_end)]
    at = core.asof_index(book.ts, t)
    cols: dict[str, np.ndarray] = {"ts": t}
    cols |= evaluate(book, trades, t, at, p, day_start, "time")
    gg, rv = rv_on_grid(book, gaps, day_start, lo + 2 * 3600 * NS, p, data_end)
    pos = np.searchsorted(gg, t)
    for w, vals in rv.items():
        cols[f"rv_ms{w}"] = vals[pos]
    cols |= clock_labels(
        book.ts,
        core.mid_ticks(book),
        book.segment_start,
        t,
        p.horizons_ms,
        p.latencies_ms,
        gaps,
        day_start + DAY_NS,
    )
    at_e = np.arange(book.n, dtype=np.int64)
    ev: dict[str, np.ndarray] = {"ts": book.ts}
    ev |= evaluate(book, trades, book.ts, at_e, p, day_start, "event")
    ev |= event_labels(core.mid_ticks(book), book.segment_start, p.horizons_events)
    return {"time": cols, "event": ev}


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("synthetic")
    write_capture(root)
    out = {}
    for venue, symbol, _ in VENUES:
        _, kept = build_day(root, venue, symbol, DAY, Params(), write=False)
        out[(venue, symbol)] = {g: _concat(v) for g, v in kept.items() if v}
    return root, out


def test_hourly_build_equals_a_single_pass(built: tuple[Path, dict[str, Any]]) -> None:
    root, out = built
    for venue, symbol, _ in VENUES:
        whole = _single_pass(root, venue, symbol, Params())
        for grid in ("time", "event"):
            hourly = out[(venue, symbol)][grid]
            assert np.array_equal(hourly["ts"], whole[grid]["ts"]), grid
            for name, vals in whole[grid].items():
                if name.startswith("rv_") and grid == "event":
                    continue  # event-grid volatility is read from the 100 ms grid, checked on time
                np.testing.assert_allclose(
                    hourly[name], vals, rtol=RTOL, atol=0, equal_nan=True, err_msg=f"{grid} {name}"
                )


def test_labels_are_null_across_the_gap(built: tuple[Path, dict[str, Any]]) -> None:
    _, out = built
    for venue, symbol, _ in VENUES:
        time = out[(venue, symbol)]["time"]
        gs = int(np.datetime64("2026-11-09T11:00:20", "ns").astype(np.int64))
        near = (time["ts"] > gs - 900 * MS) & (time["ts"] < gs)
        assert near.any()
        assert np.isnan(time["dmid_ticks_time_1000"][near]).all()
        inside = (time["ts"] >= gs) & (time["ts"] <= gs + 5 * NS)
        assert not inside.any()  # no samples while the book is down
        event = out[(venue, symbol)]["event"]
        assert len(event["ts"]) > 0 and "dmid_ticks_event_500" in event


def _rows(out: dict[str, Any]) -> list[list[str]]:
    rows = []
    for venue, symbol, _ in VENUES:
        time = out[(venue, symbol)]["time"]
        for i in range(0, len(time["ts"]), 7):
            rows.append(
                [venue, symbol, str(int(time["ts"][i]))]
                + [repr(float(time[c][i])) for c in COLUMNS]
            )
    return rows


def test_golden_values(built: tuple[Path, dict[str, Any]]) -> None:
    _, out = built
    rows = _rows(out)
    if os.environ.get("LOB_UPDATE_GOLDEN") == "1":
        with GOLDEN.open("w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle, lineterminator="\n")
            w.writerow(["venue", "symbol", "ts", *COLUMNS])
            w.writerows(rows)
    with GOLDEN.open(newline="", encoding="utf-8") as handle:
        golden = list(csv.reader(handle))[1:]
    assert len(golden) == len(rows)
    for got, want in zip(rows, golden, strict=True):
        assert got[:3] == want[:3]
        np.testing.assert_allclose(
            np.array([float(x) for x in got[3:]]),
            np.array([float(x) for x in want[3:]]),
            rtol=RTOL,
            atol=0,
            equal_nan=True,
        )
