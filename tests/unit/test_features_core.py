"""Feature kernels against hand-computed values (AC-W-13)."""

from __future__ import annotations

import numpy as np

from lob_alpha.features import core


def book(
    rows: list[tuple[int, int, float, int, float]], epochs: list[int] | None = None
) -> core.Book:
    """rows: (ts, bid_px, bid_qty, ask_px, ask_qty) at level 1; levels 2..10 empty."""
    n = len(rows)
    bpx = np.full((n, 10), -1, np.int64)
    apx = np.full((n, 10), -1, np.int64)
    bq = np.zeros((n, 10))
    aq = np.zeros((n, 10))
    for i, (_, b, qb, a, qa) in enumerate(rows):
        bpx[i, 0], bq[i, 0], apx[i, 0], aq[i, 0] = b, qb, a, qa
    ts = np.array([r[0] for r in rows], np.int64)
    ep = np.array(epochs if epochs else [1] * n, np.int64)
    return core.make_book(ts, ts, bpx, bq, apx, aq, ep)


def test_cks_ofi_hand_computed_example() -> None:
    # n=1: bid up a tick with 2.0 (new level)            e = +2.0 - 0 - 0 + 0       = +2.0
    #      ask unchanged at 101: qa 3.0 -> 1.0           e += -1.0 + 3.0           -> +4.0
    # n=2: bid down to 99 (was 100 x 2.0), ask same 1.0 e = 0 - 2.0 - 1.0 + 1.0     = -2.0
    # n=3: ask down to 100.. (101 -> 100) with 4.0, bid same 99 x 5.0 (was 99 x 1.5)
    #      e = 5.0 - 1.5 - 4.0 + 0                                                  = -0.5
    b = book(
        [
            (1, 99, 1.0, 101, 3.0),
            (2, 100, 2.0, 101, 1.0),
            (3, 99, 1.5, 101, 1.0),
            (4, 99, 5.0, 100, 4.0),
        ]
    )
    e = core.cks_e(b.bpx[:, 0], b.bq[:, 0], b.apx[:, 0], b.aq[:, 0], b.segment_start)
    assert e.tolist() == [0.0, 4.0, -2.0, -0.5]


def test_cks_resets_at_a_new_epoch_and_skips_missing_sides() -> None:
    b = book(
        [
            (1, 99, 1.0, 101, 1.0),
            (2, 100, 1.0, 101, 1.0),
            (3, 50, 9.0, 51, 9.0),
            (4, -1, 0.0, 51, 1.0),
        ],
        [1, 1, 2, 2],
    )
    e = core.cks_e(b.bpx[:, 0], b.bq[:, 0], b.apx[:, 0], b.aq[:, 0], b.segment_start)
    assert e[2] == 0.0 and e[3] == 0.0
    assert b.segment_start.tolist() == [0, 0, 2, 2]


def test_window_sums_stop_at_segment_and_lower_time() -> None:
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    at = np.array([4, 4, 2], np.int64)
    lower = np.array([0, 3, 0], np.int64)
    assert core.event_window_sum(x, at, 3, lower).tolist() == [28.0, 24.0, 7.0]
    ts = np.array([10, 20, 30, 40, 50], np.int64)
    t = np.array([50, 50, 35], np.int64)
    got = core.time_window_sum(x, ts, t, 25, np.array([0, 40, 0], np.int64))
    assert got.tolist() == [28.0, 24.0, 6.0]  # (25,50]; (39,50] by lower bound; (10,35]


def test_imbalances_spread_and_weighted_mid() -> None:
    b = book([(1, 100, 3.0, 102, 1.0)])
    assert core.spread_ticks(b)[0] == 2.0
    assert core.mid_ticks(b)[0] == 101.0
    assert core.imbalance_l1(b)[0] == 0.75
    assert core.queue_imbalance(b, 1)[0] == 0.5
    # weighted mid = (100*1 + 102*3) / 4 = 101.5 -> +0.5 ticks over the mid
    assert core.weighted_mid_adj(b)[0] == 0.5
    assert core.depth_l1(b)[0] == 2.0


def test_trade_imbalance_by_count_and_time() -> None:
    tr = core.make_trades(
        np.array([10, 20, 30, 40], np.int64),
        np.array([1, -1, 1, 1], np.int8),
        np.array([1.0, 3.0, 1.0, 2.0]),
    )
    v, c = core.trade_imbalance(tr, np.array([40], np.int64), last_n=3)
    assert v[0] == (-3 + 1 + 2) / 6 and c[0] == 1 / 3
    v, c = core.trade_imbalance(tr, np.array([40], np.int64), w_ns=15)
    assert v[0] == 1.0 and c[0] == 1.0  # (25, 40]: trades at 30 and 40
    v, c = core.trade_imbalance(tr, np.array([5], np.int64), w_ns=15)
    assert v[0] == 0.0 and c[0] == 0.0


def test_realized_vol_within_a_stretch() -> None:
    mid = np.array([100.0, 101.0, 101.0, 99.0, 98.0])
    seg = np.array([0, 0, 0, 3, 3], np.int64)  # a new stretch at index 3
    rv = core.realized_vol(mid, seg, [2, 10])
    assert rv[2].tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]
    assert np.isclose(rv[10][2], 1.0) and np.isclose(rv[10][4], 1.0)
