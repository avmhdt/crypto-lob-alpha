"""Replay kernel against the reference BookL2, and the Kraken checksum in the kernel."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from lob_alpha.book.checksum import kraken_checksum
from lob_alpha.book.l2 import ASK, BID, BookL2
from lob_alpha.book.replay import (
    CRC_TABLE,
    FLAG_CROSSED,
    FLAG_LOCKED,
    FLAG_ONE_SIDED,
    NO_LEVEL,
    TOP_N,
    BookState,
    empty_probes,
    run_batches,
    strip_raw,
)

TICK = 0.5
LO = 1000


def run(
    state: BookState,
    batches: list[tuple[bool, list[tuple[int, int, float]]]],
    *,
    depth: int = 0,
    raw: list[list[tuple[str, str]]] | None = None,
    crc: bool = False,
    probes: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    rows = [r for _, rs in batches for r in rs]
    sides = np.array([r[0] for r in rows], dtype=np.int8)
    ticks = np.array([r[1] for r in rows], dtype=np.int64)
    qtys = np.array([r[2] for r in rows], dtype=np.float64)
    bstart = np.cumsum([0] + [len(rs) for _, rs in batches]).astype(np.int64)
    bsnap = np.array([s for s, _ in batches], dtype=np.bool_)
    n = len(batches)
    if raw is not None:
        flat = [x for b in raw for x in b]
        px, pxl = strip_raw([p for p, _ in flat])
        qt, qtl = strip_raw([q for _, q in flat])
    else:
        px = qt = np.zeros((0, 1), np.uint8)
        pxl = qtl = np.zeros(0, np.uint8)
    out = {
        "bpx": np.empty((n, TOP_N), np.int64),
        "bq": np.empty((n, TOP_N), np.float64),
        "apx": np.empty((n, TOP_N), np.int64),
        "aq": np.empty((n, TOP_N), np.float64),
        "flags": np.zeros(n, np.int8),
        "crc": np.full(n, -1, np.int64),
    }
    pr = probes or empty_probes()
    run_batches(
        state.qty, state.meta, state.px_raw, state.px_len, state.qty_raw, state.qty_len,
        state.lo, depth, sides, ticks, qtys, px, pxl, qt, qtl, bstart, bsnap, 0, n,
        out["bpx"], out["bq"], out["apx"], out["aq"], out["flags"],
        np.full(n, crc, np.bool_), out["crc"], CRC_TABLE,
        pr["pr_batch"], pr["pr_snap_off"], pr["snap_tick"], pr["snap_qty"],
        pr["pr_ex_off"], pr["ex_tick"], pr["pr_out"],
    )  # fmt: skip
    return out


level = st.tuples(
    st.sampled_from([BID, ASK]),
    st.integers(LO + 5, LO + 60),
    st.sampled_from([0.0, 0.0, 1.0, 2.5, 7.0]),
)
batch = st.tuples(st.booleans(), st.lists(level, min_size=0, max_size=8))


@settings(max_examples=150, deadline=None)
@given(st.lists(batch, min_size=1, max_size=30), st.sampled_from([0, 3, 10]))
def test_kernel_matches_reference_book(batches: list, depth: int) -> None:
    state = BookState(LO, 100, raw=False)
    ref = BookL2(TICK)
    out = run(state, batches, depth=depth)
    for i, (snap, rows) in enumerate(batches):
        if snap:
            ref.clear()
        for side, tick, qty in rows:
            ref.apply(side, tick * TICK, qty)
        if depth:
            ref.truncate(depth)
        bids, asks = ref.top_n(TOP_N)
        for side_arr, px, q in (
            (bids, out["bpx"][i], out["bq"][i]),
            (asks, out["apx"][i], out["aq"][i]),
        ):
            for k in range(TOP_N):
                if np.isnan(side_arr[k, 0]):
                    assert px[k] == NO_LEVEL
                else:
                    assert px[k] == round(side_arr[k, 0] / TICK)
                    assert q[k] == side_arr[k, 1]
        flag = out["flags"][i]
        if ref.best_bid() is None or ref.best_ask() is None:
            assert flag & FLAG_ONE_SIDED
        else:
            assert bool(flag & FLAG_CROSSED) == (ref.is_crossed() and not ref.is_locked())
            assert bool(flag & FLAG_LOCKED) == ref.is_locked()
    assert state.levels(BID) == [(round(p / TICK), q) for p, q in _levels(ref, BID)]
    assert state.levels(ASK) == [(round(p / TICK), q) for p, q in _levels(ref, ASK)]


def _levels(book: BookL2, side: int) -> list[tuple[float, float]]:
    bids, asks = book.top_n(book.depth(side) or 1)
    arr = bids if side == BID else asks
    return [(float(p), float(q)) for p, q in arr if not np.isnan(p)]


def test_out_of_window_updates_are_counted_and_ignored() -> None:
    state = BookState(LO, 10, raw=False)
    run(state, [(True, [(BID, LO + 2, 1.0), (ASK, LO + 50, 1.0), (BID, LO - 1, 3.0)])])
    assert state.out_of_window == 2
    assert state.levels(BID) == [(LO + 2, 1.0)]


DOC_ASKS = [
    ("45285.2", "0.00100000"), ("45286.4", "1.54571953"), ("45286.6", "1.54571109"),
    ("45289.6", "1.54560911"), ("45290.2", "0.15890660"), ("45291.8", "1.54553491"),
    ("45294.7", "0.04454749"), ("45296.1", "0.35380000"), ("45297.5", "0.09945542"),
    ("45299.5", "0.18772827"),
]  # fmt: skip
DOC_BIDS = [
    ("45283.5", "0.10000000"), ("45283.4", "1.54582015"), ("45282.1", "0.10000000"),
    ("45281.0", "0.10000000"), ("45280.3", "1.54592586"), ("45279.0", "0.07990000"),
    ("45277.6", "0.03310103"), ("45277.5", "0.30000000"), ("45277.3", "1.54602737"),
    ("45276.6", "0.15445238"),
]  # fmt: skip


def test_kernel_checksum_matches_kraken_doc_example() -> None:
    tick = 0.1
    lo = round(45200 / tick)
    state = BookState(lo, 2000, raw=True)
    strings = [(p, q) for p, q in DOC_BIDS] + [(p, q) for p, q in DOC_ASKS]
    rows = [(BID, round(float(p) / tick), float(q)) for p, q in DOC_BIDS]
    rows += [(ASK, round(float(p) / tick), float(q)) for p, q in DOC_ASKS]
    # a deeper bid outside the top 10 must not change the checksum
    rows.append((BID, round(45270.0 / tick), 5.0))
    strings.append(("45270.0", "5.00000000"))
    out = run(state, [(True, rows)], raw=[strings], crc=True)
    assert out["crc"][0] == 3310070434 == kraken_checksum(DOC_BIDS, DOC_ASKS)


def test_checksum_follows_updates_and_removals() -> None:
    tick = 0.1
    lo = round(45200 / tick)
    state = BookState(lo, 2000, raw=True)
    snap_rows = [(BID, round(float(p) / tick), float(q)) for p, q in DOC_BIDS]
    snap_rows += [(ASK, round(float(p) / tick), float(q)) for p, q in DOC_ASKS]
    snap_raw = list(DOC_BIDS) + list(DOC_ASKS)
    diff = [(ASK, round(45285.2 / tick), 0.0), (BID, round(45283.5 / tick), 0.25)]
    diff_raw = [("45285.2", "0.00000000"), ("45283.5", "0.25000000")]
    out = run(state, [(True, snap_rows), (False, diff)], raw=[snap_raw, diff_raw], crc=True)
    bids = [("45283.5", "0.25000000"), *DOC_BIDS[1:]]
    asks = DOC_ASKS[1:]
    assert out["crc"][1] == kraken_checksum(bids, asks)


def test_probe_compares_only_the_snapshot_range_and_skips_excluded() -> None:
    state = BookState(LO, 100, raw=False)
    book = [
        (BID, LO + 10, 1.0),
        (BID, LO + 9, 2.0),
        (BID, LO + 3, 9.0),
        (ASK, LO + 12, 1.0),
        (ASK, LO + 13, 4.0),
    ]
    pr = empty_probes()
    # snapshot: bids 1009..1010 (1010 differs but is excluded), asks 1012..1013 (1013 differs)
    pr["pr_batch"] = np.array([0], np.int64)
    pr["snap_tick"] = np.array([LO + 9, LO + 10, LO + 12, LO + 13], np.int64)
    pr["snap_qty"] = np.array([2.0, 5.0, 1.0, 3.0])
    pr["pr_snap_off"] = np.array([0, 2, 4], np.int64)
    pr["ex_tick"] = np.array([LO + 10], np.int64)
    pr["pr_ex_off"] = np.array([0, 1, 1], np.int64)
    pr["pr_out"] = np.zeros((1, 5), np.int64)
    pr["pr_out"][:, 3] = NO_LEVEL
    run(state, [(True, book)], probes=pr)
    compared, bad, excluded, first_bad, oow = pr["pr_out"][0]
    assert (compared, bad, excluded, first_bad, oow) == (3, 1, 1, LO + 13, 0)


def test_strip_raw_removes_point_and_leading_zeros() -> None:
    buf, lengths = strip_raw(["0.00123400", "45285.2", None])
    assert bytes(buf[0, : lengths[0]]) == b"123400"
    assert bytes(buf[1, : lengths[1]]) == b"452852"
    assert lengths[2] == 0
