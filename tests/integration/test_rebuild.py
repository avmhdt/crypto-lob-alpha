"""Offline rebuild end to end on small synthetic captures, one per venue rule."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from lob_alpha import schemas
from lob_alpha.book.checksum import kraken_checksum
from lob_alpha.book.l2 import ASK, BID, BookL2
from lob_alpha.validate.rebuild import Rebuilder, load_stats, partition, save_stats

D1 = date(2026, 11, 9)
D2 = date(2026, 11, 10)
NS = 1_000_000_000


def t(day: date, hour: int, second: float) -> int:
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=UTC).timestamp()) * NS + int(
        second * NS
    )


def write(
    root: Path,
    dataset: str,
    venue: str,
    symbol: str,
    day: date,
    name: str,
    rows: list[dict[str, Any]],
) -> None:
    folder = partition(root, dataset, venue, symbol, day)
    folder.mkdir(parents=True, exist_ok=True)
    schema = schemas.DATASETS[dataset]
    full = [{f.name: r.get(f.name) for f in schema} for r in rows]
    pq.write_table(pa.Table.from_pylist(full, schema=schema), folder / name)


class Capture:
    """Builds book_l2 rows batch by batch while keeping a reference book."""

    def __init__(self, venue: str, symbol: str, tick: float, raw: bool = False) -> None:
        self.venue, self.symbol, self.tick, self.raw = venue, symbol, tick, raw
        self.book = BookL2(tick, keep_raw_strings=raw)
        self.rows: list[dict[str, Any]] = []
        self.batch = 0
        self.epoch = 0

    def add(
        self, ts: int, levels: list[tuple[int, str, str]], *, snap: bool = False,
        seq: int | None = None, seq_first: int | None = None, seq_prev: int | None = None,
    ) -> int:  # fmt: skip
        self.batch = max(ts, self.batch + 1)
        if snap:
            self.book.clear()
            self.epoch += 1
        for side, px, qty in levels:
            self.book.apply(side, float(px), float(qty), px, qty)
            self.rows.append(
                {"ts_exchange": ts, "ts_recv": ts, "venue": self.venue, "symbol": self.symbol,
                 "seq": seq if seq is not None else self.batch, "side": side, "price": float(px),
                 "qty": float(qty), "is_snapshot": snap, "batch_id": self.batch,
                 "seq_first": seq_first, "seq_prev": seq_prev,
                 "px_str": px if self.raw else None, "qty_str": qty if self.raw else None,
                 "sync_epoch": self.epoch}
            )  # fmt: skip
        return self.batch

    def flush(self, root: Path, day: date, name: str) -> None:
        write(root, "book_l2", self.venue, self.symbol, day, name, self.rows)
        self.rows = []


def trades(root: Path, venue: str, symbol: str, day: date, prices: list[float]) -> None:
    rows = [
        {"ts_exchange": t(day, 0, i), "ts_recv": t(day, 0, i), "venue": venue,
         "symbol": symbol, "trade_id": str(i), "price": p, "qty": 1.0, "side": 1,
         "batch_id": i, "is_aggregated": False}
        for i, p in enumerate(prices)
    ]  # fmt: skip
    write(root, "trades", venue, symbol, day, "part-00.parquet", rows)


def snapshot_levels(
    book: BookL2, n: int = 5
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    return book.raw_levels(BID, n), book.raw_levels(ASK, n)


def test_kraken_checksums_seek_back_and_outputs(tmp_path: Path) -> None:
    v, s = "kraken", "BTC/USD"
    cap = Capture(v, s, 0.1, raw=True)
    cps: list[dict[str, Any]] = []

    def checkpoint(batch: int, ts: int, *, corrupt: bool = False, ok_live: bool = True) -> None:
        bids, asks = snapshot_levels(cap.book, 10)
        value = kraken_checksum(bids, asks) + (1 if corrupt else 0)
        cps.append(
            {"checkpoint_id": str(batch), "ts_recv": ts, "venue": v, "symbol": s,
             "kind": "kraken_crc32", "seq": batch, "batch_id": batch, "expected": value,
             "computed_live": value, "ok_live": ok_live, "n_levels": 10}
        )  # fmt: skip

    snap = [(BID, f"{100 - i * 0.1:.1f}", "1.00000000") for i in range(12)]
    snap += [(ASK, f"{100.1 + i * 0.1:.1f}", "2.00000000") for i in range(12)]
    ts = t(D1, 23, 10)
    b = cap.add(ts, snap, snap=True)
    checkpoint(b, ts)
    ts = t(D1, 23, 20)
    b = cap.add(ts, [(BID, "100.0", "0.00000000"), (ASK, "100.1", "3.50000000")])
    checkpoint(b, ts)
    cap.flush(tmp_path, D1, "part-23.parquet")
    write(tmp_path, "checkpoints", v, s, D1, "part-23.parquet", cps)
    cps = []
    ts = t(D2, 0, 5)
    b = cap.add(ts, [(BID, "99.9", "4.00000000")])
    checkpoint(b, ts)
    ts = t(D2, 0, 6)
    b = cap.add(ts, [(ASK, "100.3", "0.00000000")])
    checkpoint(b, ts, corrupt=True)
    checkpoint(b + 7, ts, ok_live=False)  # failed live: no batch in book_l2
    cap.flush(tmp_path, D2, "part-00.parquet")
    write(tmp_path, "checkpoints", v, s, D2, "part-00.parquet", cps)
    for day in (D1, D2):
        trades(tmp_path, v, s, day, [100.0, 100.2])

    rebuilder = Rebuilder(tmp_path, v, s, 0.1, depth=1000, raw=True)
    (st,) = rebuilder.run(D2, D2)
    assert st.batches == 2 and st.unsynced_batches == 0  # day 2 needed day 1's snapshot
    assert st.crc_total == 3 and st.crc_live_fail == 1
    assert st.crc_compared == 2 and st.crc_match == 1 and st.crc_mismatch == 1
    assert st.crc_unmatched == 0
    assert st.failures[0]["kind"] == "kraken_crc32"
    assert st.crossed == 0 and st.locked == 0

    topn = pl.read_parquet(partition(tmp_path / "interim", "topn", v, s, D2) / "part-00.parquet")
    assert topn.height == 2
    assert topn["event_idx"].to_list() == [0, 1]
    assert topn["bid_px_1"][0] == 999 and topn["bid_qty_1"][0] == 4.0
    assert topn["ask_px_1"][1] == 1001 and topn["ask_qty_1"][1] == 3.5
    bbo = pl.read_parquet(
        partition(tmp_path / "interim", "bbo_rebuilt", v, s, D2) / "part-00.parquet"
    )
    assert bbo["bid_px"][0] == 99.9 and bbo["ask_px"][0] == 100.1
    assert set(bbo["source"]) == {"rebuilt"}
    assert not (partition(tmp_path / "interim", "topn", v, s, D1)).exists()

    save_stats(tmp_path, [st])
    (loaded,) = load_stats(tmp_path)
    assert loaded["crc_match"] == 1


def _rest_cp(
    v: str, s: str, cid: str, ts: int, seq: int, book: BookL2, ok_live: bool = True
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bids, asks = book.top_n(50)
    levels = [
        {"checkpoint_id": cid, "side": BID, "price": float(p), "qty": float(q)}
        for p, q in bids
        if not math.isnan(p)
    ]
    levels += [
        {"checkpoint_id": cid, "side": ASK, "price": float(p), "qty": float(q)}
        for p, q in asks
        if not math.isnan(p)
    ]
    row = {"checkpoint_id": cid, "ts_recv": ts, "venue": v, "symbol": s, "kind": "rest_l2",
           "seq": seq, "batch_id": 0, "expected": None, "computed_live": None,
           "ok_live": ok_live, "n_levels": len(levels)}  # fmt: skip
    return row, levels


def test_binance_rest_checkpoints_straddle_chain_and_gap(tmp_path: Path) -> None:
    v, s = "binance-usdm", "BTCUSDT"
    cap = Capture(v, s, 0.1)
    snap = [(BID, f"{100 - i * 0.1:.1f}", "1") for i in range(5)] + [
        (ASK, f"{100.1 + i * 0.1:.1f}", "1") for i in range(5)
    ]
    cap.add(t(D1, 1, 0), snap, snap=True, seq=100)
    cap.add(t(D1, 1, 1), [(BID, "100.0", "2")], seq=110, seq_first=95, seq_prev=90)
    cap.add(t(D1, 1, 2), [(ASK, "100.2", "5")], seq=120, seq_first=111, seq_prev=110)
    state_after_120 = BookL2(0.1)
    for side in (BID, ASK):
        for p, q in zip(*cap.book.top_n(50)[0 if side == BID else 1].T, strict=False):
            if not math.isnan(p):
                state_after_120.apply(side, p, q)
    # the next event touches 99.8 and straddles L=125
    cap.add(
        t(D1, 1, 3), [(BID, "99.8", "7"), (ASK, "100.4", "0")], seq=130, seq_first=121, seq_prev=120
    )
    cap.add(t(D1, 1, 4), [(BID, "99.6", "1")], seq=140, seq_first=131, seq_prev=130)
    rows_cp: list[dict[str, Any]] = []
    rows_lv: list[dict[str, Any]] = []
    # L=125 straddles 121..130: the REST book has 99.8 at an older size, must be excluded
    rest = BookL2(0.1)
    for side, arr in ((BID, state_after_120.top_n(50)[0]), (ASK, state_after_120.top_n(50)[1])):
        for p, q in arr:
            if not math.isnan(p):
                rest.apply(side, p, q)
    rest.apply(ASK, 100.4, 0.0)  # part of the straddling event already in the snapshot
    r, lv = _rest_cp(v, s, "a", t(D1, 1, 3.2), 125, rest)
    rows_cp.append(r)
    rows_lv += lv
    # L=135: between 130 and 131 chained, state after seq 130 exact
    r, lv = _rest_cp(v, s, "b", t(D1, 1, 4.5), 135, _book_after(cap, 130))
    rows_cp.append(r)
    rows_lv += lv
    # resync: a gap, then a new snapshot at 500; L=300 falls in the gap
    cap.add(t(D1, 1, 10), snap, snap=True, seq=500)
    r, lv = _rest_cp(v, s, "c", t(D1, 1, 9), 300, cap.book)
    rows_cp.append(r)
    rows_lv += lv
    r, lv = _rest_cp(v, s, "d", t(D1, 1, 11), 510, cap.book, ok_live=False)
    rows_cp.append(r)
    rows_lv += lv
    cap.flush(tmp_path, D1, "part-01.parquet")
    write(tmp_path, "checkpoints", v, s, D1, "part-01.parquet", rows_cp)
    write(tmp_path, "checkpoint_levels", v, s, D1, "part-01.parquet", rows_lv)
    trades(tmp_path, v, s, D1, [100.0])

    (st,) = Rebuilder(tmp_path, v, s, 0.1, depth=None, raw=False, write=False).run(D1, D1)
    assert st.rest_total == 4
    assert st.rest_nonlive == 1
    assert st.rest_gap == 1
    assert st.rest_compared == 2 and st.rest_ok == 2 and st.rest_bad == 0, st.failures
    assert st.rest_levels_excluded >= 1


def _book_after(cap: Capture, seq: int) -> BookL2:
    """Reference book as of the batch with this seq, rebuilt from the capture's own rows."""
    book = BookL2(0.1)
    last_batch = None
    for row in cap.rows:
        if row["seq"] > seq:
            break
        if row["is_snapshot"] and row["batch_id"] != last_batch:
            book.clear()
        last_batch = row["batch_id"]
        book.apply(row["side"], row["price"], row["qty"])
    return book


def test_coinbase_rest_checkpoint_matches_a_nearby_boundary(tmp_path: Path) -> None:
    v, s = "coinbase", "BTC-USD"
    cap = Capture(v, s, 0.01)
    snap = [(BID, f"{100 - i * 0.01:.2f}", "1") for i in range(5)] + [
        (ASK, f"{100.01 + i * 0.01:.2f}", "1") for i in range(5)
    ]
    cap.add(t(D1, 2, 0), snap, snap=True)
    cap.add(t(D1, 2, 1.00), [(BID, "100.00", "3")])
    good = BookL2(0.01)
    for side, arr in ((BID, cap.book.top_n(50)[0]), (ASK, cap.book.top_n(50)[1])):
        for p, q in arr:
            if not math.isnan(p):
                good.apply(side, p, q)
    cap.add(t(D1, 2, 1.05), [(ASK, "100.02", "9")])
    cap.add(t(D1, 2, 1.10), [(BID, "99.99", "4")])
    bad = BookL2(0.01)
    for side, arr in ((BID, good.top_n(50)[0]), (ASK, good.top_n(50)[1])):
        for p, q in arr:
            if not math.isnan(p):
                bad.apply(side, p, q)
    bad.apply(ASK, 100.05, 50.0)  # a level no batch near the checkpoint touches
    rows_cp, rows_lv = [], []
    for cid, ts, book in (("g", t(D1, 2, 1.2), good), ("x", t(D1, 2, 1.3), bad)):
        r, lv = _rest_cp(v, s, cid, ts, 0, book)
        rows_cp.append(r)
        rows_lv += lv
    cap.flush(tmp_path, D1, "part-02.parquet")
    write(tmp_path, "checkpoints", v, s, D1, "part-02.parquet", rows_cp)
    write(tmp_path, "checkpoint_levels", v, s, D1, "part-02.parquet", rows_lv)
    trades(tmp_path, v, s, D1, [100.0])

    (st,) = Rebuilder(tmp_path, v, s, 0.01, depth=None, raw=False, write=False).run(D1, D1)
    assert st.rest_total == 2 and st.rest_compared == 2
    assert st.rest_ok == 1 and st.rest_bad == 1
    assert st.failures[0]["checkpoint_id"] == "x"
