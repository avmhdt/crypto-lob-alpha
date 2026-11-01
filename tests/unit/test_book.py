from __future__ import annotations

import math

import pytest

from lob_alpha.book.checksum import kraken_checksum
from lob_alpha.book.l2 import ASK, BID, BookL2, TickError

# Worked example from Kraken's "Spot WebSocket Book v2" checksum guide, read 2026-11-01.
DOC_ASKS = [
    ("45285.2", "0.00100000"),
    ("45286.4", "1.54571953"),
    ("45286.6", "1.54571109"),
    ("45289.6", "1.54560911"),
    ("45290.2", "0.15890660"),
    ("45291.8", "1.54553491"),
    ("45294.7", "0.04454749"),
    ("45296.1", "0.35380000"),
    ("45297.5", "0.09945542"),
    ("45299.5", "0.18772827"),
]
DOC_BIDS = [
    ("45283.5", "0.10000000"),
    ("45283.4", "1.54582015"),
    ("45282.1", "0.10000000"),
    ("45281.0", "0.10000000"),
    ("45280.3", "1.54592586"),
    ("45279.0", "0.07990000"),
    ("45277.6", "0.03310103"),
    ("45277.5", "0.30000000"),
    ("45277.3", "1.54602737"),
    ("45276.6", "0.15445238"),
]


def test_kraken_checksum_doc_example() -> None:
    assert kraken_checksum(DOC_BIDS, DOC_ASKS) == 3310070434


def test_kraken_checksum_changes_with_any_top_level() -> None:
    changed = [*DOC_BIDS]
    changed[9] = ("45276.6", "0.15445239")
    assert kraken_checksum(changed, DOC_ASKS) != 3310070434


def test_kraken_checksum_uses_only_requested_levels() -> None:
    extra = [*DOC_ASKS, ("45300.0", "9.00000000")]
    assert kraken_checksum(DOC_BIDS, extra) == 3310070434


def test_apply_replace_remove_and_best() -> None:
    book = BookL2(0.1)
    book.apply(BID, 100.0, 2.0)
    book.apply(BID, 100.1, 1.0)
    book.apply(ASK, 100.2, 3.0)
    assert book.best_bid() == (100.1, 1.0)
    assert book.best_ask() == (100.2, 3.0)
    book.apply(BID, 100.1, 4.0)
    assert book.best_bid() == (100.1, 4.0)
    book.apply(BID, 100.1, 0.0)
    assert book.best_bid() == (100.0, 2.0)


def test_removing_an_unknown_level_is_not_an_error() -> None:
    book = BookL2(0.1)
    book.apply(ASK, 50.0, 0.0)
    assert book.best_ask() is None


def test_empty_book_has_no_best_and_is_not_crossed() -> None:
    book = BookL2(0.01)
    assert book.best_bid() is None
    assert book.best_ask() is None
    assert not book.is_crossed()
    assert not book.is_locked()


def test_crossed_and_locked() -> None:
    book = BookL2(0.5)
    book.apply(BID, 10.0, 1.0)
    book.apply(ASK, 10.5, 1.0)
    assert not book.is_crossed()
    book.apply(BID, 10.5, 1.0)
    assert book.is_crossed()
    assert book.is_locked()
    book.apply(BID, 11.0, 1.0)
    assert book.is_crossed()
    assert not book.is_locked()


def test_top_n_orders_best_first_and_pads_with_nan() -> None:
    book = BookL2(1.0)
    book.apply(BID, 9.0, 1.0)
    book.apply(BID, 8.0, 2.0)
    book.apply(ASK, 11.0, 3.0)
    bids, asks = book.top_n(3)
    assert bids.shape == asks.shape == (3, 2)
    assert list(bids[0]) == [9.0, 1.0]
    assert list(bids[1]) == [8.0, 2.0]
    assert math.isnan(bids[2, 0])
    assert list(asks[0]) == [11.0, 3.0]


def test_raw_levels_keep_exact_strings() -> None:
    book = BookL2(0.1, keep_raw_strings=True)
    book.apply(ASK, 100.1, 0.5, "100.1", "0.50000000")
    book.apply(ASK, 100.0, 1.0, "100.0", "1.00000000")
    assert book.raw_levels(ASK, 2) == [("100.0", "1.00000000"), ("100.1", "0.50000000")]


def test_raw_levels_need_keep_raw_strings() -> None:
    with pytest.raises(ValueError, match="keep_raw_strings"):
        BookL2(0.1).raw_levels(BID, 1)


def test_truncate_drops_the_worst_levels() -> None:
    book = BookL2(1.0)
    for price in (5.0, 6.0, 7.0):
        book.apply(BID, price, 1.0)
    for price in (8.0, 9.0, 10.0):
        book.apply(ASK, price, 1.0)
    book.truncate(2)
    assert book.depth(BID) == 2
    assert book.depth(ASK) == 2
    assert book.best_bid() == (7.0, 1.0)
    assert book.top_n(2)[0][1, 0] == 6.0
    assert book.top_n(2)[1][1, 0] == 9.0


def test_off_grid_price_raises_in_strict_mode_only() -> None:
    with pytest.raises(TickError):
        BookL2(0.1).apply(BID, 100.05, 1.0)
    relaxed = BookL2(0.1, strict=False)
    relaxed.apply(BID, 100.05, 1.0)
    assert relaxed.depth(BID) == 1


def test_tick_mapping_survives_float_noise() -> None:
    book = BookL2(0.1)
    assert book.to_ticks(77896.7) == 778967
    assert book.to_ticks(0.3) == 3


def test_tick_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        BookL2(0.0)


def test_clear() -> None:
    book = BookL2(1.0)
    book.apply(BID, 1.0, 1.0)
    book.clear()
    assert book.best_bid() is None
