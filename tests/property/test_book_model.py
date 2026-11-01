"""`BookL2` must equal a naive dict model after every batch."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from lob_alpha.book.checksum import kraken_checksum
from lob_alpha.book.l2 import ASK, BID, BookL2

TICK = 0.5
updates = st.tuples(
    st.sampled_from([BID, ASK]),
    st.integers(min_value=1, max_value=60),
    st.sampled_from([0.0, 0.25, 1.0, 2.5, 7.0]),
)
batches = st.lists(st.lists(updates, max_size=8), max_size=25)


@given(batches)
def test_book_equals_naive_model(sequence: list[list[tuple[int, int, float]]]) -> None:
    book = BookL2(TICK)
    model: dict[int, dict[int, float]] = {BID: {}, ASK: {}}
    for batch in sequence:
        for side, ticks, qty in batch:
            book.apply(side, ticks * TICK, qty)
            if qty == 0:
                model[side].pop(ticks, None)
            else:
                model[side][ticks] = qty
        bid = book.best_bid()
        ask = book.best_ask()
        assert bid == (
            (max(model[BID]) * TICK, model[BID][max(model[BID])]) if model[BID] else None
        )
        assert ask == (
            (min(model[ASK]) * TICK, model[ASK][min(model[ASK])]) if model[ASK] else None
        )
        crossed = bool(model[BID] and model[ASK] and max(model[BID]) >= min(model[ASK]))
        assert book.is_crossed() == crossed
        assert book.depth(BID) == len(model[BID])
        assert book.depth(ASK) == len(model[ASK])


level = st.tuples(
    st.integers(min_value=1, max_value=99_999), st.integers(min_value=1, max_value=99_999)
)


def _render(rows: list[tuple[int, int]]) -> list[tuple[str, str]]:
    return [(f"{p / 10:.1f}", f"{q / 1000:.8f}") for p, q in rows]


@given(st.lists(level, min_size=2, max_size=10), st.data())
def test_checksum_changes_when_any_level_changes(rows: list[tuple[int, int]], data) -> None:
    bids = _render(rows)
    asks = _render(rows[::-1])
    index = data.draw(st.integers(min_value=0, max_value=len(rows) - 1))
    changed = list(rows)
    changed[index] = (rows[index][0], rows[index][1] + 1)
    assert kraken_checksum(_render(changed), asks) != kraken_checksum(bids, asks)
