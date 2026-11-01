"""In-memory L2 order book keyed by integer ticks (DQ-04)."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from sortedcontainers import SortedDict

BID = 1
ASK = -1
TICK_TOLERANCE = 1e-6


class TickError(ValueError):
    """A price is not a whole number of ticks (DQ-04)."""


class _Level(NamedTuple):
    price: float
    qty: float
    px_str: str | None
    qty_str: str | None


class BookL2:
    """One side-sorted L2 book. State is only meaningful between batches (DQ-03).

    Quantities are absolute: `apply` replaces the size at a price, and a zero size
    removes the level. Removing a level that is not in the book is not an error,
    because venues send removals for levels outside the local view.
    """

    def __init__(
        self, tick_size: float, keep_raw_strings: bool = False, *, strict: bool = True
    ) -> None:
        if tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {tick_size}")
        self.tick_size = tick_size
        self.keep_raw_strings = keep_raw_strings
        self.strict = strict
        self._bids: SortedDict = SortedDict()
        self._asks: SortedDict = SortedDict()

    def clear(self) -> None:
        """Drop every level on both sides."""
        self._bids.clear()
        self._asks.clear()

    def to_ticks(self, price: float) -> int:
        """Map a price to integer ticks, raising `TickError` when off-grid in strict mode."""
        exact = price / self.tick_size
        ticks = round(exact)
        if self.strict and abs(exact - ticks) >= TICK_TOLERANCE:
            raise TickError(f"price {price} is not a multiple of tick {self.tick_size}")
        return ticks

    def apply(
        self,
        side: int,
        price: float,
        qty: float,
        px_str: str | None = None,
        qty_str: str | None = None,
    ) -> None:
        """Set the absolute size at one level; zero removes it."""
        ticks = self.to_ticks(price)
        book = self._bids if side == BID else self._asks
        if qty == 0:
            book.pop(ticks, None)
        elif self.keep_raw_strings:
            book[ticks] = _Level(price, qty, px_str, qty_str)
        else:
            book[ticks] = _Level(price, qty, None, None)

    def best_bid(self) -> tuple[float, float] | None:
        """Highest bid as (price, qty), or None for an empty side."""
        if not self._bids:
            return None
        level: _Level = self._bids.peekitem(-1)[1]
        return level.price, level.qty

    def best_ask(self) -> tuple[float, float] | None:
        """Lowest ask as (price, qty), or None for an empty side."""
        if not self._asks:
            return None
        level: _Level = self._asks.peekitem(0)[1]
        return level.price, level.qty

    def _walk(self, side: int, n: int) -> list[_Level]:
        book = self._bids if side == BID else self._asks
        count = min(n, len(book))
        if side == BID:
            return [book.peekitem(-1 - i)[1] for i in range(count)]
        return [book.peekitem(i)[1] for i in range(count)]

    def top_n(self, n: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Best `n` levels as two (n, 2) arrays, best first, NaN-padded when the side is short."""
        out: list[npt.NDArray[np.float64]] = []
        for side in (BID, ASK):
            rows = np.full((n, 2), np.nan, dtype=np.float64)
            for i, level in enumerate(self._walk(side, n)):
                rows[i, 0] = level.price
                rows[i, 1] = level.qty
            out.append(rows)
        return out[0], out[1]

    def is_crossed(self) -> bool:
        """True when best bid >= best ask (DQ-08 counts a locked book here as well)."""
        if not self._bids or not self._asks:
            return False
        return bool(self._bids.peekitem(-1)[0] >= self._asks.peekitem(0)[0])

    def is_locked(self) -> bool:
        """True when best bid == best ask."""
        if not self._bids or not self._asks:
            return False
        return bool(self._bids.peekitem(-1)[0] == self._asks.peekitem(0)[0])

    def raw_levels(self, side: int, n: int) -> list[tuple[str, str]]:
        """Best `n` levels as raw (price, qty) strings, needed for venue checksums."""
        if not self.keep_raw_strings:
            raise ValueError("book was created with keep_raw_strings=False")
        return [(level.px_str or "", level.qty_str or "") for level in self._walk(side, n)]

    def truncate(self, depth: int) -> None:
        """Keep the best `depth` levels per side, as Kraken requires after each update."""
        while len(self._bids) > depth:
            self._bids.popitem(0)
        while len(self._asks) > depth:
            self._asks.popitem(-1)

    def depth(self, side: int) -> int:
        """Number of levels held on one side."""
        return len(self._bids if side == BID else self._asks)
