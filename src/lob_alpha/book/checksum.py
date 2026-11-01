"""Exchange book checksums, always computed from the raw strings on the wire (NFR-06)."""

from __future__ import annotations

import zlib
from collections.abc import Sequence


def _strip(value: str) -> str:
    return value.replace(".", "").lstrip("0")


def kraken_checksum(
    bids: Sequence[tuple[str, str]], asks: Sequence[tuple[str, str]], levels: int = 10
) -> int:
    """CRC32 of Kraken's WS v2 book string.

    Each of the top `levels` is rendered as price then quantity with the decimal point
    and leading zeros removed. Asks (best first) are concatenated, then bids (best
    first), and the unsigned CRC32 of that ASCII string is the checksum.
    """
    parts = [_strip(px) + _strip(qty) for px, qty in asks[:levels]]
    parts += [_strip(px) + _strip(qty) for px, qty in bids[:levels]]
    return zlib.crc32("".join(parts).encode("ascii"))
