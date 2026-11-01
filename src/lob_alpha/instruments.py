"""Instrument metadata with effective dates (FR-13, DQ-11)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

DEFAULT_PATH = Path("config/instruments.yaml")


class Instrument(BaseModel):
    """One dated metadata entry for a (venue, symbol)."""

    venue: str
    symbol: str
    symbol_norm: str | None = None
    tick_size: float
    lot_size: float
    contract_type: Literal["perp", "spot"]
    contract_value: float
    maker_fee_bps: float
    taker_fee_bps: float
    effective_from: date
    effective_to: date | None = None
    source_url: str
    fee_source_url: str = ""
    fee_read_on: date | None = None


class Instruments:
    """Lookup of instrument metadata as of a date."""

    def __init__(self, entries: Sequence[Instrument]) -> None:
        self.entries = list(entries)

    def get(self, venue: str, symbol: str, on: date | None = None) -> Instrument:
        """Entry in force on `on` (today when None). Raises KeyError when there is none."""
        when = on or date.today()
        for entry in self.entries:
            if entry.venue != venue or entry.symbol != symbol or entry.effective_from > when:
                continue
            if entry.effective_to is None or entry.effective_to >= when:
                return entry
        raise KeyError(f"no instrument entry for {venue} {symbol} on {when}")

    def require(self, pairs: Sequence[tuple[str, str]], on: date | None = None) -> None:
        """Raise ValueError unless every (venue, symbol) has a complete entry with a fee source."""
        problems: list[str] = []
        for venue, symbol in pairs:
            try:
                entry = self.get(venue, symbol, on)
            except KeyError as exc:
                problems.append(str(exc))
                continue
            if not entry.fee_source_url or entry.fee_read_on is None:
                problems.append(f"{venue} {symbol}: fee entry has no source URL or read date")
        if problems:
            raise ValueError("; ".join(problems))


def load_instruments(path: Path = DEFAULT_PATH) -> Instruments:
    """Read and validate `config/instruments.yaml`."""
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Instruments([Instrument.model_validate(item) for item in raw["instruments"]])


def dump_instruments(instruments: Instruments, path: Path = DEFAULT_PATH) -> None:
    """Write the file back in a stable order."""
    rows = sorted(instruments.entries, key=lambda e: (e.venue, e.symbol, e.effective_from))
    payload = {"instruments": [e.model_dump(mode="json", exclude_none=True) for e in rows]}
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    path.write_text(text, encoding="utf-8", newline="\n")
