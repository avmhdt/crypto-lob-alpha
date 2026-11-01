"""`lob meta fetch`: public tick, lot and contract metadata per venue (FR-13)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import httpx

from lob_alpha.instruments import Instrument, Instruments
from lob_alpha.settings import VenueConfig

BINANCE_DOC = "https://fapi.binance.com/fapi/v1/exchangeInfo"


@dataclass(frozen=True, slots=True)
class Spec:
    """Exchange-published contract facts for one symbol."""

    symbol: str
    tick_size: float
    lot_size: float
    contract_type: str
    contract_value: float
    source_url: str


def _filter(symbol: dict[str, Any], name: str) -> dict[str, Any]:
    return next(f for f in symbol["filters"] if f["filterType"] == name)


def binance_specs(http: httpx.Client, cfg: VenueConfig) -> list[Spec]:
    url = f"{cfg.rest}/fapi/v1/exchangeInfo"
    response = http.get(url)
    response.raise_for_status()
    by_symbol = {s["symbol"]: s for s in response.json()["symbols"]}
    specs = []
    for name in cfg.symbols:
        item = by_symbol[name]
        if item["contractType"] != "PERPETUAL":
            raise ValueError(f"{name} is {item['contractType']}, expected PERPETUAL")
        specs.append(
            Spec(
                name,
                float(_filter(item, "PRICE_FILTER")["tickSize"]),
                float(_filter(item, "LOT_SIZE")["stepSize"]),
                "perp",
                1.0,  # USD-M quantities are already in base units (DQ-05)
                url,
            )
        )
    return specs


def coinbase_specs(http: httpx.Client, cfg: VenueConfig) -> list[Spec]:
    specs = []
    for name in cfg.symbols:
        url = f"{cfg.rest}/products/{name}"
        response = http.get(url)
        response.raise_for_status()
        item = response.json()
        specs.append(
            Spec(
                name,
                float(item["quote_increment"]),
                float(item["base_increment"]),
                "spot",
                1.0,
                url,
            )
        )
    return specs


def kraken_specs(http: httpx.Client, cfg: VenueConfig) -> list[Spec]:
    url = f"{cfg.rest}/0/public/AssetPairs"
    response = http.get(url, params={"pair": ",".join(s.replace("/", "") for s in cfg.symbols)})
    response.raise_for_status()
    body = response.json()
    if body["error"]:
        raise ValueError(f"kraken AssetPairs: {body['error']}")
    # AssetPairs still says XBT/USD where WS v2 publishes BTC/USD.
    by_wsname = {v["wsname"].replace("XBT/", "BTC/"): v for v in body["result"].values()}
    specs = []
    for name in cfg.symbols:
        item = by_wsname[name]
        lot = 10.0 ** -int(item["lot_decimals"])
        specs.append(Spec(name, float(item["tick_size"]), lot, "spot", 1.0, url))
    return specs


FETCHERS: dict[str, Callable[[httpx.Client, VenueConfig], list[Spec]]] = {
    "binance-usdm": binance_specs,
    "coinbase": coinbase_specs,
    "kraken": kraken_specs,
}


def _changed(entry: Instrument, spec: Spec) -> bool:
    return (
        entry.tick_size != spec.tick_size
        or entry.lot_size != spec.lot_size
        or entry.contract_type != spec.contract_type
        or entry.contract_value != spec.contract_value
    )


def merge_specs(
    instruments: Instruments, venue: str, specs: Sequence[Spec], today: date
) -> list[str]:
    """Add a dated entry for each changed symbol. Returns a line per change.

    A change closes the current entry yesterday and opens a new one today with the same
    fees, so history stays queryable by date (DQ-11). New symbols need fees entered by hand
    and are reported, not invented.
    """
    changes: list[str] = []
    for spec in specs:
        try:
            current = instruments.get(venue, spec.symbol, today)
        except KeyError:
            changes.append(f"{venue} {spec.symbol}: NEW, add a fee entry by hand: {spec}")
            continue
        if not _changed(current, spec):
            continue
        changes.append(
            f"{venue} {spec.symbol}: tick {current.tick_size} -> {spec.tick_size}, "
            f"lot {current.lot_size} -> {spec.lot_size}"
        )
        current.effective_to = today - timedelta(days=1)
        instruments.entries.append(
            current.model_copy(
                update={
                    "tick_size": spec.tick_size,
                    "lot_size": spec.lot_size,
                    "contract_type": spec.contract_type,
                    "contract_value": spec.contract_value,
                    "effective_from": today,
                    "effective_to": None,
                    "source_url": spec.source_url,
                }
            )
        )
    return changes
