"""Adapter lookup by venue id."""

from __future__ import annotations

from lob_alpha.record.venues.base import VenueAdapter
from lob_alpha.record.venues.binance_usdm import BinanceUsdm
from lob_alpha.record.venues.coinbase import Coinbase
from lob_alpha.record.venues.kraken import Kraken
from lob_alpha.settings import VenueConfig


def make_adapter(venue: str, cfg: VenueConfig) -> VenueAdapter:
    """Build the adapter for a venue id. Only the D-01 venue set exists here."""
    if venue == "binance-usdm":
        return BinanceUsdm(cfg)
    if venue == "kraken":
        return Kraken(cfg)
    if venue == "coinbase":
        return Coinbase(cfg)
    raise ValueError(f"no adapter for venue {venue!r}")
