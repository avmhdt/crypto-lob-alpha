from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest
import yaml

from lob_alpha import schemas
from lob_alpha.fetch.metadata import FETCHERS, merge_specs
from lob_alpha.instruments import (
    Instrument,
    Instruments,
    dump_instruments,
    load_instruments,
)
from lob_alpha.settings import Settings, VenueConfig, load_recorder_config

REPO = Path(__file__).resolve().parents[2]


def settings(**values: str) -> Settings:
    return Settings(_env_file=None, **values)


def test_data_dir_defaults_to_data_root_and_expands_home() -> None:
    assert settings(DATA_ROOT="~/x").data_dir == Path("~/x").expanduser()
    assert settings(DATA_ROOT="/tmp/a", LOB_DATA_DIR="/tmp/b").data_dir == Path("/tmp/b")


def test_data_under_mnt_is_refused() -> None:
    with pytest.raises(ValueError, match="ext4"):
        _ = settings(DATA_ROOT="/mnt/d/data").data_dir


def test_venue_set_splits() -> None:
    assert settings(LOB_VENUE_SET="kraken, coinbase").venues == ["kraken", "coinbase"]


def test_env_example_has_exactly_the_documented_keys() -> None:
    lines = (REPO / ".env.example").read_text(encoding="utf-8").splitlines()
    keys = [line.split("=")[0] for line in lines if line and not line.startswith("#")]
    assert keys == [
        "DATA_ROOT", "LOB_DATA_DIR", "LOB_VENUE_SET", "LOB_LOG_LEVEL", "LOB_LOG_DIR",
        "LOB_RAW_ARCHIVE_DAYS", "LOB_MIN_FREE_GB", "LOB_ALERT_WEBHOOK_URL", "COINBASE_API_KEY",
        "COINBASE_API_SECRET", "COINBASE_API_PASSPHRASE", "TARDIS_API_KEY",
    ]  # fmt: skip


def test_env_example_loads_as_a_real_env_file() -> None:
    loaded = Settings(_env_file=REPO / ".env.example")
    assert loaded.lob_data_dir == ""
    assert loaded.lob_alert_webhook_url == ""
    assert loaded.data_dir == Path("~/data/crypto-lob-alpha").expanduser()
    assert loaded.venues == ["binance-usdm", "coinbase", "kraken"]
    assert loaded.lob_raw_archive_days == 7
    assert loaded.lob_min_free_gb == 20.0


def test_recorder_config_expands_env_values() -> None:
    config = load_recorder_config(
        REPO / "config" / "recorder.yaml",
        settings(LOB_VENUE_SET="kraken", LOB_RAW_ARCHIVE_DAYS="3"),
    )
    assert config.venue_set == ["kraken"]
    assert config.raw_archive.retention_days == 3
    assert config.health.min_free_gb == 20.0
    assert config.venues["binance-usdm"].ws_public.endswith("/public/stream")
    assert config.venues["kraken"].book_depth == 1000


def test_recorder_config_rejects_a_venue_with_no_block(tmp_path: Path) -> None:
    text = (REPO / "config" / "recorder.yaml").read_text(encoding="utf-8")
    path = tmp_path / "r.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="no config block"):
        load_recorder_config(path, settings(LOB_VENUE_SET="okx"))


def test_canonical_fields_lead_every_schema_in_order() -> None:
    assert list(schemas.TRADES.names[:8]) == list(schemas.TRADES_CANONICAL)
    assert list(schemas.BOOK_L2.names[:10]) == list(schemas.BOOK_L2_CANONICAL)
    assert list(schemas.BOOK_L3.names[:11]) == list(schemas.BOOK_L3_CANONICAL)
    assert list(schemas.BBO.names[:7]) == list(schemas.BBO_CANONICAL)


def test_canonical_types_are_the_spec_types() -> None:
    expected = {
        "ts_exchange": "int64", "ts_recv": "int64", "venue": "string", "symbol": "string",
        "seq": "int64", "side": "int8", "price": "double", "qty": "double",
        "is_snapshot": "bool", "batch_id": "int64",
    }  # fmt: skip
    for name, type_name in expected.items():
        assert str(schemas.BOOK_L2.field(name).type) == type_name
    assert str(schemas.TRADES.field("trade_id").type) == "string"
    assert str(schemas.BOOK_L3.field("order_id").type) == "uint64"
    assert str(schemas.BOOK_L2.field("sync_epoch").type) == "int32"


def entry(**over: object) -> Instrument:
    base: dict[str, object] = {
        "venue": "kraken", "symbol": "BTC/USD", "tick_size": 0.1, "lot_size": 1e-8,
        "contract_type": "spot", "contract_value": 1.0, "maker_fee_bps": 40.0,
        "taker_fee_bps": 80.0, "effective_from": date(2026, 11, 1), "source_url": "u",
        "fee_source_url": "f", "fee_read_on": date(2026, 11, 1),
    }  # fmt: skip
    return Instrument.model_validate({**base, **over})


def test_instrument_lookup_is_by_effective_date() -> None:
    table = Instruments(
        [
            entry(effective_to=date(2026, 11, 30), maker_fee_bps=40.0),
            entry(effective_from=date(2026, 12, 1), maker_fee_bps=25.0),
        ]
    )
    assert table.get("kraken", "BTC/USD", date(2026, 11, 30)).maker_fee_bps == 40.0
    assert table.get("kraken", "BTC/USD", date(2026, 12, 1)).maker_fee_bps == 25.0
    with pytest.raises(KeyError):
        table.get("kraken", "BTC/USD", date(2026, 10, 31))


def test_require_reports_missing_instruments_and_undated_fees() -> None:
    table = Instruments([entry(), entry(symbol="ETH/USD", fee_source_url="")])
    table.require([("kraken", "BTC/USD")], date(2026, 11, 2))
    with pytest.raises(ValueError, match="fee entry"):
        table.require([("kraken", "ETH/USD")], date(2026, 11, 2))
    with pytest.raises(ValueError, match="no instrument entry"):
        table.require([("kraken", "SOL/USD")], date(2026, 11, 2))


def test_committed_instruments_cover_every_recorded_symbol() -> None:
    table = load_instruments(REPO / "config" / "instruments.yaml")
    raw = yaml.safe_load(
        (REPO / "config" / "recorder.yaml")
        .read_text(encoding="utf-8")
        .replace("${LOB_VENUE_SET}", "x")
        .replace("${LOB_RAW_ARCHIVE_DAYS}", "1")
        .replace("${LOB_MIN_FREE_GB}", "1")
    )
    pairs = [(v, s) for v, block in raw["venues"].items() for s in block["symbols"]]
    assert len(pairs) == 9
    table.require(pairs, date(2026, 11, 2))


def test_instruments_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "instruments.yaml"
    dump_instruments(Instruments([entry()]), path)
    assert load_instruments(path).entries == [entry()]


def client(routes: dict[str, dict[str, object]]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=routes[request.url.path])

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_binance_metadata() -> None:
    body = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "contractType": "PERPETUAL",
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                ],
            }
        ]
    }
    cfg = VenueConfig(rest="https://x", symbols=["BTCUSDT"])
    spec = FETCHERS["binance-usdm"](client({"/fapi/v1/exchangeInfo": body}), cfg)[0]
    assert (spec.tick_size, spec.lot_size, spec.contract_type) == (0.1, 0.001, "perp")


def test_coinbase_metadata() -> None:
    body = {"quote_increment": "0.01", "base_increment": "0.00000001"}
    cfg = VenueConfig(rest="https://x", symbols=["BTC-USD"])
    spec = FETCHERS["coinbase"](client({"/products/BTC-USD": body}), cfg)[0]
    assert (spec.tick_size, spec.lot_size, spec.contract_type) == (0.01, 1e-8, "spot")


def test_kraken_metadata_maps_xbt_to_btc() -> None:
    pair = {"wsname": "XBT/USD", "tick_size": "0.1", "lot_decimals": 8}
    body = {"error": [], "result": {"XXBTZUSD": pair}}
    cfg = VenueConfig(rest="https://x", symbols=["BTC/USD"])
    spec = FETCHERS["kraken"](client({"/0/public/AssetPairs": body}), cfg)[0]
    assert spec.tick_size == 0.1


def test_merge_adds_a_dated_entry_only_on_change() -> None:
    table = Instruments([entry()])
    cfg = VenueConfig(rest="https://x", symbols=["BTC/USD"])
    pair = {"wsname": "XBT/USD", "tick_size": "0.1", "lot_decimals": 8}
    http = client({"/0/public/AssetPairs": {"error": [], "result": {"X": pair}}})
    same = FETCHERS["kraken"](http, cfg)
    assert merge_specs(table, "kraken", same, date(2026, 12, 1)) == []
    pair["tick_size"] = "0.5"
    moved = FETCHERS["kraken"](http, cfg)
    changes = merge_specs(table, "kraken", moved, date(2026, 12, 1))
    assert "tick 0.1 -> 0.5" in changes[0]
    assert table.get("kraken", "BTC/USD", date(2026, 11, 30)).tick_size == 0.1
    assert table.get("kraken", "BTC/USD", date(2026, 12, 1)).tick_size == 0.5
    assert table.get("kraken", "BTC/USD", date(2026, 12, 1)).maker_fee_bps == 40.0


def test_merge_reports_new_symbols_instead_of_inventing_fees() -> None:
    table = Instruments([])
    cfg = VenueConfig(rest="https://x", symbols=["BTC/USD"])
    pair = {"wsname": "XBT/USD", "tick_size": "0.1", "lot_decimals": 8}
    http = client({"/0/public/AssetPairs": {"error": [], "result": {"X": pair}}})
    specs = FETCHERS["kraken"](http, cfg)
    assert "NEW" in merge_specs(table, "kraken", specs, date(2026, 12, 1))[0]
    assert table.entries == []
