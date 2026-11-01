"""Typed environment settings and YAML recorder config."""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)\}")


class Settings(BaseSettings):
    """Values from `.env` and the process environment (spec section 3)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_root: str = Field(default="~/data/crypto-lob-alpha", alias="DATA_ROOT")
    lob_data_dir: str = Field(default="", alias="LOB_DATA_DIR")
    lob_venue_set: str = Field(default="binance-usdm,coinbase,kraken", alias="LOB_VENUE_SET")
    lob_log_level: str = Field(default="INFO", alias="LOB_LOG_LEVEL")
    lob_log_dir: str = Field(default="", alias="LOB_LOG_DIR")
    lob_raw_archive_days: int = Field(default=7, alias="LOB_RAW_ARCHIVE_DAYS")
    lob_min_free_gb: float = Field(default=20.0, alias="LOB_MIN_FREE_GB")
    lob_alert_webhook_url: str = Field(default="", alias="LOB_ALERT_WEBHOOK_URL")

    @property
    def data_dir(self) -> Path:
        """Root of the Parquet lake: `LOB_DATA_DIR`, or `DATA_ROOT` when that is empty."""
        path = Path(self.lob_data_dir or self.data_root).expanduser()
        if PurePosixPath(path.as_posix()).is_relative_to("/mnt"):
            raise ValueError(f"data must stay on the WSL ext4 filesystem, not under /mnt: {path}")
        return path

    @property
    def venues(self) -> list[str]:
        """Venue ids from `LOB_VENUE_SET`."""
        return [v.strip() for v in self.lob_venue_set.split(",") if v.strip()]


class WriterConfig(BaseModel):
    rotation: str = "hourly"
    compression: str = "zstd"
    compression_level: int = 3
    row_group_rows: int = 250_000
    flush_interval_s: float = 60.0


class RawArchiveConfig(BaseModel):
    enabled: bool = True
    retention_days: int = 7
    flush_interval_s: float = 1.0


class HealthConfig(BaseModel):
    stale_after_s: float = 30.0
    backoff_initial_s: float = 1.0
    backoff_max_s: float = 60.0
    proactive_reconnect_h: float = 23.0
    min_free_gb: float = 20.0
    clock_probe_interval_s: float = 60.0
    status_interval_s: float = 10.0
    planned_capture_days: int = 28
    hard_stop_free_gb: float = 5.0


class VenueConfig(BaseModel):
    """Per-venue settings. Fields not used by a venue stay at their defaults."""

    rest: str
    symbols: list[str]
    ws: str = ""
    ws_public: str = ""
    ws_market: str = ""
    streams: dict[str, str] = Field(default_factory=dict)
    channels: list[str] = Field(default_factory=list)
    snapshot_limit: int = 1000
    checkpoint_interval_s: float = 600.0
    book_depth: int = 1000
    checksum_levels: int = 10
    full_channel: bool = False


class RecorderConfig(BaseModel):
    version: int
    venue_set: list[str]
    writer: WriterConfig = WriterConfig()
    raw_archive: RawArchiveConfig = RawArchiveConfig()
    health: HealthConfig = HealthConfig()
    venues: dict[str, VenueConfig]

    @field_validator("venue_set", mode="before")
    @classmethod
    def _split_venues(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value


def _env_values(settings: Settings) -> dict[str, str]:
    values = {
        "LOB_VENUE_SET": settings.lob_venue_set,
        "LOB_RAW_ARCHIVE_DAYS": str(settings.lob_raw_archive_days),
        "LOB_MIN_FREE_GB": str(settings.lob_min_free_gb),
    }
    values.update(os.environ)
    return values


def load_recorder_config(path: Path, settings: Settings | None = None) -> RecorderConfig:
    """Read `config/recorder.yaml`, expanding `${VAR}` from settings and the environment."""
    env = _env_values(settings or Settings())

    def expand(match: re.Match[str]) -> str:
        return env[match.group(1)]

    text = ENV_REF.sub(expand, path.read_text(encoding="utf-8"))
    config = RecorderConfig.model_validate(yaml.safe_load(text))
    missing = [v for v in config.venue_set if v not in config.venues]
    if missing:
        raise ValueError(f"venue_set names venues with no config block: {missing}")
    return config
