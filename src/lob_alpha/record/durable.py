"""Writes that survive losing the machine, not only losing the process."""

from __future__ import annotations

import os
from pathlib import Path

PARQUET_MAGIC = b"PAR1"


def durable_replace(tmp: Path, final: Path) -> None:
    """Rename `tmp` over `final` so a host crash leaves the old state or a complete file.

    A rename alone is not enough. On a VM shutdown the new name can reach the disk ahead of
    the file's last pages, leaving a final name on a file with no Parquet footer.
    """
    with tmp.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, final)
    if os.name != "nt":  # Windows cannot open a directory to sync it
        descriptor = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def has_parquet_footer(path: Path) -> bool:
    """False for a closed file whose tail never reached the disk."""
    try:
        with path.open("rb") as handle:
            handle.seek(-len(PARQUET_MAGIC), os.SEEK_END)
            return handle.read() == PARQUET_MAGIC
    except OSError:
        return False
