"""Validation tables from the rebuild's per-day stats, and the AC-W-02 / AC-W-03 verdicts.

AC-W-02: no LIVE batch boundary with best bid >= best ask on any rebuilt day. Crossed
(bid > ask) and locked (bid == ask) are counted separately; none of the three venues can
rest a locked book, so both count as failures (DQ-08).

AC-W-03: every compared checkpoint matches; excluded checkpoints are counted and are at most
1% of the total per (venue, symbol, kind) over the rebuilt days; REST venues have at least
144 REST checkpoints on every complete day (DQ-10).
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_EXCLUDED_PCT = 1.0
MIN_REST_PER_DAY = 144
REST_VENUES = ("binance-usdm", "coinbase")

SUMMARY_FIELDS = [
    "venue", "symbol", "date", "kind", "total", "excluded_nonlive", "excluded_gap",
    "excluded_inconclusive", "compared",
    "ok", "mismatched", "excluded_pct", "levels_compared", "levels_excluded", "levels_bad",
    "max_offset_ms", "complete_day", "meets_min_per_day",
]  # fmt: skip
CROSSED_FIELDS = [
    "venue", "symbol", "date", "files", "rows", "boundaries", "snapshots", "crossed", "locked",
    "one_sided", "first_crossed_utc", "first_locked_utc", "unsynced_batches", "out_of_window",
    "tick_errors", "order_violations", "kernel_seconds", "updates_per_s",
]  # fmt: skip


def _utc(ns: int | None) -> str:
    if ns is None:
        return ""
    return datetime.fromtimestamp(ns / 1e9, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _round(value: float | None) -> float | str:
    return "" if value is None else round(value, 1)


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 4) if whole else 0.0


def summary_rows(
    stats: Iterable[dict[str, Any]], complete: set[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    """One row per (venue, symbol, date, kind) that has checkpoints."""
    rows: list[dict[str, Any]] = []
    for st in stats:
        key = (st["venue"], st["symbol"], st["date"])
        is_complete = key in complete
        if st["crc_total"]:
            excl = st["crc_live_fail"] + st["crc_unmatched"]
            rows.append(
                {"venue": st["venue"], "symbol": st["symbol"], "date": st["date"],
                 "kind": "kraken_crc32",
                 "total": st["crc_total"], "excluded_nonlive": st["crc_live_fail"],
                 "excluded_gap": st["crc_unmatched"], "compared": st["crc_compared"],
                 "ok": st["crc_match"], "mismatched": st["crc_mismatch"],
                 "excluded_pct": _pct(excl, st["crc_total"]), "levels_compared": "",
                 "levels_excluded": "", "levels_bad": "", "max_offset_ms": "",
                 "complete_day": is_complete, "meets_min_per_day": ""}
            )  # fmt: skip
        if st["rest_total"] or st["venue"] in REST_VENUES:
            excl = st["rest_nonlive"] + st["rest_gap"] + st.get("rest_inconclusive", 0)
            meets: bool | str = ""
            if st["venue"] in REST_VENUES and is_complete:
                meets = st["rest_total"] >= MIN_REST_PER_DAY
            rows.append(
                {"venue": st["venue"], "symbol": st["symbol"], "date": st["date"],
                 "kind": "rest_l2",
                 "total": st["rest_total"], "excluded_nonlive": st["rest_nonlive"],
                 "excluded_gap": st["rest_gap"],
                 "excluded_inconclusive": st.get("rest_inconclusive", 0),
                 "compared": st["rest_compared"],
                 "ok": st["rest_ok"], "mismatched": st["rest_bad"],
                 "excluded_pct": _pct(excl, st["rest_total"]),
                 "levels_compared": st["rest_levels_compared"],
                 "levels_excluded": st["rest_levels_excluded"], "levels_bad": st["rest_levels_bad"],
                 "max_offset_ms": _round(st["rest_offset_ms_max"]),
                 "complete_day": is_complete, "meets_min_per_day": meets}
            )  # fmt: skip
    return sorted(rows, key=lambda r: (r["venue"], r["symbol"], r["kind"], r["date"]))


def crossed_rows(stats: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for st in stats:
        secs = st["kernel_seconds"]
        rows.append(
            {"venue": st["venue"], "symbol": st["symbol"], "date": st["date"], "files": st["files"],
             "rows": st["rows"], "boundaries": st["batches"], "snapshots": st["snapshots"],
             "crossed": st["crossed"], "locked": st["locked"], "one_sided": st["one_sided"],
             "first_crossed_utc": _utc(st["first_crossed_ts"]),
             "first_locked_utc": _utc(st["first_locked_ts"]),
             "unsynced_batches": st["unsynced_batches"], "out_of_window": st["out_of_window"],
             "tick_errors": st["tick_errors"], "order_violations": st["order_violations"],
             "kernel_seconds": round(secs, 3),
             "updates_per_s": round(st["rows"] / secs) if secs else ""}
        )  # fmt: skip
    return sorted(rows, key=lambda r: (r["venue"], r["symbol"], r["date"]))


def failure_rows(stats: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for st in stats:
        for item in st.get("failures", []):
            rows.append({"venue": st["venue"], "symbol": st["symbol"], "date": st["date"], **item})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verdict_ac_w_02(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    if not rows:
        return False, "no rebuilt days"
    crossed = sum(int(r["crossed"]) for r in rows)
    locked = sum(int(r["locked"]) for r in rows)
    days = len({(r["venue"], r["symbol"], r["date"]) for r in rows})
    ok = crossed == 0 and locked == 0
    return ok, f"{days} symbol-days: {crossed} crossed, {locked} locked batch boundaries"


def verdict_ac_w_03(rows: list[dict[str, Any]]) -> tuple[bool, str]:
    if not rows:
        return False, "no checkpoints"
    problems: list[str] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r["venue"], r["symbol"], r["kind"]), []).append(r)
    for (venue, symbol, kind), items in sorted(groups.items()):
        total = sum(int(r["total"]) for r in items)
        excl = sum(
            int(r["excluded_nonlive"])
            + int(r["excluded_gap"])
            + int(r.get("excluded_inconclusive") or 0)
            for r in items
        )
        bad = sum(int(r["mismatched"]) for r in items)
        if bad:
            problems.append(f"{venue} {symbol} {kind}: {bad} mismatched")
        if total and 100.0 * excl / total > MAX_EXCLUDED_PCT:
            problems.append(f"{venue} {symbol} {kind}: {_pct(excl, total)}% excluded")
        short = [r["date"] for r in items if str(r["meets_min_per_day"]) == "False"]
        if short:
            problems.append(
                f"{venue} {symbol} {kind}: under {MIN_REST_PER_DAY} on complete days {short}"
            )
    compared = sum(int(r["compared"]) for r in rows)
    text = f"{compared} checkpoints compared in {len(groups)} groups"
    return (not problems), text + ("; " + "; ".join(problems) if problems else "")
