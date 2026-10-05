from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple
from urllib.parse import quote

from .models import (
    BestBidAskRecord,
    BTCPriceSampleRecord,
    FeatureRecord,
    MarketEventRecord,
    MarketMetadata,
    MarketSnapshotRecord,
    OrderBookLevelRecord,
    parse_timestamp,
    RECORDER_VERSION,
    RawPolymarketEventRecord,
    RecorderMetricRecord,
    SCHEMA_VERSION,
    TickSizeChangeRecord,
    TradeRecord,
    to_iso,
    utc_now,
)

STARTUP_INTEGRITY_CHECK_MODES = {"off", "quick", "full"}
SQLITE_WAL_CHECKPOINT_MODES = {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}


def normalize_startup_integrity_check_mode(mode: str | None) -> str:
    normalized = (mode or "quick").strip().lower()
    if normalized not in STARTUP_INTEGRITY_CHECK_MODES:
        raise ValueError(
            "startup integrity check mode must be one of "
            f"{sorted(STARTUP_INTEGRITY_CHECK_MODES)}, got {normalized!r}"
        )
    return normalized


def normalize_sqlite_wal_checkpoint_mode(mode: str | None) -> str:
    normalized = (mode or "PASSIVE").strip().upper()
    if normalized not in SQLITE_WAL_CHECKPOINT_MODES:
        raise ValueError(
            "SQLite WAL checkpoint mode must be one of "
            f"{sorted(SQLITE_WAL_CHECKPOINT_MODES)}, got {normalized!r}"
        )
    return normalized


def run_sqlite_wal_checkpoint(
    db_path: str,
    *,
    mode: str = "PASSIVE",
    busy_timeout_ms: int = 5000,
    truncate: bool = False,
) -> dict[str, Any]:
    normalized_mode = normalize_sqlite_wal_checkpoint_mode(mode)
    path = Path(db_path)
    db_size_before = _file_size(path)
    sizes_before = _sqlite_file_sizes(path)
    if not path.exists():
        return {
            "status": "missing",
            "db_path": str(path),
            "mode": normalized_mode,
            "truncate_requested": bool(truncate),
            "sizes_before": sizes_before,
            "sizes_after": sizes_before,
            "wal_size_warnings": _wal_size_warnings(sizes_before["wal_bytes"]),
            "note": _wal_checkpoint_reader_note(),
        }
    start = time.monotonic()
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        passive_row = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        truncate_row = None
        truncate_mode = None
        if bool(truncate) or normalized_mode == "TRUNCATE":
            truncate_mode = "TRUNCATE"
            truncate_row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        elif normalized_mode != "PASSIVE":
            truncate_mode = normalized_mode
            truncate_row = conn.execute(f"PRAGMA wal_checkpoint({normalized_mode})").fetchone()
    finally:
        conn.close()
    elapsed_ms = (time.monotonic() - start) * 1000.0
    sizes_after = _sqlite_file_sizes(path)
    return {
        "status": "ok",
        "db_path": str(path),
        "mode": normalized_mode,
        "passive_checkpoint": _checkpoint_row(passive_row),
        "truncate_requested": bool(truncate) or normalized_mode == "TRUNCATE",
        "truncate_checkpoint": _checkpoint_row(truncate_row) if truncate_row is not None else None,
        "truncate_mode": truncate_mode,
        "busy": int(passive_row[0] or 0) if passive_row is not None else None,
        "log_frames": int(passive_row[1] or 0) if passive_row is not None else None,
        "checkpointed_frames": int(passive_row[2] or 0) if passive_row is not None else None,
        "elapsed_ms": round(elapsed_ms, 3),
        "db_size_before_bytes": db_size_before,
        "db_size_after_bytes": sizes_after["db_bytes"],
        "wal_size_before_bytes": sizes_before["wal_bytes"],
        "wal_size_after_bytes": sizes_after["wal_bytes"],
        "shm_size_before_bytes": sizes_before["shm_bytes"],
        "shm_size_after_bytes": sizes_after["shm_bytes"],
        "sizes_before": sizes_before,
        "sizes_after": sizes_after,
        "wal_size_warnings": _wal_size_warnings(sizes_before["wal_bytes"]),
        "note": _wal_checkpoint_reader_note(),
    }


def _checkpoint_row(row: Any) -> dict[str, int | None] | None:
    if row is None:
        return None
    return {
        "busy": int(row[0] or 0),
        "log_frames": int(row[1] or 0),
        "checkpointed_frames": int(row[2] or 0),
    }


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except FileNotFoundError:
        return 0


def _sqlite_file_sizes(path: Path) -> dict[str, int]:
    return {
        "db_bytes": _file_size(path),
        "wal_bytes": _file_size(Path(f"{path}-wal")),
        "shm_bytes": _file_size(Path(f"{path}-shm")),
    }


def _wal_size_warnings(wal_size_bytes: int) -> list[str]:
    gib = 1024 ** 3
    warnings: list[str] = []
    for threshold_gb in (10, 25, 50):
        if wal_size_bytes >= threshold_gb * gib:
            warnings.append(f"recorder.db-wal exceeds {threshold_gb}GB")
    return warnings


def _wal_checkpoint_reader_note() -> str:
    return (
        "Long-lived SQLite readers can prevent WAL truncation; this command never "
        "deletes .db-wal manually."
    )


MARKET_SNAPSHOT_INSERT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "market_id",
    "yes_price",
    "no_price",
    "best_bid_yes",
    "best_ask_yes",
    "best_bid_no",
    "best_ask_no",
    "spread_yes",
    "spread_no",
    "mid_price_yes",
    "mid_price_no",
    "volume",
    "liquidity",
    "last_trade_price",
    "last_trade_size",
    "last_trade_time",
    "has_orderbook",
    "has_trade_data",
    "snapshot_quality_status",
    "is_partial_orderbook",
    "missing_level_count",
    "time_gap_from_prev_snapshot_sec",
    "is_gap_affected",
    "feature_ready",
    "book_checksum",
    "strict_validation_passed",
    "recorder_version",
    "schema_version",
)

EXPECTED_TABLE_COLUMNS: dict[str, set[str]] = {
    "markets": {
        "run_id",
        "market_id",
        "event_id",
        "question",
        "description",
        "category",
        "outcomes",
        "resolution_source",
        "start_time",
        "end_time",
        "close_time",
        "platform_status",
        "phase",
        "status",
        "market_phase",
        "tracking_state",
        "yes_token_id",
        "no_token_id",
        "condition_id",
        "resolved",
        "resolved_at",
        "winning_asset_id",
        "winning_outcome",
        "strict_validation_passed",
        "strict_rejection_reason",
        "recorder_version",
        "schema_version",
        "created_at",
        "last_updated",
    },
    "market_snapshots": {
        "run_id",
        "timestamp",
        "market_id",
        "yes_price",
        "no_price",
        "best_bid_yes",
        "best_ask_yes",
        "best_bid_no",
        "best_ask_no",
        "spread_yes",
        "spread_no",
        "mid_price_yes",
        "mid_price_no",
        "volume",
        "liquidity",
        "last_trade_price",
        "last_trade_size",
        "last_trade_time",
        "has_orderbook",
        "has_trade_data",
        "snapshot_quality_status",
        "is_partial_orderbook",
        "missing_level_count",
        "time_gap_from_prev_snapshot_sec",
        "is_gap_affected",
        "feature_ready",
        "book_checksum",
        "strict_validation_passed",
        "recorder_version",
        "schema_version",
    },
    "order_book_levels": {
        "run_id",
        "timestamp",
        "market_id",
        "outcome_side",
        "book_side",
        "level",
        "price",
        "size",
        "recorder_version",
        "schema_version",
    },
    "trades": {
        "run_id",
        "timestamp",
        "market_id",
        "trade_id",
        "price",
        "size",
        "side",
        "maker",
        "taker",
        "recorder_version",
        "schema_version",
    },
    "features": {
        "run_id",
        "timestamp",
        "market_id",
        "price_change_1s",
        "price_change_10s",
        "price_change_60s",
        "velocity_5s",
        "velocity_30s",
        "acceleration_5s",
        "rolling_mean_60s",
        "distance_from_rolling_mean_60s",
        "total_bid_liquidity_yes",
        "total_ask_liquidity_yes",
        "liquidity_change_bid_5s",
        "orderbook_imbalance_yes",
        "buy_volume_5s",
        "sell_volume_5s",
        "net_trade_flow_5s",
        "trade_flow_ratio_5s",
        "rolling_volatility_10s",
        "rolling_volatility_60s",
        "volume_delta_1s",
        "avg_volume_60s",
        "volume_spike_ratio",
        "largest_bid_wall_size_yes",
        "largest_ask_wall_size_yes",
        "distance_to_bid_wall",
        "time_since_market_created",
        "time_until_resolution",
        "is_price_jump",
        "feature_ready",
        "is_gap_affected",
        "snapshot_quality_status",
        "strict_validation_passed",
        "recorder_version",
        "schema_version",
    },
    "market_events": {
        "run_id",
        "timestamp",
        "market_id",
        "event_type",
        "details",
        "recorder_version",
        "schema_version",
    },
    "raw_polymarket_events": {
        "id",
        "run_id",
        "local_arrival_ns",
        "local_arrival_iso",
        "exchange_timestamp",
        "event_type",
        "market_id",
        "condition_id",
        "asset_id",
        "slug",
        "parse_status",
        "parse_error",
        "raw_json",
        "recorder_version",
        "schema_version",
        "inserted_at",
    },
    "tick_size_changes": {
        "id",
        "run_id",
        "timestamp",
        "market_id",
        "condition_id",
        "asset_id",
        "old_tick_size",
        "new_tick_size",
        "raw_json",
        "recorder_version",
        "schema_version",
        "inserted_at",
    },
    "best_bid_ask_updates": {
        "id",
        "run_id",
        "timestamp",
        "market_id",
        "condition_id",
        "asset_id",
        "best_bid",
        "best_ask",
        "spread",
        "raw_json",
        "recorder_version",
        "schema_version",
        "inserted_at",
    },
    "btc_prices": {
        "id",
        "run_id",
        "source",
        "price",
        "exchange_timestamp",
        "local_arrival_ns",
        "local_arrival_iso",
        "raw_json",
        "recorder_version",
        "schema_version",
        "inserted_at",
    },
    "recorder_metrics": {
        "run_id",
        "timestamp",
        "markets_polled",
        "successful_market_fetches",
        "failed_markets",
        "api_latency_ms",
        "db_write_time_ms",
        "cycle_duration_ms",
        "rows_inserted",
        "duplicate_rows_skipped",
        "active_markets_snapshot_attempted",
        "snapshots_written",
        "snapshot_markets_skipped",
        "discovery_candidates_seen",
        "discovery_candidates_matched",
        "discovery_strict_5m_candidates",
        "discovery_broad_btc_candidates",
        "discovery_fallback_used",
        "tracked_markets_count",
        "api_call_count",
        "api_success_count",
        "api_failure_count",
        "fetched_trades",
        "accepted_trades",
        "rejected_before_start",
        "rejected_after_close",
        "skipped_already_seen",
        "last_accepted_trade_ts",
        "ws_reconnect_count",
        "raw_ws_events_seen",
        "raw_ws_events_written",
        "malformed_ws_events",
        "raw_ws_write_failures",
        "last_ws_event_age_sec",
        "subscribed_asset_count",
        "subscribed_asset_ids_json",
        "recorder_version",
        "schema_version",
    },
}

EXPECTED_TABLE_NAMES: set[str] = set(EXPECTED_TABLE_COLUMNS.keys())
_SCHEMA_TABLE_NAMES = {"sqlite_master", "sqlite_schema"}
_SCHEMA_OBJECT_TYPES = {"table", "index", "view", "trigger"}
_FALSY_SQLITE_PRAGMA_VALUES = {"0", "off", "false", "no"}
_MALFORMED_DB_ERROR_MARKERS = (
    "malformed database schema",
    "database disk image is malformed",
    "file is not a database",
)


def _sqlite_sidecar_paths(db_path: str) -> list[Path]:
    return [Path(f"{db_path}-wal"), Path(f"{db_path}-shm")]


def _sqlite_read_only_uri(db: Path) -> str:
    # Use URI mode so integrity checks do not mutate the file being inspected.
    encoded = quote(str(db.resolve()), safe="/:\\")
    return f"file:{encoded}?mode=ro"


def _normalize_schema_sql(sql: str) -> str:
    return " ".join(sql.strip().split()).upper()


def _has_valid_schema_prefix(schema_type: str, sql: str) -> bool:
    normalized = _normalize_schema_sql(sql)
    if schema_type == "table":
        return normalized.startswith("CREATE TABLE ")
    if schema_type == "index":
        return normalized.startswith("CREATE INDEX ") or normalized.startswith(
            "CREATE UNIQUE INDEX "
        )
    if schema_type == "view":
        return normalized.startswith("CREATE VIEW ")
    if schema_type == "trigger":
        return normalized.startswith("CREATE TRIGGER ")
    return False


def _validate_sqlite_master_rows(conn: sqlite3.Connection) -> tuple[list[str], int]:
    """
    Validate sqlite_master rows for semantic sanity, not just file-level integrity.
    """
    rows = conn.execute(
        """
        SELECT type, name, tbl_name, rootpage, sql
        FROM sqlite_master
        ORDER BY rowid
        """
    ).fetchall()
    issues: list[str] = []
    replay_sql: list[tuple[str, str, str]] = []
    for row in rows:
        schema_type = str(row[0] or "").strip().lower()
        name = str(row[1] or "").strip()
        table_name = str(row[2] or "").strip()
        rootpage = int(row[3] or 0)
        sql_text = row[4]
        is_internal = name.startswith("sqlite_") or table_name.startswith("sqlite_")

        if schema_type not in _SCHEMA_OBJECT_TYPES:
            issues.append(
                f"unsupported sqlite_master type={schema_type!r} for name={name!r}"
            )
            continue
        if not name:
            issues.append("sqlite_master row has empty name")
        if not table_name:
            issues.append(f"sqlite_master row has empty tbl_name for object {name!r}")

        if is_internal:
            if schema_type == "index" and table_name and table_name not in EXPECTED_TABLE_NAMES:
                issues.append(
                    f"internal index {name!r} references unexpected table {table_name!r}"
                )
            continue

        if schema_type == "table":
            if name not in EXPECTED_TABLE_NAMES:
                issues.append(f"unexpected table in sqlite_master: {name!r}")
            if rootpage <= 0:
                issues.append(f"table {name!r} has invalid rootpage={rootpage}")
        elif schema_type == "index":
            if table_name not in EXPECTED_TABLE_NAMES:
                issues.append(
                    f"index {name!r} references unexpected table {table_name!r}"
                )
            if rootpage <= 0:
                issues.append(f"index {name!r} has invalid rootpage={rootpage}")
        else:
            # Recorder schema does not define views/triggers.
            issues.append(f"unexpected {schema_type} object in sqlite_master: {name!r}")

        if sql_text is None:
            issues.append(f"{schema_type} {name!r} has NULL sql in sqlite_master")
            continue
        sql = str(sql_text).strip()
        if not sql:
            issues.append(f"{schema_type} {name!r} has empty sql in sqlite_master")
            continue
        if not _has_valid_schema_prefix(schema_type, sql):
            issues.append(
                f"invalid sql for {schema_type} {name!r}: {sql[:80]!r}"
            )
            continue
        replay_sql.append((schema_type, name, sql))

    # Run a syntax/semantic replay in-memory so malformed SQL cannot hide behind
    # writable_schema behavior on a different connection.
    replay_conn = sqlite3.connect(":memory:")
    try:
        for schema_type, name, sql in replay_sql:
            try:
                replay_conn.execute(sql)
            except sqlite3.DatabaseError as exc:
                issues.append(
                    "sqlite_master sql replay failed for "
                    f"{schema_type} {name!r}: {exc}"
                )
    finally:
        replay_conn.close()

    return issues, len(rows)


def _build_insert_sql(table: str, columns: Sequence[str], conflict_mode: str = "IGNORE") -> str:
    placeholders = ", ".join("?" for _ in columns)
    column_clause = ", ".join(columns)
    mode = conflict_mode.strip().upper()
    return (
        f"INSERT OR {mode} INTO {table} ({column_clause}) "
        f"VALUES ({placeholders})"
    )


def serialize_market_snapshot_row(
    record: MarketSnapshotRecord,
    fallback_run_id: str,
) -> tuple[object, ...]:
    return (
        record.run_id or fallback_run_id,
        to_iso(record.timestamp),
        record.market_id,
        record.yes_price,
        record.no_price,
        record.best_bid_yes,
        record.best_ask_yes,
        record.best_bid_no,
        record.best_ask_no,
        record.spread_yes,
        record.spread_no,
        record.mid_price_yes,
        record.mid_price_no,
        record.volume,
        record.liquidity,
        record.last_trade_price,
        record.last_trade_size,
        to_iso(record.last_trade_time),
        record.has_orderbook,
        record.has_trade_data,
        record.snapshot_quality_status,
        record.is_partial_orderbook,
        record.missing_level_count,
        record.time_gap_from_prev_snapshot_sec,
        record.is_gap_affected,
        record.feature_ready,
        record.book_checksum,
        record.strict_validation_passed,
        record.recorder_version,
        record.schema_version,
    )


class SnapshotInsertShapeError(ValueError):
    def __init__(
        self,
        *,
        expected_column_count: int,
        actual_row_length: int,
        columns: Sequence[str],
        row_index: int,
        row_preview: str,
    ) -> None:
        self.expected_column_count = expected_column_count
        self.actual_row_length = actual_row_length
        self.columns = list(columns)
        self.row_index = row_index
        self.row_preview = row_preview
        super().__init__(
            "market_snapshots row shape mismatch: "
            f"expected {expected_column_count} values for columns {list(columns)}, "
            f"got {actual_row_length} at row_index={row_index}; "
            f"row_preview={row_preview}"
        )


def run_sqlite_integrity_checks(
    db_path: str,
    mode: str = "full",
) -> dict[str, Any]:
    """
    Run SQLite integrity checks against an existing file.

    The default remains full so explicit validation/export safety paths keep running
    `PRAGMA integrity_check`. Recorder startup can pass mode="quick" to avoid a
    slow full scan on large live databases.
    Returns a structured report and never raises.
    """
    normalized_mode = normalize_startup_integrity_check_mode(mode)
    db = Path(db_path)
    report: dict[str, Any] = {
        "db_path": str(db),
        "db_exists": db.exists(),
        "sidecars_present": [str(path) for path in _sqlite_sidecar_paths(db_path) if path.exists()],
        "integrity_check_mode": normalized_mode,
        "integrity_check_skipped": normalized_mode == "off",
        "quick_check": None,
        "integrity_check": None,
        "quick_check_rows": [],
        "integrity_check_rows": [],
        "writable_schema": None,
        "sqlite_master_rows_checked": 0,
        "sqlite_master_issues": [],
        "error": None,
        "ok": True,
    }
    if normalized_mode == "off":
        return report
    if not db.exists():
        if report["sidecars_present"]:
            report["ok"] = False
            report["error"] = "orphan_sqlite_sidecars_without_base_db"
        return report

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(_sqlite_read_only_uri(db), uri=True)
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA writable_schema=OFF")
        writable_schema = int(conn.execute("PRAGMA writable_schema").fetchone()[0] or 0)
        report["writable_schema"] = writable_schema
        if writable_schema != 0:
            report["ok"] = False
            report["error"] = "writable_schema_enabled_during_integrity_check"
            return report
        # Avoid trusting schema-attached SQL functions while inspecting unknown files.
        try:
            conn.execute("PRAGMA trusted_schema=OFF")
        except sqlite3.DatabaseError:
            pass
        quick_rows = conn.execute("PRAGMA quick_check").fetchall()
        integrity_rows = (
            conn.execute("PRAGMA integrity_check").fetchall()
            if normalized_mode == "full"
            else []
        )
        quick_values = [str(row[0]) for row in quick_rows if row and row[0] is not None]
        integrity_values = [
            str(row[0]) for row in integrity_rows if row and row[0] is not None
        ]
        schema_issues, schema_rows_checked = _validate_sqlite_master_rows(conn)
        report["quick_check_rows"] = quick_values[:25]
        report["integrity_check_rows"] = integrity_values[:25]
        report["sqlite_master_rows_checked"] = schema_rows_checked
        report["sqlite_master_issues"] = schema_issues[:50]
        report["quick_check"] = quick_values[0] if quick_values else None
        report["integrity_check"] = integrity_values[0] if integrity_values else None
        quick_ok = bool(quick_values) and quick_values[0].lower() == "ok"
        integrity_ok = (
            normalized_mode != "full"
            or (bool(integrity_values) and integrity_values[0].lower() == "ok")
        )
        report["ok"] = quick_ok and integrity_ok and not schema_issues
        if schema_issues:
            report["error"] = "sqlite_master_validation_failed"
    except Exception as exc:
        report["ok"] = False
        report["error"] = repr(exc)
    finally:
        if conn is not None:
            conn.close()
    return report


def quarantine_corrupted_db(db_path: str) -> str:
    base = Path(db_path)
    suffix = base.suffix or ".db"
    quarantined = base.with_name(
        f"{base.stem}.corrupt.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}{suffix}"
    )
    moved_any = False
    if base.exists():
        shutil.move(str(base), str(quarantined))
        moved_any = True

    for sidecar_suffix in ("-wal", "-shm"):
        sidecar = Path(f"{db_path}{sidecar_suffix}")
        if not sidecar.exists():
            continue
        sidecar_target = quarantined.with_name(quarantined.name + sidecar_suffix)
        shutil.move(str(sidecar), str(sidecar_target))
        moved_any = True
    return str(quarantined if moved_any else base)


def ensure_startup_db_integrity(
    db_path: str,
    logger: logging.Logger | None = None,
    mode: str = "quick",
) -> dict[str, Any]:
    normalized_mode = normalize_startup_integrity_check_mode(mode)
    log = logger or logging.getLogger(__name__)
    if normalized_mode == "off":
        report = run_sqlite_integrity_checks(db_path, mode=normalized_mode)
        log.warning(
            "startup_integrity_check_skipped",
            extra={
                "db_path": db_path,
                "startup_integrity_check_mode": normalized_mode,
            },
        )
        return report

    log.info(
        "startup_integrity_check_starting",
        extra={
            "db_path": db_path,
            "startup_integrity_check_mode": normalized_mode,
        },
    )
    started = time.perf_counter()
    report = run_sqlite_integrity_checks(db_path, mode=normalized_mode)
    elapsed_sec = time.perf_counter() - started
    report["elapsed_sec"] = elapsed_sec
    log.info(
        "startup_integrity_check_finished",
        extra={
            "db_path": db_path,
            "startup_integrity_check_mode": normalized_mode,
            "startup_integrity_elapsed_sec": elapsed_sec,
            "startup_integrity_ok": bool(report.get("ok", False)),
        },
    )
    if report.get("ok", False):
        return report

    log.critical("db_corruption_detected", extra=report)

    quarantined_path = quarantine_corrupted_db(db_path)
    log.critical(
        "db_corruption_quarantined",
        extra={
            "original_db_path": db_path,
            "quarantined_db_path": quarantined_path,
        },
    )
    report["quarantined_db_path"] = quarantined_path
    return report


class SQLiteStore:
    """SQLite persistence layer for recorder runtime data."""

    def __init__(
        self,
        db_path: str,
        busy_timeout_ms: int = 5000,
        run_id: str = "legacy",
        quarantine_on_startup: bool = True,
        startup_integrity_check_mode: str = "quick",
        startup_integrity_logger: logging.Logger | None = None,
    ) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(path)
        self._startup_integrity_check_mode = normalize_startup_integrity_check_mode(
            startup_integrity_check_mode
        )
        if quarantine_on_startup:
            self.startup_integrity_report = ensure_startup_db_integrity(
                self.db_path,
                logger=startup_integrity_logger,
                mode=self._startup_integrity_check_mode,
            )
        else:
            self.startup_integrity_report = run_sqlite_integrity_checks(
                self.db_path,
                mode=self._startup_integrity_check_mode,
            )
            if (
                self._startup_integrity_check_mode != "off"
                and path.exists()
                and not self.startup_integrity_report.get("ok", False)
            ):
                raise RuntimeError(
                    "SQLite integrity check failed; schema upgrade aborted without "
                    f"quarantine or repair: {self.startup_integrity_report.get('error')}"
                )
        self.run_id = str(run_id)
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._authorizer_callback = None
        self._allow_schema_writes = False
        self._closed = False

        try:
            self.conn = self._open_connection(path)
        except sqlite3.DatabaseError as exc:
            if not self._is_malformed_db_error(exc):
                raise
            self._recover_from_malformed_db(stage="connect", exc=exc)

    def close(self) -> None:
        if self._closed:
            return

        try:
            # Commit any remaining transaction before checkpointing WAL.
            if self.conn.in_transaction:
                self.conn.commit()
            try:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.DatabaseError:
                # Best-effort fallback if the DB cannot be fully checkpointed.
                self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except sqlite3.DatabaseError:
            try:
                self.conn.rollback()
            except sqlite3.DatabaseError:
                pass
        finally:
            self.conn.close()
            self._closed = True

    def wal_checkpoint(self, mode: str = "PASSIVE") -> dict[str, Any]:
        normalized_mode = normalize_sqlite_wal_checkpoint_mode(mode)
        start = time.monotonic()
        row = self.conn.execute(f"PRAGMA wal_checkpoint({normalized_mode})").fetchone()
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return {
            "status": "ok",
            "db_path": self.db_path,
            "mode": normalized_mode,
            "busy": int(row[0] or 0) if row is not None else None,
            "log_frames": int(row[1] or 0) if row is not None else None,
            "checkpointed_frames": int(row[2] or 0) if row is not None else None,
            "elapsed_ms": round(elapsed_ms, 3),
        }

    def _open_connection(self, path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn = conn
        self._apply_runtime_pragmas()
        self._install_schema_write_guard()
        return conn

    def _apply_runtime_pragmas(self) -> None:
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA writable_schema=OFF")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA wal_autocheckpoint=1000")
        self.conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        writable_schema = int(self.conn.execute("PRAGMA writable_schema").fetchone()[0] or 0)
        if writable_schema != 0:
            raise RuntimeError("PRAGMA writable_schema must remain OFF")
        try:
            self.conn.execute("PRAGMA trusted_schema=OFF")
        except sqlite3.DatabaseError:
            pass

    def init_schema(
        self,
        *,
        backfill_existing_rows: bool = True,
        dedupe_lifecycle_events: bool = True,
        recover_malformed: bool = True,
    ) -> None:
        try:
            self._init_schema_once(
                backfill_existing_rows=backfill_existing_rows,
                dedupe_lifecycle_events=dedupe_lifecycle_events,
            )
            return
        except sqlite3.DatabaseError as exc:
            if not recover_malformed or not self._is_malformed_db_error(exc):
                raise
            self._recover_from_malformed_db(stage="init_schema", exc=exc)

        self._init_schema_once(
            backfill_existing_rows=backfill_existing_rows,
            dedupe_lifecycle_events=dedupe_lifecycle_events,
        )

    def _init_schema_once(
        self,
        *,
        backfill_existing_rows: bool = True,
        dedupe_lifecycle_events: bool = True,
    ) -> None:
        primary = Path(__file__).resolve().parent / "schema.sql"
        fallback = Path(__file__).resolve().parents[1] / "schema.sql"
        if primary.exists():
            schema_path = primary
        elif fallback.exists():
            schema_path = fallback
        else:
            raise FileNotFoundError(
                f"schema.sql not found; checked: {primary} and {fallback}"
            )

        schema_sql = schema_path.read_text(encoding="utf-8")
        with self._schema_write_window():
            with self.conn:
                try:
                    self.conn.executescript(schema_sql)
                except sqlite3.OperationalError as exc:
                    # Older DBs may fail on new index definitions before columns are added.
                    if "no such column" not in str(exc).lower():
                        raise
                self._apply_schema_upgrades(backfill_existing_rows=backfill_existing_rows)
                self._ensure_indexes(dedupe_lifecycle_events=dedupe_lifecycle_events)

        # Re-apply runtime pragmas because schema scripts can change them.
        self._apply_runtime_pragmas()
        self._assert_run_isolation_compatible()
        self._assert_expected_tables_and_columns()
        self._assert_sqlite_integrity_on_connection()

    @staticmethod
    def _is_malformed_db_error(exc: BaseException) -> bool:
        text = str(exc).lower()
        return any(marker in text for marker in _MALFORMED_DB_ERROR_MARKERS)

    def _recover_from_malformed_db(self, *, stage: str, exc: BaseException) -> None:
        logging.getLogger(__name__).critical(
            "db_corruption_detected_during_runtime",
            extra={
                "db_path": self.db_path,
                "stage": stage,
                "error": repr(exc),
            },
        )
        try:
            self.conn.close()
        except Exception:
            pass

        quarantined_path = quarantine_corrupted_db(self.db_path)
        logging.getLogger(__name__).critical(
            "db_corruption_quarantined_during_runtime",
            extra={
                "original_db_path": self.db_path,
                "quarantined_db_path": quarantined_path,
                "stage": stage,
            },
        )
        self.startup_integrity_report = {
            "ok": False,
            "error": repr(exc),
            "quarantined_db_path": quarantined_path,
            "db_path": self.db_path,
            "stage": stage,
        }
        self._authorizer_callback = None
        self._allow_schema_writes = False
        self._closed = False
        self.conn = self._open_connection(Path(self.db_path))

    @contextmanager
    def _schema_write_window(self) -> Iterable[None]:
        previous = self._allow_schema_writes
        self._allow_schema_writes = True
        try:
            yield
        finally:
            self._allow_schema_writes = previous
            try:
                self.conn.execute("PRAGMA writable_schema=OFF")
            except sqlite3.DatabaseError:
                pass

    def _assert_sqlite_integrity_on_connection(self) -> None:
        if self._startup_integrity_check_mode == "off":
            return
        quick_rows = self.conn.execute("PRAGMA quick_check").fetchall()
        quick = [str(row[0]).lower() for row in quick_rows if row and row[0] is not None]
        if not quick or quick[0] != "ok":
            raise RuntimeError(
                "SQLite quick_check failed after schema init: "
                f"{quick_rows[:5]}"
            )
        if self._startup_integrity_check_mode != "full":
            return

        integrity_rows = self.conn.execute("PRAGMA integrity_check").fetchall()
        integrity = [
            str(row[0]).lower() for row in integrity_rows if row and row[0] is not None
        ]
        if not integrity or integrity[0] != "ok":
            raise RuntimeError(
                "SQLite integrity_check failed after schema init: "
                f"{integrity_rows[:5]}"
            )

    def _assert_expected_tables_and_columns(self) -> None:
        rows = self.conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
        table_names = {str(row[0]) for row in rows}
        missing_tables = EXPECTED_TABLE_NAMES - table_names
        if missing_tables:
            raise RuntimeError(
                "SQLite schema missing required tables after init: "
                f"{sorted(missing_tables)}"
            )
        unexpected_tables = table_names - EXPECTED_TABLE_NAMES
        if unexpected_tables:
            raise RuntimeError(
                "SQLite schema contains unexpected tables after init: "
                f"{sorted(unexpected_tables)}"
            )

        schema_issues, _rows_checked = _validate_sqlite_master_rows(self.conn)
        if schema_issues:
            raise RuntimeError(
                "SQLite schema contains invalid sqlite_master rows: "
                f"{schema_issues[:10]}"
            )

        writable_schema = int(
            self.conn.execute("PRAGMA writable_schema").fetchone()[0] or 0
        )
        if writable_schema != 0:
            raise RuntimeError("PRAGMA writable_schema must be OFF")

        for table, expected_columns in EXPECTED_TABLE_COLUMNS.items():
            actual_columns = {
                str(row[1])
                for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            missing_columns = expected_columns - actual_columns
            if missing_columns:
                raise RuntimeError(
                    f"SQLite schema missing required columns for {table}: "
                    f"{sorted(missing_columns)}"
                )

    def _install_schema_write_guard(self) -> None:
        # Explicitly force OFF and reject any later attempt to toggle ON.
        self.conn.execute("PRAGMA writable_schema=OFF")

        if self._authorizer_callback is None:
            def _pragma_enables_writable_schema(value: str) -> bool:
                cleaned = value.strip().lower()
                if not cleaned:
                    return False
                if cleaned in _FALSY_SQLITE_PRAGMA_VALUES:
                    return False
                return True

            def _authorizer(
                action_code: int,
                param1: str | None,
                param2: str | None,
                _db_name: str | None,
                _trigger_name: str | None,
            ) -> int:
                target = (param1 or "").strip().lower()
                if action_code in {
                    sqlite3.SQLITE_INSERT,
                    sqlite3.SQLITE_UPDATE,
                    sqlite3.SQLITE_DELETE,
                } and target in _SCHEMA_TABLE_NAMES and not self._allow_schema_writes:
                    return sqlite3.SQLITE_DENY

                if action_code == sqlite3.SQLITE_PRAGMA:
                    pragma_name = (param1 or "").strip().lower()
                    pragma_value = (param2 or "").strip().lower()
                    if pragma_name == "writable_schema" and _pragma_enables_writable_schema(
                        pragma_value
                    ):
                        return sqlite3.SQLITE_DENY

                if action_code in {
                    sqlite3.SQLITE_ATTACH,
                    sqlite3.SQLITE_DETACH,
                }:
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK

            self._authorizer_callback = _authorizer
        self.conn.set_authorizer(self._authorizer_callback)

    def _apply_schema_upgrades(self, *, backfill_existing_rows: bool = True) -> None:
        """Idempotent upgrades for existing SQLite files created by older versions."""
        self._ensure_column("markets", "run_id", "TEXT")
        self._ensure_column("markets", "platform_status", "TEXT")
        self._ensure_column("markets", "phase", "TEXT")
        self._ensure_column("markets", "market_phase", "TEXT")
        self._ensure_column("markets", "tracking_state", "TEXT")
        self._ensure_column("markets", "yes_token_id", "TEXT")
        self._ensure_column("markets", "no_token_id", "TEXT")
        self._ensure_column("markets", "condition_id", "TEXT")
        self._ensure_column("markets", "resolved", "INTEGER DEFAULT 0")
        self._ensure_column("markets", "resolved_at", "TEXT")
        self._ensure_column("markets", "winning_asset_id", "TEXT")
        self._ensure_column("markets", "winning_outcome", "TEXT")
        self._ensure_column("markets", "strict_validation_passed", "INTEGER")
        self._ensure_column("markets", "strict_rejection_reason", "TEXT")
        self._ensure_column("markets", "recorder_version", "TEXT")
        self._ensure_column("markets", "schema_version", "TEXT")

        self._ensure_column("market_snapshots", "run_id", "TEXT")
        self._ensure_column("market_snapshots", "has_orderbook", "INTEGER")
        self._ensure_column("market_snapshots", "has_trade_data", "INTEGER")
        self._ensure_column("market_snapshots", "snapshot_quality_status", "TEXT")
        self._ensure_column("market_snapshots", "is_partial_orderbook", "INTEGER")
        self._ensure_column("market_snapshots", "missing_level_count", "INTEGER")
        self._ensure_column("market_snapshots", "time_gap_from_prev_snapshot_sec", "REAL")
        self._ensure_column("market_snapshots", "is_gap_affected", "INTEGER")
        self._ensure_column("market_snapshots", "feature_ready", "INTEGER")
        self._ensure_column("market_snapshots", "book_checksum", "TEXT")
        self._ensure_column("market_snapshots", "strict_validation_passed", "INTEGER")
        self._ensure_column("market_snapshots", "recorder_version", "TEXT")
        self._ensure_column("market_snapshots", "schema_version", "TEXT")

        self._ensure_column("order_book_levels", "run_id", "TEXT")
        self._ensure_column("order_book_levels", "recorder_version", "TEXT")
        self._ensure_column("order_book_levels", "schema_version", "TEXT")

        self._ensure_column("trades", "run_id", "TEXT")
        self._ensure_column("trades", "recorder_version", "TEXT")
        self._ensure_column("trades", "schema_version", "TEXT")

        self._ensure_column("features", "run_id", "TEXT")
        self._ensure_column("features", "feature_ready", "INTEGER")
        self._ensure_column("features", "is_gap_affected", "INTEGER")
        self._ensure_column("features", "snapshot_quality_status", "TEXT")
        self._ensure_column("features", "strict_validation_passed", "INTEGER")
        self._ensure_column("features", "recorder_version", "TEXT")
        self._ensure_column("features", "schema_version", "TEXT")

        self._ensure_column("market_events", "run_id", "TEXT")
        self._ensure_column("market_events", "recorder_version", "TEXT")
        self._ensure_column("market_events", "schema_version", "TEXT")

        self._ensure_column("raw_polymarket_events", "run_id", "TEXT")
        self._ensure_column("raw_polymarket_events", "local_arrival_ns", "INTEGER")
        self._ensure_column("raw_polymarket_events", "local_arrival_iso", "TEXT")
        self._ensure_column("raw_polymarket_events", "exchange_timestamp", "TEXT")
        self._ensure_column("raw_polymarket_events", "event_type", "TEXT")
        self._ensure_column("raw_polymarket_events", "market_id", "TEXT")
        self._ensure_column("raw_polymarket_events", "condition_id", "TEXT")
        self._ensure_column("raw_polymarket_events", "asset_id", "TEXT")
        self._ensure_column("raw_polymarket_events", "slug", "TEXT")
        self._ensure_column("raw_polymarket_events", "parse_status", "TEXT")
        self._ensure_column("raw_polymarket_events", "parse_error", "TEXT")
        self._ensure_column("raw_polymarket_events", "raw_json", "TEXT")
        self._ensure_column("raw_polymarket_events", "recorder_version", "TEXT")
        self._ensure_column("raw_polymarket_events", "schema_version", "TEXT")
        self._ensure_column("raw_polymarket_events", "inserted_at", "TEXT")

        self._ensure_column("tick_size_changes", "run_id", "TEXT")
        self._ensure_column("tick_size_changes", "timestamp", "TEXT")
        self._ensure_column("tick_size_changes", "market_id", "TEXT")
        self._ensure_column("tick_size_changes", "condition_id", "TEXT")
        self._ensure_column("tick_size_changes", "asset_id", "TEXT")
        self._ensure_column("tick_size_changes", "old_tick_size", "REAL")
        self._ensure_column("tick_size_changes", "new_tick_size", "REAL")
        self._ensure_column("tick_size_changes", "raw_json", "TEXT")
        self._ensure_column("tick_size_changes", "recorder_version", "TEXT")
        self._ensure_column("tick_size_changes", "schema_version", "TEXT")
        self._ensure_column("tick_size_changes", "inserted_at", "TEXT")

        self._ensure_column("best_bid_ask_updates", "run_id", "TEXT")
        self._ensure_column("best_bid_ask_updates", "timestamp", "TEXT")
        self._ensure_column("best_bid_ask_updates", "market_id", "TEXT")
        self._ensure_column("best_bid_ask_updates", "condition_id", "TEXT")
        self._ensure_column("best_bid_ask_updates", "asset_id", "TEXT")
        self._ensure_column("best_bid_ask_updates", "best_bid", "REAL")
        self._ensure_column("best_bid_ask_updates", "best_ask", "REAL")
        self._ensure_column("best_bid_ask_updates", "spread", "REAL")
        self._ensure_column("best_bid_ask_updates", "raw_json", "TEXT")
        self._ensure_column("best_bid_ask_updates", "recorder_version", "TEXT")
        self._ensure_column("best_bid_ask_updates", "schema_version", "TEXT")
        self._ensure_column("best_bid_ask_updates", "inserted_at", "TEXT")

        self._ensure_column("btc_prices", "run_id", "TEXT")
        self._ensure_column("btc_prices", "source", "TEXT")
        self._ensure_column("btc_prices", "price", "REAL")
        self._ensure_column("btc_prices", "exchange_timestamp", "TEXT")
        self._ensure_column("btc_prices", "local_arrival_ns", "INTEGER")
        self._ensure_column("btc_prices", "local_arrival_iso", "TEXT")
        self._ensure_column("btc_prices", "raw_json", "TEXT")
        self._ensure_column("btc_prices", "recorder_version", "TEXT")
        self._ensure_column("btc_prices", "schema_version", "TEXT")
        self._ensure_column("btc_prices", "inserted_at", "TEXT")

        self._ensure_column("recorder_metrics", "run_id", "TEXT")
        self._ensure_column(
            "recorder_metrics", "active_markets_snapshot_attempted", "INTEGER"
        )
        self._ensure_column("recorder_metrics", "snapshots_written", "INTEGER")
        self._ensure_column("recorder_metrics", "snapshot_markets_skipped", "INTEGER")
        self._ensure_column("recorder_metrics", "discovery_candidates_seen", "INTEGER")
        self._ensure_column("recorder_metrics", "discovery_candidates_matched", "INTEGER")
        self._ensure_column("recorder_metrics", "discovery_strict_5m_candidates", "INTEGER")
        self._ensure_column("recorder_metrics", "discovery_broad_btc_candidates", "INTEGER")
        self._ensure_column("recorder_metrics", "discovery_fallback_used", "INTEGER")
        self._ensure_column("recorder_metrics", "tracked_markets_count", "INTEGER")
        self._ensure_column("recorder_metrics", "api_call_count", "INTEGER")
        self._ensure_column("recorder_metrics", "api_success_count", "INTEGER")
        self._ensure_column("recorder_metrics", "api_failure_count", "INTEGER")
        self._ensure_column("recorder_metrics", "fetched_trades", "INTEGER")
        self._ensure_column("recorder_metrics", "accepted_trades", "INTEGER")
        self._ensure_column("recorder_metrics", "rejected_before_start", "INTEGER")
        self._ensure_column("recorder_metrics", "rejected_after_close", "INTEGER")
        self._ensure_column("recorder_metrics", "skipped_already_seen", "INTEGER")
        self._ensure_column("recorder_metrics", "last_accepted_trade_ts", "TEXT")
        self._ensure_column("recorder_metrics", "ws_reconnect_count", "INTEGER")
        self._ensure_column("recorder_metrics", "raw_ws_events_seen", "INTEGER")
        self._ensure_column("recorder_metrics", "raw_ws_events_written", "INTEGER")
        self._ensure_column("recorder_metrics", "malformed_ws_events", "INTEGER")
        self._ensure_column("recorder_metrics", "raw_ws_write_failures", "INTEGER")
        self._ensure_column("recorder_metrics", "last_ws_event_age_sec", "REAL")
        self._ensure_column("recorder_metrics", "subscribed_asset_count", "INTEGER")
        self._ensure_column("recorder_metrics", "subscribed_asset_ids_json", "TEXT")
        self._ensure_column("recorder_metrics", "recorder_version", "TEXT")
        self._ensure_column("recorder_metrics", "schema_version", "TEXT")
        if backfill_existing_rows:
            self._backfill_market_state_columns()

    def _backfill_market_state_columns(self) -> None:
        self.conn.execute(
            """
            UPDATE markets
            SET run_id = COALESCE(NULLIF(run_id, ''), 'legacy')
            """
        )
        self.conn.execute(
            """
            UPDATE markets
            SET resolved = COALESCE(resolved, 0)
            """
        )
        self.conn.execute(
            """
            UPDATE market_snapshots
            SET run_id = COALESCE(NULLIF(run_id, ''), 'legacy')
            """
        )
        self.conn.execute(
            """
            UPDATE order_book_levels
            SET run_id = COALESCE(NULLIF(run_id, ''), 'legacy')
            """
        )
        self.conn.execute(
            """
            UPDATE trades
            SET run_id = COALESCE(NULLIF(run_id, ''), 'legacy')
            """
        )
        self.conn.execute(
            """
            UPDATE features
            SET run_id = COALESCE(NULLIF(run_id, ''), 'legacy')
            """
        )
        self.conn.execute(
            """
            UPDATE market_events
            SET run_id = COALESCE(NULLIF(run_id, ''), 'legacy')
            """
        )
        self.conn.execute(
            """
            UPDATE raw_polymarket_events
            SET run_id = COALESCE(NULLIF(run_id, ''), 'legacy')
            """
        )
        self.conn.execute(
            """
            UPDATE btc_prices
            SET run_id = COALESCE(NULLIF(run_id, ''), 'legacy')
            """
        )
        self.conn.execute(
            """
            UPDATE recorder_metrics
            SET run_id = COALESCE(NULLIF(run_id, ''), 'legacy')
            """
        )
        self.conn.execute(
            """
            UPDATE markets
            SET
                strict_validation_passed = COALESCE(strict_validation_passed, 0),
                recorder_version = COALESCE(recorder_version, ?),
                schema_version = COALESCE(schema_version, ?)
            """,
            (RECORDER_VERSION, SCHEMA_VERSION),
        )
        self.conn.execute(
            """
            UPDATE market_snapshots
            SET
                is_partial_orderbook = COALESCE(is_partial_orderbook, 0),
                missing_level_count = COALESCE(missing_level_count, 0),
                is_gap_affected = COALESCE(is_gap_affected, 0),
                feature_ready = COALESCE(feature_ready, 0),
                strict_validation_passed = COALESCE(strict_validation_passed, 0),
                recorder_version = COALESCE(recorder_version, ?),
                schema_version = COALESCE(schema_version, ?)
            """,
            (RECORDER_VERSION, SCHEMA_VERSION),
        )
        self.conn.execute(
            """
            UPDATE order_book_levels
            SET
                recorder_version = COALESCE(recorder_version, ?),
                schema_version = COALESCE(schema_version, ?)
            """,
            (RECORDER_VERSION, SCHEMA_VERSION),
        )
        self.conn.execute(
            """
            UPDATE trades
            SET
                recorder_version = COALESCE(recorder_version, ?),
                schema_version = COALESCE(schema_version, ?)
            """,
            (RECORDER_VERSION, SCHEMA_VERSION),
        )
        self.conn.execute(
            """
            UPDATE features
            SET
                feature_ready = COALESCE(feature_ready, 0),
                is_gap_affected = COALESCE(is_gap_affected, 0),
                strict_validation_passed = COALESCE(strict_validation_passed, 0),
                recorder_version = COALESCE(recorder_version, ?),
                schema_version = COALESCE(schema_version, ?)
            """,
            (RECORDER_VERSION, SCHEMA_VERSION),
        )
        self.conn.execute(
            """
            UPDATE market_events
            SET
                recorder_version = COALESCE(recorder_version, ?),
                schema_version = COALESCE(schema_version, ?)
            """,
            (RECORDER_VERSION, SCHEMA_VERSION),
        )
        self.conn.execute(
            """
            UPDATE raw_polymarket_events
            SET
                parse_status = COALESCE(NULLIF(parse_status, ''), 'ok'),
                recorder_version = COALESCE(recorder_version, ?),
                schema_version = COALESCE(schema_version, ?),
                inserted_at = COALESCE(inserted_at, datetime('now'))
            """,
            (RECORDER_VERSION, SCHEMA_VERSION),
        )
        self.conn.execute(
            """
            UPDATE btc_prices
            SET
                recorder_version = COALESCE(recorder_version, ?),
                schema_version = COALESCE(schema_version, ?),
                inserted_at = COALESCE(inserted_at, datetime('now'))
            """,
            (RECORDER_VERSION, SCHEMA_VERSION),
        )
        self.conn.execute(
            """
            UPDATE recorder_metrics
            SET
                fetched_trades = COALESCE(fetched_trades, 0),
                accepted_trades = COALESCE(accepted_trades, 0),
                rejected_before_start = COALESCE(rejected_before_start, 0),
                rejected_after_close = COALESCE(rejected_after_close, 0),
                skipped_already_seen = COALESCE(skipped_already_seen, 0),
                recorder_version = COALESCE(recorder_version, ?),
                schema_version = COALESCE(schema_version, ?)
            """,
            (RECORDER_VERSION, SCHEMA_VERSION),
        )
        self.conn.execute(
            """
            UPDATE markets
            SET platform_status = status
            WHERE platform_status IS NULL AND status IS NOT NULL
            """
        )
        self.conn.execute(
            """
            UPDATE markets
            SET phase = market_phase
            WHERE phase IS NULL AND market_phase IS NOT NULL
            """
        )
        self.conn.execute(
            """
            UPDATE markets
            SET status = platform_status
            WHERE status IS NULL AND platform_status IS NOT NULL
            """
        )
        self.conn.execute(
            """
            UPDATE markets
            SET market_phase = phase
            WHERE market_phase IS NULL AND phase IS NOT NULL
            """
        )
        self.conn.execute(
            """
            UPDATE markets
            SET tracking_state = CASE
                WHEN tracking_state IN ('upcoming_standby') THEN 'discovered'
                WHEN tracking_state IN ('resolved', 'expired_waiting_resolution', 'inactive_untracked')
                    THEN 'inactive'
                ELSE tracking_state
            END
            WHERE tracking_state IS NOT NULL
            """
        )
        self.conn.execute(
            """
            UPDATE markets
            SET phase = CASE
                WHEN lower(COALESCE(platform_status, status, '')) IN ('closed', 'resolved', 'finalized', 'settled')
                    THEN 'resolved'
                WHEN start_time IS NOT NULL AND datetime(start_time) > datetime('now')
                    THEN 'future'
                WHEN close_time IS NOT NULL AND datetime(close_time) <= datetime('now')
                    THEN 'expired_waiting_resolution'
                ELSE 'active'
            END
            WHERE phase IS NULL
            """
        )
        self.conn.execute(
            """
            UPDATE markets
            SET market_phase = phase
            WHERE market_phase IS NULL AND phase IS NOT NULL
            """
        )
        self.conn.execute(
            """
            UPDATE markets
            SET tracking_state = 'inactive'
            WHERE tracking_state IS NULL
              AND phase IN ('resolved', 'expired_waiting_resolution')
            """
        )
        self.conn.execute(
            """
            UPDATE markets
            SET tracking_state = 'discovered'
            WHERE tracking_state IS NULL
            """
        )

    def _ensure_indexes(self, *, dedupe_lifecycle_events: bool = True) -> None:
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_snapshots_market_time "
            "ON market_snapshots (market_id, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_snapshots_time "
            "ON market_snapshots (timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_order_book_levels_market_time "
            "ON order_book_levels (market_id, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_order_book_levels_time "
            "ON order_book_levels (timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_market_time "
            "ON trades (market_id, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_time ON trades (timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_features_market_time "
            "ON features (market_id, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_events_market_time "
            "ON market_events (market_id, timestamp)"
        )
        if dedupe_lifecycle_events:
            self._dedupe_lifecycle_events()
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_market_events_lifecycle_once "
            "ON market_events (run_id, market_id, event_type) "
            "WHERE event_type IN ('market_opened', 'market_closed', 'market_resolved')"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_markets_platform_status "
            "ON markets (platform_status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_markets_status ON markets (status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_markets_phase_v2 ON markets (phase)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_markets_phase ON markets (market_phase)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_markets_tracking_state "
            "ON markets (tracking_state)"
        )

        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_markets_yes_token_id ON markets (yes_token_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_markets_no_token_id ON markets (no_token_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_markets_condition_id ON markets (condition_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_markets_run_resolved "
            "ON markets (run_id, resolved)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_markets_run_winning_asset_id "
            "ON markets (run_id, winning_asset_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_snapshots_run_market_time "
            "ON market_snapshots (run_id, market_id, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_snapshots_run_time "
            "ON market_snapshots (run_id, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_order_book_levels_run_market_time "
            "ON order_book_levels (run_id, market_id, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_run_market_time "
            "ON trades (run_id, market_id, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_features_run_market_time "
            "ON features (run_id, market_id, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_events_run_market_time "
            "ON market_events (run_id, market_id, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_polymarket_events_run_arrival "
            "ON raw_polymarket_events (run_id, local_arrival_ns)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_polymarket_events_run_event_type "
            "ON raw_polymarket_events (run_id, event_type)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_polymarket_events_run_market "
            "ON raw_polymarket_events (run_id, market_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_polymarket_events_run_condition "
            "ON raw_polymarket_events (run_id, condition_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_polymarket_events_run_asset "
            "ON raw_polymarket_events (run_id, asset_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tick_size_changes_run_market_time "
            "ON tick_size_changes (run_id, market_id, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tick_size_changes_run_asset_time "
            "ON tick_size_changes (run_id, asset_id, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_best_bid_ask_updates_run_market_time "
            "ON best_bid_ask_updates (run_id, market_id, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_best_bid_ask_updates_run_asset_time "
            "ON best_bid_ask_updates (run_id, asset_id, timestamp)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_btc_prices_run_arrival "
            "ON btc_prices (run_id, local_arrival_ns)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_btc_prices_source_arrival "
            "ON btc_prices (source, local_arrival_ns)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_markets_run_tracking_state "
            "ON markets (run_id, tracking_state)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_markets_run_strict_validation "
            "ON markets (run_id, strict_validation_passed)"
        )
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_markets_run_market "
            "ON markets (run_id, market_id)"
        )

    def _dedupe_lifecycle_events(self) -> None:
        self.conn.execute(
            """
            DELETE FROM market_events
            WHERE event_type IN ('market_opened', 'market_closed', 'market_resolved')
              AND rowid NOT IN (
                  SELECT MIN(rowid)
                  FROM market_events
                  WHERE event_type IN ('market_opened', 'market_closed', 'market_resolved')
                  GROUP BY run_id, market_id, event_type
              )
            """
        )

    def _ensure_column(self, table: str, column: str, type_def: str) -> None:
        if table not in EXPECTED_TABLE_NAMES:
            raise ValueError(f"Unexpected table in migration path: {table}")
        existing = {
            row[1]
            for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column in existing:
            return
        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {type_def}")

    def _resolved_run_id(self, run_id: str | None = None) -> str:
        return str(run_id or self.run_id or "legacy")

    @staticmethod
    def _row_preview(row: Sequence[object], max_items: int = 10) -> str:
        items = list(row[:max_items])
        preview = ", ".join(repr(item) for item in items)
        if len(row) > max_items:
            preview += ", ..."
        return f"[{preview}]"

    def _assert_insert_row_lengths(
        self,
        *,
        rows: Sequence[Tuple],
        columns: Sequence[str],
    ) -> None:
        expected = len(columns)
        for idx, row in enumerate(rows):
            actual = len(row)
            if actual != expected:
                raise SnapshotInsertShapeError(
                    expected_column_count=expected,
                    actual_row_length=actual,
                    columns=columns,
                    row_index=idx,
                    row_preview=self._row_preview(row),
                )

    def _table_pk_columns(self, table: str) -> list[str]:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        with_pos = sorted(
            ((int(row[5]), str(row[1])) for row in rows if int(row[5]) > 0),
            key=lambda item: item[0],
        )
        return [column for _pk_pos, column in with_pos]

    def _assert_run_isolation_compatible(self) -> None:
        if self.run_id in {"", "legacy"}:
            return

        expected_pks: dict[str, list[str]] = {
            "markets": ["run_id", "market_id"],
            "market_snapshots": ["run_id", "timestamp", "market_id"],
            "order_book_levels": [
                "run_id",
                "timestamp",
                "market_id",
                "outcome_side",
                "book_side",
                "level",
            ],
            "features": ["run_id", "timestamp", "market_id"],
            "recorder_metrics": ["run_id", "timestamp"],
        }
        incompatible: list[str] = []
        for table, expected in expected_pks.items():
            actual = self._table_pk_columns(table)
            if actual != expected:
                incompatible.append(f"{table}: expected PK {expected}, found {actual}")

        if incompatible:
            details = "; ".join(incompatible)
            raise RuntimeError(
                "Database uses legacy primary keys and cannot support run-isolated writes. "
                "Start a fresh DB for this run with `--fresh-run` (optionally `--archive-old-runs`) "
                f"or point `RECORDER_DB_PATH` to a new file. Details: {details}"
            )

    def upsert_markets(self, markets: Sequence[MarketMetadata], run_id: str | None = None) -> int:
        if not markets:
            return 0

        resolved_run_id = self._resolved_run_id(run_id)
        sql = """
        INSERT INTO markets (
            run_id,
            market_id, event_id, question, description, category, outcomes,
            resolution_source, start_time, end_time, close_time,
            platform_status, phase, status, market_phase, tracking_state,
            yes_token_id, no_token_id, condition_id,
            strict_validation_passed, strict_rejection_reason,
            recorder_version, schema_version,
            created_at, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, market_id) DO UPDATE SET
            event_id=excluded.event_id,
            question=excluded.question,
            description=excluded.description,
            category=excluded.category,
            outcomes=excluded.outcomes,
            resolution_source=excluded.resolution_source,
            start_time=excluded.start_time,
            end_time=excluded.end_time,
            close_time=excluded.close_time,
            platform_status=excluded.platform_status,
            phase=excluded.phase,
            status=excluded.status,
            market_phase=excluded.market_phase,
            tracking_state=excluded.tracking_state,
            yes_token_id=excluded.yes_token_id,
            no_token_id=excluded.no_token_id,
            condition_id=excluded.condition_id,
            strict_validation_passed=excluded.strict_validation_passed,
            strict_rejection_reason=excluded.strict_rejection_reason,
            recorder_version=excluded.recorder_version,
            schema_version=excluded.schema_version,
            last_updated=excluded.last_updated
        """
        rows = [
            (
                resolved_run_id,
                m.market_id,
                m.event_id,
                m.question,
                m.description,
                m.category,
                json.dumps(m.outcomes),
                m.resolution_source,
                to_iso(m.start_time),
                to_iso(m.end_time),
                to_iso(m.close_time),
                m.platform_status,
                m.phase,
                m.status,
                m.market_phase,
                m.tracking_state,
                m.yes_token_id,
                m.no_token_id,
                m.condition_id,
                m.strict_validation_passed,
                m.strict_rejection_reason,
                m.recorder_version,
                m.schema_version,
                to_iso(m.created_at),
                to_iso(m.last_updated),
            )
            for m in markets
        ]

        before = self.conn.total_changes
        with self.conn:
            self.conn.executemany(sql, rows)
        return self.conn.total_changes - before

    def insert_markets_ignore(
        self, markets: Sequence[MarketMetadata], run_id: str | None = None
    ) -> int:
        if not markets:
            return 0

        resolved_run_id = self._resolved_run_id(run_id)
        sql = """
        INSERT OR IGNORE INTO markets (
            run_id,
            market_id, event_id, question, description, category, outcomes,
            resolution_source, start_time, end_time, close_time,
            platform_status, phase, status, market_phase, tracking_state,
            yes_token_id, no_token_id, condition_id,
            strict_validation_passed, strict_rejection_reason,
            recorder_version, schema_version,
            created_at, last_updated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (
                resolved_run_id,
                m.market_id,
                m.event_id,
                m.question,
                m.description,
                m.category,
                json.dumps(m.outcomes),
                m.resolution_source,
                to_iso(m.start_time),
                to_iso(m.end_time),
                to_iso(m.close_time),
                m.platform_status,
                m.phase,
                m.status,
                m.market_phase,
                m.tracking_state,
                m.yes_token_id,
                m.no_token_id,
                m.condition_id,
                m.strict_validation_passed,
                m.strict_rejection_reason,
                m.recorder_version,
                m.schema_version,
                to_iso(m.created_at),
                to_iso(m.last_updated),
            )
            for m in markets
        ]

        before = self.conn.total_changes
        with self.conn:
            self.conn.executemany(sql, rows)
        return self.conn.total_changes - before

    def update_market_statuses(
        self,
        updates: Sequence[tuple[str, str]],
        run_id: str | None = None,
    ) -> int:
        if not updates:
            return 0
        resolved_run_id = self._resolved_run_id(run_id)
        sql = """
        UPDATE markets
        SET platform_status = ?,
            status = ?,
            phase = CASE
                WHEN lower(?) IN ('closed', 'resolved', 'finalized', 'settled') THEN 'resolved'
                ELSE phase
            END,
            market_phase = CASE
                WHEN lower(?) IN ('closed', 'resolved', 'finalized', 'settled') THEN 'resolved'
                ELSE market_phase
            END,
            tracking_state = CASE
                WHEN lower(?) IN ('closed', 'resolved', 'finalized', 'settled') THEN 'inactive'
                ELSE tracking_state
            END,
            last_updated = ?
        WHERE market_id = ? AND run_id = ?
        """
        now_iso = to_iso(utc_now())
        rows = [
            (status, status, status, status, status, now_iso, market_id, resolved_run_id)
            for market_id, status in updates
        ]
        before = self.conn.total_changes
        with self.conn:
            self.conn.executemany(sql, rows)
        return self.conn.total_changes - before

    def update_market_tracking_states(
        self,
        updates: Sequence[tuple[str, str]],
        run_id: str | None = None,
    ) -> int:
        if not updates:
            return 0
        resolved_run_id = self._resolved_run_id(run_id)
        sql = """
        UPDATE markets
        SET tracking_state = ?, last_updated = ?
        WHERE market_id = ? AND run_id = ?
        """
        now_iso = to_iso(utc_now())
        rows = [
            (tracking_state, now_iso, market_id, resolved_run_id)
            for market_id, tracking_state in updates
        ]
        before = self.conn.total_changes
        with self.conn:
            self.conn.executemany(sql, rows)
        return self.conn.total_changes - before

    def mark_non_target_open_markets_inactive(
        self, active_target_market_ids: Sequence[str], run_id: str | None = None
    ) -> int:
        """
        Mark stale open rows that are outside BTC Up/Down scope as inactive.

        This keeps historical rows and raw platform status, but removes them
        from recorder tracking.
        """
        now_iso = to_iso(utc_now())
        resolved_run_id = self._resolved_run_id(run_id)
        open_status_sql = (
            "(platform_status IS NULL OR lower(platform_status) IN ('open', 'active') "
            "OR status IS NULL OR lower(status) IN ('open', 'active'))"
        )
        non_target_sql = """
            (
                yes_token_id IS NULL OR
                no_token_id IS NULL OR
                question IS NULL OR
                lower(question) NOT LIKE '%up or down%' OR
                (
                    lower(question) NOT LIKE '%btc%' AND
                    lower(question) NOT LIKE '%bitcoin%'
                )
            )
        """

        params: list[object] = [now_iso, resolved_run_id]
        sql = f"""
        UPDATE markets
        SET tracking_state = 'inactive',
            last_updated = ?
        WHERE run_id = ?
          AND {open_status_sql}
          AND {non_target_sql}
        """

        if active_target_market_ids:
            placeholders = ", ".join("?" for _ in active_target_market_ids)
            sql += f" AND market_id NOT IN ({placeholders})"
            params.extend(active_target_market_ids)

        before = self.conn.total_changes
        with self.conn:
            self.conn.execute(sql, tuple(params))
        return self.conn.total_changes - before

    def demote_stale_primary_markets(
        self,
        current_primary_market_ids: Sequence[str],
        run_id: str | None = None,
    ) -> int:
        """
        Ensure only currently selected primary rows keep primary tracking states.

        Historical rows can remain in SQLite, but stale `selected_active`
        flags should be demoted after rotations/restarts.
        """
        now_iso = to_iso(utc_now())
        resolved_run_id = self._resolved_run_id(run_id)
        params: list[object] = [now_iso, resolved_run_id]
        sql = """
        UPDATE markets
        SET tracking_state = 'discovered',
            last_updated = ?
        WHERE run_id = ?
          AND tracking_state = 'selected_active'
        """
        if current_primary_market_ids:
            placeholders = ", ".join("?" for _ in current_primary_market_ids)
            sql += f" AND market_id NOT IN ({placeholders})"
            params.extend(current_primary_market_ids)

        before = self.conn.total_changes
        with self.conn:
            self.conn.execute(sql, tuple(params))
        return self.conn.total_changes - before

    def repair_stale_selected_active_markets(
        self,
        *,
        now: datetime | None = None,
        grace_sec: float = 60.0,
        dry_run: bool = True,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        now_dt = now or utc_now()
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        else:
            now_dt = now_dt.astimezone(timezone.utc)
        safe_grace_sec = max(0.0, float(grace_sec))
        cutoff = now_dt - timedelta(seconds=safe_grace_sec)
        now_iso = to_iso(now_dt) or ""
        cutoff_iso = to_iso(cutoff) or ""

        def _scope_clause(prefix: str = "WHERE") -> tuple[str, tuple[object, ...]]:
            if run_id is None:
                return "", ()
            return f" {prefix} run_id = ?", (str(run_id),)

        def _and_scope_clause() -> tuple[str, tuple[object, ...]]:
            if run_id is None:
                return "", ()
            return " AND run_id = ?", (str(run_id),)

        def _integrity_counts() -> dict[str, int]:
            scope_sql, scope_params = _and_scope_clause()
            selected_active = int(
                self.conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM markets
                    WHERE tracking_state = 'selected_active'
                    {scope_sql}
                    """,
                    scope_params,
                ).fetchone()[0]
                or 0
            )
            active_phase = int(
                self.conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM markets
                    WHERE COALESCE(phase, market_phase) = 'active'
                    {scope_sql}
                    """,
                    scope_params,
                ).fetchone()[0]
                or 0
            )
            invalid_active_past_close = int(
                self.conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM markets
                    WHERE COALESCE(phase, market_phase) = 'active'
                      AND close_time IS NOT NULL
                      AND datetime(close_time) <= datetime(?)
                    {scope_sql}
                    """,
                    (now_iso, *scope_params),
                ).fetchone()[0]
                or 0
            )
            return {
                "selected_active_total": selected_active,
                "active_phase_count": active_phase,
                "invalid_active_markets_past_close": invalid_active_past_close,
                "selected_active_integrity_alert": (
                    1 if active_phase > 0 and selected_active != 1 else 0
                ),
            }

        before_counts = _integrity_counts()
        scope_where, scope_params = _scope_clause(prefix="AND")
        rows = self.conn.execute(
            f"""
            SELECT
                run_id,
                market_id,
                close_time,
                phase,
                market_phase,
                platform_status,
                status
            FROM markets
            WHERE tracking_state = 'selected_active'
            {scope_where}
            ORDER BY run_id ASC, close_time ASC, market_id ASC
            """,
            scope_params,
        ).fetchall()

        candidates: list[dict[str, object]] = []
        for row in rows:
            close_ts = parse_timestamp(row[2])
            if close_ts is None or close_ts > cutoff:
                continue
            status = str(row[5] or row[6] or "").strip().lower()
            demoted_phase = (
                "resolved"
                if status in {"closed", "resolved", "finalized", "settled"}
                else "expired_waiting_resolution"
            )
            candidates.append(
                {
                    "run_id": str(row[0]),
                    "market_id": str(row[1]),
                    "close_time": row[2],
                    "previous_phase": row[3],
                    "previous_market_phase": row[4],
                    "demoted_phase": demoted_phase,
                }
            )

        demoted_count = 0
        events_inserted = 0
        if not dry_run and candidates:
            with self.conn:
                for candidate in candidates:
                    cursor = self.conn.execute(
                        """
                        UPDATE markets
                        SET tracking_state = 'inactive',
                            phase = ?,
                            market_phase = ?,
                            last_updated = ?
                        WHERE run_id = ?
                          AND market_id = ?
                          AND tracking_state = 'selected_active'
                        """,
                        (
                            candidate["demoted_phase"],
                            candidate["demoted_phase"],
                            now_iso,
                            candidate["run_id"],
                            candidate["market_id"],
                        ),
                    )
                    changed = int(cursor.rowcount or 0)
                    demoted_count += changed
                    if changed <= 0:
                        continue
                    details = json.dumps(
                        {
                            "reason": "stale_selected_active_past_close",
                            "previous_tracking_state": "selected_active",
                            "previous_phase": candidate["previous_phase"],
                            "previous_market_phase": candidate["previous_market_phase"],
                            "new_tracking_state": "inactive",
                            "new_phase": candidate["demoted_phase"],
                            "close_time": candidate["close_time"],
                            "cutoff_time": cutoff_iso,
                        },
                        sort_keys=True,
                    )
                    event_cursor = self.conn.execute(
                        """
                        INSERT OR IGNORE INTO market_events (
                            run_id, timestamp, market_id, event_type, details,
                            recorder_version, schema_version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            candidate["run_id"],
                            now_iso,
                            candidate["market_id"],
                            "stale_selected_active_demoted",
                            details,
                            RECORDER_VERSION,
                            SCHEMA_VERSION,
                        ),
                    )
                    events_inserted += int(event_cursor.rowcount or 0)

        after_counts = _integrity_counts()
        return {
            "dry_run": bool(dry_run),
            "run_id_scope": str(run_id) if run_id is not None else "all",
            "now": now_iso,
            "grace_sec": safe_grace_sec,
            "cutoff_time": cutoff_iso,
            "stale_selected_active_count": len(candidates),
            "demoted_count": demoted_count,
            "market_events_inserted": events_inserted,
            "candidate_market_ids": [candidate["market_id"] for candidate in candidates],
            "before": before_counts,
            "after": after_counts,
        }

    def normalize_market_resolutions(
        self,
        *,
        dry_run: bool = True,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        def _table_exists(table: str) -> bool:
            row = self.conn.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = ?
                LIMIT 1
                """,
                (table,),
            ).fetchone()
            return row is not None

        def _has_column(table: str, column: str) -> bool:
            if not _table_exists(table):
                return False
            rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            return any(str(row[1]) == column for row in rows)

        def _json_obj(value: object) -> dict[str, Any]:
            if not value:
                return {}
            try:
                parsed = json.loads(str(value))
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}

        def _as_str(value: object) -> str | None:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        def _list_str(value: object) -> list[str]:
            if value is None:
                return []
            if isinstance(value, str):
                try:
                    decoded = json.loads(value)
                except json.JSONDecodeError:
                    return [value]
                value = decoded
            if isinstance(value, list):
                return [str(item) for item in value if item is not None]
            return [str(value)]

        def _looks_like_condition_hash(value: str | None) -> bool:
            return bool(value and value.startswith("0x") and len(value) > 20)

        def _scope_sql(prefix: str = "AND") -> tuple[str, tuple[object, ...]]:
            if run_id is None:
                return "", ()
            return f" {prefix} run_id = ?", (str(run_id),)

        resolution_columns = (
            "resolved",
            "resolved_at",
            "winning_asset_id",
            "winning_outcome",
        )
        columns_present = {
            column: _has_column("markets", column) for column in resolution_columns
        }

        if _table_exists("market_events"):
            scope_sql, scope_params = _scope_sql("AND")
            event_rows = self.conn.execute(
                f"""
                SELECT
                    'market_events' AS source_table,
                    rowid AS source_id,
                    run_id,
                    timestamp,
                    market_id,
                    details
                FROM market_events
                WHERE event_type = 'market_resolved'
                {scope_sql}
                ORDER BY run_id ASC, timestamp ASC, rowid ASC
                """,
                scope_params,
            ).fetchall()
        else:
            event_rows = []

        if _table_exists("raw_polymarket_events"):
            raw_scope_sql, raw_scope_params = _scope_sql("AND")
            raw_rows = self.conn.execute(
                f"""
                SELECT
                    'raw_polymarket_events' AS source_table,
                    id AS source_id,
                    run_id,
                    COALESCE(exchange_timestamp, local_arrival_iso) AS timestamp,
                    COALESCE(market_id, condition_id) AS market_id,
                    raw_json AS details
                FROM raw_polymarket_events
                WHERE event_type = 'market_resolved'
                  AND COALESCE(parse_status, 'ok') = 'ok'
                {raw_scope_sql}
                ORDER BY run_id ASC, local_arrival_ns ASC, id ASC
                """,
                raw_scope_params,
            ).fetchall()
        else:
            raw_rows = []

        seen_event_keys: set[tuple[str, str, str, str, str]] = set()
        mappings: list[dict[str, object]] = []
        unmapped: list[dict[str, object]] = []
        ambiguous: list[dict[str, object]] = []

        for row in [*event_rows, *raw_rows]:
            source_table = str(row[0])
            source_id = int(row[1])
            event_run_id = str(row[2])
            event_timestamp = _as_str(row[3])
            event_market_id = _as_str(row[4])
            details = _json_obj(row[5])

            numeric_id = _as_str(
                details.get("id")
                or details.get("market_id")
                or details.get("marketId")
            )
            condition_id = _as_str(
                details.get("condition_id")
                or details.get("conditionId")
                or details.get("market")
            )
            winning_asset_id = _as_str(
                details.get("winning_asset_id")
                or details.get("winningAssetId")
                or details.get("winning_asset")
            )
            winning_outcome_raw = _as_str(
                details.get("winning_outcome")
                or details.get("winningOutcome")
                or details.get("outcome")
            )
            asset_ids = _list_str(details.get("assets_ids") or details.get("asset_ids"))
            if winning_asset_id is not None:
                asset_ids.append(winning_asset_id)

            event_key = (
                event_run_id,
                numeric_id or "",
                condition_id or event_market_id or "",
                winning_asset_id or "",
                event_timestamp or "",
            )
            if event_key in seen_event_keys:
                continue
            seen_event_keys.add(event_key)

            identifiers = {
                value
                for value in (numeric_id, condition_id, event_market_id)
                if value is not None
            }
            asset_set = {value for value in asset_ids if value}
            where_parts: list[str] = []
            params: list[object] = [event_run_id]
            if identifiers:
                placeholders = ", ".join("?" for _ in identifiers)
                where_parts.append(f"market_id IN ({placeholders})")
                params.extend(sorted(identifiers))
                where_parts.append(f"condition_id IN ({placeholders})")
                params.extend(sorted(identifiers))
            if asset_set:
                placeholders = ", ".join("?" for _ in asset_set)
                where_parts.append(f"yes_token_id IN ({placeholders})")
                params.extend(sorted(asset_set))
                where_parts.append(f"no_token_id IN ({placeholders})")
                params.extend(sorted(asset_set))
            if not where_parts:
                unmapped.append(
                    {
                        "source_table": source_table,
                        "source_id": source_id,
                        "reason": "no_identifiers",
                    }
                )
                continue

            candidates = self.conn.execute(
                f"""
                SELECT
                    run_id,
                    market_id,
                    condition_id,
                    yes_token_id,
                    no_token_id,
                    question
                FROM markets
                WHERE run_id = ?
                  AND ({" OR ".join(where_parts)})
                """,
                tuple(params),
            ).fetchall()

            scored: list[tuple[int, tuple[object, ...]]] = []
            for candidate in candidates:
                market_id = _as_str(candidate[1])
                candidate_condition_id = _as_str(candidate[2])
                yes_token_id = _as_str(candidate[3])
                no_token_id = _as_str(candidate[4])
                score = 0
                if numeric_id is not None and market_id == numeric_id:
                    score += 120
                if event_market_id is not None and market_id == event_market_id:
                    score += 20 if _looks_like_condition_hash(event_market_id) else 100
                if condition_id is not None and candidate_condition_id == condition_id:
                    score += 100
                if (
                    event_market_id is not None
                    and _looks_like_condition_hash(event_market_id)
                    and candidate_condition_id == event_market_id
                ):
                    score += 100
                if winning_asset_id is not None and winning_asset_id in {
                    yes_token_id,
                    no_token_id,
                }:
                    score += 80
                if asset_set and ({yes_token_id, no_token_id} & asset_set):
                    score += 40
                if yes_token_id or no_token_id:
                    score += 30
                if market_id is not None and not _looks_like_condition_hash(market_id):
                    score += 20
                if candidate[5]:
                    score += 5
                scored.append((score, candidate))

            scored.sort(key=lambda item: (-item[0], str(item[1][1])))
            if not scored:
                unmapped.append(
                    {
                        "source_table": source_table,
                        "source_id": source_id,
                        "reason": "no_matching_market",
                    }
                )
                continue
            if len(scored) > 1 and scored[0][0] == scored[1][0]:
                ambiguous.append(
                    {
                        "source_table": source_table,
                        "source_id": source_id,
                        "candidate_market_ids": [
                            str(item[1][1]) for item in scored if item[0] == scored[0][0]
                        ],
                    }
                )
                continue

            candidate = scored[0][1]
            target_market_id = str(candidate[1])
            yes_token_id = _as_str(candidate[3])
            no_token_id = _as_str(candidate[4])
            winning_outcome = winning_outcome_raw
            if winning_outcome is None and winning_asset_id is not None:
                if winning_asset_id == yes_token_id:
                    winning_outcome = "YES"
                elif winning_asset_id == no_token_id:
                    winning_outcome = "NO"

            resolved_at = to_iso(
                parse_timestamp(event_timestamp)
            ) or event_timestamp

            mappings.append(
                {
                    "source_table": source_table,
                    "source_id": source_id,
                    "run_id": event_run_id,
                    "event_market_id": event_market_id,
                    "target_market_id": target_market_id,
                    "condition_id": condition_id or candidate[2],
                    "resolved_at": resolved_at,
                    "winning_asset_id": winning_asset_id,
                    "winning_outcome": winning_outcome,
                    "score": scored[0][0],
                }
            )

        updated = 0
        if not dry_run and mappings:
            missing_columns = [
                column for column, present in columns_present.items() if not present
            ]
            if missing_columns:
                raise RuntimeError(
                    "Resolution columns are missing. Run schema initialization first: "
                    + ",".join(missing_columns)
                )
            now_iso = to_iso(utc_now()) or ""
            with self.conn:
                for mapping in mappings:
                    cursor = self.conn.execute(
                        """
                        UPDATE markets
                        SET resolved = 1,
                            resolved_at = COALESCE(?, resolved_at),
                            winning_asset_id = COALESCE(?, winning_asset_id),
                            winning_outcome = COALESCE(?, winning_outcome),
                            platform_status = 'resolved',
                            status = 'resolved',
                            phase = 'resolved',
                            market_phase = 'resolved',
                            tracking_state = 'inactive',
                            last_updated = ?
                        WHERE run_id = ?
                          AND market_id = ?
                          AND (
                                COALESCE(resolved, 0) <> 1
                             OR resolved_at IS NULL
                             OR (? IS NOT NULL AND winning_asset_id IS NULL)
                             OR (? IS NOT NULL AND winning_outcome IS NULL)
                             OR COALESCE(phase, market_phase, '') <> 'resolved'
                          )
                        """,
                        (
                            mapping["resolved_at"],
                            mapping["winning_asset_id"],
                            mapping["winning_outcome"],
                            now_iso,
                            mapping["run_id"],
                            mapping["target_market_id"],
                            mapping["winning_asset_id"],
                            mapping["winning_outcome"],
                        ),
                    )
                    updated += int(cursor.rowcount or 0)

        return {
            "dry_run": bool(dry_run),
            "run_id_scope": str(run_id) if run_id is not None else "all",
            "resolution_columns_present": columns_present,
            "resolution_events_seen": len(seen_event_keys),
            "mapped_resolution_events": len(mappings),
            "unmapped_resolution_events": len(unmapped),
            "ambiguous_resolution_events": len(ambiguous),
            "markets_to_update": len({str(row["target_market_id"]) for row in mappings}),
            "markets_updated": updated,
            "mapped_market_ids": sorted(
                {str(row["target_market_id"]) for row in mappings}
            ),
            "unmapped": unmapped[:20],
            "ambiguous": ambiguous[:20],
        }

    def refresh_market_phases(
        self, now: datetime | None = None, run_id: str | None = None
    ) -> int:
        """
        Recompute phase from time bounds + platform status and clean stale tracking state.

        This keeps app-level phase semantics consistent even when a market times out
        between discovery cycles.
        """
        now_iso = to_iso(now or utc_now())
        if now_iso is None:
            now_iso = to_iso(utc_now()) or ""
        resolved_run_id = self._resolved_run_id(run_id)

        before = self.conn.total_changes
        with self.conn:
            self.conn.execute(
                """
                UPDATE markets
                SET phase = CASE
                    WHEN lower(COALESCE(platform_status, status, '')) IN ('closed', 'resolved', 'finalized', 'settled')
                        THEN 'resolved'
                    WHEN start_time IS NOT NULL AND datetime(start_time) > datetime(?)
                        THEN 'future'
                    WHEN close_time IS NOT NULL AND datetime(close_time) <= datetime(?)
                        THEN 'expired_waiting_resolution'
                    ELSE 'active'
                END,
                market_phase = CASE
                    WHEN lower(COALESCE(platform_status, status, '')) IN ('closed', 'resolved', 'finalized', 'settled')
                        THEN 'resolved'
                    WHEN start_time IS NOT NULL AND datetime(start_time) > datetime(?)
                        THEN 'future'
                    WHEN close_time IS NOT NULL AND datetime(close_time) <= datetime(?)
                        THEN 'expired_waiting_resolution'
                    ELSE 'active'
                END
                WHERE run_id = ?
                """,
                (now_iso, now_iso, now_iso, now_iso, resolved_run_id),
            )
            self.conn.execute(
                """
                UPDATE markets
                SET tracking_state = 'inactive'
                WHERE tracking_state = 'selected_active'
                  AND COALESCE(phase, market_phase) <> 'active'
                  AND run_id = ?
                """
                ,
                (resolved_run_id,),
            )
            self.conn.execute(
                """
                UPDATE markets
                SET tracking_state = 'inactive'
                WHERE tracking_state <> 'selected_active'
                  AND COALESCE(phase, market_phase) IN ('resolved', 'expired_waiting_resolution')
                  AND run_id = ?
                """
                ,
                (resolved_run_id,),
            )
        return self.conn.total_changes - before

    def snapshot_trade_integrity_counts(self, run_id: str | None = None) -> dict[str, int]:
        resolved_run_id = self._resolved_run_id(run_id)
        row = self.conn.execute(
            """
            SELECT
                SUM(
                    CASE
                        WHEN s.last_trade_time IS NOT NULL
                         AND m.start_time IS NOT NULL
                         AND datetime(s.last_trade_time) < datetime(m.start_time)
                        THEN 1 ELSE 0
                    END
                ) AS before_start,
                SUM(
                    CASE
                        WHEN s.last_trade_time IS NOT NULL
                         AND datetime(s.last_trade_time) > datetime(s.timestamp)
                        THEN 1 ELSE 0
                    END
                ) AS after_snapshot
            FROM market_snapshots s
            LEFT JOIN markets m ON m.market_id = s.market_id AND m.run_id = s.run_id
            WHERE s.run_id = ?
            """
            ,
            (resolved_run_id,),
        ).fetchone()
        before_start = int((row[0] if row is not None else 0) or 0)
        after_snapshot = int((row[1] if row is not None else 0) or 0)
        return {
            "last_trade_before_market_start": before_start,
            "last_trade_after_snapshot_time": after_snapshot,
            "total_invalid": before_start + after_snapshot,
        }

    def snapshot_trade_integrity_by_market(
        self, limit: int = 200, run_id: str | None = None
    ) -> List[tuple[str, int, int]]:
        resolved_run_id = self._resolved_run_id(run_id)
        rows = self.conn.execute(
            """
            SELECT
                s.market_id,
                SUM(
                    CASE
                        WHEN s.last_trade_time IS NOT NULL
                         AND m.start_time IS NOT NULL
                         AND datetime(s.last_trade_time) < datetime(m.start_time)
                        THEN 1 ELSE 0
                    END
                ) AS before_start,
                SUM(
                    CASE
                        WHEN s.last_trade_time IS NOT NULL
                         AND datetime(s.last_trade_time) > datetime(s.timestamp)
                        THEN 1 ELSE 0
                    END
                ) AS after_snapshot
            FROM market_snapshots s
            LEFT JOIN markets m ON m.market_id = s.market_id AND m.run_id = s.run_id
            WHERE s.run_id = ?
            GROUP BY s.market_id
            HAVING before_start > 0 OR after_snapshot > 0
            ORDER BY (before_start + after_snapshot) DESC, s.market_id ASC
            LIMIT ?
            """,
            (resolved_run_id, int(limit)),
        ).fetchall()
        return [
            (str(row[0]), int(row[1] or 0), int(row[2] or 0))
            for row in rows
        ]

    def repair_invalid_snapshot_trade_fields(self, run_id: str | None = None) -> int:
        resolved_run_id = self._resolved_run_id(run_id)
        before = self.conn.total_changes
        with self.conn:
            self.conn.execute(
                """
                UPDATE market_snapshots
                SET
                    last_trade_price = NULL,
                    last_trade_size = NULL,
                    last_trade_time = NULL,
                    has_trade_data = 0
                WHERE last_trade_time IS NOT NULL
                  AND run_id = ?
                  AND (
                        datetime(last_trade_time) > datetime(timestamp)
                        OR EXISTS (
                            SELECT 1
                            FROM markets m
                            WHERE m.market_id = market_snapshots.market_id
                              AND m.run_id = market_snapshots.run_id
                              AND m.start_time IS NOT NULL
                              AND datetime(market_snapshots.last_trade_time) < datetime(m.start_time)
                        )
                  )
                """,
                (resolved_run_id,),
            )
        return self.conn.total_changes - before

    def has_market_event(
        self, market_id: str, event_type: str, run_id: str | None = None
    ) -> bool:
        resolved_run_id = self._resolved_run_id(run_id)
        row = self.conn.execute(
            "SELECT 1 FROM market_events WHERE run_id = ? AND market_id = ? AND event_type = ? LIMIT 1",
            (resolved_run_id, market_id, event_type),
        ).fetchone()
        return row is not None

    def insert_market_snapshots(
        self, records: Sequence[MarketSnapshotRecord]
    ) -> Tuple[int, int]:
        columns = MARKET_SNAPSHOT_INSERT_COLUMNS
        sql = _build_insert_sql(
            table="market_snapshots",
            columns=columns,
            conflict_mode="IGNORE",
        )
        rows = [
            serialize_market_snapshot_row(
                r,
                fallback_run_id=self.run_id,
            )
            for r in records
        ]
        self._assert_insert_row_lengths(rows=rows, columns=columns)
        return self._insert_many_count(sql, rows)

    def insert_order_book_levels(
        self, records: Sequence[OrderBookLevelRecord]
    ) -> Tuple[int, int]:
        sql = """
        INSERT OR IGNORE INTO order_book_levels (
            run_id, timestamp, market_id, outcome_side, book_side, level, price, size,
            recorder_version, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (
                r.run_id or self.run_id,
                to_iso(r.timestamp),
                r.market_id,
                r.outcome_side,
                r.book_side,
                r.level,
                r.price,
                r.size,
                r.recorder_version,
                r.schema_version,
            )
            for r in records
        ]
        return self._insert_many_count(sql, rows)

    def insert_trades(self, records: Sequence[TradeRecord]) -> Tuple[int, int]:
        sql = """
        INSERT OR IGNORE INTO trades (
            run_id, timestamp, market_id, trade_id, price, size, side, maker, taker,
            recorder_version, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (
                r.run_id or self.run_id,
                to_iso(r.timestamp),
                r.market_id,
                r.trade_id,
                r.price,
                r.size,
                r.side,
                r.maker,
                r.taker,
                r.recorder_version,
                r.schema_version,
            )
            for r in records
        ]
        return self._insert_many_count(sql, rows)

    def insert_features(self, records: Sequence[FeatureRecord]) -> Tuple[int, int]:
        sql = """
        INSERT OR IGNORE INTO features (
            run_id, timestamp, market_id,
            price_change_1s, price_change_10s, price_change_60s,
            velocity_5s, velocity_30s, acceleration_5s,
            rolling_mean_60s, distance_from_rolling_mean_60s,
            total_bid_liquidity_yes, total_ask_liquidity_yes,
            liquidity_change_bid_5s, orderbook_imbalance_yes,
            buy_volume_5s, sell_volume_5s, net_trade_flow_5s, trade_flow_ratio_5s,
            rolling_volatility_10s, rolling_volatility_60s,
            volume_delta_1s, avg_volume_60s, volume_spike_ratio,
            largest_bid_wall_size_yes, largest_ask_wall_size_yes, distance_to_bid_wall,
            time_since_market_created, time_until_resolution,
            is_price_jump,
            feature_ready, is_gap_affected, snapshot_quality_status,
            strict_validation_passed, recorder_version, schema_version
        ) VALUES (
            ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?, ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?, ?,
            ?, ?,
            ?,
            ?, ?, ?,
            ?, ?, ?
        )
        """

        rows = [
            (
                r.run_id or self.run_id,
                to_iso(r.timestamp),
                r.market_id,
                r.price_change_1s,
                r.price_change_10s,
                r.price_change_60s,
                r.velocity_5s,
                r.velocity_30s,
                r.acceleration_5s,
                r.rolling_mean_60s,
                r.distance_from_rolling_mean_60s,
                r.total_bid_liquidity_yes,
                r.total_ask_liquidity_yes,
                r.liquidity_change_bid_5s,
                r.orderbook_imbalance_yes,
                r.buy_volume_5s,
                r.sell_volume_5s,
                r.net_trade_flow_5s,
                r.trade_flow_ratio_5s,
                r.rolling_volatility_10s,
                r.rolling_volatility_60s,
                r.volume_delta_1s,
                r.avg_volume_60s,
                r.volume_spike_ratio,
                r.largest_bid_wall_size_yes,
                r.largest_ask_wall_size_yes,
                r.distance_to_bid_wall,
                r.time_since_market_created,
                r.time_until_resolution,
                r.is_price_jump,
                r.feature_ready,
                r.is_gap_affected,
                r.snapshot_quality_status,
                r.strict_validation_passed,
                r.recorder_version,
                r.schema_version,
            )
            for r in records
        ]
        return self._insert_many_count(sql, rows)

    def insert_market_events(self, records: Sequence[MarketEventRecord]) -> Tuple[int, int]:
        sql = """
        INSERT OR IGNORE INTO market_events (
            run_id, timestamp, market_id, event_type, details,
            recorder_version, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (
                r.run_id or self.run_id,
                to_iso(r.timestamp),
                r.market_id,
                r.event_type,
                r.details,
                r.recorder_version,
                r.schema_version,
            )
            for r in records
        ]
        return self._insert_many_count(sql, rows)

    def insert_raw_polymarket_events(
        self,
        records: Sequence[RawPolymarketEventRecord],
    ) -> Tuple[int, int]:
        sql = """
        INSERT INTO raw_polymarket_events (
            run_id, local_arrival_ns, local_arrival_iso, exchange_timestamp,
            event_type, market_id, condition_id, asset_id, slug,
            parse_status, parse_error, raw_json,
            recorder_version, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (
                r.run_id or self.run_id,
                int(r.local_arrival_ns),
                r.local_arrival_iso,
                to_iso(r.exchange_timestamp),
                r.event_type,
                r.market_id,
                r.condition_id,
                r.asset_id,
                r.slug,
                r.parse_status,
                r.parse_error,
                r.raw_json,
                r.recorder_version,
                r.schema_version,
            )
            for r in records
        ]
        return self._insert_many_count(sql, rows)

    def prune_raw_polymarket_events(
        self,
        *,
        retention_sec: float = 0.0,
        max_rows: int = 0,
        batch_size: int = 50000,
        now_ns: int | None = None,
    ) -> dict[str, Any]:
        safe_batch_size = max(1, int(batch_size or 50000))
        total_deleted = 0
        deleted_by_retention = 0
        deleted_by_max_rows = 0
        started = time.monotonic()

        if float(retention_sec or 0.0) > 0:
            threshold_ns = int(
                (now_ns if now_ns is not None else time.time_ns())
                - float(retention_sec) * 1_000_000_000
            )
            before = self.conn.total_changes
            with self.conn:
                self.conn.execute(
                    """
                    DELETE FROM raw_polymarket_events
                    WHERE id IN (
                        SELECT id
                        FROM raw_polymarket_events
                        WHERE local_arrival_ns < ?
                        ORDER BY local_arrival_ns ASC, id ASC
                        LIMIT ?
                    )
                    """,
                    (threshold_ns, safe_batch_size),
                )
            deleted_by_retention = self.conn.total_changes - before
            total_deleted += deleted_by_retention

        if int(max_rows or 0) > 0 and total_deleted < safe_batch_size:
            current_count = int(
                self.conn.execute("SELECT COUNT(*) FROM raw_polymarket_events").fetchone()[0]
                or 0
            )
            excess_rows = max(0, current_count - int(max_rows))
            max_row_delete_limit = min(safe_batch_size - total_deleted, excess_rows)
            if max_row_delete_limit > 0:
                before = self.conn.total_changes
                with self.conn:
                    self.conn.execute(
                        """
                        DELETE FROM raw_polymarket_events
                        WHERE id IN (
                            SELECT id
                            FROM raw_polymarket_events
                            ORDER BY local_arrival_ns ASC, id ASC
                            LIMIT ?
                        )
                        """,
                        (max_row_delete_limit,),
                    )
                deleted_by_max_rows = self.conn.total_changes - before
                total_deleted += deleted_by_max_rows

        elapsed_ms = (time.monotonic() - started) * 1000.0
        return {
            "deleted_rows": int(total_deleted),
            "deleted_by_retention": int(deleted_by_retention),
            "deleted_by_max_rows": int(deleted_by_max_rows),
            "retention_sec": float(retention_sec or 0.0),
            "max_rows": int(max_rows or 0),
            "batch_size": safe_batch_size,
            "elapsed_ms": round(elapsed_ms, 3),
        }

    def insert_tick_size_changes(
        self,
        records: Sequence[TickSizeChangeRecord],
    ) -> Tuple[int, int]:
        sql = """
        INSERT INTO tick_size_changes (
            run_id, timestamp, market_id, condition_id, asset_id,
            old_tick_size, new_tick_size, raw_json,
            recorder_version, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (
                r.run_id or self.run_id,
                to_iso(r.timestamp),
                r.market_id,
                r.condition_id,
                r.asset_id,
                r.old_tick_size,
                r.new_tick_size,
                r.raw_json,
                r.recorder_version,
                r.schema_version,
            )
            for r in records
        ]
        return self._insert_many_count(sql, rows)

    def insert_best_bid_ask_updates(
        self,
        records: Sequence[BestBidAskRecord],
    ) -> Tuple[int, int]:
        sql = """
        INSERT INTO best_bid_ask_updates (
            run_id, timestamp, market_id, condition_id, asset_id,
            best_bid, best_ask, spread, raw_json,
            recorder_version, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (
                r.run_id or self.run_id,
                to_iso(r.timestamp),
                r.market_id,
                r.condition_id,
                r.asset_id,
                r.best_bid,
                r.best_ask,
                r.spread,
                r.raw_json,
                r.recorder_version,
                r.schema_version,
            )
            for r in records
        ]
        return self._insert_many_count(sql, rows)

    def insert_btc_price_samples(
        self,
        records: Sequence[BTCPriceSampleRecord],
    ) -> Tuple[int, int]:
        sql = """
        INSERT INTO btc_prices (
            run_id, source, price, exchange_timestamp,
            local_arrival_ns, local_arrival_iso, raw_json,
            recorder_version, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (
                r.run_id or self.run_id,
                r.source,
                r.price,
                to_iso(r.exchange_timestamp),
                int(r.local_arrival_ns),
                r.local_arrival_iso,
                r.raw_json,
                r.recorder_version,
                r.schema_version,
            )
            for r in records
        ]
        return self._insert_many_count(sql, rows)

    def get_latest_btc_price_sample(
        self,
        *,
        source: str | None = None,
        run_id: str | None = None,
    ) -> BTCPriceSampleRecord | None:
        resolved_run_id = self._resolved_run_id(run_id)
        params: list[object] = [resolved_run_id]
        where = "run_id = ?"
        if source is not None:
            where += " AND source = ?"
            params.append(source)
        row = self.conn.execute(
            f"""
            SELECT
                run_id,
                source,
                price,
                exchange_timestamp,
                local_arrival_ns,
                local_arrival_iso,
                raw_json,
                recorder_version,
                schema_version
            FROM btc_prices
            WHERE {where}
            ORDER BY local_arrival_ns DESC, id DESC
            LIMIT 1
            """,
            tuple(params),
        ).fetchone()
        if row is None:
            return None
        return BTCPriceSampleRecord(
            run_id=str(row[0]),
            source=str(row[1]),
            price=float(row[2]),
            exchange_timestamp=parse_timestamp(row[3]),
            local_arrival_ns=int(row[4]),
            local_arrival_iso=str(row[5]),
            raw_json=str(row[6]),
            recorder_version=str(row[7] or RECORDER_VERSION),
            schema_version=str(row[8] or SCHEMA_VERSION),
        )

    def insert_recorder_metric(self, record: RecorderMetricRecord) -> None:
        sql = """
        INSERT OR REPLACE INTO recorder_metrics (
            run_id, timestamp, markets_polled, successful_market_fetches, failed_markets,
            api_latency_ms, db_write_time_ms, cycle_duration_ms,
            rows_inserted, duplicate_rows_skipped,
            active_markets_snapshot_attempted, snapshots_written, snapshot_markets_skipped,
            discovery_candidates_seen, discovery_candidates_matched,
            discovery_strict_5m_candidates, discovery_broad_btc_candidates, discovery_fallback_used,
            tracked_markets_count,
            api_call_count, api_success_count, api_failure_count,
            fetched_trades, accepted_trades, rejected_before_start, rejected_after_close,
            skipped_already_seen, last_accepted_trade_ts,
            ws_reconnect_count, raw_ws_events_seen, raw_ws_events_written,
            malformed_ws_events, raw_ws_write_failures, last_ws_event_age_sec,
            subscribed_asset_count, subscribed_asset_ids_json,
            recorder_version, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        row = (
            record.run_id or self.run_id,
            to_iso(record.timestamp),
            record.markets_polled,
            record.successful_market_fetches,
            record.failed_markets,
            record.api_latency_ms,
            record.db_write_time_ms,
            record.cycle_duration_ms,
            record.rows_inserted,
            record.duplicate_rows_skipped,
            record.active_markets_snapshot_attempted,
            record.snapshots_written,
            record.snapshot_markets_skipped,
            record.discovery_candidates_seen,
            record.discovery_candidates_matched,
            record.discovery_strict_5m_candidates,
            record.discovery_broad_btc_candidates,
            record.discovery_fallback_used,
            record.tracked_markets_count,
            record.api_call_count,
            record.api_success_count,
            record.api_failure_count,
            record.fetched_trades,
            record.accepted_trades,
            record.rejected_before_start,
            record.rejected_after_close,
            record.skipped_already_seen,
            to_iso(record.last_accepted_trade_ts),
            record.ws_reconnect_count,
            record.raw_ws_events_seen,
            record.raw_ws_events_written,
            record.malformed_ws_events,
            record.raw_ws_write_failures,
            record.last_ws_event_age_sec,
            record.subscribed_asset_count,
            record.subscribed_asset_ids_json,
            record.recorder_version,
            record.schema_version,
        )
        with self.conn:
            self.conn.execute(sql, row)

    def _insert_many_count(self, sql: str, rows: Sequence[Tuple]) -> Tuple[int, int]:
        if not rows:
            return (0, 0)
        start = time.monotonic()
        before = self.conn.total_changes
        with self.conn:
            self.conn.executemany(sql, rows)
        inserted = self.conn.total_changes - before
        skipped = max(0, len(rows) - inserted)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        logging.getLogger(__name__).debug(
            "sqlite_write_batch_latency_ms",
            extra={
                "sqlite_write_batch_latency_ms": round(elapsed_ms, 3),
                "sqlite_write_batch_rows": len(rows),
                "sqlite_write_batch_inserted": inserted,
                "sqlite_write_batch_skipped": skipped,
            },
        )
        return inserted, skipped
