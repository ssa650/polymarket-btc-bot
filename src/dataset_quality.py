from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

from .btc_price_feed import (
    POLYMARKET_RTDS_BINANCE_SOURCE,
    POLYMARKET_RTDS_CHAINLINK_SOURCE,
)
from .models import parse_timestamp, to_iso


KEY_TABLES: tuple[str, ...] = (
    "markets",
    "market_snapshots",
    "features",
    "trades",
    "btc_prices",
    "market_events",
    "order_book_levels",
    "best_bid_ask_updates",
    "raw_polymarket_events",
)

_RECENT_TIME_COLUMNS: dict[str, str] = {
    "markets": "COALESCE(last_updated, created_at, close_time, start_time)",
    "market_snapshots": "timestamp",
    "features": "timestamp",
    "trades": "timestamp",
    "btc_prices": "local_arrival_iso",
    "market_events": "timestamp",
    "order_book_levels": "timestamp",
    "best_bid_ask_updates": "timestamp",
    "raw_polymarket_events": "local_arrival_iso",
    "recorder_metrics": "timestamp",
}


def build_dataset_quality_report(
    db_path: str,
    *,
    recent_minutes: float | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = _utc(now or datetime.now(timezone.utc))
    cutoff = (
        now_dt - timedelta(minutes=max(0.0, float(recent_minutes)))
        if recent_minutes is not None
        else None
    )
    conn = _connect_read_only(db_path)
    try:
        row_counts = {
            table: _table_count(conn, table, cutoff=cutoff)
            for table in KEY_TABLES
        }
        raw_ws_events_written = _latest_raw_ws_events_written(conn, cutoff=cutoff)
        btc_source_health = _btc_source_health(conn, cutoff=cutoff, now=now_dt)
        feature_readiness = _feature_readiness(conn, cutoff=cutoff)
        feature_quality_counts = _feature_quality_counts(conn, cutoff=cutoff)
        snapshot_coverage = _snapshot_coverage(conn, cutoff=cutoff)
        per_market_coverage = _per_market_coverage(conn, cutoff=cutoff)
    finally:
        conn.close()

    warnings = _quality_warnings(
        btc_source_health=btc_source_health,
        raw_ws_events_written=raw_ws_events_written,
        feature_readiness=feature_readiness,
        snapshot_coverage=snapshot_coverage,
    )
    return {
        "mode": "recent_only" if cutoff is not None else "full",
        "recent_minutes": recent_minutes,
        "cutoff_iso": to_iso(cutoff) if cutoff is not None else None,
        "now": now_dt.isoformat(),
        "db_path": db_path,
        "file_sizes": _database_file_sizes(db_path),
        "row_counts": row_counts,
        "raw_ws_events_written": raw_ws_events_written,
        "btc_source_health": btc_source_health,
        "feature_readiness": feature_readiness,
        "feature_quality_counts": feature_quality_counts,
        "snapshot_coverage": snapshot_coverage,
        "per_market_coverage": per_market_coverage,
        "warnings": warnings,
    }


def render_dataset_quality_report(report: dict[str, Any]) -> str:
    lines: list[str] = ["=== Dataset Quality ==="]
    lines.append(f"mode={report.get('mode')}")
    lines.append(f"db_path={report.get('db_path')}")
    lines.append(f"now={report.get('now')}")
    if report.get("mode") == "recent_only":
        lines.append(f"recent_minutes={report.get('recent_minutes')}")
        lines.append(f"recent_cutoff={report.get('cutoff_iso')}")

    sizes = report.get("file_sizes", {})
    lines.extend(
        [
            "",
            "=== Database Files ===",
            f"db_file_size_bytes={sizes.get('db_file_size_bytes', 0)}",
            f"wal_file_size_bytes={sizes.get('wal_file_size_bytes', 0)}",
            f"shm_file_size_bytes={sizes.get('shm_file_size_bytes', 0)}",
        ]
    )

    lines.extend(["", "=== Row Counts ==="])
    for table in KEY_TABLES:
        value = report.get("row_counts", {}).get(table)
        lines.append(f"{table}={'missing' if value is None else value}")

    lines.extend(
        [
            "",
            "=== Recorder Metrics ===",
            f"raw_ws_events_written={report.get('raw_ws_events_written')}",
        ]
    )

    lines.extend(["", "=== BTC Source Health ==="])
    btc_rows = report.get("btc_source_health", [])
    if not btc_rows:
        lines.append("none")
    for row in btc_rows:
        lines.append(
            " | ".join(
                [
                    f"source={row.get('source')}",
                    f"canonical={str(bool(row.get('canonical'))).lower()}",
                    f"row_count={row.get('row_count')}",
                    f"first_local_arrival={row.get('first_local_arrival')}",
                    f"latest_local_arrival={row.get('latest_local_arrival')}",
                    f"sample_age_sec={row.get('sample_age_sec')}",
                    f"rows_per_sec={row.get('rows_per_sec')}",
                ]
            )
        )

    readiness = report.get("feature_readiness", {})
    lines.extend(
        [
            "",
            "=== Feature Readiness ===",
            f"total_feature_rows={readiness.get('total_feature_rows', 0)}",
            f"feature_ready_rows={readiness.get('feature_ready_rows', 0)}",
            f"feature_ready_pct={readiness.get('feature_ready_pct', 0.0)}",
            f"gap_affected_rows={readiness.get('gap_affected_rows', 0)}",
            f"gap_affected_pct={readiness.get('gap_affected_pct', 0.0)}",
            f"latest_feature_timestamp={readiness.get('latest_feature_timestamp')}",
        ]
    )

    lines.extend(["", "=== Feature Quality Status Counts ==="])
    quality_rows = report.get("feature_quality_counts", [])
    if not quality_rows:
        lines.append("none")
    for row in quality_rows:
        lines.append(
            f"snapshot_quality_status={row.get('snapshot_quality_status')} | "
            f"count={row.get('count')}"
        )

    snapshot = report.get("snapshot_coverage", {})
    lines.extend(
        [
            "",
            "=== Snapshot Coverage ===",
            f"total_snapshots={snapshot.get('total_snapshots', 0)}",
            f"snapshots_with_orderbook={snapshot.get('snapshots_with_orderbook', 0)}",
            f"snapshots_with_trade_data={snapshot.get('snapshots_with_trade_data', 0)}",
            f"latest_snapshot_timestamp={snapshot.get('latest_snapshot_timestamp')}",
        ]
    )

    lines.extend(["", "=== Per Market Coverage ==="])
    market_rows = report.get("per_market_coverage", [])
    if not market_rows:
        lines.append("none")
    for row in market_rows:
        lines.append(
            " | ".join(
                [
                    f"market_id={row.get('market_id')}",
                    f"question={row.get('question')}",
                    f"start_time={row.get('start_time')}",
                    f"close_time={row.get('close_time')}",
                    f"snapshot_count={row.get('snapshot_count')}",
                    f"feature_count={row.get('feature_count')}",
                    f"ready_feature_count={row.get('ready_feature_count')}",
                    f"ready_pct={row.get('ready_pct')}",
                    f"first_snapshot={row.get('first_snapshot')}",
                    f"latest_snapshot={row.get('latest_snapshot')}",
                ]
            )
        )

    lines.extend(["", "=== Warnings ==="])
    warnings = report.get("warnings", [])
    if not warnings:
        lines.append("none")
    else:
        lines.extend(f"warning={warning}" for warning in warnings)
    return "\n".join(lines)


def _connect_read_only(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(db_path)
    encoded = quote(str(path.resolve()), safe="/:\\")
    conn = sqlite3.connect(f"file:{encoded}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


def _database_file_sizes(db_path: str) -> dict[str, int]:
    path = Path(db_path)
    return {
        "db_file_size_bytes": _file_size(path),
        "wal_file_size_bytes": _file_size(Path(f"{db_path}-wal")),
        "shm_file_size_bytes": _file_size(Path(f"{db_path}-shm")),
    }


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (table,),
    ).fetchone()
    return row is not None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    if not _table_exists(conn, table):
        return False
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row[1]) == column for row in rows)


def _table_count(
    conn: sqlite3.Connection,
    table: str,
    *,
    cutoff: datetime | None,
) -> int | None:
    if not _table_exists(conn, table):
        return None
    where_sql, params = _recent_where(table, cutoff)
    row = conn.execute(
        f"SELECT COUNT(*) FROM {table}{where_sql}",
        params,
    ).fetchone()
    return int(row[0] or 0)


def _recent_where(
    table: str,
    cutoff: datetime | None,
    *,
    alias: str | None = None,
) -> tuple[str, tuple[Any, ...]]:
    if cutoff is None:
        return "", ()
    column = _RECENT_TIME_COLUMNS.get(table)
    if column is None:
        return "", ()
    qualified = f"{alias}.{column}" if alias and column.isidentifier() else column
    return f" WHERE datetime({qualified}) >= datetime(?)", (to_iso(cutoff),)


def _recent_and(
    table: str,
    cutoff: datetime | None,
    *,
    alias: str | None = None,
) -> tuple[str, tuple[Any, ...]]:
    where, params = _recent_where(table, cutoff, alias=alias)
    if not where:
        return "", ()
    return where.replace(" WHERE ", " AND ", 1), params


def _latest_raw_ws_events_written(
    conn: sqlite3.Connection,
    *,
    cutoff: datetime | None,
) -> int | None:
    if not _table_exists(conn, "recorder_metrics") or not _has_column(
        conn, "recorder_metrics", "raw_ws_events_written"
    ):
        return None
    where_sql, params = _recent_where("recorder_metrics", cutoff)
    row = conn.execute(
        f"""
        SELECT raw_ws_events_written
        FROM recorder_metrics
        {where_sql}
        ORDER BY datetime(timestamp) DESC
        LIMIT 1
        """,
        params,
    ).fetchone()
    return int(row[0] or 0) if row is not None else None


def _btc_source_health(
    conn: sqlite3.Connection,
    *,
    cutoff: datetime | None,
    now: datetime,
) -> list[dict[str, Any]]:
    if not _table_exists(conn, "btc_prices"):
        return []
    where_sql, params = _recent_where("btc_prices", cutoff)
    rows = conn.execute(
        f"""
        SELECT
          source,
          COUNT(*) AS row_count,
          MIN(local_arrival_iso) AS first_local_arrival,
          MAX(local_arrival_iso) AS latest_local_arrival
        FROM btc_prices
        {where_sql}
        GROUP BY source
        ORDER BY source ASC
        """,
        params,
    ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        first = parse_timestamp(row["first_local_arrival"])
        latest = parse_timestamp(row["latest_local_arrival"])
        span_sec = (
            max(0.0, (latest - first).total_seconds())
            if first is not None and latest is not None
            else 0.0
        )
        row_count = int(row["row_count"] or 0)
        sample_age_sec = (
            round((now - latest).total_seconds(), 3)
            if latest is not None
            else None
        )
        rows_per_sec = (
            round(row_count / span_sec, 6)
            if span_sec > 0
            else (float(row_count) if row_count > 0 else 0.0)
        )
        source = str(row["source"] or "")
        result.append(
            {
                "source": source,
                "canonical": source == POLYMARKET_RTDS_CHAINLINK_SOURCE,
                "row_count": row_count,
                "first_local_arrival": row["first_local_arrival"],
                "latest_local_arrival": row["latest_local_arrival"],
                "sample_age_sec": sample_age_sec,
                "rows_per_sec": rows_per_sec,
            }
        )
    return result


def _feature_readiness(
    conn: sqlite3.Connection,
    *,
    cutoff: datetime | None,
) -> dict[str, Any]:
    if not _table_exists(conn, "features"):
        return {
            "total_feature_rows": 0,
            "feature_ready_rows": 0,
            "feature_ready_pct": 0.0,
            "gap_affected_rows": 0,
            "gap_affected_pct": 0.0,
            "latest_feature_timestamp": None,
        }
    where_sql, params = _recent_where("features", cutoff)
    row = conn.execute(
        f"""
        SELECT
          COUNT(*) AS total_rows,
          SUM(CASE WHEN COALESCE(feature_ready, 0) = 1 THEN 1 ELSE 0 END) AS ready_rows,
          SUM(CASE WHEN COALESCE(is_gap_affected, 0) = 1 THEN 1 ELSE 0 END) AS gap_rows,
          MAX(timestamp) AS latest_timestamp
        FROM features
        {where_sql}
        """,
        params,
    ).fetchone()
    total = int(row["total_rows"] or 0)
    ready = int(row["ready_rows"] or 0)
    gaps = int(row["gap_rows"] or 0)
    return {
        "total_feature_rows": total,
        "feature_ready_rows": ready,
        "feature_ready_pct": _pct(ready, total),
        "gap_affected_rows": gaps,
        "gap_affected_pct": _pct(gaps, total),
        "latest_feature_timestamp": row["latest_timestamp"],
    }


def _feature_quality_counts(
    conn: sqlite3.Connection,
    *,
    cutoff: datetime | None,
) -> list[dict[str, Any]]:
    if not _table_exists(conn, "features"):
        return []
    where_sql, params = _recent_where("features", cutoff)
    rows = conn.execute(
        f"""
        SELECT
          COALESCE(snapshot_quality_status, 'unknown') AS snapshot_quality_status,
          COUNT(*) AS count
        FROM features
        {where_sql}
        GROUP BY COALESCE(snapshot_quality_status, 'unknown')
        ORDER BY count DESC, snapshot_quality_status ASC
        """,
        params,
    ).fetchall()
    return [
        {
            "snapshot_quality_status": str(row["snapshot_quality_status"]),
            "count": int(row["count"] or 0),
        }
        for row in rows
    ]


def _snapshot_coverage(
    conn: sqlite3.Connection,
    *,
    cutoff: datetime | None,
) -> dict[str, Any]:
    if not _table_exists(conn, "market_snapshots"):
        return {
            "total_snapshots": 0,
            "snapshots_with_orderbook": 0,
            "snapshots_with_trade_data": 0,
            "latest_snapshot_timestamp": None,
        }
    where_sql, params = _recent_where("market_snapshots", cutoff)
    row = conn.execute(
        f"""
        SELECT
          COUNT(*) AS total_rows,
          SUM(CASE WHEN COALESCE(has_orderbook, 0) = 1 THEN 1 ELSE 0 END) AS orderbook_rows,
          SUM(CASE WHEN COALESCE(has_trade_data, 0) = 1 THEN 1 ELSE 0 END) AS trade_rows,
          MAX(timestamp) AS latest_timestamp
        FROM market_snapshots
        {where_sql}
        """,
        params,
    ).fetchone()
    return {
        "total_snapshots": int(row["total_rows"] or 0),
        "snapshots_with_orderbook": int(row["orderbook_rows"] or 0),
        "snapshots_with_trade_data": int(row["trade_rows"] or 0),
        "latest_snapshot_timestamp": row["latest_timestamp"],
    }


def _per_market_coverage(
    conn: sqlite3.Connection,
    *,
    cutoff: datetime | None,
) -> list[dict[str, Any]]:
    if (
        not _table_exists(conn, "markets")
        or not _table_exists(conn, "market_snapshots")
        or not _table_exists(conn, "features")
    ):
        return []
    snapshot_where_sql, snapshot_params = _recent_where("market_snapshots", cutoff)
    feature_where_sql, feature_params = _recent_where("features", cutoff)
    rows = conn.execute(
        f"""
        WITH snapshot_counts AS (
          SELECT
            run_id,
            market_id,
            COUNT(*) AS snapshot_count,
            MIN(timestamp) AS first_snapshot,
            MAX(timestamp) AS latest_snapshot
          FROM market_snapshots
          {snapshot_where_sql}
          GROUP BY run_id, market_id
        ),
        feature_counts AS (
          SELECT
            run_id,
            market_id,
            COUNT(*) AS feature_count,
            SUM(CASE WHEN COALESCE(feature_ready, 0) = 1 THEN 1 ELSE 0 END)
              AS ready_feature_count,
            MAX(timestamp) AS latest_feature
          FROM features
          {feature_where_sql}
          GROUP BY run_id, market_id
        )
        SELECT
          m.market_id,
          m.question,
          m.start_time,
          m.close_time,
          COALESCE(s.snapshot_count, 0) AS snapshot_count,
          COALESCE(f.feature_count, 0) AS feature_count,
          COALESCE(f.ready_feature_count, 0) AS ready_feature_count,
          s.first_snapshot,
          s.latest_snapshot
        FROM markets m
        LEFT JOIN snapshot_counts s
          ON s.run_id = m.run_id
         AND s.market_id = m.market_id
        LEFT JOIN feature_counts f
          ON f.run_id = m.run_id
         AND f.market_id = m.market_id
        WHERE COALESCE(s.snapshot_count, 0) > 0
           OR COALESCE(f.feature_count, 0) > 0
        ORDER BY COALESCE(s.latest_snapshot, f.latest_feature, m.close_time, m.start_time) DESC
        LIMIT 50
        """,
        (*snapshot_params, *feature_params),
    ).fetchall()
    result = []
    for row in rows:
        feature_count = int(row["feature_count"] or 0)
        ready = int(row["ready_feature_count"] or 0)
        result.append(
            {
                "market_id": row["market_id"],
                "question": row["question"],
                "start_time": row["start_time"],
                "close_time": row["close_time"],
                "snapshot_count": int(row["snapshot_count"] or 0),
                "feature_count": feature_count,
                "ready_feature_count": ready,
                "ready_pct": _pct(ready, feature_count),
                "first_snapshot": row["first_snapshot"],
                "latest_snapshot": row["latest_snapshot"],
            }
        )
    return result


def _quality_warnings(
    *,
    btc_source_health: Sequence[dict[str, Any]],
    raw_ws_events_written: int | None,
    feature_readiness: dict[str, Any],
    snapshot_coverage: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    sources = {str(row.get("source")): row for row in btc_source_health}
    stale_sources = [
        source
        for source, row in sources.items()
        if _float_or_none(row.get("sample_age_sec")) is not None
        and float(row.get("sample_age_sec")) > 3.0
    ]
    for source in stale_sources:
        warnings.append(f"btc_source_stale source={source}")
    if {
        POLYMARKET_RTDS_CHAINLINK_SOURCE,
        POLYMARKET_RTDS_BINANCE_SOURCE,
    }.issubset(set(stale_sources)):
        warnings.append("both_btc_sources_stale")
    if POLYMARKET_RTDS_CHAINLINK_SOURCE not in sources:
        warnings.append("missing_chainlink_source")
    if POLYMARKET_RTDS_BINANCE_SOURCE not in sources:
        warnings.append("missing_binance_source")
    if int(raw_ws_events_written or 0) > 0:
        warnings.append(f"raw_ws_events_written_gt_zero value={raw_ws_events_written}")
    if float(feature_readiness.get("feature_ready_pct") or 0.0) < 50.0:
        warnings.append(
            "feature_ready_pct_below_50 "
            f"value={feature_readiness.get('feature_ready_pct')}"
        )
    if float(feature_readiness.get("gap_affected_pct") or 0.0) > 5.0:
        warnings.append(
            "gap_pct_above_5 "
            f"value={feature_readiness.get('gap_affected_pct')}"
        )
    if int(snapshot_coverage.get("total_snapshots") or 0) == 0:
        warnings.append("no_recent_snapshots")
    if int(feature_readiness.get("total_feature_rows") or 0) == 0:
        warnings.append("no_recent_features")
    return warnings


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * float(numerator) / float(denominator), 2)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "KEY_TABLES",
    "build_dataset_quality_report",
    "render_dataset_quality_report",
]
