from __future__ import annotations

import sqlite3
from typing import Any, Dict, Optional

from .db import run_sqlite_integrity_checks


def validate_db_quality(db_path: str, run_id: Optional[str] = None) -> dict[str, Any]:
    integrity = run_sqlite_integrity_checks(db_path)
    if not integrity.get("ok", False):
        return {
            "run_id": run_id,
            "issues": ["sqlite_integrity_failed"],
            "is_training_safe": 0,
            "integrity": integrity,
        }

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if run_id is None:
            row = conn.execute(
                """
                SELECT run_id
                FROM recorder_metrics
                WHERE run_id IS NOT NULL AND run_id <> ''
                ORDER BY timestamp DESC
                LIMIT 1
                """
            ).fetchone()
            run_id = str(row[0]) if row is not None else "legacy"

        params = (run_id,)
        strict_failed = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM markets
                WHERE run_id = ?
                  AND COALESCE(strict_validation_passed, 0) = 0
                """,
                params,
            ).fetchone()[0]
            or 0
        )
        snapshots_total = int(
            conn.execute(
                "SELECT COUNT(*) FROM market_snapshots WHERE run_id = ?",
                params,
            ).fetchone()[0]
            or 0
        )
        partial_books = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM market_snapshots
                WHERE run_id = ?
                  AND COALESCE(is_partial_orderbook, 0) = 1
                """,
                params,
            ).fetchone()[0]
            or 0
        )
        gap_rows = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM market_snapshots
                WHERE run_id = ?
                  AND COALESCE(is_gap_affected, 0) = 1
                """,
                params,
            ).fetchone()[0]
            or 0
        )
        feature_ready_rows = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM features
                WHERE run_id = ?
                  AND COALESCE(feature_ready, 0) = 1
                """,
                params,
            ).fetchone()[0]
            or 0
        )
        invalid_snapshot_trades = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM market_snapshots s
                JOIN markets m
                  ON m.run_id = s.run_id
                 AND m.market_id = s.market_id
                WHERE s.run_id = ?
                  AND s.last_trade_time IS NOT NULL
                  AND (
                        (m.start_time IS NOT NULL AND datetime(s.last_trade_time) < datetime(m.start_time))
                        OR datetime(s.last_trade_time) > datetime(s.timestamp)
                      )
                """,
                params,
            ).fetchone()[0]
            or 0
        )

        issues: list[str] = []
        if strict_failed > 0:
            issues.append("non_strict_markets_present")
        if partial_books > 0:
            issues.append("partial_orderbook_rows_present")
        if gap_rows > 0:
            issues.append("gap_affected_rows_present")
        if invalid_snapshot_trades > 0:
            issues.append("invalid_snapshot_trade_alignment")
        if snapshots_total > 0 and feature_ready_rows == 0:
            issues.append("no_feature_ready_rows")

        return {
            "run_id": run_id,
            "snapshots_total": snapshots_total,
            "partial_books": partial_books,
            "gap_affected_rows": gap_rows,
            "feature_ready_rows": feature_ready_rows,
            "strict_failed_markets": strict_failed,
            "invalid_snapshot_trade_rows": invalid_snapshot_trades,
            "issues": issues,
            "is_training_safe": 1 if not issues else 0,
        }
    finally:
        conn.close()
